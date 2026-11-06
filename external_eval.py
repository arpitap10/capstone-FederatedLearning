"""
external_eval.py - Tier-C independent-institution evaluation (plan 6.4 / E5).

Zero-shot-evaluate a trained global model on an EXTERNAL dataset to test whether
the robustness gains generalise beyond the simulated RSNA federation.

Hard caveats (stated in the paper):
  * RSNA Pneumonia is a 30k subset of NIH ChestX-ray14, so NIH ChestX-ray14 is
    NOT an independent dataset and must not be used here.
  * Candidate external sets (MIMIC-CXR, CheXpert, VinDr-CXR, PadChest) use
    different label definitions; harmonisation to the RSNA "pneumonia-like
    opacity" positive is a THREAT TO VALIDITY.  We document the mapping, drop
    "uncertain" labels in the primary analysis, and report a sensitivity analysis.
  * Access needs credentialing / data-use agreements; nothing is redistributed.
  * If no external set is obtainable, the real-world-generality claim is dropped.

The model is evaluated zero-shot (no fine-tuning).  Image-cue cutpoints fit on
RSNA do not transfer, so worst-group AUROC here uses only metadata cues that the
external set actually provides.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from cue_bank import CueBank
from evaluate import model_report
from hospital_dataset import _BaseFrameDataset, make_loader
from hospital_transforms import get_transform
from model import build_model, load_state, predict_proba
from utils import get_logger, save_json

LOG = get_logger()


# ======================================================================
# LABEL HARMONISATION  (documented, conservative)
# ======================================================================
# For each external dataset: findings mapped to RSNA positive (pneumonia-like
# opacity), and labels EXCLUDED from the primary analysis.  Extend as needed.
LABEL_HARMONIZATION: Dict[str, Dict] = {
    "mimic_cxr": {
        # CheXpert-labeller columns on MIMIC-CXR-JPG.
        "positive_if_any": ["Pneumonia", "Lung Opacity", "Consolidation"],
        "negative_if": "No Finding",
        "exclude_uncertain": True,     # label == -1 dropped in primary analysis
        "caveat": "NLP-derived labels; 'Pneumonia' and 'Lung Opacity' are not "
                  "identical to RSNA's radiologist opacity annotation.",
    },
    "chexpert": {
        "positive_if_any": ["Pneumonia", "Lung Opacity", "Consolidation"],
        "negative_if": "No Finding",
        "exclude_uncertain": True,
        "caveat": "Uncertain (-1) labels dropped primarily; sensitivity analysis "
                  "maps uncertain->positive and ->negative.",
    },
    "vindr_cxr": {
        "positive_if_any": ["Pneumonia", "Infiltration", "Consolidation"],
        "negative_if": "No finding",
        "exclude_uncertain": False,
        "caveat": "Radiologist findings; map local finding names to opacity.",
    },
    "padchest": {
        "positive_if_any": ["pneumonia", "infiltrates", "consolidation"],
        "negative_if": "normal",
        "exclude_uncertain": False,
        "caveat": "Spanish-language reports; label strings vary - verify mapping.",
    },
}


def harmonize_labels(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Add a 0/1 `Target` column from an external dataset's native labels."""
    if dataset_name not in LABEL_HARMONIZATION:
        raise ValueError(f"no harmonization rule for '{dataset_name}'. "
                         f"Known: {list(LABEL_HARMONIZATION)}")
    rule = LABEL_HARMONIZATION[dataset_name]
    pos_cols = [c for c in rule["positive_if_any"] if c in df.columns]
    if not pos_cols:
        raise KeyError(f"none of {rule['positive_if_any']} present in columns "
                       f"{list(df.columns)[:20]}... - check the external schema")
    pos = df[pos_cols].copy()
    if rule.get("exclude_uncertain"):
        # CheXpert convention: 1 positive, 0 negative, -1 uncertain, blank absent.
        uncertain = (pos == -1).any(axis=1)
        df = df[~uncertain].copy()
        pos = pos[~uncertain]
    target = (pos == 1).any(axis=1).astype(int)
    out = df.copy()
    out["Target"] = target.values
    LOG.info(f"[{dataset_name}] harmonized: {int(target.sum())} pos / "
             f"{len(target) - int(target.sum())} neg (cols {pos_cols}); "
             f"caveat: {rule['caveat']}")
    return out


def build_external_manifest(dataset_name: str, images_dir: str, labels_csv: str,
                            out_csv: str, path_col: str = "Path",
                            id_col: Optional[str] = None) -> pd.DataFrame:
    """Produce a standard manifest CSV (patientId, image_file, Target, +cues)."""
    df = pd.read_csv(labels_csv)
    df = harmonize_labels(df, dataset_name)
    if path_col not in df.columns:
        raise KeyError(f"expected image path column '{path_col}' in {labels_csv}")
    man = pd.DataFrame({
        "patientId": df[id_col].astype(str) if (id_col and id_col in df) else
                     df[path_col].astype(str),
        "image_file": df[path_col].astype(str),
        "Target": df["Target"].astype(int),
    })
    # carry any available cue columns
    for src, dst in (("ViewPosition", "view"), ("view", "view"),
                     ("PatientSex", "sex"), ("sex", "sex"),
                     ("PatientAge", "age"), ("age", "age")):
        if src in df.columns and dst not in man.columns:
            man[dst] = df[src].values
    for col in ("view", "sex", "age", "age_band", "class", "nih_patient_id"):
        if col not in man.columns:
            man[col] = np.nan
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    man.to_csv(out_csv, index=False)
    LOG.info(f"wrote external manifest ({len(man)} rows) -> {out_csv}")
    return man


# ======================================================================
# DATASET WITH A CUSTOM IMAGE DIRECTORY
# ======================================================================
class ExternalCsvDataset(_BaseFrameDataset):
    """Like CsvImageDataset but resolves each row's image under `images_dir`
    using an explicit `image_file` column (external filenames differ from RSNA)."""

    def __init__(self, csv_path: str, images_dir: str, file_col: str = "image_file"):
        df = pd.read_csv(csv_path, dtype={"patientId": str})
        super().__init__(df, train=False)
        self.images_dir = images_dir
        self.file_col = file_col
        self.files = self.df[file_col].astype(str).tolist()

    def __getitem__(self, idx):
        path = os.path.join(self.images_dir, self.files[idx])
        img = self.transform(Image.open(path).convert("L"))
        target = float(self.targets[idx])
        from hospital_dataset import _METADATA_CUE_NAMES
        cues = {name: int(self.cue_arrays[name][idx]) for name in _METADATA_CUE_NAMES}
        return img, target, cues, self.ids[idx]


# ======================================================================
# EVALUATION
# ======================================================================
def zero_shot_evaluate(model_state_path: str, external_csv: str, images_dir: str,
                       device=None, norm_layer: str = "batchnorm",
                       batch_size: int = 32) -> Dict:
    """Load a saved global_model.pt and evaluate it zero-shot on an external set."""
    from utils import get_device
    device = device or get_device()
    model = build_model(pretrained=False, norm_layer=norm_layer).to(device)
    state = torch.load(model_state_path, map_location=device)
    load_state(model, state)

    ds = ExternalCsvDataset(external_csv, images_dir)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    # metadata cues only (RSNA image-cue cutpoints do not transfer)
    res = predict_proba(model, loader, device, cue_bank=None)
    rep = model_report(res)
    rep["n"] = int(len(res["targets"]))
    rep["prevalence"] = float(np.mean(res["targets"])) if len(res["targets"]) else float("nan")
    out_json = model_state_path.replace(".pt", "_external_eval.json")
    save_json(rep, out_json)
    LOG.info(f"[zero-shot] {os.path.basename(model_state_path)} on "
             f"{os.path.basename(external_csv)}: AUROC={rep['auroc']:.3f} "
             f"wg={rep.get('worst_group_auroc', float('nan')):.3f} (n={rep['n']})")
    return rep


def evaluate_all_methods(runs_dir: str, external_csv: str, images_dir: str,
                         device=None) -> pd.DataFrame:
    """Zero-shot-evaluate every runs/<tag>/global_model.pt for the Tier-C table."""
    rows = []
    for tag in sorted(os.listdir(runs_dir)):
        mpath = os.path.join(runs_dir, tag, "global_model.pt")
        if not os.path.exists(mpath):
            continue
        method = tag.split("__")[1] if "__" in tag else "unknown"
        try:
            rep = zero_shot_evaluate(mpath, external_csv, images_dir, device)
            rows.append({"tag": tag, "method": method, "ext_auroc": rep["auroc"],
                         "ext_wg_auroc": rep.get("worst_group_auroc", float("nan")),
                         "n": rep["n"], "prevalence": rep["prevalence"]})
        except Exception as e:
            LOG.warning(f"skip {tag}: {e}")
    return pd.DataFrame(rows)


def measure_natural_cue_association(external_csv: str) -> Dict:
    """Report the natural cue-label association present in the external set.

    The method's premise is that such associations occur in the wild; this
    quantifies them (e.g. P(view=AP | Y=1) vs P(view=AP | Y=0))."""
    df = pd.read_csv(external_csv)
    out: Dict[str, Dict] = {}
    y = df["Target"].astype(int).to_numpy()
    for cue, mapper in (("view", {"AP": 1, "PA": 0}), ("sex", {"M": 1, "F": 0})):
        if cue not in df.columns:
            continue
        cv = df[cue].map(mapper).astype("float").to_numpy()
        valid = ~np.isnan(cv)
        if valid.sum() == 0:
            continue
        p1_pos = float(np.mean(cv[valid & (y == 1)] == 1)) if (valid & (y == 1)).sum() else float("nan")
        p1_neg = float(np.mean(cv[valid & (y == 0)] == 1)) if (valid & (y == 0)).sum() else float("nan")
        out[cue] = {"P(cue=1|Y=1)": p1_pos, "P(cue=1|Y=0)": p1_neg,
                    "association": p1_pos - p1_neg}
    LOG.info(f"natural cue associations in {os.path.basename(external_csv)}: {out}")
    return out
