"""
synthetic_probe.py - OPTIONAL appendix probe (plan section 17).  NOT load-bearing.

This is a thin wrapper around the pre-existing generator `synthetic_pneumonia_v2.py`
(at the project root, one level above src/), which Poisson-blends real RSNA opacity
crops into pneumonia-negative films and adds marker/text/tag/line/contrast artifact
variants.

Why this is only an appendix (see plan section 2):
  * The inserted opacities are image-processing composites whose realism was never
    validated by a radiologist; several looked smooth / rounded / mis-placed.
  * The probe can itself become a shortcut (insertion fingerprint).
  * Lesion diversity is limited (7 donor ROIs) and the set is small.

Therefore every metric here is reported ONLY as "change in predicted probability
to a SYNTHETIC opacity" and NEVER as "detects genuine pneumonia".  The main
results do not depend on this module.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import config
from evaluate import auroc
from external_eval import ExternalCsvDataset  # reused generic-path dataset
from hospital_dataset import make_loader
from model import build_model, load_state, predict_proba
from utils import get_logger, save_json

LOG = get_logger()

# project root (parent of src/) holds synthetic_pneumonia_v2.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLD_DIR = os.path.join(config.RSNA_DIR, "hospitals", "golden_server_set")
_SYN_DIR = os.path.join(_GOLD_DIR, "synthetic_pneumonia_50")


def _import_generator():
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    try:
        import synthetic_pneumonia_v2 as gen
        return gen
    except Exception as e:
        raise ImportError(
            f"could not import synthetic_pneumonia_v2 from {_PROJECT_ROOT}: {e}. "
            "This optional probe needs the original generator and its RSNA golden "
            "set CSVs; skip it for the main results.")


def generate_probe_set(mode: str = "generate", strength_range=(0.45, 0.55),
                       make_artifacts: bool = True) -> str:
    """Run the existing generator to produce the synthetic opacity + artifact set.

    Returns the output directory.  Requires the generator's golden-set CSVs to be
    present under the RSNA dataset; raises a clear error otherwise.
    """
    gen = _import_generator()
    try:
        if mode == "calibrate":
            gen.main(["calibrate"])
        else:
            lo, hi = strength_range
            gen.main(["generate", str(lo), str(hi)])
            if make_artifacts:
                gen.main(["artifacts"])
    except SystemExit as e:  # the generator uses SystemExit for guidance messages
        LOG.warning(f"generator exited: {e}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"generator could not find its golden-set inputs: {e}. The synthetic "
            "probe is optional; the main pipeline does not need it.")
    LOG.info(f"synthetic probe set at {_SYN_DIR}")
    return _SYN_DIR


def _build_probe_manifest(out_csv: str) -> pd.DataFrame:
    """Combine clean / synthetic-opacity / artifact images into one manifest with a
    `group` column and an `image_file` path relative to the synthetic dir."""
    rows: List[Dict] = []
    syn_csv = os.path.join(_SYN_DIR, "synthetic_pneumonia_50.csv")
    art_csv = os.path.join(_SYN_DIR, "artifacts.csv")

    if os.path.exists(syn_csv):
        syn = pd.read_csv(syn_csv, dtype={"patientId": str})
        for _, r in syn.iterrows():
            rows.append({"patientId": r["patientId"], "image_file": r["image_file"],
                         "group": "synthetic_opacity", "Target": 1})
            # the clean original of the same patient (RSNA golden copy)
            clean_rel = os.path.relpath(config.image_path(r["patientId"]), _SYN_DIR)
            rows.append({"patientId": r["patientId"], "image_file": clean_rel,
                         "group": "clean", "Target": 0})
    if os.path.exists(art_csv):
        art = pd.read_csv(art_csv, dtype={"patientId": str})
        for _, r in art.iterrows():
            rows.append({"patientId": r["patientId"], "image_file": r["image_file"],
                         "group": f"artifact_{r.get('kind', 'x')}", "Target": 0})

    man = pd.DataFrame(rows)
    for col in ("view", "sex", "age", "age_band", "class", "nih_patient_id"):
        man[col] = np.nan
    man.to_csv(out_csv, index=False)
    return man


def probe_sensitivity(model_state_path: str, probe_dir: Optional[str] = None,
                      device=None, batch_size: int = 32) -> Dict:
    """Measure the model's SYNTHETIC-opacity sensitivity (appendix metric).

    Returns mean predicted P(pneumonia) for clean vs synthetic-opacity vs artifact
    images and their deltas.  All keys say 'synthetic'; this is NOT a measure of
    genuine-pneumonia detection (see module docstring).
    """
    from utils import get_device
    device = device or get_device()
    probe_dir = probe_dir or _SYN_DIR
    man_csv = os.path.join(probe_dir, "probe_manifest.csv")
    man = _build_probe_manifest(man_csv)
    if len(man) == 0:
        raise FileNotFoundError(
            f"no probe images found under {probe_dir}; run generate_probe_set first.")

    model = build_model(pretrained=False).to(device)
    load_state(model, torch.load(model_state_path, map_location=device))
    ds = ExternalCsvDataset(man_csv, images_dir=probe_dir)
    loader = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    res = predict_proba(model, loader, device, cue_bank=None)

    groups = man["group"].to_numpy()
    probs = res["probs"]
    out: Dict[str, float] = {}
    for g in sorted(set(groups)):
        out[f"mean_prob__{g}"] = float(np.mean(probs[groups == g]))
    clean = out.get("mean_prob__clean", float("nan"))
    out["delta_synthetic_minus_clean"] = out.get("mean_prob__synthetic_opacity", float("nan")) - clean
    for g in sorted(set(groups)):
        if g.startswith("artifact_"):
            out[f"delta_{g}_minus_clean"] = out[f"mean_prob__{g}"] - clean
    # discrimination between clean and synthetic-opacity (NOT genuine pneumonia)
    mask = np.isin(groups, ["clean", "synthetic_opacity"])
    if mask.sum() > 0:
        yb = (groups[mask] == "synthetic_opacity").astype(int)
        out["synthetic_vs_clean_auroc"] = auroc(yb, probs[mask])
    out["_disclaimer"] = ("Measures response to SYNTHETIC opacities, not genuine "
                          "pneumonia detection. Appendix only (plan section 17).")
    save_json(out, model_state_path.replace(".pt", "_synthetic_probe.json"))
    LOG.info(f"[synthetic probe] {out}")
    return out
