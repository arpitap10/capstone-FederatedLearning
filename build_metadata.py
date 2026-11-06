"""
build_metadata.py - assemble per-patient metadata and a feasibility report.

Steps
-----
1. Read ViewPosition / PatientSex / PatientAge from every RSNA DICOM header
   (headers only, pixels skipped -> fast).
2. Join Target (one row per patient) from stage_2_train_labels.csv and the
   3-way class from stage_2_detailed_class_info.csv.
3. Join the RSNA->NIH mapping so every patient carries its NIH patient id
   (the prefix of `img_id`, e.g. "00012773_000.png" -> "00012773").  This is the
   identifier used for patient-level disjointness, because RSNA is a subset of
   NIH ChestX-ray14 and the same NIH patient can appear under several RSNA ids.
4. Write metadata/patient_metadata.csv.
5. Print and save a feasibility report: (Target x view), (Target x sex),
   (Target x age_band) cross-tabs and the largest matched-prevalence hospital
   achievable for each cue at each coupling strength.

Optionally pre-render DICOMs to JPG (`--render-jpg`) so later training is fast.

Usage
-----
    python build_metadata.py                 # metadata + feasibility report
    python build_metadata.py --render-jpg    # also write stage_2_train_images_jpg/
    python build_metadata.py --limit 500     # quick smoke test on 500 patients
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from config import ensure_dirs
from cue_bank import age_to_band
from utils import get_logger, load_json, save_json

LOG = get_logger()

_AGE_RE = re.compile(r"(\d+)")


# ----------------------------------------------------------------------
def _parse_age(raw) -> float:
    """DICOM PatientAge is e.g. '059Y'; return years as float, NaN if unknown."""
    if raw is None:
        return float("nan")
    s = str(raw)
    m = _AGE_RE.search(s)
    if not m:
        return float("nan")
    val = float(m.group(1))
    # Occasionally stored in months/days with a suffix; treat implausible as NaN.
    if "M" in s.upper() and val > 24:
        val = val / 12.0
    if val <= 0 or val > 120:
        return float("nan")
    return val


def load_rsna_to_nih() -> Dict[str, str]:
    """subset_img_id (RSNA patientId) -> NIH patient id (prefix of img_id)."""
    if not os.path.exists(config.RSNA_NIH_MAPPING_JSON):
        LOG.warning(f"mapping file not found: {config.RSNA_NIH_MAPPING_JSON}; "
                    "nih_patient_id will fall back to patientId (NO cross-id "
                    "disjointness). Place the 2018 mapping json to enable it.")
        return {}
    data = load_json(config.RSNA_NIH_MAPPING_JSON)
    mapping = {}
    for rec in data:
        rsna_id = str(rec["subset_img_id"])
        nih_id = str(rec["img_id"]).split("_")[0]
        mapping[rsna_id] = nih_id
    LOG.info(f"loaded RSNA->NIH mapping for {len(mapping)} images")
    return mapping


def read_dicom_headers(patient_ids: List[str],
                       limit: Optional[int] = None) -> pd.DataFrame:
    import pydicom

    if limit:
        patient_ids = patient_ids[:limit]
    rows = []
    n = len(patient_ids)
    for i, pid in enumerate(patient_ids):
        path = os.path.join(config.DICOM_DIR, f"{pid}.dcm")
        view = sex = None
        age = float("nan")
        if os.path.exists(path):
            try:
                d = pydicom.dcmread(path, stop_before_pixels=True)
                view = getattr(d, "ViewPosition", None)
                sex = getattr(d, "PatientSex", None)
                age = _parse_age(getattr(d, "PatientAge", None))
            except Exception as e:  # corrupt header -> leave as missing
                LOG.warning(f"header read failed for {pid}: {e}")
        rows.append({"patientId": pid, "view": view, "sex": sex, "age": age})
        if (i + 1) % 2000 == 0:
            LOG.info(f"  headers {i + 1}/{n}")
    return pd.DataFrame(rows)


def metadata_from_nih(patient_ids: List[str], mapping: Dict[str, str],
                      limit: Optional[int] = None) -> pd.DataFrame:
    """Get view / sex / age by joining NIH ChestX-ray14's Data_Entry_2017.csv.

    Used when the RSNA DICOMs are absent (only JPGs on disk).  The RSNA->NIH
    mapping gives each RSNA patientId its NIH `img_id` (= NIH "Image Index"),
    which keys the NIH metadata CSV.  NIH columns:
        Image Index, Patient Age, Patient Gender (M/F), View Position (AP/PA).
    """
    if not os.path.exists(config.NIH_DATA_ENTRY_CSV):
        raise FileNotFoundError(
            f"{config.NIH_DATA_ENTRY_CSV} not found. The RSNA DICOMs are absent "
            "(only JPGs), so cues must come from NIH ChestX-ray14 metadata. "
            "Download 'Data_Entry_2017.csv' from the NIH ChestX-ray14 dataset "
            "(e.g. Kaggle: nih-chest-xrays/data) and place it in the "
            "RSNA-pneumonia-dataset folder.")
    # need RSNA patientId -> NIH img_id; build the reverse of `mapping`
    # (mapping is rsna_id -> nih_patient_id prefix; we need the full img_id).
    nih_imgid = _rsna_to_nih_imgid()
    nih = pd.read_csv(config.NIH_DATA_ENTRY_CSV)
    nih = nih.rename(columns={
        "Image Index": "img_id", "Patient Age": "age",
        "Patient Gender": "sex", "View Position": "view"})
    nih = nih.drop_duplicates("img_id").set_index("img_id")

    if limit:
        patient_ids = patient_ids[:limit]
    rows = []
    missing = 0
    for pid in patient_ids:
        img_id = nih_imgid.get(pid)
        if img_id is not None and img_id in nih.index:
            r = nih.loc[img_id]
            view = str(r["view"]).upper() if pd.notna(r["view"]) else None
            sex = str(r["sex"]).upper()[:1] if pd.notna(r["sex"]) else None
            age = _parse_age(r["age"])
        else:
            view = sex = None
            age = float("nan")
            missing += 1
        rows.append({"patientId": pid, "view": view, "sex": sex, "age": age})
    if missing:
        LOG.warning(f"{missing}/{len(patient_ids)} RSNA ids had no NIH metadata match")
    return pd.DataFrame(rows)


def _rsna_to_nih_imgid() -> Dict[str, str]:
    """subset_img_id (RSNA patientId) -> full NIH img_id (e.g. 00012773_000.png)."""
    data = load_json(config.RSNA_NIH_MAPPING_JSON)
    return {str(rec["subset_img_id"]): str(rec["img_id"]) for rec in data}


def _dicoms_present() -> bool:
    if not os.path.isdir(config.DICOM_DIR):
        return False
    return any(f.endswith(".dcm") for f in os.listdir(config.DICOM_DIR)[:50])


def render_jpgs(patient_ids: List[str], limit: Optional[int] = None) -> None:
    """Pre-render DICOMs to 8-bit JPG for fast training (optional)."""
    import pydicom
    from PIL import Image

    os.makedirs(config.JPG_DIR, exist_ok=True)
    if limit:
        patient_ids = patient_ids[:limit]
    for i, pid in enumerate(patient_ids):
        out = os.path.join(config.JPG_DIR, f"{pid}.jpg")
        if os.path.exists(out):
            continue
        path = os.path.join(config.DICOM_DIR, f"{pid}.dcm")
        if not os.path.exists(path):
            continue
        d = pydicom.dcmread(path)
        arr = d.pixel_array.astype(np.float32)
        if str(getattr(d, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        Image.fromarray(arr.astype(np.uint8), mode="L").save(out, quality=95)
        if (i + 1) % 2000 == 0:
            LOG.info(f"  jpg {i + 1}/{len(patient_ids)}")


# ----------------------------------------------------------------------
def feasibility_report(df: pd.DataFrame,
                       couplings=(0.5, 0.6, 0.8, 0.9, 0.95),
                       prevalence: float = 0.40) -> pd.DataFrame:
    """Cross-tabs + the largest matched-prevalence hospital per (cue, coupling).

    For a binary cue, a coupling-c, prevalence-p hospital of size n needs:
        pos&cue1 = c*p*n     pos&cue0 = (1-c)*p*n
        neg&cue1 = (1-c)*(1-p)*n   neg&cue0 = c*(1-p)*n
    The achievable n is bounded by the scarcest of the four cells in the pool.
    This report assumes a SINGLE hospital drawing from the whole pool; the real
    builder splits the pool across hospitals + tests, so treat this as an upper
    bound to sanity-check that the roster is buildable.
    """
    LOG.info("=" * 60)
    LOG.info("FEASIBILITY REPORT")
    LOG.info("=" * 60)
    for cue, mapper in (("view", {"AP": 1, "PA": 0}),
                        ("sex", {"M": 1, "F": 0}),
                        ("age_band", None)):
        if cue == "age_band":
            cv = (df["age"] >= 65).astype("Int64")
            cv[df["age"].isna()] = pd.NA
        else:
            cv = df[cue].map(mapper).astype("Int64")
        ct = pd.crosstab(df["Target"], cv, dropna=False)
        LOG.info(f"\n[{cue}] Target x cue counts:\n{ct.to_string()}")

    rows = []
    for cue, mapper in (("view", {"AP": 1, "PA": 0}), ("sex", {"M": 1, "F": 0}),
                        ("age_band", None)):
        if cue == "age_band":
            cv = (df["age"] >= 65).astype("float")
            cv[df["age"].isna()] = np.nan
        else:
            cv = df[cue].map(mapper).astype("float")
        y = df["Target"].astype(int).to_numpy()
        valid = ~np.isnan(cv.to_numpy())
        yv, cvv = y[valid], cv.to_numpy()[valid]
        n_pos1 = int(np.sum((yv == 1) & (cvv == 1)))
        n_pos0 = int(np.sum((yv == 1) & (cvv == 0)))
        n_neg1 = int(np.sum((yv == 0) & (cvv == 1)))
        n_neg0 = int(np.sum((yv == 0) & (cvv == 0)))
        for c in couplings:
            p = prevalence
            # n bounded by each cell: cell_available / cell_fraction
            fracs = [c * p, (1 - c) * p, (1 - c) * (1 - p), c * (1 - p)]
            avail = [n_pos1, n_pos0, n_neg1, n_neg0]
            caps = [a / f if f > 0 else np.inf for a, f in zip(avail, fracs)]
            n_max = int(min(caps))
            rows.append({"cue": cue, "coupling": c, "prevalence": p,
                         "max_hospital_n": n_max,
                         "pos_cue1": n_pos1, "pos_cue0": n_pos0,
                         "neg_cue1": n_neg1, "neg_cue0": n_neg0})
    report = pd.DataFrame(rows)
    save_json({"generated_from": "build_metadata.py"}, os.path.join(config.METADATA_DIR, "_feasibility_meta.json"))
    report.to_csv(config.FEASIBILITY_CSV, index=False)
    LOG.info(f"\nmax single-hospital size (upper bound):\n{report.to_string(index=False)}")
    LOG.info(f"\nsaved feasibility report -> {config.FEASIBILITY_CSV}")
    return report


# ----------------------------------------------------------------------
def build(limit: Optional[int] = None, do_render: bool = False) -> pd.DataFrame:
    ensure_dirs()
    labels = pd.read_csv(config.LABELS_CSV, dtype={"patientId": str})
    labels = labels.drop_duplicates("patientId")[["patientId", "Target"]]
    cls = (pd.read_csv(config.CLASS_CSV, dtype={"patientId": str})
           .drop_duplicates("patientId")[["patientId", "class"]])

    patient_ids = labels["patientId"].astype(str).tolist()
    LOG.info(f"{len(patient_ids)} unique patients in labels csv")

    mapping = load_rsna_to_nih()

    # Choose the metadata source: DICOM headers if the DICOMs are on disk,
    # otherwise join NIH ChestX-ray14 metadata (JPG-only installs).
    if _dicoms_present():
        LOG.info("DICOMs present -> reading view/sex/age from headers")
        headers = read_dicom_headers(patient_ids, limit=limit)
    else:
        LOG.info("no DICOMs found -> deriving view/sex/age from NIH Data_Entry_2017.csv")
        headers = metadata_from_nih(patient_ids, mapping, limit=limit)

    df = headers.merge(labels, on="patientId", how="left").merge(cls, on="patientId", how="left")

    df["nih_patient_id"] = df["patientId"].map(mapping).fillna(df["patientId"])
    df["age_band"] = df["age"].map(age_to_band)

    df = df[["patientId", "nih_patient_id", "Target", "view", "sex", "age",
             "age_band", "class"]]
    df.to_csv(config.PATIENT_METADATA_CSV, index=False)
    LOG.info(f"wrote {len(df)} rows -> {config.PATIENT_METADATA_CSV}")

    n_nih = df["nih_patient_id"].nunique()
    LOG.info(f"distinct NIH patients: {n_nih} (vs {df['patientId'].nunique()} RSNA ids) "
             f"-> {df['patientId'].nunique() - n_nih} collisions matter for disjointness")

    feasibility_report(df.dropna(subset=["Target"]))

    if do_render:
        LOG.info("pre-rendering DICOM -> JPG ...")
        render_jpgs(patient_ids, limit=limit)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N patients (smoke test)")
    ap.add_argument("--render-jpg", action="store_true", help="also pre-render JPGs")
    args = ap.parse_args()
    build(limit=args.limit, do_render=args.render_jpg)


if __name__ == "__main__":
    main()
