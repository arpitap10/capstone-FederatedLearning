"""
cue_bank.py - the set of candidate shortcut cues the audit probes for.

Design rules (from the plan, section 6.2):
  * Cues are PRE-REGISTERED and client-agnostic.  Every cue uses a FIXED GLOBAL
    cutpoint, never a client-local median (a client-local cutpoint would leak the
    client's own distribution into the audit and make scores incomparable).
  * Two families:
      - metadata cues   : derived from DICOM header fields already in the
                          federation CSVs (view, sex, age band).
      - image cues      : derived from pixels with cutpoints fitted ONCE on a
                          reference pool and frozen to disk (outer-frame mean
                          intensity, global brightness).
  * A cue returns a binary value in {0, 1} or None when it is undefined for a
    sample (e.g. missing header).  None-valued samples are dropped per-cue by
    the audit's minimum-cell rule.
  * The bank is configurable so a sweep can HOLD OUT the cue a federation was
    actually coupled on - testing the method when the leakage signal is blind.

The image-cue cutpoints are stored in metadata/cue_cutpoints.json; call
`fit_image_cutpoints` once (build_federation does this) before auditing.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from utils import load_json, save_json, get_logger

LOG = get_logger()

CUTPOINTS_PATH = os.path.join(config.METADATA_DIR, "cue_cutpoints.json")

# The default active bank.  "frame_intensity" and "brightness" are image cues.
DEFAULT_BANK: List[str] = ["view", "sex", "age_band", "frame_intensity", "brightness"]

# Fixed metadata cutpoints (never client-local).
AGE_BAND_CHILD_MAX = 18       # age <= 18 -> child
AGE_BAND_ELDERLY_MIN = 65     # age >= 65 -> elderly
AGE_CUE_ELDERLY_MIN = 65      # the binary "age" cue = elderly vs not


# ======================================================================
# METADATA CUES  (row of patient_metadata.csv -> {0,1} or None)
# ======================================================================
def _cue_view(row) -> Optional[int]:
    v = str(row.get("view", "")).upper()
    if v == "AP":
        return 1
    if v == "PA":
        return 0
    return None


def _cue_sex(row) -> Optional[int]:
    s = str(row.get("sex", "")).upper()
    if s in ("M", "MALE"):
        return 1
    if s in ("F", "FEMALE"):
        return 0
    return None


def _cue_age_band(row) -> Optional[int]:
    """Binary cue: elderly (>=65) vs not.  (age_band column is the 3-way label.)"""
    age = row.get("age", None)
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if np.isnan(age):
        return None
    return int(age >= AGE_CUE_ELDERLY_MIN)


METADATA_CUES: Dict[str, Callable] = {
    "view": _cue_view,
    "sex": _cue_sex,
    "age_band": _cue_age_band,
}


def age_to_band(age) -> str:
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if np.isnan(age):
        return "unknown"
    if age <= AGE_BAND_CHILD_MAX:
        return "child"
    if age >= AGE_BAND_ELDERLY_MIN:
        return "elderly"
    return "adult"


# ======================================================================
# IMAGE CUES  (float feature from a HxW uint8/float image in [0,255])
# ======================================================================
def _feat_frame_intensity(img: np.ndarray) -> float:
    """Mean intensity of the outer 10% frame - captures collimation / border."""
    h, w = img.shape[:2]
    b = max(1, int(0.10 * min(h, w)))
    frame = np.concatenate([
        img[:b, :].ravel(), img[-b:, :].ravel(),
        img[:, :b].ravel(), img[:, -b:].ravel(),
    ])
    return float(frame.mean())


def _feat_brightness(img: np.ndarray) -> float:
    """Global mean intensity."""
    return float(np.asarray(img, dtype=np.float32).mean())


IMAGE_CUE_FEATURES: Dict[str, Callable[[np.ndarray], float]] = {
    "frame_intensity": _feat_frame_intensity,
    "brightness": _feat_brightness,
}


# ======================================================================
# FIXED CUTPOINTS FOR IMAGE CUES
# ======================================================================
def fit_image_cutpoints(image_arrays: List[np.ndarray],
                        quantile: float = 0.5) -> Dict[str, float]:
    """Compute and persist a global threshold for every image cue.

    `image_arrays` should come from a REFERENCE pool (e.g. the dev federation),
    not from any single client.  The threshold is the given quantile of the
    feature; value >= threshold -> cue 1.  Stored to CUTPOINTS_PATH.
    """
    cutpoints: Dict[str, float] = {}
    for name, feat in IMAGE_CUE_FEATURES.items():
        vals = np.array([feat(im) for im in image_arrays], dtype=np.float64)
        cutpoints[name] = float(np.quantile(vals, quantile))
    save_json(cutpoints, CUTPOINTS_PATH)
    LOG.info(f"[cue_bank] fitted image cutpoints on {len(image_arrays)} "
             f"reference images -> {CUTPOINTS_PATH}: {cutpoints}")
    return cutpoints


def load_image_cutpoints() -> Dict[str, float]:
    if not os.path.exists(CUTPOINTS_PATH):
        raise FileNotFoundError(
            f"{CUTPOINTS_PATH} not found. Run cue_bank.fit_image_cutpoints "
            "(build_federation does this) before auditing with image cues."
        )
    return load_json(CUTPOINTS_PATH)


# ======================================================================
# CUE BANK
# ======================================================================
class CueBank:
    """Evaluates a configurable set of cues on metadata rows and images."""

    def __init__(self, active: Optional[List[str]] = None,
                 image_cutpoints: Optional[Dict[str, float]] = None):
        self.active = list(active) if active is not None else list(DEFAULT_BANK)
        unknown = [c for c in self.active
                   if c not in METADATA_CUES and c not in IMAGE_CUE_FEATURES]
        if unknown:
            raise ValueError(f"unknown cues requested: {unknown}")
        self.metadata_cues = [c for c in self.active if c in METADATA_CUES]
        self.image_cues = [c for c in self.active if c in IMAGE_CUE_FEATURES]
        self._cutpoints = image_cutpoints
        if self.image_cues and self._cutpoints is None:
            self._cutpoints = load_image_cutpoints()

    # -- metadata --------------------------------------------------------
    def eval_metadata_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of binary (nullable Int) cue columns for df rows.

        Column names are prefixed `cue_` to avoid clashing with metadata columns.
        """
        out = {}
        for name in self.metadata_cues:
            fn = METADATA_CUES[name]
            out[f"cue_{name}"] = [fn(r) for _, r in df.iterrows()]
        res = pd.DataFrame(out, index=df.index)
        for c in res.columns:
            res[c] = res[c].astype("Int64")  # nullable int keeps None as <NA>
        return res

    def eval_metadata_row(self, row) -> Dict[str, Optional[int]]:
        return {name: METADATA_CUES[name](row) for name in self.metadata_cues}

    # -- images ----------------------------------------------------------
    def eval_image(self, img: np.ndarray) -> Dict[str, int]:
        """Binary image-cue values for a single HxW image in [0,255]."""
        out = {}
        for name in self.image_cues:
            val = IMAGE_CUE_FEATURES[name](img)
            out[name] = int(val >= self._cutpoints[name])
        return out

    def eval_image_batch(self, imgs: np.ndarray) -> Dict[str, np.ndarray]:
        """`imgs`: (N,H,W) in [0,255].  Returns name -> int array of length N."""
        out = {name: np.empty(len(imgs), dtype=np.int64) for name in self.image_cues}
        for i, im in enumerate(imgs):
            for name in self.image_cues:
                out[name][i] = int(IMAGE_CUE_FEATURES[name](im) >= self._cutpoints[name])
        return out

    def __repr__(self):
        return f"CueBank(active={self.active})"
