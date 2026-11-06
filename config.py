"""
config.py - single source of truth for paths and default hyper-parameters.

Every other module imports from here.  Nothing in this file touches the GPU,
reads an image, or has side effects on import, so it is safe to import anywhere
(including from a Colab cell, a worker process, or a unit test).

PATHS
-----
`PROJECT_DIR` defaults to the Google-Drive location used in Colab but can be
overridden with an environment variable so the same code runs on a laptop or a
CI box:

    PROJECT_DIR=/tmp/capstone python build_metadata.py

Directory layout expected under PROJECT_DIR::

    PROJECT_DIR/
      RSNA-pneumonia-dataset/
        stage_2_train_images/            # original DICOMs (*.dcm)
        stage_2_train_images_jpg/        # optional pre-rendered JPGs (faster)
        stage_2_train_labels.csv
        stage_2_detailed_class_info.csv
        pneumonia-challenge-dataset-mappings_2018.json   # RSNA -> NIH mapping
      metadata/                          # written by build_metadata.py
      federations/                       # written by build_federation.py
      runs/                              # written by fed_train.py / experiments
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ======================================================================
# PATHS
# ======================================================================

PROJECT_DIR = os.environ.get(
    "PROJECT_DIR", "/content/drive/MyDrive/capstone_project"
)

RSNA_DIR = os.path.join(PROJECT_DIR, "RSNA-pneumonia-dataset")

# RSNA source files -----------------------------------------------------
DICOM_DIR = os.path.join(RSNA_DIR, "stage_2_train_images")
JPG_DIR = os.path.join(RSNA_DIR, "stage_2_train_images_jpg")
LABELS_CSV = os.path.join(RSNA_DIR, "stage_2_train_labels.csv")
CLASS_CSV = os.path.join(RSNA_DIR, "stage_2_detailed_class_info.csv")
RSNA_NIH_MAPPING_JSON = os.path.join(
    RSNA_DIR, "pneumonia-challenge-dataset-mappings_2018.json"
)
# NIH ChestX-ray14 metadata CSV (Image Index, Patient Age/Gender, View Position).
# Used to obtain the cues (view/sex/age) when the RSNA DICOMs are not available
# and only pre-rendered JPGs are on disk.  Joined to RSNA via the mapping json.
NIH_DATA_ENTRY_CSV = os.path.join(RSNA_DIR, "Data_Entry_2017.csv")

# Derived / output directories -----------------------------------------
METADATA_DIR = os.path.join(PROJECT_DIR, "metadata")
PATIENT_METADATA_CSV = os.path.join(METADATA_DIR, "patient_metadata.csv")
FEASIBILITY_CSV = os.path.join(METADATA_DIR, "feasibility_report.csv")

FEDERATIONS_DIR = os.path.join(PROJECT_DIR, "federations")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "external")  # Tier-C datasets

# Image resolution the model consumes.
IMG_SIZE = 224
# ImageNet statistics (DenseNet-121 pretrained on ImageNet expects these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    """Create the output directories.  Call once from a script's main()."""
    for d in (METADATA_DIR, FEDERATIONS_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)


def image_path(patient_id: str) -> str:
    """Return the on-disk path for a patientId, preferring a pre-rendered JPG.

    build_metadata.py can pre-render DICOMs to JPG for speed; if that has not
    been done we fall back to the DICOM and the dataset loader will decode it.
    """
    jpg = os.path.join(JPG_DIR, f"{patient_id}.jpg")
    if os.path.exists(jpg):
        return jpg
    return os.path.join(DICOM_DIR, f"{patient_id}.dcm")


# ======================================================================
# FEDERATION / EXPERIMENT DEFAULTS
# ======================================================================

HOSPITALS_DEFAULT: List[str] = ["H1_general", "H2_viewcue", "H3_ICU", "H4_democue"]

# Which hospitals carry a planted shortcut, and on which cue, at the default
# roster.  `is_shortcut` is ground truth: used ONLY by the oracle baseline and
# to score the method, never by any reliability computation.
ROSTER_DEFAULT: Dict[str, Dict] = {
    "H1_general": {"cue": None, "coupling": 0.5, "is_shortcut": False},
    "H2_viewcue": {"cue": "view", "coupling": 0.90, "is_shortcut": True},
    "H3_ICU": {"cue": None, "coupling": 0.5, "is_shortcut": False},
    "H4_democue": {"cue": "sex", "coupling": 0.90, "is_shortcut": True},
}


@dataclass
class SplitConfig:
    """Controls build_federation.py."""

    hospitals: List[str] = field(default_factory=lambda: list(HOSPITALS_DEFAULT))
    roster: Dict[str, Dict] = field(default_factory=lambda: dict(ROSTER_DEFAULT))
    # Target (pneumonia-positive) prevalence, held equal across hospitals so
    # prevalence is never the confound under study.
    prevalence: float = 0.40
    # Sizes are constrained by the scarcest (label x cue) cell AND by NIH-patient
    # disjointness.  In RSNA the binding cell is positive+PA view (~1.3k images),
    # so defaults are deliberately modest; raise them if `--cells` shows headroom.
    # Disjointness unit is the NIH PATIENT; with group_by_patient=True a patient
    # is assigned wholly to one split but ALL its images are usable, so the budget
    # is image-level (RSNA ~2.3 imgs/patient).  The binding cell is positive+PA
    # view (~1.1-1.3k images in the main pool).  These sizes use ~70% of it ->
    # clean coupling with margin.  Run `--cells` to see the exact budget.
    group_by_patient: bool = True      # keep multiple images per patient (recovers ~2.5x data)
    # The naturally-confounded cue, balanced at 0.5 in every hospital that is not
    # itself biased on it (RSNA: pneumonia is naturally AP-dominant, so view must
    # be neutralised everywhere; sex is naturally ~balanced and left alone).  This
    # keeps hospitals off the scarce positive+female+PA joint cell.
    balance_cue: Optional[str] = "view"
    train_per_hospital: int = 540      # images per hospital train
    val_per_hospital: int = 135        # local validation V_i (never leaves client)
    test_size: int = 420               # each of test_id / test_balanced / test_flipped
    dev_fraction: float = 0.15         # fraction of PATIENTS reserved for the dev federation
    dev_scale: float = 0.20            # dev federation sizes = this * main sizes
    seed: int = 2026


@dataclass
class TrainConfig:
    """Controls local_train.py / fed_train.py."""

    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4        # L2 regularisation (raised for small clients)
    dropout: float = 0.2              # dropout before the single-logit head
    optimizer: str = "adam"            # "adam" | "sgd"
    momentum: float = 0.9             # sgd only
    fedprox_mu: float = 0.0           # >0 enables the FedProx proximal term
    pretrained: bool = True           # ImageNet init for DenseNet-121
    norm_layer: str = "batchnorm"     # "batchnorm" | "groupnorm" (record-level DP)
    amp: bool = True                  # mixed precision
    num_workers: int = 2
    grad_clip: Optional[float] = None
    # Early stopping on validation AUROC (0 = off, used by FedAvg which does fixed
    # short local epochs; >0 used by the local-only baseline to avoid overfitting).
    early_stop_patience: int = 0
    local_mitigation: bool = False    # audit-triggered reweighting (§8.5)
    mitigation_tau_leak: float = 0.30


@dataclass
class ReliabilityConfig:
    """Controls client_audit.py / reliability.py / aggregate.py."""

    # fusion exponents a_k for each signal (geometric mean); 0 disables a signal.
    w_leak: float = 1.0        # label-conditional cue-leakage  -> s_leak = 1 - l
    w_gap: float = 1.0         # subgroup gap                   -> s_gap  = 1 - G
    w_ctx: float = 1.0         # context reliance (border-only) -> s_ctx  = 1 - leak
    w_tau: float = 1.0         # transfer retention (P2 only)   -> s_tau  = tau
    # aggregation weighting w_i ~ n_i * max(r_i, r_min)^gamma
    gamma: float = 2.0
    r_min: float = 0.10        # floor so no hospital is silenced
    warmup_rounds: int = 3     # FedAvg rounds before scores are trusted
    ema: float = 0.5           # EMA smoothing of r across rounds (0 = no smoothing)
    # shrinkage r = lambda*r_hat + (1-lambda)*r0 with lambda = n_eff/(n_eff+kappa)
    shrink_kappa: float = 50.0
    r0: float = 0.5            # neutral prior for shrinkage / abstention
    # minimum samples per cue x label cell before a score is released
    min_cell: int = 20
    n_bootstrap: int = 200     # bootstrap resamples for worst-case score bounds


@dataclass
class PrivacyConfig:
    """Controls dp.py / protocols.py."""

    protocol: str = "P1"               # "P0" | "P1" | "P2" | "P3"
    secure_agg: bool = True            # simulate secure aggregation (summation)
    update_clip_C: Optional[float] = None   # L2 clip on Delta_i; None = off
    update_dp_sigma: float = 0.0       # Gaussian noise multiplier on the update sum
    score_dp_sigma: float = 0.0        # Gaussian noise on released scalar scores
    score_dp_delta: float = 1e-5
    report_noisy_max: bool = True      # release argmax cue via RNM, not every score


# A frozen registry of the dataclasses so run_experiments.py can build configs
# from YAML by name.
CONFIG_REGISTRY = {
    "split": SplitConfig,
    "train": TrainConfig,
    "reliability": ReliabilityConfig,
    "privacy": PrivacyConfig,
}


def config_to_dict(cfg) -> dict:
    """Serialise a dataclass config for logging to JSON."""
    return asdict(cfg)
