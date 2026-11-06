"""
utils.py - small, dependency-light helpers shared across the codebase.

Seeding, device selection, JSON/CSV IO, a run logger, and a couple of numeric
helpers.  Nothing here imports torch at module load except lazily inside the
functions that need it, so `import utils` stays cheap.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

import numpy as np


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and (if available) torch.

    `deterministic=True` also sets cuDNN deterministic flags.  This slows
    training a little but is required for the reproducibility guarantees the
    paper claims.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer_gpu: bool = True):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def save_json(obj: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    """Make numpy scalars/arrays JSON-serialisable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serialisable")


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(name: str = "fedcxr", logfile: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextmanager
def timer(logger: logging.Logger, msg: str):
    t0 = time.time()
    logger.info(f"{msg} ...")
    yield
    logger.info(f"{msg} done in {time.time() - t0:.1f}s")


class AverageMeter:
    """Running mean of a scalar (e.g. loss)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, k: int = 1):
        self.sum += float(value) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.sum / self.n if self.n else 0.0


# ----------------------------------------------------------------------
# Numeric helpers
# ----------------------------------------------------------------------
def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def shrink(r_hat: float, n_eff: int, kappa: float, r0: float) -> float:
    """Shrink an estimate toward a prior r0 when the sample is small.

    lambda = n_eff / (n_eff + kappa); r = lambda*r_hat + (1-lambda)*r0.
    """
    lam = n_eff / (n_eff + kappa) if (n_eff + kappa) > 0 else 0.0
    return clip01(lam * r_hat + (1.0 - lam) * r0)
