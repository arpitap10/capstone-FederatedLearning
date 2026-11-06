"""
reliability.py - fuse the client audit signals into a reliability score r in
[0,1] (higher = cleaner), with shrinkage, EMA smoothing and abstention
(plan 8.3).

Signal -> cleanliness mapping (each in [0,1]):
    s_leak = 1 - leak      s_gap = 1 - gap      s_ctx = 1 - ctx      s_tau = tau

Fusion is a weighted geometric mean (normalised by the exponent sum, which keeps
r on a stable [0,1] scale and reduces to the plan's Pi s_k^{a_k} up to a monotone
transform):

    r_hat = exp( sum_k a_k log s_k / sum_k a_k )

NaN signals are dropped (a missing signal is neutral, not penalising).  Small
samples are shrunk toward the prior r0; scores are EMA-smoothed across rounds; a
client that cannot form any audit cell ABSTAINS and gets r0.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from utils import clip01, shrink

_EPS = 1e-3


def _signals_to_clean(signals: Dict) -> Dict[str, float]:
    """Map raw audit signals to cleanliness scores in [0,1] (NaN preserved)."""
    def inv(x):
        return float("nan") if (x is None or np.isnan(x)) else clip01(1.0 - x)

    def ident(x):
        return float("nan") if (x is None or np.isnan(x)) else clip01(x)

    return {
        "leak": inv(signals.get("leak")),
        "gap": inv(signals.get("gap")),
        "ctx": inv(signals.get("ctx")),
        "tau": ident(signals.get("tau")),
    }


def fuse(signals: Dict, rcfg, prev_r: Optional[float] = None) -> Dict:
    """Return {'r', 'r_hat', 's', 'used'} for one client in one round."""
    s = _signals_to_clean(signals)
    weights = {"leak": rcfg.w_leak, "gap": rcfg.w_gap,
               "ctx": rcfg.w_ctx, "tau": rcfg.w_tau}

    if signals.get("abstain", False):
        r_hat = rcfg.r0
        used = []
    else:
        num, den, used = 0.0, 0.0, []
        for k, sv in s.items():
            a = weights[k]
            if a <= 0 or sv is None or np.isnan(sv):
                continue
            num += a * math.log(max(sv, _EPS))
            den += a
            used.append(k)
        r_hat = math.exp(num / den) if den > 0 else rcfg.r0

    # shrink toward prior for small validation sets
    r_shrunk = shrink(r_hat, signals.get("n_eff", 0), rcfg.shrink_kappa, rcfg.r0)

    # EMA across rounds
    if prev_r is not None and rcfg.ema > 0:
        r = rcfg.ema * prev_r + (1.0 - rcfg.ema) * r_shrunk
    else:
        r = r_shrunk

    return {"r": clip01(r), "r_hat": clip01(r_hat), "s": s, "used": used}


def aggregation_weight(n_i: int, r_i: float, rcfg) -> float:
    """w_i = n_i * max(r_i, r_min)^gamma  (plan 8.4)."""
    return float(n_i) * (max(r_i, rcfg.r_min) ** rcfg.gamma)
