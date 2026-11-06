"""
aggregate.py - server-side aggregation rules (plan 8.4).

Every client contributes a (clipped, possibly noised) update
    Delta_i = state_i - global_state
and a scalar weight.  The new global state is
    global + sum_i w_i Delta_i / sum_i w_i .

Implemented:
  fedavg            w = n
  fedprox           w = n            (the proximal term lives in local_train)
  valperf           w = n * val_auroc           (clients self-report)
  fedar             w = n * softmax(val_auroc / T)   (FedAR-style performance)
  qffl              w = n * (1 - val_auroc)^q        (q-FedAvg-style fairness)
  update_agreement  AND-mask-style: keep coordinates whose update sign agrees
                    across >= tau of clients, zero the rest (then n-weighted)
  oracle_drop       drop is_shortcut clients, then fedavg   (uses ground truth)
  reliability       w = n * max(r, r_min)^gamma            (OURS)

A client dict must contain: state, delta (StateDict), n_samples, val_auroc,
r (float or None), is_shortcut (bool or None).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from model import state_zeros_like, state_axpy, state_add, clone_state, _is_float
from reliability import aggregation_weight


# ----------------------------------------------------------------------
# weight schedules
# ----------------------------------------------------------------------
def _n(clients):           return np.array([c["n_samples"] for c in clients], float)
def _val(clients):         return np.array([c.get("val_auroc", 0.5) for c in clients], float)


def compute_weights(method: str, clients: List[Dict], rcfg,
                    fedar_T: float = 0.05, qffl_q: float = 1.0) -> np.ndarray:
    n = _n(clients)
    if method in ("fedavg", "fedprox"):
        w = n
    elif method == "valperf":
        w = n * np.clip(_val(clients), 0.0, None)
    elif method == "fedar":
        a = _val(clients) / max(fedar_T, 1e-6)
        soft = np.exp(a - a.max())
        w = n * soft / soft.sum()
    elif method == "qffl":
        loss = np.clip(1.0 - _val(clients), 1e-6, None)
        w = n * (loss ** qffl_q)
    elif method == "oracle_drop":
        keep = np.array([0.0 if c.get("is_shortcut") else 1.0 for c in clients])
        w = n * keep
    elif method == "reliability":
        w = np.array([aggregation_weight(c["n_samples"],
                                         c.get("r", rcfg.r0), rcfg) for c in clients])
    else:
        raise ValueError(f"unknown aggregation method '{method}'")
    if w.sum() <= 0:
        w = n  # safety fallback (e.g. oracle dropped everyone)
    return w


# ----------------------------------------------------------------------
# weighted delta application
# ----------------------------------------------------------------------
def apply_weighted_deltas(global_state, clients: List[Dict], weights: np.ndarray):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    acc = state_zeros_like(global_state)
    for c, w in zip(clients, weights):
        acc = state_axpy(acc, float(w), c["delta"])
    return state_add(global_state, acc)


# ----------------------------------------------------------------------
# AND-mask-style update agreement
# ----------------------------------------------------------------------
def aggregate_update_agreement(global_state, clients: List[Dict],
                               tau: float = 0.6, weights: np.ndarray = None):
    """Keep each coordinate's n-weighted mean update only where the sign agrees
    across at least `tau` fraction of clients; zero disputed coordinates."""
    if weights is None:
        weights = _n(clients)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    out = clone_state(global_state)
    deltas = [c["delta"] for c in clients]
    for k in global_state:
        if not _is_float(global_state[k]):
            continue
        stack = torch.stack([d[k] for d in deltas], 0)        # (C, *shape)
        wmean = torch.einsum("c,c...->...", torch.tensor(weights, dtype=stack.dtype), stack)
        signs = torch.sign(stack)
        pos = (signs > 0).float().mean(0)
        neg = (signs < 0).float().mean(0)
        agree = torch.maximum(pos, neg) >= tau
        out[k] = global_state[k] + wmean * agree.to(wmean.dtype)
    return out


# ----------------------------------------------------------------------
# robust aggregation baselines (coordinate median / trimmed mean)
# ----------------------------------------------------------------------
# These are standard Byzantine-robust FL rules.  They down-weight *outlier*
# updates - but a shortcut-biased client's update is NOT an outlier (its gradient
# looks like everyone else's), so these should FAIL to remove the bias.  That
# contrast is the point: robust-FL != shortcut-aware.
def aggregate_coordinate_median(global_state, clients: List[Dict]):
    out = clone_state(global_state)
    deltas = [c["delta"] for c in clients]
    for k in global_state:
        if not _is_float(global_state[k]):
            continue
        stack = torch.stack([d[k] for d in deltas], 0)
        med = torch.median(stack, dim=0).values
        out[k] = global_state[k] + med
    return out


def aggregate_trimmed_mean(global_state, clients: List[Dict], trim: float = 0.1):
    out = clone_state(global_state)
    deltas = [c["delta"] for c in clients]
    C = len(deltas)
    k_trim = int(trim * C)
    for key in global_state:
        if not _is_float(global_state[key]):
            continue
        stack = torch.stack([d[key] for d in deltas], 0)
        s, _ = torch.sort(stack, dim=0)
        if k_trim > 0 and C - 2 * k_trim > 0:
            s = s[k_trim:C - k_trim]
        out[key] = global_state[key] + s.mean(0)
    return out


# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------
def aggregate(method: str, global_state, clients: List[Dict], rcfg,
              agreement_tau: float = 0.6, trim: float = 0.1):
    if method == "update_agreement":
        return aggregate_update_agreement(global_state, clients, agreement_tau)
    if method == "median":
        return aggregate_coordinate_median(global_state, clients)
    if method == "trimmed_mean":
        return aggregate_trimmed_mean(global_state, clients, trim)
    weights = compute_weights(method, clients, rcfg)
    return apply_weighted_deltas(global_state, clients, weights)
