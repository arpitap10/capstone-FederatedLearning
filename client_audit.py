"""
client_audit.py - the behavioural audit a client runs LOCALLY on its private
validation set V_i after each local training phase (plan 8.2).  **Heart of the
method.**  Nothing here leaves the client except the final scalar signals.

Signals produced (each returned raw; reliability.py maps them to [0,1]):
  (a) leak  - label-conditional cue-leakage: within one true label, can the
              model's pneumonia score recover the cue?  ell = max_{c,y} 2|AUC-0.5|.
  (b) gap   - subgroup gap: overall AUROC - worst-group AUROC (max over cues).
  (c) ctx   - context reliance: border-only-image AUROC leakage (context_audit).

Minimum-cell rule: a cue x label cell must have at least `min_cell` samples of
EACH cue value, else that cue is skipped (and if no cue qualifies the client
ABSTAINS -> neutral prior).  A bootstrap gives an uncertainty band on ell.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from evaluate import auroc, worst_group_auroc
from context_audit import context_leak
from model import predict_proba


# ----------------------------------------------------------------------
# (a) label-conditional cue-leakage
# ----------------------------------------------------------------------
def _cue_leak_one(probs: np.ndarray, cue_vals: np.ndarray, targets: np.ndarray,
                  y: int, min_cell: int) -> Optional[float]:
    """2|AUROC(score -> cue) - 0.5| within the y-labelled subgroup, or None."""
    m = (targets == y) & ~np.isnan(cue_vals)
    cv = cue_vals[m]
    if cv.size == 0:
        return None
    n1 = int(np.sum(cv == 1))
    n0 = int(np.sum(cv == 0))
    if n1 < min_cell or n0 < min_cell:
        return None
    a = auroc(cv, probs[m])            # predict cue from the model's score
    if np.isnan(a):
        return None
    return float(2.0 * abs(a - 0.5))


def cue_leakage(result: Dict, active_cues: List[str], min_cell: int,
                n_bootstrap: int = 0, rng: Optional[np.random.Generator] = None
                ) -> Dict:
    """Max label-conditional leakage over active cues and labels.

    Returns {'ell': float|nan, 'per_cue': {cue: {y: L}}, 'n_cells': int,
             'ell_boot_hi': float}.  nan ell means abstain (no qualifying cell).
    """
    probs = result["probs"]
    targets = result["targets"]
    cues = result.get("cues", {})
    per_cue: Dict[str, Dict[int, float]] = {}
    vals: List[float] = []
    for c in active_cues:
        if c not in cues:
            continue
        cv = np.asarray(cues[c], dtype=np.float64)
        per_cue[c] = {}
        for y in (0, 1):
            L = _cue_leak_one(probs, cv, targets, y, min_cell)
            if L is not None:
                per_cue[c][y] = L
                vals.append(L)
    if not vals:
        return {"ell": float("nan"), "per_cue": per_cue, "n_cells": 0,
                "ell_boot_hi": float("nan")}
    ell = float(np.max(vals))

    ell_boot_hi = float("nan")
    if n_bootstrap and rng is not None:
        boots = []
        n = len(probs)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            rb = {"probs": probs[idx], "targets": targets[idx],
                  "cues": {c: np.asarray(cues[c])[idx] for c in cues}}
            sub = cue_leakage(rb, active_cues, min_cell, 0)
            if not np.isnan(sub["ell"]):
                boots.append(sub["ell"])
        if boots:
            ell_boot_hi = float(np.quantile(boots, 0.95))
    return {"ell": ell, "per_cue": per_cue, "n_cells": len(vals),
            "ell_boot_hi": ell_boot_hi}


# ----------------------------------------------------------------------
# (b) subgroup gap
# ----------------------------------------------------------------------
def subgroup_gap(result: Dict, active_cues: List[str], min_group: int) -> Dict:
    probs, targets = result["probs"], result["targets"]
    cues = result.get("cues", {})
    overall = auroc(targets, probs)
    gaps, per_cue = [], {}
    for c in active_cues:
        if c not in cues:
            continue
        wg = worst_group_auroc(targets, probs, np.asarray(cues[c], dtype=np.float64),
                               min_group)
        if not np.isnan(wg["gap"]):
            per_cue[c] = wg["gap"]
            gaps.append(wg["gap"])
    G = float(np.max(gaps)) if gaps else float("nan")
    return {"gap": G, "overall_auroc": overall, "per_cue": per_cue}


# ----------------------------------------------------------------------
# full local audit
# ----------------------------------------------------------------------
def compute_local_signals(model, val_loader, cue_bank, device, rcfg) -> Dict:
    """Run the complete client-side audit on V_i.

    `rcfg` is a config.ReliabilityConfig.  Returns raw signals + diagnostics;
    reliability.fuse() turns these into r.  `tau` (transfer retention) is NOT
    computed here - it needs peer models and is injected by fed_train under P2.
    """
    rng = np.random.default_rng(0)
    result = predict_proba(model, val_loader, device, cue_bank=cue_bank)
    active = list(cue_bank.active)

    leak = cue_leakage(result, active, rcfg.min_cell, rcfg.n_bootstrap, rng)
    gap = subgroup_gap(result, active, rcfg.min_cell)
    ctx = context_leak(model, val_loader, device)

    y = result["targets"]
    n_eff = int(min((y == 1).sum(), (y == 0).sum()))
    # Abstain ONLY if NO signal could be computed. The old rule abstained whenever
    # cue-leakage was unavailable (common under tight coupling / small val), which
    # discarded the usable border + gap signals and silently reverted to FedAvg.
    abstain = bool(np.isnan(leak["ell"]) and np.isnan(gap["gap"]) and np.isnan(ctx))

    return {
        "leak": leak["ell"],
        "leak_boot_hi": leak["ell_boot_hi"],
        "gap": gap["gap"],
        "ctx": ctx,
        "tau": float("nan"),                 # filled by fed_train under P2
        "val_auroc": gap["overall_auroc"],   # self-reported perf (baseline uses this)
        "n_eff": n_eff,
        "abstain": abstain,
        "per_cue_leak": leak["per_cue"],
        "per_cue_gap": gap["per_cue"],
    }
