"""
evaluate.py - metrics for model performance and for the detection method.

Model metrics (on a labelled test set with cue annotations):
  * AUROC, AUPRC, sensitivity @ fixed specificity, expected calibration error
  * worst-group AUROC = min over cue values of the within-group AUROC
    (a model that works regardless of the cue has a high worst-group AUROC)

Detection metrics (does the reliability score find the biased clients):
  * detection AUROC of r vs the ground-truth is_shortcut flag
  * Spearman(r, true coupling strength)
  * "all biased clients ranked below all clean clients" boolean
  * false-positive behaviour when no client is biased

All functions take plain numpy arrays / dicts so they are trivially testable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - surfaced clearly at call time
    roc_auc_score = None


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC that returns NaN when a group has a single class."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ======================================================================
# MODEL METRICS
# ======================================================================
def auroc(targets: np.ndarray, probs: np.ndarray) -> float:
    return _safe_auroc(targets, probs)


def auprc(targets: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(average_precision_score(targets, probs))


def sensitivity_at_specificity(targets: np.ndarray, probs: np.ndarray,
                               specificity: float = 0.90) -> float:
    """Max sensitivity achievable at >= the requested specificity."""
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        return float("nan")
    neg = probs[targets == 0]
    thr = np.quantile(neg, specificity)   # threshold giving `specificity` on negatives
    pred_pos = probs >= thr
    tp = np.sum(pred_pos & (targets == 1))
    fn = np.sum((~pred_pos) & (targets == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def expected_calibration_error(targets: np.ndarray, probs: np.ndarray,
                               n_bins: int = 15) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = targets[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def worst_group_auroc(targets: np.ndarray, probs: np.ndarray,
                      cue_values: np.ndarray,
                      min_group: int = 20) -> Dict[str, float]:
    """Within-group AUROC for each value of a single cue + the worst (min).

    `cue_values` may contain NaN (missing); those samples are dropped.
    Returns {'group_0':auroc, 'group_1':auroc, 'worst':min, 'gap':all-worst}.
    """
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    cue_values = np.asarray(cue_values, dtype=np.float64)
    valid = ~np.isnan(cue_values)
    out: Dict[str, float] = {}
    group_aucs = []
    for g in np.unique(cue_values[valid]):
        m = valid & (cue_values == g)
        if m.sum() < min_group:
            continue
        a = _safe_auroc(targets[m], probs[m])
        out[f"group_{int(g)}"] = a
        if not np.isnan(a):
            group_aucs.append(a)
    overall = _safe_auroc(targets[valid], probs[valid])
    out["overall"] = overall
    out["worst"] = float(np.min(group_aucs)) if group_aucs else float("nan")
    out["gap"] = (overall - out["worst"]) if group_aucs and not np.isnan(overall) else float("nan")
    return out


def model_report(result: Dict, cue_for_worstgroup: Optional[str] = None,
                 min_group: int = 20) -> Dict[str, float]:
    """Summarise a model.predict_proba() result dict into scalar metrics.

    If `cue_for_worstgroup` is given and present, worst-group AUROC is computed
    over that cue; otherwise worst-group is the min over every available cue.
    """
    y, p = result["targets"], result["probs"]
    rep = {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "sens@spec90": sensitivity_at_specificity(y, p, 0.90),
        "ece": expected_calibration_error(y, p),
        "n": int(len(y)),
    }
    cues = result.get("cues", {})
    if cue_for_worstgroup and cue_for_worstgroup in cues:
        wg = worst_group_auroc(y, p, cues[cue_for_worstgroup], min_group)
        rep["worst_group_auroc"] = wg["worst"]
        rep["worst_group_gap"] = wg["gap"]
        rep[f"wg_detail_{cue_for_worstgroup}"] = wg
    elif cues:
        worsts = []
        for name, cv in cues.items():
            wg = worst_group_auroc(y, p, cv, min_group)
            if not np.isnan(wg["worst"]):
                worsts.append(wg["worst"])
                rep[f"wg_{name}"] = wg["worst"]
        rep["worst_group_auroc"] = float(np.min(worsts)) if worsts else float("nan")
    return rep


# ======================================================================
# DETECTION METRICS  (method quality)
# ======================================================================
def detection_metrics(r_by_client: Dict[str, float],
                      is_shortcut: Dict[str, bool],
                      coupling: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """How well does the reliability score find the biased clients?

    Lower r should indicate a biased client, so we score is_shortcut with (1-r).
    """
    clients = sorted(r_by_client)
    r = np.array([r_by_client[c] for c in clients], dtype=np.float64)
    bias = np.array([1 if is_shortcut[c] else 0 for c in clients], dtype=np.int64)
    out: Dict[str, float] = {}

    if len(np.unique(bias)) == 2:
        out["detection_auroc"] = float(roc_auc_score(bias, 1.0 - r))
        clean_max = r[bias == 0].max()
        biased_max = r[bias == 1].max()
        out["all_biased_below_clean"] = bool(biased_max < r[bias == 0].min())
        out["margin"] = float(r[bias == 0].min() - biased_max)
    else:
        out["detection_auroc"] = float("nan")
        out["all_biased_below_clean"] = bool("nan")

    if coupling is not None:
        c = np.array([coupling[cl] for cl in clients], dtype=np.float64)
        if len(np.unique(c)) > 1:
            rho, _ = spearmanr(r, c)
            out["spearman_r_vs_coupling"] = float(rho)  # expect negative
    out["r_by_client"] = {cl: float(r_by_client[cl]) for cl in clients}
    return out


def false_positive_rate(r_by_client: Dict[str, float], r_min_flag: float) -> float:
    """Fraction of clients flagged (r below threshold) - used in the m=0 run."""
    vals = np.array(list(r_by_client.values()))
    return float(np.mean(vals < r_min_flag)) if len(vals) else float("nan")
