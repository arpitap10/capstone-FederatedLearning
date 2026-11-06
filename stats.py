"""
stats.py - statistics utilities for the paper's result tables (plan section 10).

Pure numpy / scipy / (pandas for the one table helper).  Nothing here touches
the GPU, reads an image, or has side effects on import, so it is safe to import
anywhere (a Colab cell, a worker, a unit test).

VARIANCE SOURCES
----------------
Every reported number varies with the random seed.  A single seed in this
codebase simultaneously perturbs THREE things:

  * model initialisation (the classifier head, and - if `pretrained=False` -
    the backbone);
  * client/minibatch ordering during local training (DataLoader shuffling and
    client visiting order);
  * the federation draw itself (which patients land in which hospital, which
    cue cells are populated - see build_federation.py's `seed`).

Where the experiment grid lets us hold one of these fixed while varying another
(e.g. re-drawing the federation with the model seed fixed, or re-seeding
training on a fixed federation) these are reported as SEPARATE variance
sources.  Where the grid only varies the top-level seed, the reported CI is the
*combined* variance of all three and is labelled as such in the table caption.
The functions below are agnostic to which source a given 1-D array of per-seed
values represents; the caller decides what each array means.

Two complementary tools are provided:

  * bootstrap CIs (`bootstrap_ci`, `paired_bootstrap_diff`, `summarize_runs`)
    quantify uncertainty of a statistic by resampling;
  * signed-rank tests (`wilcoxon_vs_baseline`) give a distribution-free
    hypothesis test of each method against a baseline across *paired* seeds,
    with Holm-Bonferroni control of the family-wise error rate and a
    nonparametric effect size.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

try:  # pandas is only needed by summarize_runs
    import pandas as pd
except Exception:  # pragma: no cover - surfaced clearly at call time
    pd = None

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:  # pragma: no cover - surfaced clearly at call time
    _scipy_wilcoxon = None


# ======================================================================
# BOOTSTRAP
# ======================================================================
def _as_rng(rng) -> np.random.Generator:
    """Coerce None / int / Generator into a numpy Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 agg: Callable[[np.ndarray], float] = np.mean,
                 rng=None) -> Dict[str, float]:
    """Percentile bootstrap CI of a statistic over a 1-D array.

    Resamples `values` with replacement `n_boot` times, applies `agg` to each
    resample, and returns the point estimate on the observed sample plus the
    (alpha/2, 1-alpha/2) percentile interval of the bootstrap distribution and
    its standard error (std of the bootstrap statistics).

    Returns {'mean','lo','hi','se'} where 'mean' is agg(values) (NaN-safe only
    insofar as `agg` is).  NaNs in `values` are dropped first.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "se": float("nan")}
    point = float(agg(v))
    if n == 1:
        return {"mean": point, "lo": point, "hi": point, "se": 0.0}
    gen = _as_rng(rng)
    idx = gen.integers(0, n, size=(n_boot, n))
    stats = np.array([float(agg(v[row])) for row in idx], dtype=np.float64)
    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(stats, ddof=1))
    return {"mean": point, "lo": lo, "hi": hi, "se": se}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 2000, alpha: float = 0.05,
                          rng=None) -> Dict[str, float]:
    """Bootstrap CI of the paired difference a - b.

    `a` and `b` must be aligned (element i is the same seed / federation draw
    for both methods); the pairing is what removes the shared across-seed
    variance.  We resample *pairs* (the same index for a and b) so the
    correlation between the two methods is preserved.

    Returns {'mean','lo','hi','se','p_gt'} where 'mean' is mean(a-b) on the
    observed sample and 'p_gt' is the fraction of bootstrap resamples whose mean
    difference is > 0 (i.e. the resampled evidence that a beats b).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be aligned, got {a.shape} vs {b.shape}")
    d = a - b
    mask = ~np.isnan(d)
    d = d[mask]
    n = d.size
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "p_gt": float("nan")}
    point = float(np.mean(d))
    if n == 1:
        return {"mean": point, "lo": point, "hi": point, "se": 0.0,
                "p_gt": float(point > 0)}
    gen = _as_rng(rng)
    idx = gen.integers(0, n, size=(n_boot, n))
    means = np.mean(d[idx], axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(means, ddof=1))
    p_gt = float(np.mean(means > 0.0))
    return {"mean": point, "lo": lo, "hi": hi, "se": se, "p_gt": p_gt}


# ======================================================================
# EFFECT SIZE
# ======================================================================
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta, a nonparametric effect size in [-1, 1].

    delta = P(a > b) - P(a < b) over all pairs.  +1 means every a exceeds every
    b, -1 the reverse, 0 full overlap.  Does not require pairing.  NaNs dropped.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    gt = float(np.sum(diff > 0))
    lt = float(np.sum(diff < 0))
    return (gt - lt) / (a.size * b.size)


def _rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from paired differences `d`.

    r = (sum of positive signed ranks - sum of negative signed ranks) /
        (sum of all signed ranks); zero differences are dropped (Wilcoxon
        convention).  Lies in [-1, 1]; +1 means every pair favours a.
    """
    d = d[d != 0.0]
    if d.size == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(d))
    r_plus = float(np.sum(ranks[d > 0]))
    r_minus = float(np.sum(ranks[d < 0]))
    total = r_plus + r_minus
    if total == 0:
        return float("nan")
    return (r_plus - r_minus) / total


# ======================================================================
# MULTIPLE-COMPARISON CORRECTION
# ======================================================================
def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down correction of a family of raw p-values.

    `pvals` maps a comparison name -> raw p.  Returns name -> corrected p, where
    corrected p is monotone non-decreasing in the sorted order and clipped to 1.
    NaN p-values are passed through unchanged (they do not consume a step).
    """
    items = [(name, p) for name, p in pvals.items() if not np.isnan(p)]
    m = len(items)
    out: Dict[str, float] = {name: p for name, p in pvals.items() if np.isnan(p)}
    if m == 0:
        return out
    items.sort(key=lambda kv: kv[1])
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = (m - rank) * p
        running = max(running, adj)          # enforce monotonicity
        out[name] = float(min(1.0, running))
    return out


# ======================================================================
# SIGNED-RANK TEST VS BASELINE
# ======================================================================
def wilcoxon_vs_baseline(method_scores: Dict[str, Sequence[float]],
                         baseline_key: str) -> Dict[str, Dict]:
    """Wilcoxon signed-rank test of each method against the baseline.

    `method_scores` maps a method name -> a 1-D array of per-seed metric values,
    ALIGNED across seeds (index i is the same seed for every method, which is
    what makes the test paired).  For each non-baseline method we run
    `scipy.stats.wilcoxon` on the paired differences (method - baseline),
    Holm-Bonferroni-correct the p-values across all such comparisons, and report
    an effect size (matched-pairs rank-biserial correlation, with Cliff's delta
    alongside) and the median paired difference.

    Returns name -> {statistic, p_raw, p_holm, significant, rank_biserial,
    cliffs_delta, median_diff, n}.  `significant` uses alpha=0.05 on the
    Holm-corrected p.
    """
    if _scipy_wilcoxon is None:
        raise RuntimeError("scipy.stats.wilcoxon is unavailable")
    if baseline_key not in method_scores:
        raise KeyError(f"baseline_key {baseline_key!r} not in method_scores")

    base = np.asarray(method_scores[baseline_key], dtype=np.float64)
    raw: Dict[str, float] = {}
    partial: Dict[str, Dict] = {}

    for name, scores in method_scores.items():
        if name == baseline_key:
            continue
        vals = np.asarray(scores, dtype=np.float64)
        if vals.shape != base.shape:
            raise ValueError(
                f"method {name!r} ({vals.shape}) not aligned with baseline "
                f"({base.shape})")
        d = vals - base
        mask = ~np.isnan(d)
        d = d[mask]
        rec: Dict = {
            "n": int(d.size),
            "median_diff": float(np.median(d)) if d.size else float("nan"),
            "cliffs_delta": cliffs_delta(vals[mask], base[mask]),
            "rank_biserial": _rank_biserial(d),
        }
        # Wilcoxon is undefined when every difference is zero / too few pairs.
        if d.size == 0 or np.all(d == 0.0):
            rec["statistic"] = float("nan")
            p = float("nan")
        else:
            try:
                stat, p = _scipy_wilcoxon(d, zero_method="wilcox",
                                          alternative="two-sided")
                rec["statistic"] = float(stat)
                p = float(p)
            except ValueError:
                rec["statistic"] = float("nan")
                p = float("nan")
        rec["p_raw"] = p
        raw[name] = p
        partial[name] = rec

    corrected = holm_bonferroni(raw)
    for name, rec in partial.items():
        ph = corrected.get(name, float("nan"))
        rec["p_holm"] = float(ph)
        rec["significant"] = bool((not np.isnan(ph)) and ph < 0.05)
    return partial


# ======================================================================
# HEADLINE TABLE HELPER
# ======================================================================
def summarize_runs(df, group_cols: Sequence[str], metric_cols: Sequence[str],
                   n_boot: int = 2000):
    """Group a tidy results DataFrame and return mean + bootstrap CI per metric.

    `df` has one row per seed/run.  For each unique combination of `group_cols`
    (e.g. method x protocol) and each metric in `metric_cols` we report the mean
    and the percentile bootstrap CI of the mean over the runs in that group.

    Returns a tidy DataFrame with the group columns plus, for every metric,
    `<metric>_mean`, `<metric>_lo`, `<metric>_hi`, `<metric>_se`, `<metric>_n`.
    """
    if pd is None:
        raise RuntimeError("pandas is required for summarize_runs")
    group_cols = list(group_cols)
    metric_cols = list(metric_cols)
    rows: List[Dict] = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in grouped:
        if not group_cols:
            key = ()
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict = {col: val for col, val in zip(group_cols, key)}
        for metric in metric_cols:
            vals = sub[metric].to_numpy(dtype=np.float64)
            ci = bootstrap_ci(vals, n_boot=n_boot)
            row[f"{metric}_mean"] = ci["mean"]
            row[f"{metric}_lo"] = ci["lo"]
            row[f"{metric}_hi"] = ci["hi"]
            row[f"{metric}_se"] = ci["se"]
            row[f"{metric}_n"] = int(np.sum(~np.isnan(vals)))
        rows.append(row)
    return pd.DataFrame(rows)
