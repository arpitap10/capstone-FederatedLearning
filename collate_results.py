"""
collate_results.py - turn runs/ into paper-ready tables and figures.

Scans every runs/<tag>/ folder, averages the last `--last` rounds of
round_metrics.csv (kills round-to-round noise), groups runs that share a config
across seeds, and reports mean +/- 95% bootstrap CI per group plus a
Wilcoxon-vs-FedAvg significance test and a bar chart with error bars.

Group label = the run tag with a trailing `__seedN` stripped, so
    main__fedavg__seed0, main__fedavg__seed1, ...  ->  group "main__fedavg"
    dev__rel_locked (no seed suffix)                ->  group "dev__rel_locked"
Seed and method come from each run's summary.json.

Usage
-----
    python src/collate_results.py                      # all runs
    python src/collate_results.py --fed main           # only main-federation runs
    python src/collate_results.py --last 5 --out runs/_summary
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from utils import load_json, get_logger

LOG = get_logger()

# metrics we report, in display order
METRICS = ["test_flipped_auroc", "test_balanced_wg_auroc", "flip_drop",
           "test_id_auroc", "test_balanced_auroc"]
SHORT = {"test_flipped_auroc": "flipped", "test_balanced_wg_auroc": "bal_wg",
         "flip_drop": "flip_drop", "test_id_auroc": "id",
         "test_balanced_auroc": "balanced"}

_SEED_RE = re.compile(r"__seed\d+$")


def _group_label(tag: str) -> str:
    return _SEED_RE.sub("", tag)


def scan_runs(runs_dir: str, last: int = 5,
              fed_filter: Optional[str] = None) -> pd.DataFrame:
    """One row per run: group label, method, seed, fed, + last-`last`-round means."""
    rows = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*", ""))):
        rm = os.path.join(d, "round_metrics.csv")
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(rm):
            continue
        tag = os.path.basename(d.rstrip("/\\"))
        meta = load_json(sj) if os.path.exists(sj) else {}
        fed = meta.get("fed_name", tag.split("__")[0])
        if fed_filter and fed != fed_filter:
            continue
        m = pd.read_csv(rm).tail(last).mean(numeric_only=True)
        row = {"group": _group_label(tag), "tag": tag,
               "method": meta.get("method", "?"), "seed": meta.get("seed", -1),
               "fed": fed, "protocol": meta.get("protocol", "?")}
        for k in METRICS:
            row[SHORT[k]] = float(m[k]) if k in m else float("nan")
        rows.append(row)
    if not rows:
        LOG.warning(f"no runs with round_metrics.csv under {runs_dir}")
    return pd.DataFrame(rows)


def summarize(per_run: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Mean + 95% bootstrap CI per group across seeds."""
    try:
        from stats import bootstrap_ci
    except Exception:
        bootstrap_ci = None
    out = []
    for group, g in per_run.groupby("group"):
        rec = {"group": group, "method": g["method"].iloc[0],
               "fed": g["fed"].iloc[0], "n_seeds": len(g)}
        for short in SHORT.values():
            vals = g[short].dropna().to_numpy()
            if len(vals) == 0:
                rec[short] = float("nan")
                continue
            if bootstrap_ci is not None and len(vals) >= 2:
                ci = bootstrap_ci(vals, n_boot=n_boot)
                rec[short] = round(ci["mean"], 3)
                rec[f"{short}_lo"] = round(ci["lo"], 3)
                rec[f"{short}_hi"] = round(ci["hi"], 3)
            else:
                rec[short] = round(float(np.mean(vals)), 3)
        out.append(rec)
    return pd.DataFrame(out).sort_values(["fed", "method"]).reset_index(drop=True)


def wilcoxon_table(per_run: pd.DataFrame, fed: str,
                   baseline: str = "fedavg", metric: str = "flipped") -> Dict:
    """Wilcoxon signed-rank of each method vs FedAvg on one metric (paired by seed)."""
    try:
        from stats import wilcoxon_vs_baseline
    except Exception:
        return {}
    sub = per_run[per_run["fed"] == fed]
    # align per-seed metric arrays by GROUP (tag-based), not `method` - otherwise
    # all reliability variants (default + gamma sweep + ablations) would be pooled.
    by_group = {}
    for group, g in sub.groupby("group"):
        s = g.sort_values("seed")[metric].dropna().to_numpy()
        if len(s):
            by_group[group] = s
    # baseline group is the one ending in the baseline method name (e.g. main__fedavg)
    base_key = next((k for k in by_group if k.endswith(f"__{baseline}")), None)
    if base_key is None or len(by_group) < 2:
        return {}
    # keep only groups with the same number of seeds as the baseline (paired)
    nb = len(by_group[base_key])
    aligned = {k: v for k, v in by_group.items() if len(v) == nb}
    baseline = base_key
    if baseline not in aligned or len(aligned) < 2:
        return {}
    out = {}
    # paired t-test on per-seed differences (primary; no discrete p-floor)
    try:
        from scipy.stats import ttest_rel
        b = aligned[baseline]
        for m, v in aligned.items():
            if m == baseline:
                continue
            diff = v - b
            t, p = ttest_rel(v, b)
            out[m] = {"n": len(v), "mean_diff": float(np.mean(diff)),
                      "all_positive": bool(np.all(diff > 0)),
                      "t": float(t), "p_ttest": float(p)}
    except Exception as e:
        LOG.warning(f"t-test skipped: {e}")
    # Wilcoxon as nonparametric support
    try:
        w = wilcoxon_vs_baseline(aligned, baseline)
        for m in out:
            if m in w:
                out[m]["p_wilcoxon"] = w[m].get("p_raw")
    except Exception as e:
        LOG.warning(f"wilcoxon skipped: {e}")
    return out


def bar_plot(summary: pd.DataFrame, fed: str, out_path: str,
             metrics=("flipped", "bal_wg")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = summary[summary["fed"] == fed].copy()
    if sub.empty:
        return
    methods = sub["method"].tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        means = sub[metric].to_numpy()
        lo = sub.get(f"{metric}_lo", sub[metric]).to_numpy()
        hi = sub.get(f"{metric}_hi", sub[metric]).to_numpy()
        yerr = np.vstack([means - lo, hi - means])
        ax.bar(x, means, yerr=yerr, capsize=4, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{fed}: {metric}")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOG.info(f"saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=config.RUNS_DIR)
    ap.add_argument("--fed", default=None, help="only this federation (e.g. main)")
    ap.add_argument("--last", type=int, default=5, help="rounds to average per run")
    ap.add_argument("--out", default=None, help="output dir (default runs/_summary)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.runs, "_summary")
    os.makedirs(out_dir, exist_ok=True)

    per_run = scan_runs(args.runs, last=args.last, fed_filter=args.fed)
    if per_run.empty:
        return
    per_run.to_csv(os.path.join(out_dir, "per_run.csv"), index=False)
    LOG.info("PER-RUN (last-%d-round means):\n%s" % (args.last, per_run.to_string(index=False)))

    summary = summarize(per_run)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    LOG.info("SUMMARY (mean [95%% CI] across seeds):\n%s" % summary.to_string(index=False))

    for fed in sorted(per_run["fed"].unique()):
        w = wilcoxon_table(per_run, fed, baseline="fedavg", metric="flipped")
        if w:
            LOG.info(f"[{fed}] Wilcoxon vs fedavg on flipped: {w}")
        bar_plot(summary, fed, os.path.join(out_dir, f"bars_{fed}.png"))

    LOG.info(f"saved tables + plots -> {out_dir}")


if __name__ == "__main__":
    main()
