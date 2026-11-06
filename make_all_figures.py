"""
make_all_figures.py - ONE script that regenerates every results figure for the paper
straight from runs/*/ (no dependency on the stale runs/_summary/per_run.csv).

Put it in src/ and run from Colab (CPU is fine):
    %cd /content/drive/MyDrive/capstone_project/src
    !python make_all_figures.py
Output: figures/P*.png + .pdf   (prefix P = "paper", so nothing old is overwritten)

Metric per run = mean of the LAST 5 rounds of round_metrics.csv (same as the paper / my analysis).
CIs = 95% t-interval over seeds.

Figures
  P1_all_methods_main      all methods, main federation: cue-flipped | balanced worst-group | in-dist
  P2_second_federation     draw 2 (FedAvg, q-FFL, Reliability g=2, g=8), same 3 metrics
  P3_both_draws_flipped    flipped AUROC of the key methods on draw 1 vs draw 2 side by side
  P4_method_heatmap        heatmap: methods x metrics (main federation)
  P5_round_curves          flipped AUROC vs round (FedAvg / q-FFL / Reliability / Oracle)
  P6_r_heatmap             reliability r per hospital x round (biased vs clean)
  P7_val_inflation         WHY q-FFL works: local validation AUROC of biased vs clean hospitals
  P8_signal_separation     the audit signals (leak, gap, ctx) per hospital
  P9_gamma_tradeoff        gamma sweep on both draws (flipped vs in-dist)
  P10_ablation             which signal carries the gain
  P11_coupling_sweep       flipped AUROC vs coupling strength c (FedAvg / q-FFL / Reliability)
Anything whose runs are missing is skipped with a message.
"""
from __future__ import annotations
import os, re, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    import config
    PROJECT = config.PROJECT_DIR
    RUNS = config.RUNS_DIR
except Exception:
    PROJECT = "/content/drive/MyDrive/capstone_project"
    RUNS = os.path.join(PROJECT, "runs")
OUT = os.path.join(PROJECT, "figures")
os.makedirs(OUT, exist_ok=True)

LAST_N = 5
SHOW_FEDAR = False            # FedAR scored below chance: fix the baseline before showing it
SHOW_UPDATE_AGREEMENT = False # only 2 seeds
COL = {"fedavg": "#404040", "fedprox": "#9E9E9E", "median": "#BDBDBD", "valperf": "#8C6BB1",
       "fedar": "#E69F00", "update_agreement": "#56B4E9", "qffl": "#D55E00",
       "reliability": "#2166AC", "rel_g8": "#08306B", "oracle_drop": "#1A9850",
       "biased": "#D6604D", "clean": "#2C7FB8", "ink": "#1A1A1A", "muted": "#6E6E6E", "grid": "#E9E9E9"}
NAME = {"fedavg": "FedAvg", "fedprox": "FedProx", "median": "Robust median", "valperf": "Val-weighted",
        "fedar": "FedAR", "update_agreement": "Update-agreement", "qffl": "q-FFL",
        "reliability": "Reliability γ=2 (ours)", "rel_g8": "Reliability γ=8 (ours)",
        "oracle_drop": "Oracle (drop)"}
METRICS = [("flipped", "Cue-flipped AUROC"), ("bal_wg", "Balanced worst-group AUROC"), ("id", "In-distribution AUROC")]

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.edgecolor": "#444444", "axes.grid": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------- data
def parse_tag(tag):
    """-> (federation, method, seed) or None. Handles main__x__seed0, E9_...__main_draw2__x__s1."""
    m = re.match(r"^(E9b?)_.*?__(main_draw2)__(\w+)__s(\d+)$", tag)
    if m:
        exp, fed, method, seed = m.groups()
        if exp == "E9b" and method == "reliability":
            method = "rel_g8"          # E9b is the gamma=8 run
        return fed, method, int(seed)
    m = re.match(r"^(\w+?)__([\w.]+)__seed(\d+)$", tag)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


def load_runs():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        tag = os.path.basename(d)
        p = parse_tag(tag)
        rm = os.path.join(d, "round_metrics.csv")
        if not p or not os.path.exists(rm):
            continue
        try:
            df = pd.read_csv(rm)
        except Exception:
            continue
        last = df.tail(LAST_N).mean(numeric_only=True)
        rows.append(dict(tag=tag, fed=p[0], method=p[1], seed=p[2], dir=d,
                         flipped=last.get("test_flipped_auroc", np.nan),
                         bal_wg=last.get("test_balanced_wg_auroc", np.nan),
                         id=last.get("test_id_auroc", np.nan),
                         balanced=last.get("test_balanced_auroc", np.nan)))
    return pd.DataFrame(rows)


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, 0.0
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def stat_of(R, fed, method, col):
    s = R[(R.fed == fed) & (R.method == method)][col]
    return ci(s) + (len(s),)


def dots(ax, R, fed, methods, col, title, xlim=None, sort=True):
    rows = []
    for k in methods:
        m, h, n = stat_of(R, fed, k, col)
        if n:
            rows.append((k, m, h, n))
    if not rows:
        ax.axis("off"); return False
    if sort:
        rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="none", ecolor="#888", capsize=3, lw=1.2, zorder=2)
    ax.scatter([r[1] for r in rows], y, c=[COL[r[0]] for r in rows], s=70, zorder=3, edgecolors="white", linewidths=.8)
    for yi, (k, m, h, n) in zip(y, rows):
        ax.text(m, yi + 0.3, f"{m:.3f}", ha="center", fontsize=7.5, color=COL["ink"])
    ax.set_yticks(y); ax.set_yticklabels([f"{NAME[r[0]]} (n={r[3]})" for r in rows], fontsize=8.5)
    fa = stat_of(R, fed, "fedavg", col)
    if fa[2]:
        ax.axvline(fa[0], color=COL["fedavg"], ls="--", lw=.9)
    ax.set_ylim(-0.6, len(rows) - 0.1 + 0.5)
    lo, hi = ax.get_xlim(); ax.set_xlim(lo - 0.01, hi + 0.012)
    ax.set_title(title, fontsize=10, pad=10)
    ax.xaxis.grid(True, color=COL["grid"]); ax.set_axisbelow(True); despine(ax)
    if xlim: ax.set_xlim(*xlim)
    return True


# ---------------------------------------------------------------- P1 / P2
def p1_all_methods(R):
    order = ["fedavg", "fedprox", "median", "valperf", "fedar", "update_agreement", "qffl",
             "reliability", "rel_g8", "oracle_drop"]
    order = [k for k in order if not (k == "fedar" and not SHOW_FEDAR) and not (k == "update_agreement" and not SHOW_UPDATE_AGREEMENT)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ok = False
    for ax, (c, t) in zip(axes, METRICS):
        ok |= dots(ax, R, "main", order, c, t)
    if not ok:
        print("skip P1"); plt.close(fig); return
    fig.suptitle("All methods, main federation (mean ± 95% CI over seeds; dashed = FedAvg)", fontsize=12)
    save(fig, "P1_all_methods_main")


def p2_second_federation(R):
    order = ["fedavg", "qffl", "reliability", "rel_g8"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), constrained_layout=True)
    ok = False
    for ax, (c, t) in zip(axes, METRICS):
        ok |= dots(ax, R, "main_draw2", order, c, t)
    if not ok:
        print("skip P2"); plt.close(fig); return
    fig.suptitle("Second independent federation draw (split seed 2027)", fontsize=12)
    save(fig, "P2_second_federation")


def p3_both_draws(R):
    keys = ["fedavg", "qffl", "reliability", "rel_g8"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    w = 0.36
    for ax, (c, t) in zip(axes, METRICS):
        x = np.arange(len(keys))
        for j, (fed, lab, al) in enumerate([("main", "Draw 1 (seed 2026)", 1.0), ("main_draw2", "Draw 2 (seed 2027)", 0.55)]):
            st = [stat_of(R, fed, k, c) for k in keys]
            ax.bar(x + (j - .5) * w, [s[0] if s[2] else np.nan for s in st], w,
                   yerr=[s[1] if s[2] else 0 for s in st], capsize=2,
                   color=[COL[k] for k in keys], alpha=al, label=lab, zorder=3,
                   edgecolor="white", hatch=None if j == 0 else "//")
        ax.set_xticks(x); ax.set_xticklabels([NAME[k].replace(" (ours)", "") for k in keys], rotation=20, ha="right", fontsize=8)
        lo = np.nanmin([stat_of(R, f, k, c)[0] if stat_of(R, f, k, c)[2] else np.nan
                        for f in ("main", "main_draw2") for k in keys])
        ax.set_ylim(max(0.4, lo - 0.08), None); ax.set_title(t, fontsize=10)
        ax.yaxis.grid(True, color=COL["grid"]); ax.set_axisbelow(True); despine(ax)
    axes[0].legend(["Draw 1 (solid)", "Draw 2 (hatched)"], fontsize=8)
    fig.suptitle("Replication across federation draws (bars truncated; read the numbers, not the heights)", fontsize=11)
    save(fig, "P3_both_draws")


# ---------------------------------------------------------------- P4 heatmap
def p4_heatmap(R):
    order = ["fedavg", "fedprox", "median", "valperf", "fedar", "update_agreement", "qffl",
             "reliability", "rel_g8", "oracle_drop"]
    order = [k for k in order if not (k == "fedar" and not SHOW_FEDAR) and not (k == "update_agreement" and not SHOW_UPDATE_AGREEMENT)]
    rows, lab = [], []
    for k in order:
        v = [stat_of(R, "main", k, c)[0] for c, _ in METRICS] + [stat_of(R, "main", k, "balanced")[0]]
        if not np.isnan(v).all():
            rows.append(v); lab.append(NAME[k])
    if not rows:
        print("skip P4"); return
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.8), constrained_layout=True)
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.35, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Cue-flipped", "Balanced\nworst-group", "In-dist", "Cue-balanced"], fontsize=9)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.75 else "black")
    fig.colorbar(im, ax=ax, label="AUROC", fraction=0.04)
    ax.set_title("Methods × test conditions (main federation)")
    save(fig, "P4_method_heatmap")


# ---------------------------------------------------------------- P5 rounds
def p5_round_curves(R):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    keys = ["fedavg", "qffl", "reliability", "rel_g8", "oracle_drop"]
    done = False
    for ax, (fed, t) in zip(axes, [("main", "Main federation"), ("main_draw2", "Second draw")]):
        for k in keys:
            sub = R[(R.fed == fed) & (R.method == k)]
            curves = []
            for d in sub.dir:
                try:
                    curves.append(pd.read_csv(os.path.join(d, "round_metrics.csv")).set_index("round")["test_flipped_auroc"])
                except Exception:
                    pass
            if not curves:
                continue
            C = pd.concat(curves, axis=1)
            m = C.mean(1); s = C.std(1).fillna(0)
            ax.plot(m.index, m.values, color=COL[k], lw=2, label=NAME[k])
            ax.fill_between(m.index, m - s, m + s, color=COL[k], alpha=.12)
            done = True
        ax.axhline(.5, color="#C0C0C0", ls=":", lw=.8)
        ax.set_xlabel("federated round"); ax.set_title(t, fontsize=10); despine(ax)
        ax.yaxis.grid(True, color=COL["grid"])
    axes[0].set_ylabel("cue-flipped AUROC"); axes[1].legend(fontsize=8)
    if not done:
        print("skip P5"); plt.close(fig); return
    save(fig, "P5_round_curves")


# ---------------------------------------------------------------- signals
def load_signals(R, fed, method):
    fr = []
    for _, r in R[(R.fed == fed) & (R.method == method)].iterrows():
        p = os.path.join(r.dir, "client_signals.csv")
        if os.path.exists(p):
            d = pd.read_csv(p); d["seed"] = r.seed; fr.append(d)
    return pd.concat(fr) if fr else None


def p6_r_heatmap(R):
    sigs = [("main", "reliability", "Main, γ=2"), ("main_draw2", "rel_g8", "Draw 2, γ=8"), ("m0", "reliability", "Clean federation (m0)")]
    sigs = [(f, m, t, load_signals(R, f, m)) for f, m, t in sigs]
    sigs = [s for s in sigs if s[3] is not None]
    if not sigs:
        print("skip P6"); return
    fig, axes = plt.subplots(1, len(sigs), figsize=(5.2 * len(sigs), 3.4), constrained_layout=True, squeeze=False)
    for ax, (f, m, t, df) in zip(axes[0], sigs):
        piv = df.pivot_table(index="client", columns="round", values="r")
        isb = {c: bool(df[df.client == c]["is_shortcut"].iloc[0]) for c in piv.index}
        piv = piv.loc[sorted(piv.index, key=lambda c: (isb[c], c))]
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(piv))); ax.set_yticklabels([c.split("_")[0] + (" ✗" if isb[c] else "") for c in piv.index])
        for tl, c in zip(ax.get_yticklabels(), piv.index):
            tl.set_color(COL["biased"] if isb[c] else COL["clean"])
        ax.set_xticks(range(0, piv.shape[1], 4)); ax.set_xticklabels(piv.columns[::4])
        ax.set_xlabel("round"); ax.set_title(t, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label="reliability r", fraction=0.025)
    save(fig, "P6_r_heatmap")


def p7_val_inflation(R):
    """Mechanism: biased hospitals' own validation AUROC is inflated, so q-FFL (which up-weights LOW-AUROC clients)
    ends up down-weighting the shortcut hospitals."""
    panels = [("main", "reliability", "Main"), ("main_draw2", "rel_g8", "Draw 2"), ("m0", "reliability", "Clean federation")]
    panels = [(f, m, t, load_signals(R, f, m)) for f, m, t in panels]
    panels = [p for p in panels if p[3] is not None and "val_auroc" in p[3]]
    if not panels:
        print("skip P7"); return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4), sharey=True, constrained_layout=True, squeeze=False)
    for ax, (f, m, t, df) in zip(axes[0], panels):
        d = df[df["round"] >= df["round"].max() - LAST_N + 1]
        cl = sorted(d.client.unique(), key=lambda c: (bool(d[d.client == c]["is_shortcut"].iloc[0]), c))
        for i, c in enumerate(cl):
            s = d[d.client == c]; b = bool(s["is_shortcut"].iloc[0])
            ax.scatter(np.full(len(s), i) + np.random.default_rng(i).normal(0, .06, len(s)), s["val_auroc"], s=14,
                       color=COL["biased"] if b else COL["clean"], alpha=.4, edgecolors="none")
            ax.scatter([i], [s["val_auroc"].mean()], s=110, marker="D", color=COL["biased"] if b else COL["clean"],
                       edgecolors="white", linewidths=1.3, zorder=4)
        ax.set_xticks(range(len(cl))); ax.set_xticklabels([c.split("_")[0] for c in cl])
        ax.set_title(t, fontsize=10); despine(ax); ax.yaxis.grid(True, color=COL["grid"])
    axes[0][0].set_ylabel("hospital's own validation AUROC (last rounds)")
    fig.suptitle("Shortcut hospitals look BETTER on their own validation set; performance-aware weighting exploits that", fontsize=11)
    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color=COL["clean"], label="clean"), mp.Patch(color=COL["biased"], label="biased")],
               loc="lower center", ncol=2, bbox_to_anchor=(.5, -.07))
    save(fig, "P7_val_inflation")


def p8_signals(R):
    df = load_signals(R, "main", "reliability")
    if df is None:
        print("skip P8"); return
    cols = [c for c in ["s_leak", "s_gap", "s_ctx"] if c in df]
    if not cols:
        print("skip P8 (no s_ columns)"); return
    d = df[df["round"] >= df["round"].max() - LAST_N + 1]
    cl = sorted(d.client.unique(), key=lambda c: (bool(d[d.client == c]["is_shortcut"].iloc[0]), c))
    M = np.array([[d[d.client == c][s].mean() for s in cols] for c in cl])
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([{"s_leak": "cue leakage", "s_gap": "subgroup gap", "s_ctx": "border-only\ncontext"}[c] for c in cols])
    ax.set_yticks(range(len(cl))); ax.set_yticklabels([c.split("_")[0] for c in cl])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] < .6 else "black")
    fig.colorbar(im, ax=ax, label="trust score (1 = clean)", fraction=.05)
    ax.set_title("Audit signals per hospital (main, γ=2)")
    save(fig, "P8_signal_separation")


# ---------------------------------------------------------------- gamma, ablation, coupling
def p9_gamma(R):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharey=True)
    ok = False
    for ax, (fed, gm, t) in zip(axes, [("main", [(1, "rel_g1"), (2, "reliability"), (4, "rel_g4"), (8, "rel_g8")], "Main federation"),
                                       ("main_draw2", [(2, "reliability"), (8, "rel_g8")], "Second draw")]):
        gm = [(g, k) for g, k in gm if stat_of(R, fed, k, "flipped")[2]]
        for col, c, lab, mk in [("flipped", COL["reliability"], "cue-flipped", "o"), ("id", COL["muted"], "in-distribution", "s")]:
            if gm:
                ax.errorbar([g for g, _ in gm], [stat_of(R, fed, k, col)[0] for _, k in gm],
                            yerr=[stat_of(R, fed, k, col)[1] for _, k in gm], color=c, marker=mk, ms=7, lw=2, capsize=3, label=lab)
                ok = True
        for k, lab, ls in [("fedavg", "FedAvg", "--"), ("qffl", "q-FFL", ":")]:
            if stat_of(R, fed, k, "flipped")[2]:
                ax.axhline(stat_of(R, fed, k, "flipped")[0], color=COL[k], ls=ls, lw=1.2)
                ax.text(8.4, stat_of(R, fed, k, "flipped")[0], f"{lab} (flipped)", fontsize=7.5, color=COL[k], va="center")
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlim(.8, 14); ax.set_xlabel("γ"); ax.set_title(t, fontsize=10); despine(ax); ax.yaxis.grid(True, color=COL["grid"])
    axes[0].set_ylabel("AUROC"); axes[0].legend(fontsize=8, loc="center left")
    if not ok:
        print("skip P9"); plt.close(fig); return
    save(fig, "P9_gamma_tradeoff")


def p10_ablation(R):
    order = [("fedavg", "FedAvg"), ("rel_leakonly", "Leakage only"), ("rel_gaponly", "Gap only"),
             ("rel_ctxonly", "Border only"), ("reliability", "All signals")]
    order = [o for o in order if stat_of(R, "main", o[0], "flipped")[2]]
    if len(order) < 2:
        print("skip P10"); return
    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    x = np.arange(len(order)); w = .26
    for j, ((c, lab), colr) in enumerate(zip(METRICS, ["#D55E00", "#0072B2", "#999999"])):
        st = [stat_of(R, "main", k, c) for k, _ in order]
        ax.bar(x + (j - 1) * w, [s[0] for s in st], w, yerr=[s[1] for s in st], capsize=2, color=colr, label=lab, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in order]); ax.set_ylim(.5, 1.0); ax.set_ylabel("AUROC")
    ax.set_title("Signal ablation (main federation; ablation variants have only 3 seeds)")
    ax.legend(fontsize=8, ncol=3, loc="upper center"); despine(ax); ax.yaxis.grid(True, color=COL["grid"]); ax.set_axisbelow(True)
    save(fig, "P10_ablation")


def p11_coupling(R):
    fed_c = {"c060": .60, "c075": .75, "c080": .80, "main": .90, "c095": .95}
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ok = False
    for k in ["fedavg", "qffl", "reliability", "rel_g8", "oracle_drop"]:
        cs, ms, hs = [], [], []
        for f, c in sorted(fed_c.items(), key=lambda t: t[1]):
            s = stat_of(R, f, k, "flipped")
            if s[2]:
                cs.append(c); ms.append(s[0]); hs.append(s[1])
        if len(cs) >= 2:
            ax.errorbar(cs, ms, yerr=hs, marker="o", lw=2, capsize=3, color=COL[k], label=NAME[k]); ok = True
    if not ok:
        print("skip P11"); plt.close(fig); return
    ax.axhline(.5, color="#C0C0C0", ls=":"); ax.set_xlabel("coupling strength c"); ax.set_ylabel("cue-flipped AUROC")
    ax.set_title("Robustness vs shortcut strength (points have 1–5 seeds)"); ax.legend(fontsize=8); despine(ax)
    ax.yaxis.grid(True, color=COL["grid"])
    save(fig, "P11_coupling_sweep")


# ---------------------------------------------------------------- stats table
def stats_table(R):
    out = []
    for fed in ["main", "main_draw2"]:
        for k in R[R.fed == fed].method.unique():
            row = dict(fed=fed, method=k, n=int((R[(R.fed == fed) & (R.method == k)]).shape[0]))
            for c, _ in METRICS:
                m, h, _ = stat_of(R, fed, k, c)
                row[c] = f"{m:.3f} ± {h:.3f}"
            out.append(row)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "P0_summary_table.csv"), index=False)
    print("saved P0_summary_table.csv")


def main():
    R = load_runs()
    if R.empty:
        print("no runs found under", RUNS); sys.exit(1)
    print(R.groupby(["fed", "method"]).size().to_string())
    for f in (stats_table, p1_all_methods, p2_second_federation, p3_both_draws, p4_heatmap, p5_round_curves,
              p6_r_heatmap, p7_val_inflation, p8_signals, p9_gamma, p10_ablation, p11_coupling):
        try:
            f(R)
        except Exception as e:
            print(f"!! {f.__name__} failed: {type(e).__name__}: {e}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
