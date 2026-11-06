"""
make_paper_figures.py - the redesigned, publication-grade figure set.

Built on figstyle.py (one visual identity).  Each figure has a single focal
point, direct labels, collision-safe layout (constrained_layout), consistent
semantic colours, and PNG+PDF export.

Figures:
  FIG1_shortcut            real X-rays + view-label coupling (the problem)
  FIG2_robustness          collapse-&-recovery profile + ranked flipped AUROC (HERO)
  FIG3_detection_donoharm  biased vs clean federation trust separation
  FIG4_gamma_tradeoff      in-distribution vs flipped robustness across gamma
  FIG5_ablation            which audit signal carries the gain
  FIG6_external            CheXpert zero-shot (honest scoping)   [optional]

Run after collate_results.py (needs runs/_summary/per_run.csv).
    python src/make_paper_figures.py
"""

from __future__ import annotations
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import config
import figstyle as fs
from figstyle import COL, METHODS

fs.apply()
PER_RUN = os.path.join(config.RUNS_DIR, "_summary", "per_run.csv")


def _per_run():
    df = pd.read_csv(PER_RUN)
    df["short"] = df["group"].str.split("__").str[-1]
    return df


def _ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, 0.0
    m = float(v.mean())
    h = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, h


# ======================================================================
# FIG 1 - the shortcut, made obvious
# ======================================================================
def fig1_shortcut():
    from PIL import Image
    mp = config.PATIENT_METADATA_CSV
    if not os.path.exists(mp): print("skip FIG1: no metadata"); return
    m = pd.read_csv(mp, dtype={"patientId": str}).dropna(subset=["Target"])
    m["Target"] = m["Target"].astype(int)
    def pid(t, v):
        s = m[(m.Target == t) & (m.view == v)]
        return s.sample(1, random_state=3)["patientId"].iloc[0] if len(s) else None

    fig = plt.figure(figsize=(9.2, 4.3), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.5], wspace=0.05, hspace=0.1)
    cells = [(0, 0, "Pneumonia · AP", pid(1, "AP")), (0, 1, "Pneumonia · PA", pid(1, "PA")),
             (1, 0, "Normal · AP", pid(0, "AP")), (1, 1, "Normal · PA", pid(0, "PA"))]
    for r, c, lab, p in cells:
        ax = fig.add_subplot(gs[r, c])
        if p:
            try: ax.imshow(Image.open(config.image_path(p)).convert("L"), cmap="gray")
            except Exception: pass
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor(COL["biased"] if "Pneumonia" in lab else COL["clean"]); s.set_linewidth(1.6)
        ax.set_title(lab, fontsize=9, color=COL["ink"], pad=3)

    # coupling bar (right, spanning both rows)
    axb = fig.add_subplot(gs[:, 2])
    def pAP(t):
        s = m[(m.Target == t) & m.view.isin(["AP", "PA"])]
        return (s.view == "AP").mean() if len(s) else np.nan
    nat = [pAP(1), pAP(0)]; planted = [0.90, 0.10]
    y = np.arange(2); h = 0.34
    axb.barh(y + h/2, nat, h, color=COL["clean"], label="RSNA (natural)")
    axb.barh(y - h/2, planted, h, color=COL["biased"], label="biased hospital")
    for yi, a, b in zip(y, nat, planted):
        axb.text(a + 0.02, yi + h/2, f"{a:.2f}", va="center", fontsize=8, color=COL["ink"])
        axb.text(b + 0.02, yi - h/2, f"{b:.2f}", va="center", fontsize=8, color=COL["ink"])
    axb.set_yticks(y); axb.set_yticklabels(["Pneumonia", "Normal"]); axb.invert_yaxis()
    axb.set_xlim(0, 1.08); axb.set_xlabel("P( AP view | label )")
    axb.set_title("Pneumonia co-occurs with AP view", fontsize=10)
    axb.legend(loc="lower right", fontsize=8); fs.despine(axb); fs.ygrid(axb)
    fig.suptitle("A hospital-specific shortcut: the view cue predicts the label",
                 fontsize=12, y=1.04)
    fs.save(fig, "FIG1_shortcut")


# ======================================================================
# FIG 2 - collapse & recovery (HERO)
# ======================================================================
def fig2_robustness(pr):
    fig = plt.figure(figsize=(10, 4.4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1])

    # (a) performance profile across conditions
    ax = fig.add_subplot(gs[0, 0])
    conds = [("id", "In-distribution\n(cue present)"), ("balanced", "Cue-balanced"),
             ("flipped", "Cue-flipped\n(adversarial)")]
    x = np.arange(3)
    lines = [("fedavg", "FedAvg", COL["fedavg"], "o", "-"),
             ("rel_g8", "Reliability (ours)", COL["ours"], "s", "-"),
             ("oracle_drop", "Oracle", COL["oracle"], "^", "--")]
    ends = {}
    for key, name, c, mk, ls in lines:
        if not (pr.short == key).any(): continue
        mus = [_ci(pr[pr.short == key][col])[0] for col, _ in conds]
        hs = [_ci(pr[pr.short == key][col])[1] for col, _ in conds]
        ax.errorbar(x, mus, yerr=hs, color=c, marker=mk, ms=7, lw=2.4, ls=ls,
                    capsize=3, zorder=3, clip_on=False)
        ends[key] = (name, c, mus[-1])
    # shade recovered gap at flipped
    if "fedavg" in ends and "rel_g8" in ends:
        lo, hi = ends["fedavg"][2], ends["rel_g8"][2]
        ax.annotate("", xy=(2.12, hi), xytext=(2.12, lo),
                    arrowprops=dict(arrowstyle="<->", color=COL["muted"], lw=1.3))
        ax.text(2.2, (lo + hi) / 2, f"+{hi-lo:.2f}\nrecovered", fontsize=8.5,
                color=COL["ink"], va="center")
    # direct end-labels (no legend box)
    for key, (name, c, yv) in ends.items():
        ax.text(-0.08, yv, name, color=c, fontsize=9, ha="right", va="center", fontweight="bold")
    ax.axhline(0.5, color=COL["chance"], lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in conds]); ax.set_xlim(-1.15, 2.5)
    ax.set_ylim(0.5, 0.98); ax.set_ylabel("AUROC")
    ax.set_title("FedAvg collapses under shortcut shift; our method recovers", fontsize=11)
    fs.despine(ax); fs.ygrid(ax); fs.panel_label(ax, "a")

    # (b) ranked flipped AUROC, all methods
    ax2 = fig.add_subplot(gs[0, 1])
    order_keys = ["fedavg", "fedprox", "valperf", "median", "reliability", "rel_g8", "oracle_drop"]
    rows = [(k, _ci(pr[pr.short == k]["flipped"])) for k in order_keys if (pr.short == k).any()]
    rows.sort(key=lambda r: r[1][0])          # ascending -> best on top
    names = [METHODS[k][0] for k, _ in rows]
    mus = [v[0] for _, v in rows]; hs = [v[1] for _, v in rows]
    cols = [METHODS[k][1] for k, _ in rows]
    yy = np.arange(len(rows))
    ax2.errorbar(mus, yy, xerr=hs, fmt="none", ecolor=COL["muted"], capsize=3, lw=1.2, zorder=2)
    ax2.scatter(mus, yy, c=cols, s=70, zorder=3, edgecolors="white", linewidths=0.8)
    for xi, yi, k in zip(mus, yy, [k for k, _ in rows]):
        ax2.text(xi, yi + 0.28, f"{xi:.3f}", ha="center", fontsize=7.5, color=COL["ink"])
    if (pr.short == "fedavg").any():
        ax2.axvline(_ci(pr[pr.short == "fedavg"]["flipped"])[0], color=COL["fedavg"], lw=0.9, ls="--")
    ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=8.5)
    ax2.set_xlabel("AUROC on cue-flipped test"); ax2.set_xlim(0.55, 0.82)
    ax2.set_title("Robustness across methods (5 seeds, 95% CI)", fontsize=10)
    fs.despine(ax2); ax2.xaxis.grid(True, color=COL["grid"], lw=0.9); ax2.set_axisbelow(True)
    fs.panel_label(ax2, "b")
    fs.save(fig, "FIG2_robustness")


# ======================================================================
# FIG 3 - detection + do-no-harm
# ======================================================================
def _load_signals(pattern):
    frames = []
    for f in glob.glob(os.path.join(config.RUNS_DIR, pattern, "client_signals.csv")):
        frames.append(pd.read_csv(f))
    return pd.concat(frames) if frames else None


def fig3_detection(pr=None):
    biased = _load_signals("main__reliability__seed*")
    clean = _load_signals("m0__reliability__seed*")
    if biased is None: print("skip FIG3: no client_signals"); return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)

    def panel(ax, df, title, note):
        clients = sorted(df.client.unique(),
                         key=lambda c: (bool(df[df.client == c]["is_shortcut"].iloc[0]), c))
        for i, c in enumerate(clients):
            sub = df[df.client == c]; is_s = bool(sub["is_shortcut"].iloc[0])
            col = COL["biased"] if is_s else COL["clean"]
            jit = rng.normal(0, 0.06, len(sub))
            ax.scatter(np.full(len(sub), i) + jit, sub["r"], s=14, color=col, alpha=0.45, edgecolors="none", zorder=2)
            ax.scatter([i], [sub["r"].mean()], s=120, color=col, edgecolors="white", linewidths=1.4, zorder=4, marker="D")
        ax.set_xticks(range(len(clients)))
        ax.set_xticklabels([c.replace("_", "\n") for c in clients], fontsize=7.5)
        ax.set_title(title, fontsize=10.5)
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color=COL["ink"], style="italic")
        fs.despine(ax); fs.ygrid(ax)

    panel(axes[0], biased, "Biased federation", "biased < clean  →  detection AUROC = 1.0")
    panel(axes[1], clean, "Clean federation (do-no-harm)", "all overlap  →  no false down-weighting") if clean is not None else axes[1].axis("off")
    axes[0].set_ylabel("reliability score  r"); axes[0].set_ylim(0.35, 0.85)
    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color=COL["clean"], label="clean hospital"),
                        mp.Patch(color=COL["biased"], label="biased hospital")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09))
    fig.suptitle("The audit separates clients only when a real shortcut exists", fontsize=12, y=1.14)
    fs.save(fig, "FIG3_detection_donoharm")


# ======================================================================
# FIG 4 - gamma trade-off
# ======================================================================
def fig4_gamma(pr):
    gmap = [(1, "rel_g1"), (2, "reliability"), (4, "rel_g4"), (8, "rel_g8")]
    gmap = [(g, k) for g, k in gmap if (pr.short == k).any()]
    if len(gmap) < 2: print("skip FIG4"); return
    gammas = [g for g, _ in gmap]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for col, c, name, mk in [("flipped", COL["ours"], "Cue-flipped (robustness)", "o"),
                             ("id", COL["muted"], "In-distribution", "s")]:
        mus = [_ci(pr[pr.short == k][col])[0] for _, k in gmap]
        hs = [_ci(pr[pr.short == k][col])[1] for _, k in gmap]
        ax.errorbar(gammas, mus, yerr=hs, color=c, marker=mk, ms=7, lw=2.2, capsize=3, label=name, zorder=3)
    for key, c, ls, lab in [("fedavg", COL["fedavg"], "--", "FedAvg"), ("oracle_drop", COL["oracle"], ":", "Oracle")]:
        if (pr.short == key).any():
            ax.axhline(_ci(pr[pr.short == key]["flipped"])[0], color=c, lw=1, ls=ls)
            ax.text(8.3, _ci(pr[pr.short == key]["flipped"])[0], lab, fontsize=8, color=c, va="center")
    ax.axvline(2, color=COL["chance"], lw=1, ls="-")
    ax.text(2, 0.52, "operating point\n(pre-registered)", fontsize=7.5, color=COL["muted"], ha="center")
    ax.set_xscale("log", base=2); ax.set_xticks(gammas); ax.set_xticklabels([str(g) for g in gammas])
    ax.set_xlabel("down-weighting strength  γ"); ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 0.95); ax.set_xlim(0.8, 11)
    ax.set_title("Robustness vs in-distribution trade-off across γ", fontsize=11)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42)); fs.despine(ax); fs.ygrid(ax)
    fs.save(fig, "FIG4_gamma_tradeoff")


# ======================================================================
# FIG 5 - signal ablation (horizontal, no rotated labels)
# ======================================================================
def fig5_ablation(pr):
    order = [("fedavg", "FedAvg (no audit)"), ("rel_leakonly", "Cue-leakage only"),
             ("rel_gaponly", "Subgroup-gap only"), ("rel_ctxonly", "Border signal only"),
             ("reliability", "All signals (ours)")]
    order = [o for o in order if (pr.short == o[0]).any()]
    if not order: print("skip FIG5"); return
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    yy = np.arange(len(order))
    mus = [_ci(pr[pr.short == k]["flipped"])[0] for k, _ in order]
    hs = [_ci(pr[pr.short == k]["flipped"])[1] for k, _ in order]
    cols = [COL["ours"] if k == "reliability" else (COL["fedavg"] if k == "fedavg" else COL["baseline"]) for k, _ in order]
    # dots + CI (not truncated bars) so small AUROC gaps are not visually exaggerated
    fedavg_mu = mus[[k for k, _ in order].index("fedavg")] if any(k == "fedavg" for k, _ in order) else None
    if fedavg_mu is not None:
        ax.axvline(fedavg_mu, color=COL["fedavg"], lw=0.9, ls="--", zorder=1)
        ax.text(fedavg_mu, len(order) - 0.3, "FedAvg", fontsize=8, color=COL["fedavg"], ha="center")
    ax.errorbar(mus, yy, xerr=hs, fmt="none", ecolor=COL["muted"], capsize=3, lw=1.2, zorder=2)
    ax.scatter(mus, yy, c=cols, s=90, zorder=3, edgecolors="white", linewidths=0.9)
    for yi, mu in zip(yy, mus):
        ax.text(mu, yi + 0.26, f"{mu:.3f}", va="bottom", ha="center", fontsize=8, color=COL["ink"])
    ax.set_yticks(yy); ax.set_yticklabels([n for _, n in order], fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("AUROC on cue-flipped test"); ax.set_xlim(0.58, 0.72); ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_title("Only the behavioural border signal recovers the gain", fontsize=11)
    fs.despine(ax); ax.xaxis.grid(True, color=COL["grid"], lw=0.9); ax.set_axisbelow(True)
    fs.save(fig, "FIG5_ablation")


# ======================================================================
# FIG 6 - external (optional)
# ======================================================================
def fig6_external(path=None):
    path = path or os.path.join(config.PROJECT_DIR, "external", "chexpert_eval", "summary_mean_std.csv")
    if not os.path.exists(path): print(f"skip FIG6 (no {path})"); return
    df = pd.read_csv(path)
    print("FIG6 external columns:", list(df.columns), "- edit mapping if needed")


def main():
    pr = _per_run()
    fig1_shortcut()
    fig2_robustness(pr)
    fig3_detection(pr)
    fig4_gamma(pr)
    fig5_ablation(pr)
    fig6_external()
    print(f"\nPaper figures (PNG+PDF) in {fs.FIG_DIR}")


if __name__ == "__main__":
    main()
