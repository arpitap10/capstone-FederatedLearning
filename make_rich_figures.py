"""
make_rich_figures.py - the "polished" figure set (matches the inspiration deck):

  F1  shortcut_coupling_xrays.png   real RSNA X-rays grouped by (label x view) +
                                     the coupling bar  (the shortcut, made visible)
  F2  trust_heatmap.png             reliability r as a hospital x round heatmap
                                     (viridis), biased rows stay dark
  F3  robustness_vs_coupling.png    flipped-AUROC vs coupling strength c, per method
                                     (the dose-response line)
  F4  ablation_multimetric.png      grouped bars (flipped / balanced-wg / in-dist)
                                     per signal variant, with error bars
  F5  robustness_boxplots.png       per-method box plots of flipped AUROC across
                                     seeds, FedAvg/Oracle as reference bands

Colour-blind-safe Okabe-Ito palette; single axis; CIs; recessive grid.
Run after collate_results.py (needs runs/_summary/per_run.csv).

    python src/make_rich_figures.py
"""

from __future__ import annotations

import os, glob, json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import config
from utils import load_json

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.edgecolor": "#333333",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

FIG = os.path.join(config.PROJECT_DIR, "figures")
SUMMARY = os.path.join(config.RUNS_DIR, "_summary", "per_run.csv")
C = {"gray": "#999999", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
     "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "ink": "#222222",
     "muted": "#666666"}
METHOD = {"fedavg": ("FedAvg", C["gray"]), "fedprox": ("FedProx", C["orange"]),
          "valperf": ("Val-weighted", C["purple"]), "median": ("Robust median", C["sky"]),
          "reliability": ("Reliability γ=2", C["blue"]), "rel_g8": ("Reliability γ=8", C["verm"]),
          "oracle_drop": ("Oracle", C["green"])}
FED_COUPLING = {"c060": 0.60, "c080": 0.80, "main": 0.90, "c095": 0.95}


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("saved", p)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e4e4e4", lw=0.9); ax.set_axisbelow(True)


def _ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return np.nan, 0.0
    return v.mean(), (1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


# ======================================================================
# F1: shortcut coupling made visible with REAL X-rays
# ======================================================================
def fig_shortcut_xrays(n_per=1):
    from PIL import Image
    meta_p = config.PATIENT_METADATA_CSV
    if not os.path.exists(meta_p):
        print("no patient_metadata.csv; skipping F1"); return
    m = pd.read_csv(meta_p, dtype={"patientId": str}).dropna(subset=["Target"])
    m["Target"] = m["Target"].astype(int)
    def pick(tgt, view):
        s = m[(m.Target == tgt) & (m.view == view)]
        return s.sample(min(n_per, len(s)), random_state=1)["patientId"].tolist() if len(s) else []
    cells = [("Pneumonia", 1, "AP"), ("Pneumonia", 1, "PA"),
             ("Normal", 0, "AP"), ("Normal", 0, "PA")]

    fig = plt.figure(figsize=(11, 5.2))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 0.12, 1, 1.4], wspace=0.08, hspace=0.25)
    # left 2x2: pneumonia row (AP,PA), normal row (AP,PA)
    pos_ap, pos_pa, neg_ap, neg_pa = [pick(t, v) for _, t, v in cells]
    layout = [(0, 0, "Pneumonia · AP", pos_ap), (0, 1, "Pneumonia · PA", pos_pa),
              (1, 0, "Normal · AP", neg_ap), (1, 1, "Normal · PA", neg_pa)]
    for r, cidx, lab, pids in layout:
        ax = fig.add_subplot(gs[r, cidx])
        if pids:
            try:
                img = Image.open(config.image_path(pids[0])).convert("L")
                ax.imshow(img, cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "img n/a", ha="center")
        ax.set_title(lab, fontsize=9, color=C["ink"]); ax.axis("off")
    fig.text(0.30, 0.95, "What the model can cheat on", ha="center", fontsize=13, weight="bold")

    # right: the coupling bar P(AP | label) natural vs planted (0.9)
    ax2 = fig.add_subplot(gs[:, 4])
    # natural RSNA association from metadata
    def p_ap(tgt):
        s = m[(m.Target == tgt) & m.view.isin(["AP", "PA"])]
        return (s.view == "AP").mean() if len(s) else np.nan
    nat = [p_ap(1), p_ap(0)]
    planted = [0.90, 0.10]   # H2_viewcue planted coupling
    x = np.arange(2); w = 0.38
    ax2.bar(x - w / 2, nat, w, color=C["sky"], label="RSNA natural", zorder=3)
    ax2.bar(x + w / 2, planted, w, color=C["verm"], label="biased hospital (planted)", zorder=3)
    for xi, a, b in zip(x, nat, planted):
        ax2.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax2.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(["Pneumonia", "Normal"], fontsize=9)
    ax2.set_ylabel("P(AP view | label)", fontsize=10); ax2.set_ylim(0, 1.05)
    ax2.set_title("Pneumonia ↔ view coupling", fontsize=11, weight="bold")
    ax2.legend(frameon=False, fontsize=8, loc="upper right"); _clean(ax2)
    fig.text(0.5, 0.015, "Representative RSNA frontal X-rays. In the biased hospital, pneumonia is 90% AP-view — a model can predict the label from view instead of the lungs.",
             ha="center", fontsize=8.5, color=C["muted"])
    _save(fig, "F1_shortcut_coupling_xrays.png")


# ======================================================================
# F2: reliability r as a hospital x round heatmap
# ======================================================================
def fig_trust_heatmap():
    sig = None
    for s in range(5):
        p = os.path.join(config.RUNS_DIR, f"main__reliability__seed{s}", "client_signals.csv")
        if os.path.exists(p): sig = p; break
    if not sig:
        print("no client_signals.csv; skipping F2"); return
    df = pd.read_csv(sig)
    piv = df.pivot_table(index="client", columns="round", values="r")
    is_short = {c: bool(df[df.client == c]["is_shortcut"].iloc[0]) for c in piv.index}
    piv = piv.loc[sorted(piv.index, key=lambda c: (is_short[c], c))]  # clean first

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(0, piv.shape[1], max(1, piv.shape[1] // 10)))
    ax.set_xticklabels([piv.columns[i] for i in range(0, piv.shape[1], max(1, piv.shape[1] // 10))])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{c}  {'(biased)' if is_short[c] else '(clean)'}" for c in piv.index], fontsize=9)
    for i, c in enumerate(piv.index):
        ax.get_yticklabels()[i].set_color(C["verm"] if is_short[c] else C["green"])
    ax.set_xlabel("federated round"); ax.set_title("Reliability score r per hospital over rounds\n(biased hospitals stay dark = low trust)", weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("reliability  r")
    _save(fig, "F2_trust_heatmap.png")


# ======================================================================
# F3: robustness vs coupling strength (dose-response)
# ======================================================================
def _scan_runs():
    rows = []
    for d in sorted(glob.glob(os.path.join(config.RUNS_DIR, "*", ""))):
        rm = os.path.join(d, "round_metrics.csv")
        if not os.path.exists(rm): continue
        tag = os.path.basename(d.rstrip("/\\")); parts = tag.split("__")
        if len(parts) < 2: continue
        fed, method = parts[0], parts[1]
        mlast = pd.read_csv(rm).tail(5).mean(numeric_only=True)
        rows.append({"fed": fed, "method": method, "tag": tag,
                     "flipped": mlast.get("test_flipped_auroc", np.nan),
                     "bal_wg": mlast.get("test_balanced_wg_auroc", np.nan)})
    return pd.DataFrame(rows)


def fig_robustness_vs_coupling():
    df = _scan_runs()
    df = df[df.fed.isin(FED_COUPLING)]
    if df.empty: print("no coupling-sweep runs; skipping F3"); return
    df["c"] = df.fed.map(FED_COUPLING)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method in ["fedavg", "reliability", "rel_g8", "oracle_drop"]:
        sub = df[df.method == method]
        if sub.empty: continue
        cs = sorted(sub.c.unique())
        mus = [_ci(sub[sub.c == c]["flipped"])[0] for c in cs]
        hs = [_ci(sub[sub.c == c]["flipped"])[1] for c in cs]
        ax.errorbar(cs, mus, yerr=hs, marker="o", ms=6, lw=2, capsize=3,
                    color=METHOD[method][1], label=METHOD[method][0])
    ax.axhline(0.5, ls="-", lw=0.8, color="#cccccc")
    ax.set_xlabel("shortcut coupling strength  c"); ax.set_ylabel("AUROC on cue-flipped test")
    ax.set_title("Robustness vs shortcut strength\n(FedAvg degrades as the shortcut strengthens; ours holds)", weight="bold")
    ax.legend(frameon=False, fontsize=9); _clean(ax)
    _save(fig, "F3_robustness_vs_coupling.png")


# ======================================================================
# F4: multi-metric ablation grouped bars
# ======================================================================
def fig_ablation_multimetric():
    if not os.path.exists(SUMMARY): print("no per_run.csv; skipping F4"); return
    pr = pd.read_csv(SUMMARY); pr["short"] = pr.group.str.split("__").str[-1]
    variants = [("fedavg", "FedAvg"), ("rel_leakonly", "Leak only"),
                ("rel_gaponly", "Gap only"), ("rel_ctxonly", "Border only"),
                ("reliability", "All (ours)")]
    variants = [v for v in variants if (pr.short == v[0]).any()]
    metrics = [("flipped", "cue-flipped AUROC", C["verm"]),
               ("bal_wg", "balanced worst-group", C["blue"]),
               ("id", "in-distribution", C["gray"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(variants)); w = 0.26
    for j, (col, lab, c) in enumerate(metrics):
        mus = [_ci(pr[pr.short == k][col])[0] for k, _ in variants]
        hs = [_ci(pr[pr.short == k][col])[1] for k, _ in variants]
        ax.bar(x + (j - 1) * w, mus, w, yerr=hs, capsize=2, color=c, label=lab,
               zorder=3, error_kw=dict(ecolor=C["muted"], lw=1))
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in variants], fontsize=9)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC")
    ax.set_title("Signal ablation: only the behavioural border signal recovers the gain", weight="bold")
    leg = ax.legend(frameon=True, fontsize=9, loc="upper left", edgecolor="#cccccc")
    leg.get_frame().set_facecolor("white"); _clean(ax)
    _save(fig, "F4_ablation_multimetric.png")


# ======================================================================
# F5: per-method robustness box plots
# ======================================================================
def fig_robustness_boxplots():
    if not os.path.exists(SUMMARY): print("no per_run.csv; skipping F5"); return
    pr = pd.read_csv(SUMMARY); pr["short"] = pr.group.str.split("__").str[-1]
    order = [m for m in ["fedavg", "fedprox", "valperf", "median", "reliability", "rel_g8", "oracle_drop"]
             if (pr.short == m).any()]
    data = [pr[pr.short == m]["flipped"].dropna().values for m in order]
    cols = [METHOD[m][1] for m in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    medianprops=dict(color=C["ink"], lw=1.5),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=C["ink"], ms=5),
                    flierprops=dict(marker="o", ms=3, mfc=C["muted"], mec="none"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor(C["ink"])
    # reference bands
    if "fedavg" in order:
        y = pr[pr.short == "fedavg"]["flipped"].mean()
        ax.axhline(y, ls="--", lw=1, color=C["gray"]); ax.text(len(order) + 0.4, y, " FedAvg", va="center", fontsize=8, color=C["gray"])
    if "oracle_drop" in order:
        y = pr[pr.short == "oracle_drop"]["flipped"].mean()
        ax.axhline(y, ls=":", lw=1.2, color=C["green"]); ax.text(len(order) + 0.4, y, " Oracle", va="center", fontsize=8, color=C["green"])
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([METHOD[m][0] for m in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AUROC on cue-flipped test (5 seeds)"); ax.set_ylim(0.45, 0.85)
    ax.set_title("Robustness across methods (box = seed spread)", weight="bold"); _clean(ax)
    _save(fig, "F5_robustness_boxplots.png")


# ======================================================================
# F6: feature-space t-SNE - the model's representation organised by cue vs disease
# ======================================================================
def _load_model_robust(path, device, norm="batchnorm"):
    import torch
    from model import build_model
    st = torch.load(path, map_location="cpu")
    mdl = build_model(pretrained=False, norm_layer=norm); msd = mdl.state_dict()
    if "classifier.weight" in st and "classifier.1.weight" in msd:
        st["classifier.1.weight"] = st.pop("classifier.weight"); st["classifier.1.bias"] = st.pop("classifier.bias")
    elif "classifier.1.weight" in st and "classifier.weight" in msd:
        st["classifier.weight"] = st.pop("classifier.1.weight"); st["classifier.bias"] = st.pop("classifier.1.bias")
    mdl.load_state_dict(st, strict=True)
    return mdl.to(device).eval()


def _features(model, loader, device):
    import torch, torch.nn.functional as F
    feats, ys, views = [], [], []
    with torch.no_grad():
        for images, y, cue, _ in loader:
            images = images.to(device)
            f = F.relu(model.features(images))
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            feats.append(f.cpu().numpy()); ys.append(np.asarray(y))
            v = cue["view"].numpy() if "view" in cue else np.full(len(y), -1)
            views.append(v)
    return np.concatenate(feats), np.concatenate(ys), np.concatenate(views)


def fig_tsne():
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import torch
        from hospital_dataset import CsvImageDataset, make_loader
        from utils import get_device
    except Exception as e:
        print(f"skipping F6 (deps): {e}"); return
    test_csv = os.path.join(config.FEDERATIONS_DIR, "main", "tests", "test_balanced.csv")
    if not os.path.exists(test_csv): print("no test_balanced.csv; skipping F6"); return
    models = []
    for tag, name in [("main__fedavg__seed0", "FedAvg (shortcut-prone)"),
                      ("main__rel_g8__seed0", "Reliability γ=8 (ours)")]:
        p = os.path.join(config.RUNS_DIR, tag, "global_model.pt")
        if os.path.exists(p): models.append((p, name))
    if not models: print("no models for F6"); return
    dev = get_device()
    loader = make_loader(CsvImageDataset(test_csv), batch_size=32, shuffle=False, num_workers=0)

    fig, axes = plt.subplots(len(models), 2, figsize=(10, 4.6 * len(models)))
    axes = np.atleast_2d(axes)
    for r, (p, name) in enumerate(models):
        mdl = _load_model_robust(p, dev)
        F_, y, view = _features(mdl, loader, dev)
        emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
                   learning_rate="auto").fit_transform(StandardScaler().fit_transform(F_))
        # colour by disease label
        ax = axes[r, 0]
        for lab, c, nm in [(0, C["sky"], "Normal"), (1, C["verm"], "Pneumonia")]:
            mk = y == lab
            ax.scatter(emb[mk, 0], emb[mk, 1], s=10, c=c, alpha=0.7, label=nm, edgecolors="none")
        ax.set_title(f"{name}\ncoloured by DISEASE", fontsize=10, weight="bold"); ax.axis("off")
        ax.legend(frameon=False, fontsize=8, markerscale=1.5)
        # colour by view cue
        ax = axes[r, 1]
        for v, c, nm in [(0, C["green"], "PA view"), (1, C["orange"], "AP view")]:
            mk = view == v
            ax.scatter(emb[mk, 0], emb[mk, 1], s=10, c=c, alpha=0.7, label=nm, edgecolors="none")
        ax.set_title(f"{name}\ncoloured by VIEW (the cue)", fontsize=10, weight="bold"); ax.axis("off")
        ax.legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.suptitle("Feature-space t-SNE: the shortcut model organises its representation by VIEW;\nthe robust model organises it by DISEASE",
                 fontsize=12.5, weight="bold", y=1.01)
    _save(fig, "F6_tsne_feature_space.png")


# ======================================================================
# F7: score-level DP privacy-utility curve
# ======================================================================
def fig_privacy_utility():
    import re
    rows = []
    for d in sorted(glob.glob(os.path.join(config.RUNS_DIR, "*", ""))):
        tag = os.path.basename(d.rstrip("/\\"))
        m = re.match(r"main__rel_dp([\d.]+)__seed\d+$", tag)
        if m:
            sigma = float(m.group(1))
        elif re.match(r"main__reliability__seed\d+$", tag):
            sigma = 0.0
        else:
            continue
        rm = os.path.join(d, "round_metrics.csv")
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(rm):
            continue
        flipped = pd.read_csv(rm).tail(5)["test_flipped_auroc"].mean()
        det = np.nan
        if os.path.exists(sj):
            det = load_json(sj).get("detection", {}).get("detection_auroc", np.nan)
        rows.append({"sigma": sigma, "flipped": flipped, "detection": det})
    df = pd.DataFrame(rows)
    if df.empty or df.sigma.nunique() < 2:
        print("no score-DP sweep runs (need main__rel_dp<sigma>__seed*); skipping F7"); return
    g = df.groupby("sigma").agg(flipped_m=("flipped", "mean"), flipped_s=("flipped", "std"),
                                det_m=("detection", "mean"), det_s=("detection", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(g.sigma, g.det_m, yerr=g.det_s.fillna(0), marker="o", ms=7, lw=2,
                color=C["blue"], capsize=3, label="detection AUROC (finds cheaters)")
    ax.errorbar(g.sigma, g.flipped_m, yerr=g.flipped_s.fillna(0), marker="s", ms=7, lw=2,
                color=C["verm"], capsize=3, label="flipped AUROC (robustness)")
    ax.set_xlabel("score-level DP noise  σ  (more private →)", fontsize=10)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 1.05)
    ax.set_title("Privacy–utility: adding DP noise to the released scores\ndegrades the audit gracefully", weight="bold")
    ax.legend(frameon=False, fontsize=9); _clean(ax)
    _save(fig, "F7_privacy_utility.png")


def main():
    fig_shortcut_xrays()
    fig_trust_heatmap()
    fig_robustness_vs_coupling()
    fig_ablation_multimetric()
    fig_robustness_boxplots()
    fig_tsne()
    fig_privacy_utility()
    print(f"\nRich figures in {FIG}")


if __name__ == "__main__":
    main()
