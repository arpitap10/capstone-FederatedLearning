"""
make_gradcam_all.py - Grad-CAM for the GLOBAL models of every key method (needs Colab GPU + saved global_model.pt).
Reuses GradCAM / border_mass / loaders from your existing make_gradcam.py. No local training (fast).

    %cd /content/drive/MyDrive/capstone_project/src
    !python make_gradcam_all.py            # 4 examples
    !python make_gradcam_all.py --n 6

Outputs figures/G4_gradcam_methods.png/.pdf and G5_bordermass_methods.png/.pdf
Grad-CAM is ILLUSTRATIVE only (Arun et al. 2021: saliency is not a reliable audit); say so in the caption.
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from make_gradcam import GradCAM, border_mass, _load_global, _pick_examples
from utils import get_device

FIG = os.path.join(config.PROJECT_DIR, "figures")
METHODS = [("fedavg", "FedAvg"), ("qffl", "q-FFL"), ("reliability", "Reliability γ=2"), ("rel_g8", "Reliability γ=8"),
           ("oracle_drop", "Oracle")]
COL = {"fedavg": "#404040", "qffl": "#D55E00", "reliability": "#2166AC", "rel_g8": "#08306B", "oracle_drop": "#1A9850"}


def first_tag(method):
    c = sorted(glob.glob(os.path.join(config.RUNS_DIR, f"main__{method}__seed*", "global_model.pt")))
    return os.path.basename(os.path.dirname(c[0])) if c else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4); a = ap.parse_args()
    dev = get_device(); ex = _pick_examples(a.n)
    if not ex:
        print("no positive+AP examples"); return
    models = []
    for k, name in METHODS:
        t = first_tag(k)
        m = _load_global(t, dev) if t else None
        if m is None:
            print("missing checkpoint for", k); continue
        models.append((k, name, m)); print("loaded", t)
    fig, axes = plt.subplots(len(ex), 1 + len(models), figsize=(2.4 * (1 + len(models)), 2.5 * len(ex)), squeeze=False)
    bm = {k: [] for k, _, _ in models}
    for r, (x, gray, pid) in enumerate(ex):
        axes[r, 0].imshow(gray, cmap="gray"); axes[r, 0].axis("off")
        if r == 0: axes[r, 0].set_title("Original\n(pneumonia, AP)", fontsize=9, weight="bold")
        for c, (k, name, m) in enumerate(models, 1):
            cam, p = GradCAM(m)(x.to(dev)); b = border_mass(cam); bm[k].append(b)
            axes[r, c].imshow(gray, cmap="gray"); axes[r, c].imshow(cam, cmap="jet", alpha=.5); axes[r, c].axis("off")
            axes[r, c].set_title((name + "\n" if r == 0 else "") + f"p={p:.2f}  border={b:.0%}", fontsize=8,
                                 weight="bold" if r == 0 else "normal")
    fig.suptitle("Grad-CAM of the global models (illustrative)", fontsize=12, y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"G4_gradcam_methods.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    mu = [np.mean(bm[k]) for k, _, _ in models]; sd = [np.std(bm[k]) for k, _, _ in models]
    ax.bar(range(len(models)), mu, yerr=sd, capsize=3, color=[COL[k] for k, _, _ in models], zorder=3)
    for i, v in enumerate(mu): ax.text(i, v + .02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([n for _, n, _ in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("fraction of CAM mass in border ring"); ax.set_title(f"Border attention (n={a.n} images, one seed per method)")
    ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color="#e9e9e9"); ax.set_axisbelow(True)
    for e in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"G5_bordermass_methods.{e}"), dpi=200, bbox_inches="tight")
    print("saved G4 / G5 in", FIG)


if __name__ == "__main__":
    main()
