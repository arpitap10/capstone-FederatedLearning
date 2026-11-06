"""
make_confusion.py - confusion matrix + full classification metrics for a trained
GLOBAL model on an RSNA test set.  Threshold-free AUROC/AUPRC plus F1 / precision
/ recall / specificity at a stated threshold (fixed 0.5 by default, and the
F1-optimal threshold shown alongside).  PNG + PDF via figstyle.

    python src/make_confusion.py                      # reliability global on test_balanced
    python src/make_confusion.py --model main__fedavg__seed0 --test test_flipped
    python src/make_confusion.py --thr 0.5
"""

from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

import config
import figstyle as fs
from figstyle import COL
from hospital_dataset import CsvImageDataset, make_loader
from model import build_model, predict_proba
from utils import get_device, get_logger

fs.apply()
LOG = get_logger()

try:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 confusion_matrix, precision_recall_curve,
                                 precision_recall_fscore_support)
except Exception:
    roc_auc_score = None


def _load(path, device):
    import torch
    st = torch.load(path, map_location="cpu")
    m = build_model(pretrained=False); msd = m.state_dict()
    if "classifier.weight" in st and "classifier.1.weight" in msd:
        st["classifier.1.weight"] = st.pop("classifier.weight"); st["classifier.1.bias"] = st.pop("classifier.bias")
    elif "classifier.1.weight" in st and "classifier.weight" in msd:
        st["classifier.weight"] = st.pop("classifier.1.weight"); st["classifier.bias"] = st.pop("classifier.1.bias")
    m.load_state_dict(st, strict=True)
    return m.to(device).eval()


def _metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return dict(precision=pr, recall=rc, f1=f1, specificity=spec,
                accuracy=(tp + tn) / len(y)), np.array([[tn, fp], [fn, tp]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="main__reliability__seed0")
    ap.add_argument("--test", default="test_balanced")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    device = get_device()
    mpath = os.path.join(config.RUNS_DIR, args.model, "global_model.pt")
    tcsv = os.path.join(config.FEDERATIONS_DIR, "main", "tests", f"{args.test}.csv")
    model = _load(mpath, device)
    loader = make_loader(CsvImageDataset(tcsv), batch_size=32, shuffle=False, num_workers=0)
    res = predict_proba(model, loader, device)
    y, p = res["targets"].astype(int), res["probs"]

    auroc = float(roc_auc_score(y, p)); auprc = float(average_precision_score(y, p))
    pr_c, rc_c, thr_c = precision_recall_curve(y, p)
    f1s = 2 * pr_c * rc_c / (pr_c + rc_c + 1e-12)
    f1opt_thr = float(thr_c[int(np.nanargmax(f1s[:-1]))]) if len(thr_c) else 0.5
    m_fixed, cm = _metrics(y, p, args.thr)
    m_opt, _ = _metrics(y, p, f1opt_thr)
    LOG.info(f"{args.model} on {args.test}: AUROC={auroc:.3f} AUPRC={auprc:.3f} "
             f"F1@{args.thr}={m_fixed['f1']:.3f} F1opt={m_opt['f1']:.3f}")

    # ---- figure: confusion matrix (left) + metrics (right) ----
    fig, (axc, axm) = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    # confusion matrix: counts + row-normalised %
    row = cm.sum(1, keepdims=True); norm = cm / np.clip(row, 1, None)
    im = axc.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = ["Normal", "Pneumonia"]
    for i in range(2):
        for j in range(2):
            txt = f"{cm[i, j]}\n({norm[i, j]*100:.0f}%)"
            axc.text(j, i, txt, ha="center", va="center", fontsize=11,
                     color="white" if norm[i, j] > 0.5 else COL["ink"], weight="bold")
    axc.set_xticks([0, 1]); axc.set_xticklabels(["Pred Normal", "Pred Pneumonia"], fontsize=9)
    axc.set_yticks([0, 1]); axc.set_yticklabels(["True Normal", "True Pneumonia"], fontsize=9)
    axc.set_title(f"Confusion matrix  (thr={args.thr})", fontsize=11)
    for s in axc.spines.values(): s.set_visible(False)

    # metrics as a horizontal dot plot (threshold-dependent + AUROC/AUPRC)
    names = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Specificity"]
    vals = [auroc, auprc, m_fixed["f1"], m_fixed["precision"], m_fixed["recall"], m_fixed["specificity"]]
    yy = np.arange(len(names))[::-1]
    axm.hlines(yy, 0, vals, color=COL["grid"], lw=6)
    axm.scatter(vals, yy, s=90, color=COL["ours"], zorder=3, edgecolors="white", linewidths=0.8)
    for v, y_ in zip(vals, yy):
        axm.text(v + 0.02, y_, f"{v:.3f}", va="center", fontsize=9, color=COL["ink"])
    axm.set_yticks(yy); axm.set_yticklabels(names, fontsize=9.5)
    axm.set_xlim(0, 1.12); axm.set_xlabel("score")
    axm.set_title(f"Metrics  (n={len(y)}, prev={y.mean():.2f})", fontsize=11)
    fs.despine(axm); axm.xaxis.grid(True, color=COL["grid"], lw=0.9); axm.set_axisbelow(True)

    disp = args.model.split("__")[1] if "__" in args.model else args.model
    fig.suptitle(f"{disp} global model — {args.test} (RSNA held-out)", fontsize=12.5, y=1.06)
    fs.save(fig, f"FIG7_confusion_{disp}_{args.test}")


if __name__ == "__main__":
    main()
