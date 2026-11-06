"""
local_baseline.py - independent per-hospital ("local-only") training.

This is both a required baseline (plan section 10) and the E0 motivating
diagnostic.  Each hospital trains its OWN model on its OWN data (starting from
ImageNet, no federation, no aggregation), and we look at:

  * own-val AUROC                : how good the model looks "at home"
  * test_id / test_balanced AUROC: performance when the cue is preserved vs balanced
  * test_flipped AUROC + flip_drop: the collapse when the cue is reversed
  * context_border_auroc         : behavioural border-only AUROC (does the model
                                    read the periphery instead of the lungs?)
  * transfer matrix S[i,j]        : AUROC of hospital i's model on hospital j's
                                    validation set, and transfer-retention tau_i
                                    (a preview of the peer-transfer reliability
                                    signal: biased models transfer worse)

Expected story: a biased hospital scores high on its own/id test but drops sharply
on test_flipped and transfers poorly to peers, while a clean hospital holds up.

Usage
-----
    python src/local_baseline.py --fed main --epochs 8
    python src/local_baseline.py --fed main --epochs 8 --gradcam   # + saliency panels
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd

import config
from config import TrainConfig
from hospital_dataset import HospitalDataset, CsvImageDataset, make_loader
from model import build_model, get_state, load_state, predict_proba
from local_train import train_client
from context_audit import context_auroc
from protocols import peer_eval_matrix, transfer_retention
from evaluate import model_report
from fed_train import _build_cue_bank, _load_manifest
from utils import get_logger, get_device, set_seed, save_json

LOG = get_logger()


def run_local_baseline(fed_name: str = "main", epochs: int = 8, seed: int = 0,
                       do_gradcam: bool = False) -> pd.DataFrame:
    set_seed(seed)
    device = get_device()
    fed_dir = os.path.join(config.FEDERATIONS_DIR, fed_name)
    manifest = _load_manifest(fed_dir)
    hospitals = sorted(manifest["hospitals"])
    is_shortcut = {h: manifest["hospitals"][h]["is_shortcut"] for h in hospitals}
    coupling = {h: manifest["hospitals"][h]["coupling"] for h in hospitals}
    cue = {h: manifest["hospitals"][h]["cue"] for h in hospitals}

    out_dir = os.path.join(config.RUNS_DIR, f"local_baseline__{fed_name}__seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    cue_bank = _build_cue_bank()

    tcfg = TrainConfig()
    tcfg.local_epochs = epochs          # max epochs; early stopping picks the best
    tcfg.pretrained = True
    tcfg.early_stop_patience = 3        # stop when val AUROC stalls (anti-overfit)

    # fixed loaders for evaluation
    val_loaders = {h: make_loader(HospitalDataset(h, "val", federation=fed_name),
                                  tcfg.batch_size, shuffle=False,
                                  num_workers=tcfg.num_workers)
                   for h in hospitals}
    tests = {}
    for t in ("test_id", "test_balanced", "test_flipped"):
        p = os.path.join(fed_dir, "tests", f"{t}.csv")
        if os.path.exists(p):
            tests[t] = make_loader(CsvImageDataset(p), tcfg.batch_size,
                                   shuffle=False, num_workers=tcfg.num_workers)

    init_state = get_state(build_model(pretrained=True, norm_layer=tcfg.norm_layer))
    eval_model = build_model(pretrained=False, norm_layer=tcfg.norm_layer).to(device)

    client_states: Dict[str, object] = {}
    rows = []
    for h in hospitals:
        LOG.info(f"=== local-only training: {h} ({epochs} epochs) ===")
        res = train_client(init_state, h, tcfg, device, federation=fed_name)  # ImageNet -> H's data
        client_states[h] = res["state"]
        load_state(eval_model, res["state"])

        row = {"hospital": h, "is_shortcut": is_shortcut[h], "cue": cue[h],
               "coupling": coupling[h], "own_val_auroc": res["val_auroc"]}
        for tname, loader in tests.items():
            rep = model_report(predict_proba(eval_model, loader, device, cue_bank=cue_bank))
            row[f"{tname}_auroc"] = rep["auroc"]
            row[f"{tname}_wg_auroc"] = rep.get("worst_group_auroc", float("nan"))
        if "test_balanced_auroc" in row and "test_flipped_auroc" in row:
            row["flip_drop"] = row["test_balanced_auroc"] - row["test_flipped_auroc"]
        # behavioural border-only AUROC on the hospital's own val
        row["context_border_auroc"] = context_auroc(eval_model, val_loaders[h],
                                                     device, region="border")
        rows.append(row)
        LOG.info("  " + " | ".join(f"{k}={v:.3f}" for k, v in row.items()
                                   if isinstance(v, float)))

    # transfer matrix + retention (preview of the peer-transfer signal)
    LOG.info("computing cross-hospital transfer matrix ...")
    S = peer_eval_matrix(client_states, val_loaders, device, tcfg)
    tau = transfer_retention(S)
    names = sorted(client_states)
    pd.DataFrame(S, index=names, columns=names).to_csv(
        os.path.join(out_dir, "transfer_matrix.csv"))
    df = pd.DataFrame(rows)
    df["transfer_retention_tau"] = [float(tau[names.index(h)]) for h in df["hospital"]]
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    # optional Grad-CAM vs behavioural panels (the Arun et al. comparison)
    if do_gradcam:
        try:
            from explain import compare_gradcam_vs_behavioural
            for h in hospitals:
                load_state(eval_model, client_states[h])
                g_dir = os.path.join(out_dir, f"gradcam_{h}")
                compare_gradcam_vs_behavioural(eval_model, val_loaders[h], device,
                                               g_dir, n_examples=6)
        except Exception as e:
            LOG.warning(f"gradcam step skipped: {e}")

    # headline observation
    biased = df[df["is_shortcut"]]
    clean = df[~df["is_shortcut"]]
    summary = {
        "fed": fed_name, "epochs": epochs, "seed": seed,
        "biased_mean_flip_drop": float(biased["flip_drop"].mean()) if len(biased) else float("nan"),
        "clean_mean_flip_drop": float(clean["flip_drop"].mean()) if len(clean) else float("nan"),
        "biased_mean_transfer_tau": float(biased["transfer_retention_tau"].mean()) if len(biased) else float("nan"),
        "clean_mean_transfer_tau": float(clean["transfer_retention_tau"].mean()) if len(clean) else float("nan"),
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    LOG.info("=" * 60)
    LOG.info("LOCAL-ONLY DIAGNOSTIC (E0):")
    LOG.info(df.to_string(index=False))
    LOG.info(f"mean flip_drop  - biased: {summary['biased_mean_flip_drop']:.3f} | "
             f"clean: {summary['clean_mean_flip_drop']:.3f}  "
             "(biased should be LARGER -> shortcut is real)")
    LOG.info(f"mean transfer tau - biased: {summary['biased_mean_transfer_tau']:.3f} | "
             f"clean: {summary['clean_mean_transfer_tau']:.3f}  "
             "(biased should be LOWER -> generalises worse)")
    LOG.info(f"saved -> {out_dir}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed", default="main")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gradcam", action="store_true", help="also save Grad-CAM vs behavioural panels")
    args = ap.parse_args()
    run_local_baseline(args.fed, epochs=args.epochs, seed=args.seed, do_gradcam=args.gradcam)


if __name__ == "__main__":
    main()
