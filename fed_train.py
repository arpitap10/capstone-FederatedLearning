"""
fed_train.py - the federated round loop (plan 8.7) and the entry point for a
single federated run.

For each round:
  1. every client trains locally from the current global weights
  2. its update Delta_i = state_i - global is formed and (optionally) clipped
  3. after a warm-up, each client runs the behavioural audit on its PRIVATE V_i,
     signals are fused into r_i (+ EMA, + optional score-DP noise); under P2 the
     peer-transfer signal tau is added from a scalar AUROC matrix
  4. the server aggregates with the selected rule (secure-aggregation simulated
     as a sum; optional update-DP noise)
  5. the global model is evaluated on test_id / test_balanced / test_flipped

Outputs (under runs/<tag>/):
  global_model.pt, round_metrics.csv, client_signals.csv, summary.json
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import config
from config import TrainConfig, ReliabilityConfig, PrivacyConfig, config_to_dict
from cue_bank import CueBank, DEFAULT_BANK
from hospital_dataset import HospitalDataset, CsvImageDataset, make_loader
from model import (build_model, load_state, get_state, clone_state, state_sub,
                   clip_state, predict_proba, denormalize_to_gray)
from local_train import train_client
from client_audit import compute_local_signals
from reliability import fuse, aggregation_weight
from protocols import secure_aggregate, peer_eval_matrix, transfer_retention
from aggregate import aggregate, compute_weights
from evaluate import model_report, detection_metrics
from utils import get_logger, set_seed, save_json, timer, get_device

LOG = get_logger()


# ----------------------------------------------------------------------
def _build_cue_bank() -> CueBank:
    """Full bank if image cutpoints exist, else metadata-only (graceful)."""
    try:
        return CueBank(active=list(DEFAULT_BANK))
    except FileNotFoundError:
        LOG.warning("image-cue cutpoints missing -> auditing with metadata cues "
                    "only. Run build_federation (which fits cutpoints) for the "
                    "full bank.")
        return CueBank(active=["view", "sex", "age_band"])


def _load_manifest(fed_dir: str) -> Dict:
    from utils import load_json
    return load_json(os.path.join(fed_dir, "hospital_manifest.json"))


def _maybe_noise_signals(signals: Dict, pcfg: PrivacyConfig, rng) -> Dict:
    if pcfg.score_dp_sigma <= 0:
        return signals
    from dp import noisy_clip01
    out = dict(signals)
    for k in ("leak", "gap", "ctx", "tau"):
        v = signals.get(k)
        if v is not None and not np.isnan(v):
            out[k] = noisy_clip01(float(v), pcfg.score_dp_sigma, rng)
    return out


# ----------------------------------------------------------------------
def run_federation(fed_name: str, method: str = "reliability",
                   tcfg: Optional[TrainConfig] = None,
                   rcfg: Optional[ReliabilityConfig] = None,
                   pcfg: Optional[PrivacyConfig] = None,
                   rounds: Optional[int] = None, seed: int = 0,
                   out_tag: Optional[str] = None) -> Dict:
    tcfg = tcfg or TrainConfig()
    rcfg = rcfg or ReliabilityConfig()
    pcfg = pcfg or PrivacyConfig()
    rounds = rounds or tcfg.rounds
    out_tag = out_tag or f"{fed_name}__{method}__seed{seed}"
    set_seed(seed)
    device = get_device()
    rng = np.random.default_rng(seed)

    fed_dir = os.path.join(config.FEDERATIONS_DIR, fed_name)
    manifest = _load_manifest(fed_dir)
    hospitals = sorted(manifest["hospitals"])
    is_shortcut = {h: manifest["hospitals"][h]["is_shortcut"] for h in hospitals}
    coupling = {h: manifest["hospitals"][h]["coupling"] for h in hospitals}

    run_dir = os.path.join(config.RUNS_DIR, out_tag)
    os.makedirs(run_dir, exist_ok=True)
    LOG.info(f"RUN {out_tag} | method={method} protocol={pcfg.protocol} "
             f"device={device} rounds={rounds}")

    cue_bank = _build_cue_bank()

    # fixed loaders for audit (val) and evaluation (tests)
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

    needs_audit = (method == "reliability")
    global_state = get_state(build_model(pretrained=tcfg.pretrained,
                                         norm_layer=tcfg.norm_layer))
    audit_model = build_model(pretrained=False, norm_layer=tcfg.norm_layer).to(device)

    prev_r: Dict[str, float] = {}
    mitigation_cue: Dict[str, Optional[str]] = {h: None for h in hospitals}
    round_rows, signal_rows = [], []

    for t in range(1, rounds + 1):
        LOG.info(f"--- round {t}/{rounds} ---")
        clients: List[Dict] = []
        client_states: Dict[str, object] = {}
        for h in hospitals:
            res = train_client(global_state, h, tcfg, device,
                               mitigation_cue=mitigation_cue[h], federation=fed_name)
            delta = clip_state(state_sub(res["state"], global_state),
                               pcfg.update_clip_C)
            client_states[h] = res["state"]
            clients.append({"name": h, "state": res["state"], "delta": delta,
                            "n_samples": res["n_samples"],
                            "val_auroc": res["val_auroc"],
                            "balanced_val_auroc": res["balanced_val_auroc"],
                            "is_shortcut": is_shortcut[h], "r": None})

        # ---------------- audit + reliability ----------------
        do_audit = needs_audit and t > rcfg.warmup_rounds
        if do_audit:
            tau_vec = None
            if pcfg.protocol == "P2":
                S = peer_eval_matrix(client_states, val_loaders, device, tcfg)
                tau_vec = transfer_retention(S)
            for idx, c in enumerate(clients):
                load_state(audit_model, c["state"])
                sig = compute_local_signals(audit_model, val_loaders[c["name"]],
                                            cue_bank, device, rcfg)
                if tau_vec is not None:
                    sig["tau"] = float(tau_vec[idx])
                sig = _maybe_noise_signals(sig, pcfg, rng)
                fused = fuse(sig, rcfg, prev_r.get(c["name"]))
                c["r"] = fused["r"]
                prev_r[c["name"]] = fused["r"]
                # audit-triggered local mitigation for the next round
                if tcfg.local_mitigation and sig["leak"] is not None \
                        and not np.isnan(sig["leak"]) and sig["leak"] > tcfg.mitigation_tau_leak:
                    mitigation_cue[c["name"]] = _top_leak_cue(sig)
                signal_rows.append({"round": t, "client": c["name"],
                                    "is_shortcut": is_shortcut[c["name"]],
                                    "coupling": coupling[c["name"]],
                                    "r": fused["r"], **{f"s_{k}": v for k, v in fused["s"].items()},
                                    "leak": sig["leak"], "gap": sig["gap"],
                                    "ctx": sig["ctx"], "tau": sig.get("tau"),
                                    "val_auroc": sig["val_auroc"], "abstain": sig["abstain"]})
            agg_method = method
        else:
            agg_method = "fedavg" if needs_audit else method

        # ---------------- aggregation ----------------
        if agg_method == "reliability":
            items = [{"w": aggregation_weight(c["n_samples"], c["r"], rcfg),
                      "delta": c["delta"]} for c in clients]
            if pcfg.secure_agg:
                global_state = secure_aggregate(global_state, items)
            else:
                w = np.array([it["w"] for it in items])
                from aggregate import apply_weighted_deltas
                global_state = apply_weighted_deltas(global_state, clients, w)
        else:
            global_state = aggregate(agg_method, global_state, clients, rcfg)

        # ---------------- optional update-DP noise ----------------
        if pcfg.update_dp_sigma > 0 and pcfg.update_clip_C:
            from dp import add_state_noise
            global_state = add_state_noise(global_state, pcfg.update_dp_sigma,
                                           pcfg.update_clip_C, len(clients), rng)

        # ---------------- evaluate ----------------
        load_state(audit_model, global_state)
        row = {"round": t, "agg_method": agg_method}
        for tname, loader in tests.items():
            res = predict_proba(audit_model, loader, device, cue_bank=cue_bank)
            rep = model_report(res)
            row[f"{tname}_auroc"] = rep["auroc"]
            row[f"{tname}_wg_auroc"] = rep.get("worst_group_auroc", float("nan"))
        if "test_balanced_auroc" in row and "test_flipped_auroc" in row:
            row["flip_drop"] = row["test_balanced_auroc"] - row["test_flipped_auroc"]
        round_rows.append(row)
        LOG.info("  " + " | ".join(f"{k}={v:.3f}" for k, v in row.items()
                                   if isinstance(v, float)))

    # ---------------- persist ----------------
    torch.save(global_state, os.path.join(run_dir, "global_model.pt"))
    pd.DataFrame(round_rows).to_csv(os.path.join(run_dir, "round_metrics.csv"), index=False)
    if signal_rows:
        pd.DataFrame(signal_rows).to_csv(os.path.join(run_dir, "client_signals.csv"), index=False)

    summary = {"tag": out_tag, "method": method, "protocol": pcfg.protocol,
               "fed_name": fed_name, "seed": seed, "rounds": rounds, "final": round_rows[-1],
               "configs": {"train": config_to_dict(tcfg),
                           "reliability": config_to_dict(rcfg),
                           "privacy": config_to_dict(pcfg)}}
    if needs_audit and prev_r:
        summary["detection"] = detection_metrics(prev_r, is_shortcut, coupling)
    save_json(summary, os.path.join(run_dir, "summary.json"))
    LOG.info(f"saved run -> {run_dir}")
    return summary


def _top_leak_cue(sig: Dict) -> Optional[str]:
    """Pick the cue with the largest per-cue leakage for local mitigation."""
    best, best_cue = -1.0, None
    for cue, ys in sig.get("per_cue_leak", {}).items():
        m = max(ys.values()) if ys else -1.0
        if m > best:
            best, best_cue = m, cue
    return best_cue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed", default="main")
    ap.add_argument("--method", default="reliability",
                    help="fedavg|fedprox|valperf|fedar|qffl|update_agreement|"
                         "oracle_drop|reliability")
    ap.add_argument("--protocol", default="P1")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=None)
    # per-signal fusion weights (0 disables a signal); None keeps the config default
    ap.add_argument("--w-leak", type=float, default=None)
    ap.add_argument("--w-gap", type=float, default=None)
    ap.add_argument("--w-ctx", type=float, default=None)
    ap.add_argument("--w-tau", type=float, default=None)
    ap.add_argument("--shrink-kappa", type=float, default=None,
                    help="shrinkage toward r0 (lower = preserve signal separation; 0 = off)")
    ap.add_argument("--r-min", type=float, default=None, help="weight floor on r")
    ap.add_argument("--ema", type=float, default=None, help="EMA smoothing of r across rounds")
    ap.add_argument("--warmup", type=int, default=None, help="FedAvg warm-up rounds before auditing")
    ap.add_argument("--score-dp-sigma", type=float, default=None,
                    help="Gaussian noise std added to each released reliability signal (score-level DP)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tcfg, rcfg = TrainConfig(), ReliabilityConfig()
    pcfg = PrivacyConfig(protocol=args.protocol)
    if args.score_dp_sigma is not None:
        pcfg.score_dp_sigma = args.score_dp_sigma
    if args.gamma is not None:
        rcfg.gamma = args.gamma
    if args.shrink_kappa is not None:
        rcfg.shrink_kappa = args.shrink_kappa
    if args.r_min is not None:
        rcfg.r_min = args.r_min
    if args.ema is not None:
        rcfg.ema = args.ema
    if args.warmup is not None:
        rcfg.warmup_rounds = args.warmup
    for name in ("w_leak", "w_gap", "w_ctx", "w_tau"):
        v = getattr(args, name)
        if v is not None:
            setattr(rcfg, name, v)
    if args.method == "fedprox":
        tcfg.fedprox_mu = 0.1
    run_federation(args.fed, method=args.method, tcfg=tcfg, rcfg=rcfg, pcfg=pcfg,
                   rounds=args.rounds, seed=args.seed, out_tag=args.tag)


if __name__ == "__main__":
    main()
