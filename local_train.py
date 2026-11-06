"""
local_train.py - one client's local training step (plan 8.1, 8.5).

`train_client` loads the current global weights, trains for `local_epochs` on the
client's private training pool, and returns the new weights plus self-reported
validation metrics.  The same function serves FedAvg and FedProx (set
`fedprox_mu > 0`).  Audit-triggered local mitigation (plan 8.5) reweights the
sampler by (label x cue) cell so a flagged client stops exploiting its cue.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from evaluate import auroc, worst_group_auroc
from hospital_dataset import HospitalDataset, collate_batch, make_loader
from model import build_model, load_state, get_state, predict_proba
from utils import AverageMeter, get_logger

LOG = get_logger()


def _make_optimizer(model: nn.Module, tcfg):
    if tcfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=tcfg.lr,
                                weight_decay=tcfg.weight_decay)
    if tcfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=tcfg.lr,
                               momentum=tcfg.momentum, weight_decay=tcfg.weight_decay)
    raise ValueError(f"unknown optimizer {tcfg.optimizer}")


def _cell_sample_weights(dataset: HospitalDataset, cue: str) -> np.ndarray:
    """Inverse-frequency weights over (Target, cue) cells for mitigation."""
    y = dataset.targets
    cv = dataset.cue_arrays[cue].astype(float)
    cv[cv < 0] = np.nan
    weights = np.ones(len(y), dtype=np.float64)
    cells = {}
    for i in range(len(y)):
        key = (int(y[i]), -1 if np.isnan(cv[i]) else int(cv[i]))
        cells.setdefault(key, []).append(i)
    for key, idxs in cells.items():
        for i in idxs:
            weights[i] = 1.0 / len(idxs)
    weights *= len(y) / weights.sum()
    return weights


def train_client(global_state, hospital_name: str, tcfg, device,
                 mitigation_cue: Optional[str] = None,
                 federation: str = "main") -> Dict:
    """Train one client; return {state, n_samples, val_auroc, balanced_val_auroc}."""
    model = build_model(pretrained=tcfg.pretrained, norm_layer=tcfg.norm_layer,
                        dropout=tcfg.dropout).to(device)
    load_state(model, global_state)
    global_params = [p.detach().clone() for p in model.parameters()]

    train_ds = HospitalDataset(hospital_name, "train", federation=federation)
    val_ds = HospitalDataset(hospital_name, "val", federation=federation)

    if tcfg.local_mitigation and mitigation_cue:
        w = _cell_sample_weights(train_ds, mitigation_cue)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, sampler=sampler,
                                  num_workers=tcfg.num_workers, pin_memory=True,
                                  collate_fn=collate_batch)
    else:
        train_loader = make_loader(train_ds, tcfg.batch_size, shuffle=True,
                                   num_workers=tcfg.num_workers)
    val_loader = make_loader(val_ds, tcfg.batch_size, shuffle=False,
                             num_workers=tcfg.num_workers)

    optimizer = _make_optimizer(model, tcfg)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = bool(tcfg.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state, best_auroc, bad_epochs = None, -1.0, 0
    for epoch in range(tcfg.local_epochs):
        model.train()
        meter = AverageMeter()
        for images, y, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(images).squeeze(1)
                loss = criterion(logits, y)
                if tcfg.fedprox_mu > 0:
                    prox = sum(((p - gp) ** 2).sum()
                               for p, gp in zip(model.parameters(), global_params))
                    loss = loss + 0.5 * tcfg.fedprox_mu * prox
            scaler.scale(loss).backward()
            if tcfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item(), len(y))
        msg = f"    [{hospital_name}] epoch {epoch + 1}/{tcfg.local_epochs} loss={meter.avg:.4f}"

        # early stopping on validation AUROC (keeps the best-generalising epoch)
        if tcfg.early_stop_patience > 0:
            _r = predict_proba(model, val_loader, device)
            va = auroc(_r["targets"], _r["probs"])
            msg += f" val_auroc={va:.4f}"
            if not np.isnan(va) and va > best_auroc + 1e-4:
                best_auroc, best_state, bad_epochs = va, get_state(model), 0
            else:
                bad_epochs += 1
            LOG.info(msg)
            if bad_epochs >= tcfg.early_stop_patience:
                LOG.info(f"    [{hospital_name}] early stop (no val gain for "
                         f"{bad_epochs} epochs); best val_auroc={best_auroc:.4f}")
                break
        else:
            LOG.info(msg)

    if best_state is not None:          # restore best-val weights
        load_state(model, best_state)

    # self-reported validation metrics (on the selected weights)
    res = predict_proba(model, val_loader, device)
    val_auroc = auroc(res["targets"], res["probs"])
    bal = []
    for name, cv in res["cues"].items():
        wg = worst_group_auroc(res["targets"], res["probs"],
                               np.asarray(cv, dtype=np.float64))
        if not np.isnan(wg["worst"]):
            bal.append(wg["worst"])
    balanced_val_auroc = float(np.min(bal)) if bal else val_auroc

    return {
        "state": get_state(model),
        "n_samples": len(train_ds),
        "val_auroc": float(val_auroc),
        "balanced_val_auroc": float(balanced_val_auroc),
    }
