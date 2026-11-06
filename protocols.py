"""
protocols.py - privacy protocol mechanics (plan 7.3) used by fed_train.

The protocols differ in WHAT is revealed, not in the arithmetic of a simulated
run on one machine:

  P0  non-private reference  - audit uses a server-held cue-annotated probe set
                               (upper bound; measures the price of privacy).
  P1  default                - every signal computed on the client's private
                               V_i; server sees only Sum(w_i Delta_i), Sum(w_i).
  P2  peer evaluation        - adds the transfer-retention signal tau from a
                               scalar peer-AUROC matrix S[i,j].
  P3  subset attribution     - (stub hook) contribution via random subsets.

This module provides the secure-aggregation SIMULATION (summation: the server
never sees an individual update) and the peer-evaluation matrix + tau.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from evaluate import auroc
from model import (build_model, load_state, predict_proba, state_zeros_like,
                   state_axpy, state_add)


# ----------------------------------------------------------------------
# secure aggregation (simulated by summation)
# ----------------------------------------------------------------------
def secure_aggregate(global_state, weighted_items: List[Dict]):
    """Each item: {'w': float, 'delta': StateDict}.  Returns the FedAvg update
    global + Sum(w*delta)/Sum(w).  In a real deployment the per-item w*delta and
    w are masked and only their sums decrypt; here we add them directly."""
    total_w = sum(it["w"] for it in weighted_items)
    if total_w <= 0:
        total_w = float(len(weighted_items)) or 1.0
    acc = state_zeros_like(global_state)
    for it in weighted_items:
        acc = state_axpy(acc, it["w"] / total_w, it["delta"])
    return state_add(global_state, acc)


# ----------------------------------------------------------------------
# P2 peer evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def peer_eval_matrix(client_states: Dict[str, "OrderedDict"],
                     val_loaders: Dict[str, object], device, tcfg) -> np.ndarray:
    """S[i,j] = AUROC of client i's model on client j's local validation set.

    O(K^2) evaluations; fine for small K.  Clients are the sorted keys.
    """
    names = sorted(client_states)
    K = len(names)
    S = np.full((K, K), np.nan)
    model = build_model(pretrained=False, norm_layer=tcfg.norm_layer).to(device)
    for i, ni in enumerate(names):
        load_state(model, client_states[ni])
        for j, nj in enumerate(names):
            res = predict_proba(model, val_loaders[nj], device)
            S[i, j] = auroc(res["targets"], res["probs"])
    return S


def transfer_retention(S: np.ndarray) -> np.ndarray:
    """tau_i = clip( (mean_{j!=i} S[i,j] - 0.5) / (S[i,i] - 0.5), 0, 1 ).

    High when a model generalises to peers as well as to its own data; low for a
    shortcut model that only works at home.
    """
    K = S.shape[0]
    tau = np.full(K, np.nan)
    for i in range(K):
        own = S[i, i]
        others = np.array([S[i, j] for j in range(K) if j != i], float)
        others = others[~np.isnan(others)]
        if np.isnan(own) or own <= 0.5 or others.size == 0:
            continue
        tau[i] = float(np.clip((others.mean() - 0.5) / (own - 0.5), 0.0, 1.0))
    return tau
