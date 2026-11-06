"""
model.py - DenseNet-121 (CheXNet-style) binary classifier and the state-dict
arithmetic the federated aggregator needs.

A single-logit head is used (BCEWithLogits), which is the standard CheXNet
setup for one finding.  A GroupNorm variant is provided because record-level
DP-SGD (Opacus) cannot handle BatchNorm's cross-sample running statistics.

State-dict arithmetic (`state_sub`, `state_axpy`, ...) operates ONLY on floating
tensors (weights + BN running stats).  Integer buffers such as
`num_batches_tracked` are copied from a reference, never averaged.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import config

StateDict = "OrderedDict[str, torch.Tensor]"


# ======================================================================
# BUILD
# ======================================================================
def build_model(pretrained: bool = True, norm_layer: str = "batchnorm",
                dropout: float = 0.0) -> nn.Module:
    """DenseNet-121 with a single-logit classifier head.

    The head is always Dropout+Linear so the state_dict structure is identical
    regardless of the dropout rate (dropout has no parameters) - models built for
    evaluation (dropout=0) can load weights trained with dropout>0 and vice versa.
    """
    try:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        net = models.densenet121(weights=weights)
    except (AttributeError, TypeError):
        # Older torchvision without the weights enum.
        net = models.densenet121(pretrained=pretrained)
    in_features = net.classifier.in_features
    net.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    if norm_layer == "groupnorm":
        net = convert_bn_to_gn(net)
    elif norm_layer != "batchnorm":
        raise ValueError(f"norm_layer must be 'batchnorm' or 'groupnorm', got {norm_layer}")
    return net


def convert_bn_to_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """Recursively replace every BatchNorm2d with GroupNorm (DP-safe).

    Weights/bias (affine params) are copied so an ImageNet-initialised net keeps
    its learned scale/shift; running stats are discarded (GroupNorm has none).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            g = num_groups
            while C % g != 0 and g > 1:
                g //= 2
            gn = nn.GroupNorm(g, C, affine=True)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)
    return module


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ======================================================================
# STATE-DICT ARITHMETIC  (for FedAvg-style aggregation)
# ======================================================================
def _is_float(t: torch.Tensor) -> bool:
    return t.is_floating_point()


def clone_state(state: StateDict) -> StateDict:
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def get_state(model: nn.Module) -> StateDict:
    """Snapshot weights on CPU so all aggregation arithmetic is device-consistent
    (a GPU-trained client and the CPU global state must subtract cleanly).  A CPU
    state loads back into a GPU model fine - load_state_dict copies across."""
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def load_state(model: nn.Module, state: StateDict) -> None:
    model.load_state_dict(state, strict=True)


def state_sub(a: StateDict, b: StateDict) -> StateDict:
    """a - b on float tensors; non-float entries carried from `a`."""
    out = OrderedDict()
    for k in a:
        out[k] = (a[k] - b[k]) if _is_float(a[k]) else a[k].clone()
    return out


def state_zeros_like(ref: StateDict) -> StateDict:
    out = OrderedDict()
    for k, v in ref.items():
        out[k] = torch.zeros_like(v) if _is_float(v) else v.clone()
    return out


def state_axpy(acc: StateDict, scale: float, delta: StateDict) -> StateDict:
    """acc += scale * delta on float tensors (in place on a copy)."""
    out = OrderedDict()
    for k in acc:
        if _is_float(acc[k]):
            out[k] = acc[k] + scale * delta[k]
        else:
            out[k] = acc[k].clone()
    return out


def state_add(a: StateDict, b: StateDict) -> StateDict:
    return state_axpy(a, 1.0, b)


def state_scale(a: StateDict, scale: float) -> StateDict:
    out = OrderedDict()
    for k, v in a.items():
        out[k] = (v * scale) if _is_float(v) else v.clone()
    return out


def state_l2_norm(delta: StateDict) -> float:
    """L2 norm over the concatenation of all float tensors."""
    sq = 0.0
    for v in delta.values():
        if _is_float(v):
            sq += float(torch.sum(v.double() ** 2).item())
    return float(np.sqrt(sq))


def clip_state(delta: StateDict, max_norm: float) -> StateDict:
    """Scale a delta so its global L2 norm <= max_norm (DP update clipping)."""
    n = state_l2_norm(delta)
    if max_norm is None or n <= max_norm or n == 0.0:
        return clone_state(delta)
    return state_scale(delta, max_norm / n)


# ======================================================================
# IMAGE DE-NORMALISATION  (normalised tensor -> [0,255] grayscale)
# ======================================================================
_MEAN = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)


def denormalize_to_gray(batch: torch.Tensor) -> np.ndarray:
    """(N,3,H,W) normalised -> (N,H,W) uint8-scale float in [0,255].

    CXRs are grayscale replicated across 3 channels; channel 0 is taken.
    This must match the representation cue_bank.fit_image_cutpoints was fit on.
    """
    x = batch.detach().cpu() * _STD + _MEAN       # back to [0,1]
    x = x[:, 0, :, :].clamp(0, 1) * 255.0
    return x.numpy()


# ======================================================================
# INFERENCE
# ======================================================================
@torch.no_grad()
def predict_proba(model: nn.Module, loader, device, cue_bank=None) -> Dict:
    """Run the model over a loader and collect everything the audit needs.

    The loader must yield batches of (image, target, cue_dict, patient_id) as
    produced by hospital_dataset.collate_batch.

    Returns a dict with numpy arrays:
        probs   (N,)      sigmoid probabilities
        targets (N,)      0/1 ground truth
        ids     list[str] patient ids
        cues    {name: (N,) float array}  metadata cues (from CSV) +
                image cues (computed here via cue_bank); missing values are NaN
    """
    model.eval()
    probs, targets, ids = [], [], []
    cue_cols: Dict[str, list] = {}

    for images, y, cue_dict, pid in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        targets.append(np.asarray(y, dtype=np.float64))
        ids.extend(list(pid))

        # metadata cues come batched from the dataset (nullable -> NaN)
        for name, vals in cue_dict.items():
            arr = _to_float_with_nan(vals)
            cue_cols.setdefault(name, []).append(arr)

        # image cues computed on the fly from the de-normalised batch
        if cue_bank is not None and cue_bank.image_cues:
            grays = denormalize_to_gray(images)
            img_cues = cue_bank.eval_image_batch(grays)
            for name, arr in img_cues.items():
                cue_cols.setdefault(name, []).append(arr.astype(np.float64))

    out = {
        "probs": np.concatenate(probs) if probs else np.array([]),
        "targets": np.concatenate(targets) if targets else np.array([]),
        "ids": ids,
        "cues": {k: np.concatenate(v) for k, v in cue_cols.items()},
    }
    return out


def _to_float_with_nan(vals) -> np.ndarray:
    """Convert a batched cue column (tensor / list, possibly with sentinel -1 or
    None for missing) to a float array with NaN for missing."""
    if isinstance(vals, torch.Tensor):
        arr = vals.detach().cpu().numpy().astype(np.float64)
    else:
        arr = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
    arr[arr < 0] = np.nan  # dataset encodes "missing" as -1
    return arr
