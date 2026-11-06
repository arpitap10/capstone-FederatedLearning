"""
context_audit.py - cue-agnostic context-reliance test (plan 8.2c).

Blank out either the lung region (keep the border) or the border (keep the
centre), replacing the removed pixels with a FIXED constant gray.  A per-image
fill (e.g. the image median) would leak label information, so a constant is
mandatory.

A model that reads lung evidence keeps its AUROC on the centre-only image and
drops to chance on the border-only image.  A model exploiting peripheral cues
(markers, collimation, text) keeps AUROC on the border-only image.

    context_auroc(model, loader, device, region="border")  ->  AUROC
    context_leak = max(0, 2*(border_only_AUROC - 0.5))      in [0,1]
"""

from __future__ import annotations

import numpy as np
import torch

import config
from evaluate import auroc

# normalised value of a constant gray g in [0,1], per channel
_MEAN = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)


def _fill_value(gray: float) -> torch.Tensor:
    g = torch.full((1, 3, 1, 1), float(gray))
    return (g - _MEAN) / _STD


def mask_batch(images: torch.Tensor, region: str, border_frac: float = 0.25,
               fill_gray: float = 0.5) -> torch.Tensor:
    """Return a copy of `images` with one region replaced by constant gray.

    region="border"  keeps the outer ring, blanks the centre  (border-only).
    region="centre"  keeps the inner box,  blanks the ring     (centre-only).
    """
    n, c, h, w = images.shape
    bh, bw = int(border_frac * h), int(border_frac * w)
    fill = _fill_value(fill_gray).to(images.device, images.dtype)
    out = images.clone()
    centre = (slice(None), slice(None), slice(bh, h - bh), slice(bw, w - bw))
    if region == "border":
        out[centre] = fill
    elif region == "centre":
        # blank everything, then restore the centre box
        out[:] = fill
        out[centre] = images[centre]
    else:
        raise ValueError("region must be 'border' or 'centre'")
    return out


@torch.no_grad()
def context_auroc(model, loader, device, region: str = "border",
                  border_frac: float = 0.25, fill_gray: float = 0.5) -> float:
    model.eval()
    probs, targets = [], []
    for images, y, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masked = mask_batch(images, region, border_frac, fill_gray)
        p = torch.sigmoid(model(masked).squeeze(1)).cpu().numpy()
        probs.append(p)
        targets.append(np.asarray(y, dtype=np.float64))
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    return auroc(targets, probs)


def context_leak(model, loader, device, border_frac: float = 0.25,
                 fill_gray: float = 0.5) -> float:
    a = context_auroc(model, loader, device, "border", border_frac, fill_gray)
    if np.isnan(a):
        return float("nan")
    return float(max(0.0, 2.0 * (a - 0.5)))
