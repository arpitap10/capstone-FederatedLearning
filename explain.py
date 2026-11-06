"""
explain.py - explanation analysis for the shortcut study (plan 8 / E11).

Two kinds of evidence about *where* the model looks are contrasted here:

  * Grad-CAM, a gradient-based saliency map over the last convolutional block;
  * the behavioural context-ablation test of context_audit.py, which measures
    how AUROC survives blanking the lung centre vs the border.

Following Arun et al. (2021, "Assessing the (Un)Trustworthiness of Saliency Maps
for Localizing Abnormalities in Medical Imaging"), saliency maps such as
Grad-CAM are unreliable localisers: a map that looks lung-centred is NOT
evidence that the model actually relies on lung content.  We therefore treat the
behavioural ablation test as the PRIMARY evidence of cue reliance and use
Grad-CAM only as an illustrative, secondary signal.  `compare_gradcam_vs_
behavioural` makes that contrast explicit and will flag the case where the
behavioural test detects border reliance that the saliency map fails to show.

matplotlib runs under the headless Agg backend; nothing in this module has side
effects on import beyond selecting that backend.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

from model import denormalize_to_gray
from context_audit import context_auroc, context_leak
from utils import get_logger

_LOG = get_logger("explain")


# ======================================================================
# GRAD-CAM
# ======================================================================
class GradCAM:
    """Grad-CAM for the DenseNet-121 single-logit binary model.

    The target layer defaults to the final feature block `net.features.norm5`
    (the BatchNorm after the last denseblock, 1024 channels at 7x7), which is
    the standard choice of "last conv feature block" for DenseNet.  Pass an
    explicit `nn.Module` as `target_layer` to override.

    Usage::

        cam_fn = GradCAM(model)
        cam, prob = cam_fn(input_tensor)   # input_tensor: (1,3,H,W) or (3,H,W)
        cam_fn.remove()                    # or use as a context manager

    The head is a single logit, so the target score is that one logit (its
    sigmoid is the pneumonia probability).  CAM is returned as an HxW float array
    normalised to [0, 1] at the input resolution.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None \
            else self._default_target_layer(model)
        if self.target_layer is None:
            raise ValueError("could not locate a Grad-CAM target layer; pass "
                             "target_layer explicitly")
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register()

    @staticmethod
    def _default_target_layer(model):
        feats = getattr(model, "features", None)
        if feats is None:
            return None
        # Prefer the final BatchNorm (norm5); fall back to the last denseblock.
        if hasattr(feats, "norm5"):
            return feats.norm5
        last = None
        for name, child in feats.named_children():
            if name.startswith("denseblock"):
                last = child
        return last

    def _register(self):
        def fwd_hook(_module, _inp, out):
            self._activations = out

        def bwd_hook(_module, _grad_in, grad_out):
            self._gradients = grad_out[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        # full_backward_hook is the non-deprecated API on modern torch.
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._handles.append(
                self.target_layer.register_full_backward_hook(bwd_hook))
        else:  # pragma: no cover - very old torch
            self._handles.append(
                self.target_layer.register_backward_hook(bwd_hook))

    def remove(self):
        """Detach all hooks.  Safe to call more than once."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def __call__(self, input_tensor: torch.Tensor):
        """Return (cam: HxW float in [0,1], prob: float) for one image.

        Gradients are REQUIRED here, so this deliberately runs outside any
        torch.no_grad context.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.size(0) != 1:
            raise ValueError("GradCAM.__call__ expects a single image "
                             f"(got batch of {input_tensor.size(0)})")

        device = next(self.model.parameters()).device
        x = input_tensor.to(device)

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logit = self.model(x).squeeze(1)        # (1,)
        score = logit[0]
        score.backward()

        acts = self._activations.detach()        # (1,C,h,w)
        grads = self._gradients.detach()         # (1,C,h,w)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP over spatial dims
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)

        H, W = x.shape[-2:]
        cam = torch.nn.functional.interpolate(
            cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        denom = float(cam.max().item())
        if denom > 0:
            cam = cam / denom
        cam_np = cam.detach().cpu().numpy().astype(np.float64)

        prob = float(torch.sigmoid(logit)[0].item())
        if was_training:
            self.model.train()
        return cam_np, prob


# ======================================================================
# CAM SUMMARIES
# ======================================================================
def gradcam_border_mass(cam: np.ndarray, border_frac: float = 0.25) -> float:
    """Fraction of total Grad-CAM mass in the outer border ring.

    Mirrors the geometry of context_audit.mask_batch: the "centre" is the inner
    box after trimming `border_frac` off each side, the "border" is everything
    outside it.  A high value => attention concentrated on the periphery
    (markers, collimation, text) rather than the lungs.
    """
    cam = np.asarray(cam, dtype=np.float64)
    cam = np.clip(cam, 0.0, None)
    h, w = cam.shape
    bh, bw = int(border_frac * h), int(border_frac * w)
    total = float(cam.sum())
    if total <= 0:
        return float("nan")
    centre = cam[bh:h - bh, bw:w - bw]
    centre_mass = float(centre.sum())
    border_mass = total - centre_mass
    return border_mass / total


# ======================================================================
# VISUALISATION
# ======================================================================
def overlay_and_save(image_gray: np.ndarray, cam: np.ndarray, path: str,
                     title: Optional[str] = None) -> None:
    """Overlay a Grad-CAM heatmap on a de-normalised grayscale CXR and save.

    `image_gray` is an HxW array in [0,255] (as returned by
    model.denormalize_to_gray for one image); `cam` is an HxW array in [0,1].
    """
    image_gray = np.asarray(image_gray, dtype=np.float64)
    cam = np.asarray(cam, dtype=np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_gray, cmap="gray", vmin=0, vmax=255)
    ax.imshow(cam, cmap="jet", alpha=0.4, vmin=0.0, vmax=1.0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# GRAD-CAM vs BEHAVIOURAL CONTRAST  (plan E11)
# ======================================================================
def compare_gradcam_vs_behavioural(model, loader, device, out_dir: str,
                                   n_examples: int = 8) -> Dict:
    """Contrast Grad-CAM saliency with the behavioural context-ablation test.

    For up to `n_examples` images we compute the per-image Grad-CAM border-mass
    and save an overlay panel.  Independently, over the WHOLE loader we compute
    the model's behavioural context signals: border-only and centre-only AUROC
    (context_audit.context_auroc) and the derived border context-leak.

    The behavioural test is the primary evidence (Arun et al. 2021); the verdict
    highlights when it flags border/cue reliance that the averaged saliency map
    does not obviously show.

    Returns a dict with mean_gradcam_border_mass, per_image_gradcam_border_mass,
    border_only_auroc, centre_only_auroc, context_leak, saved panel paths, and a
    short textual `verdict`.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)

    # --- behavioural test over the full loader (no grads needed) ----------
    border_auroc = context_auroc(model, loader, device, region="border")
    centre_auroc = context_auroc(model, loader, device, region="centre")
    leak = context_leak(model, loader, device)

    # --- Grad-CAM on a handful of example images (grads needed) -----------
    cam_fn = GradCAM(model)
    border_masses: List[float] = []
    panels: List[str] = []
    try:
        collected = 0
        for images, y, _cues, pids in loader:
            grays = denormalize_to_gray(images)   # (N,H,W) in [0,255]
            for i in range(images.size(0)):
                if collected >= n_examples:
                    break
                single = images[i:i + 1]
                cam, prob = cam_fn(single)
                bm = gradcam_border_mass(cam)
                border_masses.append(bm)
                pid = pids[i] if i < len(pids) else collected
                path = os.path.join(out_dir, f"gradcam_{collected:02d}_{pid}.png")
                title = (f"p={prob:.2f} y={int(y[i])} "
                         f"border_mass={bm:.2f}")
                overlay_and_save(grays[i], cam, path, title=title)
                panels.append(path)
                collected += 1
            if collected >= n_examples:
                break
    finally:
        cam_fn.remove()

    valid = [m for m in border_masses if not np.isnan(m)]
    mean_bm = float(np.mean(valid)) if valid else float("nan")

    # --- verdict ----------------------------------------------------------
    verdict = _verdict(mean_bm, border_auroc, centre_auroc, leak)

    out = {
        "mean_gradcam_border_mass": mean_bm,
        "per_image_gradcam_border_mass": [float(m) for m in border_masses],
        "border_only_auroc": float(border_auroc),
        "centre_only_auroc": float(centre_auroc),
        "context_leak": float(leak),
        "n_examples": int(len(panels)),
        "panels": panels,
        "verdict": verdict,
    }
    _LOG.info("gradcam-vs-behavioural: border_only_auroc=%.3f centre_only_auroc="
              "%.3f leak=%.3f mean_cam_border_mass=%.3f",
              border_auroc, centre_auroc, leak, mean_bm)
    _LOG.info("verdict: %s", verdict)
    return out


def _verdict(mean_border_mass: float, border_auroc: float,
             centre_auroc: float, leak: float,
             leak_thresh: float = 0.30, cam_thresh: float = 0.5) -> str:
    """Human-readable interpretation contrasting the two signals."""
    behav_flags = (not np.isnan(leak)) and leak >= leak_thresh
    cam_shows = (not np.isnan(mean_border_mass)) and mean_border_mass >= cam_thresh

    if behav_flags and not cam_shows:
        return ("Behavioural test FLAGS border/cue reliance "
                f"(border-only AUROC={border_auroc:.3f}, leak={leak:.3f}) that "
                f"Grad-CAM does NOT show (mean border-mass={mean_border_mass:.3f}): "
                "a lung-looking saliency map is not evidence of lung reliance - "
                "trust the behavioural test (Arun et al. 2021).")
    if behav_flags and cam_shows:
        return ("Both signals agree the model relies on peripheral cues "
                f"(border-only AUROC={border_auroc:.3f}, leak={leak:.3f}; "
                f"CAM border-mass={mean_border_mass:.3f}).")
    if (not behav_flags) and cam_shows:
        return ("Grad-CAM looks border-heavy "
                f"(border-mass={mean_border_mass:.3f}) but the behavioural test "
                f"does NOT flag reliance (border-only AUROC={border_auroc:.3f}, "
                f"leak={leak:.3f}); the saliency map is likely misleading.")
    return ("No border/cue reliance detected: behavioural centre-only AUROC="
            f"{centre_auroc:.3f} is retained while border-only AUROC="
            f"{border_auroc:.3f} is near chance (leak={leak:.3f}).")
