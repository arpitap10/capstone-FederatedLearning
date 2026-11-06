"""
hospital_transforms.py - preprocessing / augmentation, plus the pixel-artifact
generators that are retained ONLY for the optional appendix experiment (§17).

The main cue-coupling pillar does NOT inject pixels: the shortcut there is a
real-image sampling bias.  The artifact generators below reproduce the earlier
synthetic-shortcut study for comparison and are not used by default.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torchvision.transforms as T

import config

# ImageNet normalisation (DenseNet-121 pretrained).
_NORM = T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)


def get_transform(train: bool, img_size: int = config.IMG_SIZE) -> T.Compose:
    """Standard CheXNet-style preprocessing.

    Augmentation is deliberately mild (small rotation + affine jitter, no
    left-right flip, which would swap laterality and interact with the view
    cue).  Grayscale is replicated to 3 channels for the ImageNet backbone.
    """
    common_tail = [
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        _NORM,
    ]
    if train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomRotation(7),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            *common_tail,
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        *common_tail,
    ])


# ======================================================================
# OPTIONAL pixel-artifact generators (appendix only)
# Each takes a PIL grayscale image and returns a PIL grayscale image.
# ======================================================================
def _to_arr(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L")).copy()


def add_corner_marker(img: Image.Image, size: int = 32, value: int = 255) -> Image.Image:
    arr = _to_arr(img)
    h, w = arr.shape[:2]
    s = min(size, h, w)
    arr[0:s, 0:s] = value
    return Image.fromarray(arr)


def add_side_border(img: Image.Image, width: int = 28, value: int = 0) -> Image.Image:
    arr = _to_arr(img)
    arr[:, :width] = value
    return Image.fromarray(arr)


def adjust_gamma(img: Image.Image, gamma: float = 1.5) -> Image.Image:
    arr = _to_arr(img).astype(np.float32) / 255.0
    arr = 255.0 * (arr ** (1.0 / gamma))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_gaussian_noise(img: Image.Image, sigma: float = 12.0) -> Image.Image:
    arr = _to_arr(img).astype(np.float32)
    arr = arr + np.random.normal(0, sigma, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


ARTIFACTS = {
    "corner_marker": add_corner_marker,
    "side_border": add_side_border,
    "gamma": adjust_gamma,
    "noise": add_gaussian_noise,
}
