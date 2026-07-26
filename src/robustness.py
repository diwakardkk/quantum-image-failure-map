"""Deterministic image transformations for spatial robustness."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate, shift


def translate_images(x: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    return np.stack([shift(img, shift=(0, dy, dx), order=1, mode="constant", cval=0.0) for img in x]).astype(np.float32)


def rotate_images(x: np.ndarray, degrees: float) -> np.ndarray:
    return np.stack([rotate(img, degrees, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0) for img in x]).astype(np.float32)


def central_occlusion(x: np.ndarray, percent: float) -> np.ndarray:
    out = x.copy()
    if percent <= 0:
        return out
    _, _, h, w = out.shape
    side = int(round(np.sqrt(percent / 100.0) * min(h, w)))
    r0 = (h - side) // 2
    c0 = (w - side) // 2
    out[:, :, r0 : r0 + side, c0 : c0 + side] = 0.0
    return out


def global_pixel_permutation(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = x.reshape(len(x), -1)
    order = np.arange(flat.shape[1])
    rng.shuffle(order)
    return flat[:, order].reshape(x.shape).astype(np.float32)

