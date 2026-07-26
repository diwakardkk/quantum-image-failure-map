"""Sampling utilities for nested and balanced subsets."""

from __future__ import annotations

import numpy as np


def nested_balanced_indices(y: np.ndarray, sizes_per_class: list[int], seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class = {}
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        by_class[int(label)] = idx
    result = {}
    for size in sizes_per_class:
        chosen = np.concatenate([idx[: min(size, len(idx))] for idx in by_class.values()])
        rng.shuffle(chosen)
        result[size] = chosen
    return result

