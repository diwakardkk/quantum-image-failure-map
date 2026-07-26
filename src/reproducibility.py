"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic_torch: bool
    notes: list[str]


def seed_everything(seed: int, deterministic_torch: bool = True) -> SeedState:
    notes: list[str] = []
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception as exc:  # pragma: no cover - depends on optional torch install
        notes.append(f"PyTorch deterministic seeding unavailable: {exc}")
    return SeedState(seed=seed, deterministic_torch=deterministic_torch, notes=notes)


def worker_seed(worker_id: int) -> None:
    seed = (np.random.get_state()[1][0] + worker_id) % 2**32
    random.seed(int(seed))
    np.random.seed(int(seed))

