"""Dataset loading and deterministic subset construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, Any]


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _normalize_images(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim == 4 and x.shape[-1] == 1:
        x = np.transpose(x, (0, 3, 1, 2))
    if x.max() > 1.0:
        x = x / 255.0
    return x.astype(np.float32)


def _balanced_indices(labels: np.ndarray, per_class: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for label in np.unique(labels):
        cls = np.flatnonzero(labels == label)
        rng.shuffle(cls)
        indices.append(cls[: per_class or len(cls)])
    chosen = np.concatenate(indices)
    rng.shuffle(chosen)
    return chosen


def _split_train_val(labels: np.ndarray, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, val_idx


def load_fashion_mnist(data_root: Path, config: dict[str, Any], mode: str, seed: int) -> DatasetBundle:
    from torchvision.datasets import FashionMNIST

    train = FashionMNIST(root=str(data_root), train=True, download=True)
    test = FashionMNIST(root=str(data_root), train=False, download=True)
    classes = config.get("datasets", {}).get("fashion_mnist", {}).get("classes", [0, 6])

    train_x = train.data.numpy()
    train_y_raw = np.asarray(train.targets)
    test_x = test.data.numpy()
    test_y_raw = np.asarray(test.targets)

    train_mask = np.isin(train_y_raw, classes)
    test_mask = np.isin(test_y_raw, classes)
    train_x, train_y_raw = train_x[train_mask], train_y_raw[train_mask]
    test_x, test_y_raw = test_x[test_mask], test_y_raw[test_mask]
    mapping = {classes[0]: 0, classes[1]: 1}
    train_y = np.vectorize(mapping.get)(train_y_raw).astype(int)
    test_y = np.vectorize(mapping.get)(test_y_raw).astype(int)

    samples = config.get("subsets", {}).get("samples_per_class", {}).get(mode)
    subset_idx = _balanced_indices(train_y, samples, seed)
    train_x, train_y = train_x[subset_idx], train_y[subset_idx]
    inner_train, val_idx = _split_train_val(
        train_y, float(config.get("subsets", {}).get("validation_fraction", 0.2)), seed
    )
    test_idx = _balanced_indices(test_y, samples, seed + 1)
    return DatasetBundle(
        name="fashion_mnist",
        x_train=_normalize_images(train_x[inner_train]),
        y_train=train_y[inner_train],
        x_val=_normalize_images(train_x[val_idx]),
        y_val=train_y[val_idx],
        x_test=_normalize_images(test_x[test_idx]),
        y_test=test_y[test_idx],
        train_indices=subset_idx[inner_train],
        val_indices=subset_idx[val_idx],
        test_indices=test_idx,
        metadata={"classes": classes, "official_test_preserved": True},
    )


def load_pneumoniamnist(data_root: Path, config: dict[str, Any], mode: str, seed: int) -> DatasetBundle:
    import medmnist
    from medmnist import INFO

    data_cls = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
    train = data_cls(split="train", root=str(data_root), download=True)
    val = data_cls(split="val", root=str(data_root), download=True)
    test = data_cls(split="test", root=str(data_root), download=True)
    x_train, y_train = train.imgs, np.asarray(train.labels).reshape(-1).astype(int)
    x_val, y_val = val.imgs, np.asarray(val.labels).reshape(-1).astype(int)
    x_test, y_test = test.imgs, np.asarray(test.labels).reshape(-1).astype(int)
    samples = config.get("subsets", {}).get("samples_per_class", {}).get(mode)
    train_idx = _balanced_indices(y_train, samples, seed)
    # Primary evaluation keeps official, naturally imbalanced validation/test distributions.
    max_eval = samples * 2 if samples else None
    val_idx = np.arange(len(y_val))[:max_eval]
    test_idx = np.arange(len(y_test))[:max_eval]
    return DatasetBundle(
        name="pneumoniamnist",
        x_train=_normalize_images(x_train[train_idx]),
        y_train=y_train[train_idx],
        x_val=_normalize_images(x_val[val_idx]),
        y_val=y_val[val_idx],
        x_test=_normalize_images(x_test[test_idx]),
        y_test=y_test[test_idx],
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        metadata={"official_partitions": True, "primary_test_distribution": "natural"},
    )


def load_dataset(name: str, data_root: Path, config: dict[str, Any], mode: str, seed: int) -> DatasetBundle:
    if name == "fashion_mnist":
        return load_fashion_mnist(data_root, config, mode, seed)
    if name == "pneumoniamnist":
        return load_pneumoniamnist(data_root, config, mode, seed)
    raise ValueError(f"Unknown dataset: {name}")


def save_dataset_artifacts(bundle: DatasetBundle, run_dir: Path) -> None:
    raw = run_dir / "raw" / bundle.name
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw / "split_indices.npz",
        train_indices=bundle.train_indices,
        val_indices=bundle.val_indices,
        test_indices=bundle.test_indices,
    )
    summary = pd.DataFrame(
        [
            {
                "dataset": bundle.name,
                "split": split,
                "n_samples": len(y),
                "label_counts": dict(zip(*np.unique(y, return_counts=True))),
                "image_mean": float(x.mean()),
                "image_std": float(x.std()),
                "split_hash": array_hash(idx),
            }
            for split, x, y, idx in [
                ("train", bundle.x_train, bundle.y_train, bundle.train_indices),
                ("val", bundle.x_val, bundle.y_val, bundle.val_indices),
                ("test", bundle.x_test, bundle.y_test, bundle.test_indices),
            ]
        ]
    )
    summary.to_csv(raw / "dataset_summary.csv", index=False)

