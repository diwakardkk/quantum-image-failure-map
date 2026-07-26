"""Image preprocessing fitted strictly on training data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSet:
    name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    metadata: dict[str, Any]
    reconstruction_train: np.ndarray | None = None
    reconstruction_test: np.ndarray | None = None


def flatten_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def resize_features(x: np.ndarray, size: int) -> np.ndarray:
    resized = np.stack([resize(img[0], (size, size), anti_aliasing=True) for img in x])
    return resized.reshape(len(x), -1).astype(np.float32)


def patch_statistics(x: np.ndarray, grid: int) -> np.ndarray:
    n, _, h, w = x.shape
    rows = np.array_split(np.arange(h), grid)
    cols = np.array_split(np.arange(w), grid)
    feats = []
    for img in x[:, 0]:
        values = []
        for r in rows:
            for c in cols:
                patch = img[np.ix_(r, c)]
                values.extend([patch.mean(), patch.std()])
        feats.append(values)
    return np.asarray(feats, dtype=np.float32)


def fit_transform_variant(
    variant: str, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, artifact_dir: Path
) -> FeatureSet:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if variant.startswith("resize_"):
        size = int(variant.split("_")[1])
        return FeatureSet(
            variant,
            resize_features(x_train, size),
            resize_features(x_val, size),
            resize_features(x_test, size),
            {"fit_on": "none", "size": size},
        )
    if variant.startswith("pca_"):
        n_components = int(variant.split("_")[1])
        scaler = StandardScaler()
        train_flat = flatten_images(x_train)
        scaler.fit(train_flat)
        max_components = min(n_components, len(x_train), train_flat.shape[1])
        pca = PCA(n_components=max_components, random_state=0)
        pca.fit(scaler.transform(train_flat))
        joblib.dump({"scaler": scaler, "pca": pca}, artifact_dir / f"{variant}.joblib")
        train_z = pca.transform(scaler.transform(train_flat))
        val_z = pca.transform(scaler.transform(flatten_images(x_val)))
        test_z = pca.transform(scaler.transform(flatten_images(x_test)))
        rec_test = scaler.inverse_transform(pca.inverse_transform(test_z)).reshape(x_test.shape)
        rec_train = scaler.inverse_transform(pca.inverse_transform(train_z)).reshape(x_train.shape)
        return FeatureSet(
            variant,
            train_z.astype(np.float32),
            val_z.astype(np.float32),
            test_z.astype(np.float32),
            {
                "fit_on": "train",
                "requested_components": n_components,
                "components": max_components,
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
            },
            rec_train.astype(np.float32),
            rec_test.astype(np.float32),
        )
    if variant.startswith("patch_"):
        grid = int(variant.split("_")[1].replace("x", ""))
        return FeatureSet(
            variant,
            patch_statistics(x_train, grid),
            patch_statistics(x_val, grid),
            patch_statistics(x_test, grid),
            {"fit_on": "none", "grid": grid},
        )
    if variant == "flatten":
        return FeatureSet(variant, flatten_images(x_train), flatten_images(x_val), flatten_images(x_test), {"fit_on": "none"})
    raise ValueError(f"Unknown preprocessing variant: {variant}")


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray | None) -> dict[str, float]:
    if reconstructed is None:
        return {}
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    mse = float(np.mean((original - reconstructed) ** 2))
    norm = float(mse / (np.mean(original**2) + 1e-12))
    psnr = float(peak_signal_noise_ratio(original, reconstructed, data_range=1.0))
    ssim_values = [
        structural_similarity(original[i, 0], reconstructed[i, 0], data_range=1.0)
        for i in range(min(len(original), 50))
    ]
    return {"mse": mse, "normalized_reconstruction_error": norm, "psnr": psnr, "ssim": float(np.mean(ssim_values))}


def save_features(feature_set: FeatureSet, run_dir: Path, dataset: str) -> Path:
    path = run_dir / "processed" / f"{dataset}_{feature_set.name}.npz"
    np.savez_compressed(
        path,
        x_train=feature_set.x_train,
        x_val=feature_set.x_val,
        x_test=feature_set.x_test,
        reconstruction_train=feature_set.reconstruction_train if feature_set.reconstruction_train is not None else np.array([]),
        reconstruction_test=feature_set.reconstruction_test if feature_set.reconstruction_test is not None else np.array([]),
    )
    return path

