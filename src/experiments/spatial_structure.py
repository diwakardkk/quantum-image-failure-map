"""Problem 3: spatial robustness diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..robustness import central_occlusion, rotate_images, translate_images


def run_spatial_diagnostics(run_dir: Path, dataset: str, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    # Mechanism-only image statistics; model-specific transformed predictions are added in full runs.
    for kind, value, transformed in [
        ("rotation", 15, rotate_images(x_test, 15)),
        ("translation", 2, translate_images(x_test, dx=2, dy=0)),
        ("occlusion", 20, central_occlusion(x_test, 20)),
    ]:
        np.savez_compressed(run_dir / "processed" / f"{dataset}_{kind}_{value}.npz", x=transformed, y=y_test)
        rows.append(
            {
                "problem": 3,
                "dataset": dataset,
                "model": "image_transform",
                "transformation": kind,
                "strength": value,
                "mean_absolute_change": float(np.mean(np.abs(transformed - x_test))),
                "robust_accuracy": np.nan,
                "failure_indicator": float(np.mean(np.abs(transformed - x_test)) > 0.1),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / f"{dataset}_P3_spatial_metrics.csv", index=False)
    return df

