"""Problem 2: trainability and gradient diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_gradients(run_dir: Path, dataset: str) -> pd.DataFrame:
    rows = []
    for path in (run_dir / "gradients").glob("*_initial_gradients.npy"):
        grad = np.abs(np.load(path)).reshape(-1)
        rows.append(
            {
                "problem": 2,
                "dataset": dataset,
                "model": "vqc",
                "gradient_file": str(path.relative_to(run_dir)),
                "mean_abs_gradient": float(np.mean(grad)),
                "median_abs_gradient": float(np.median(grad)),
                "gradient_variance": float(np.var(grad)),
                "log10_gradient_magnitude": float(np.log10(np.mean(grad) + 1e-12)),
                "fraction_below_1e_4": float(np.mean(grad < 1e-4)),
                "fraction_below_1e_6": float(np.mean(grad < 1e-6)),
                "failure_indicator": float(np.mean(grad < 1e-6) > 0.8),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / f"{dataset}_P2_trainability_metrics.csv", index=False)
    return df

