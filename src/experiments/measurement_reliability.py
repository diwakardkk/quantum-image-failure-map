"""Problem 4: finite-shot reliability diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_reliability_diagnostics(run_dir: Path, dataset: str, seed: int, repeats: int = 5, shots: list[int] | None = None) -> pd.DataFrame:
    shots = shots or [64, 256]
    prediction_files = list((run_dir / "predictions").glob("*_predictions.csv"))
    rows = []
    rng = np.random.default_rng(seed)
    for path in prediction_files:
        pred = pd.read_csv(path)
        if pred["dataset"].iloc[0] != dataset:
            continue
        p = pred["probability"].to_numpy()
        analytic_label = p >= 0.5
        for shot in shots:
            repeated = rng.binomial(shot, np.clip(p, 0, 1), size=(repeats, len(p))) / shot
            labels = repeated >= 0.5
            flip = np.mean(labels != analytic_label[None, :])
            rows.append(
                {
                    "problem": 4,
                    "dataset": dataset,
                    "model": pred["model"].iloc[0],
                    "shots": shot,
                    "repeats": repeats,
                    "flip_rate": float(flip),
                    "probability_std": float(np.mean(np.std(repeated, axis=0))),
                    "shot_normalized_cost": int(shot * repeats * len(p)),
                    "failure_indicator": float(flip > 0.05),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / f"{dataset}_P4_reliability_metrics.csv", index=False)
    return df

