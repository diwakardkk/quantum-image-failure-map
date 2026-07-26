"""Calibration bin generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    pred = prob >= 0.5
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (prob >= left) & (prob < right if right < 1 else prob <= right)
        rows.append(
            {
                "bin": i,
                "left": left,
                "right": right,
                "count": int(mask.sum()),
                "confidence": float(prob[mask].mean()) if mask.any() else np.nan,
                "accuracy": float((pred[mask] == y_true[mask]).mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)

