"""Problem 5: practical utility summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_utility_summary(run_dir: Path, dataset: str) -> pd.DataFrame:
    metrics_path = run_dir / "metrics" / "primary_metrics.csv"
    if not metrics_path.exists():
        out = pd.DataFrame()
    else:
        df = pd.read_csv(metrics_path)
        df = df[df["dataset"] == dataset].copy()
        if df.empty:
            out = pd.DataFrame()
        else:
            params = df["trainable_parameters"] if "trainable_parameters" in df else 0
            out = df[["dataset", "model", "accuracy", "macro_f1"]].copy()
            out["trainable_parameters"] = params
            out["problem"] = 5
            out["accuracy_per_parameter"] = out["accuracy"] / (out["trainable_parameters"].fillna(0) + 1)
            out["wall_time_seconds"] = df["wall_time_seconds"] if "wall_time_seconds" in df else 0.0
            out["failure_indicator"] = 0.0
            classical = out[~out["model"].str.contains("vqc", case=False, na=False)]
            best_classical = classical["accuracy"].max() if not classical.empty else out["accuracy"].max()
            quantum_mask = out["model"].str.contains("vqc", case=False, na=False)
            out.loc[quantum_mask & (out["accuracy"] < best_classical), "failure_indicator"] = 1.0
    out.to_csv(run_dir / "metrics" / f"{dataset}_P5_utility_metrics.csv", index=False)
    return out

