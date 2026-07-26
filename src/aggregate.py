"""Result aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_metric_files(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in (run_dir / "metrics").glob("*_P*_*.csv"):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    primary = run_dir / "metrics" / "primary_metrics.csv"
    if primary.exists():
        frames.append(pd.read_csv(primary))
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    df.to_csv(run_dir / "metrics" / "aggregated_metrics.csv", index=False)
    return df

