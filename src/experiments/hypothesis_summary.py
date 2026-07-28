"""Transparent v2 hypothesis support classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "v2_failure_isolation"


def build_hypothesis_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    datasets = _datasets(run_dir)
    for dataset in datasets:
        rows.extend(_dataset_rows(run_dir, dataset))
    df = pd.DataFrame(rows)
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics / "hypothesis_summary.csv", index=False)
    (metrics / "hypothesis_summary.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")
    return df


def _datasets(run_dir: Path) -> list[str]:
    names: set[str] = set()
    for path in (run_dir / "metrics").glob("*_P*_v2_*.csv"):
        names.add(path.name.split("_P")[0])
    return sorted(names)


def _status(value: float, strong: float, weak: float, reverse: bool = False) -> str:
    if np.isnan(value):
        return "inconclusive"
    if reverse:
        return "supported" if value <= strong else "partially_supported" if value <= weak else "unsupported"
    return "supported" if value >= strong else "partially_supported" if value >= weak else "unsupported"


def _dataset_rows(run_dir: Path, dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    p1 = _read(run_dir, f"{dataset}_P1_v2_probe_accuracy.csv")
    best_probe = float(p1["accuracy"].max()) if not p1.empty else np.nan
    rows.append(_row(dataset, "H1_representation", "partially_supported" if best_probe >= 0.7 else "inconclusive", "best_probe_accuracy", best_probe, "High probe accuracy means information remains; inspect utilisation gap separately."))
    p2 = _read(run_dir, f"{dataset}_P2_v2_gradient_scaling.csv")
    near_zero = float(p2["near_zero_fraction_1e_6"].mean()) if not p2.empty and "near_zero_fraction_1e_6" in p2 else np.nan
    rows.append(_row(dataset, "H2_trainability", _status(near_zero, 0.5, 0.2), "mean_near_zero_fraction_1e_6", near_zero, "Reference small VQC should remain separately interpreted."))
    p3 = _read(run_dir, f"{dataset}_P3_v2_spatial_robustness.csv")
    flip = float(p3["prediction_flip_rate_correct_only"].mean()) if not p3.empty else np.nan
    rows.append(_row(dataset, "H3_spatial", _status(flip, 0.10, 0.05), "mean_correct_only_flip_rate", flip, "Model-based transformed prediction diagnostic."))
    p4 = _read(run_dir, f"{dataset}_P4_v2_margin_reliability.csv")
    flip_correct = float(p4["Flip_correct"].mean()) if not p4.empty else np.nan
    rows.append(_row(dataset, "H4_measurement", _status(flip_correct, 0.10, 0.05), "mean_Flip_correct", flip_correct, "Finite-shot instability on analytically correct subset."))
    p5 = _read(run_dir, f"{dataset}_P5_v2_low_data_utility.csv")
    best_gap = np.nan
    if not p5.empty:
        best = p5.groupby("model")["accuracy"].mean()
        best_gap = float(best.max() - best.get("vqc", np.nan)) if "vqc" in best else np.nan
    rows.append(_row(dataset, "H5_utility", _status(best_gap, 0.05, 0.025), "best_classical_minus_vqc_accuracy", best_gap, "Matched-information and low-data utility comparison."))
    return rows


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / "metrics" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _row(dataset: str, hypothesis: str, status: str, metric: str, value: float, notes: str) -> dict[str, object]:
    return {"dataset": dataset, "hypothesis": hypothesis, "status": status, "primary_metric": metric, "primary_value": value, "CI_low": np.nan, "CI_high": np.nan, "supporting_metrics": "{}", "failure_boundary": "", "notes": notes, "protocol_version": PROTOCOL}
