"""Output verification for completed runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .result_store import RUN_SUBDIRS


EXPECTED_FIGURES = [
    "P1_F03_accuracy_vs_feature_count.png",
    "P2_F01_gradient_variance_vs_depth.png",
    "P3_F02_rotation_robustness.png",
    "P4_F01_flip_rate_vs_shots.png",
    "P5_F02_accuracy_vs_wall_time.png",
    "UF_F01_five_problem_dashboard.png",
]


def verify_run(run_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, message: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "message": message})

    add("run_dir_exists", run_dir.exists())
    for subdir in RUN_SUBDIRS:
        add(f"subdir_{subdir}", (run_dir / subdir).is_dir())
    add("config_resolved", (run_dir / "config_resolved.yaml").exists())
    manifest = run_dir / "metrics" / "experiment_manifest.parquet"
    add("manifest_exists", manifest.exists())
    if manifest.exists():
        mf = pd.read_parquet(manifest)
        add("manifest_no_duplicate_ids", not mf.get("configuration_id", pd.Series()).duplicated().any())
    pred_files = list((run_dir / "predictions").glob("*_predictions.csv"))
    add("prediction_file_exists", bool(pred_files))
    for path in pred_files:
        df = pd.read_csv(path)
        add(f"prediction_columns_{path.name}", {"label", "probability", "predicted_label", "configuration_id"}.issubset(df.columns))
        add(f"prediction_probabilities_{path.name}", np.isfinite(df["probability"]).all() and df["probability"].between(0, 1).all())
    primary = run_dir / "metrics" / "primary_metrics.csv"
    add("primary_metrics_exists", primary.exists())
    if primary.exists():
        df = pd.read_csv(primary)
        for col in ["accuracy", "balanced_accuracy", "macro_f1", "brier_score"]:
            add(f"primary_metric_{col}", col in df.columns)
    for fig in EXPECTED_FIGURES:
        path = run_dir / "figures" / fig
        ok = path.exists()
        if ok:
            try:
                img = mpimg.imread(path)
                ok = img.size > 0 and img.shape[0] > 0 and img.shape[1] > 0
            except Exception as exc:
                ok = False
                add(f"figure_readable_{fig}", False, str(exc))
                continue
        add(f"figure_{fig}", ok)
    _verify_v2(run_dir, add)
    passed = sum(c["ok"] for c in checks)
    report = {"passed": passed, "failed": len(checks) - passed, "checks": checks}
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Verification Report", "", f"Passed: {passed}", f"Failed: {len(checks) - passed}", ""]
    lines += [f"- [{'x' if c['ok'] else ' '}] {c['check']}: {c['message']}" for c in checks]
    (artifacts / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _verify_v2(run_dir: Path, add) -> None:  # type: ignore[no-untyped-def]
    cfg = run_dir / "config_resolved.yaml"
    is_v2 = False
    if cfg.exists():
        try:
            is_v2 = "v2_failure_isolation" in cfg.read_text(encoding="utf-8")
        except Exception:
            is_v2 = False
    if not is_v2 and not (run_dir / "metrics" / "hypothesis_summary.csv").exists():
        return
    required = [
        "problem1/compression_metrics.parquet",
        "problem1/probe_predictions.parquet",
        "problem1/utilisation_gap.parquet",
        "problem1/fidelity_summary.parquet",
        "problem1/compression_failure_boundary.parquet",
        "problem2/gradient_summary.parquet",
        "problem2/scaling_fit_results.parquet",
        "problem2/selected_training_runs.parquet",
        "problem3/transformed_predictions.parquet",
        "problem3/transformation_metrics.parquet",
        "problem3/failure_boundaries.parquet",
        "problem4/shot_predictions.parquet",
        "problem4/margin_bins.parquet",
        "problem4/flip_correct_summary.parquet",
        "problem4/shot_stability_boundary.parquet",
        "problem4/noise_results.parquet",
        "problem5/matched_information_results.parquet",
        "problem5/low_data_results.parquet",
        "problem5/resource_metrics.parquet",
        "problem5/pareto_results.parquet",
        "problem5/dominance_matrix.parquet",
        "metrics/hypothesis_summary.csv",
        "metrics/hypothesis_summary.json",
        "metrics/statistical_tests_v2.parquet",
    ]
    for rel in required:
        add(f"v2_file_{rel}", (run_dir / rel).exists())
    p1 = run_dir / "problem1" / "compression_metrics.parquet"
    if p1.exists():
        df = pd.read_parquet(p1)
        add("v2_p1_protocol_version", "protocol_version" in df.columns and df["protocol_version"].notna().all())
        add("v2_p1_pca_train_fit_metadata", {"pca_dimension", "cumulative_explained_variance"}.issubset(df.columns))
    p2 = run_dir / "problem2" / "gradient_summary.parquet"
    if p2.exists():
        df = pd.read_parquet(p2)
        add("v2_p2_raw_gradient_reference", "gradient_file" in df.columns)
        raw_ok = True
        for rel in df.get("gradient_file", pd.Series(dtype=str)).dropna():
            if rel and not (run_dir / rel).exists():
                raw_ok = False
                break
        add("v2_p2_raw_gradients_exist", raw_ok)
    p3 = run_dir / "problem3" / "transformed_predictions.parquet"
    if p3.exists():
        df = pd.read_parquet(p3)
        add("v2_p3_prediction_schema", {"sample_id", "label", "original_probability", "transformed_probability", "prediction_flip"}.issubset(df.columns))
    p4 = run_dir / "problem4" / "shot_predictions.parquet"
    if p4.exists():
        df = pd.read_parquet(p4)
        add("v2_p4_margin_schema", {"analytic_probability", "analytic_margin", "sample_flip_rate"}.issubset(df.columns))
        if "analytic_margin" in df:
            add("v2_p4_margin_valid", df["analytic_margin"].between(0, 0.5).all())
    p5 = run_dir / "problem5" / "low_data_results.parquet"
    if p5.exists():
        df = pd.read_parquet(p5)
        add("v2_p5_nested_size_schema", {"train_samples_per_class", "comparison_regime"}.issubset(df.columns))
