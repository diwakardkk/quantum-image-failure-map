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
    passed = sum(c["ok"] for c in checks)
    report = {"passed": passed, "failed": len(checks) - passed, "checks": checks}
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Verification Report", "", f"Passed: {passed}", f"Failed: {len(checks) - passed}", ""]
    lines += [f"- [{'x' if c['ok'] else ' '}] {c['check']}: {c['message']}" for c in checks]
    (artifacts / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report

