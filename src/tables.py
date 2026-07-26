"""Publication table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_NAMES = [
    "T1_dataset_summary",
    "T2_reference_model_configuration",
    "T3_encoding_information_and_resource_results",
    "T4_trainability_and_gradient_results",
    "T5_spatial_robustness_results",
    "T6_shot_and_noise_reliability_results",
    "T7_classical_quantum_performance",
    "T8_end_to_end_resource_comparison",
    "T9_statistical_tests",
    "T10_failure_boundaries",
]


def _write_table(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    path_base.with_suffix(".md").write_text(_markdown_table(df), encoding="utf-8")
    path_base.with_suffix(".tex").write_text(df.to_latex(index=False, escape=True, longtable=False), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| status |\n| --- |\n| no rows |\n"
    columns = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def generate_tables(run_dir: Path) -> list[str]:
    primary = pd.read_csv(run_dir / "metrics" / "primary_metrics.csv") if (run_dir / "metrics" / "primary_metrics.csv").exists() else pd.DataFrame()
    aggregated = pd.read_csv(run_dir / "metrics" / "aggregated_metrics.csv") if (run_dir / "metrics" / "aggregated_metrics.csv").exists() else primary
    outputs = []
    content = {
        "T1_dataset_summary": _read_dataset_summaries(run_dir),
        "T2_reference_model_configuration": primary[["dataset", "model", "preprocessing"]].drop_duplicates() if not primary.empty else pd.DataFrame(),
        "T3_encoding_information_and_resource_results": aggregated[aggregated.get("problem", pd.Series(dtype=int)) == 1] if not aggregated.empty else pd.DataFrame(),
        "T4_trainability_and_gradient_results": aggregated[aggregated.get("problem", pd.Series(dtype=int)) == 2] if not aggregated.empty else pd.DataFrame(),
        "T5_spatial_robustness_results": aggregated[aggregated.get("problem", pd.Series(dtype=int)) == 3] if not aggregated.empty else pd.DataFrame(),
        "T6_shot_and_noise_reliability_results": aggregated[aggregated.get("problem", pd.Series(dtype=int)) == 4] if not aggregated.empty else pd.DataFrame(),
        "T7_classical_quantum_performance": primary,
        "T8_end_to_end_resource_comparison": primary[[c for c in ["dataset", "model", "accuracy", "wall_time_seconds", "trainable_parameters"] if c in primary]] if not primary.empty else pd.DataFrame(),
        "T9_statistical_tests": pd.DataFrame(columns=["comparison", "effect_size", "p_value", "ci_low", "ci_high"]),
        "T10_failure_boundaries": aggregated[[c for c in ["dataset", "problem", "model", "failure_indicator"] if c in aggregated]] if not aggregated.empty else pd.DataFrame(),
    }
    for name in TABLE_NAMES:
        _write_table(content[name], run_dir / "tables" / name)
        outputs.append(name)
    try:
        with pd.ExcelWriter(run_dir / "tables" / "paper_tables.xlsx") as writer:
            for name, df in content.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception:
        pass
    return outputs


def _read_dataset_summaries(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in (run_dir / "raw").glob("*/dataset_summary.csv"):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
