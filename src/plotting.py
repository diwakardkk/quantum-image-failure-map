"""Publication-style plotting from saved result files only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, run_dir: Path, name: str, data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    paper_style()
    fig_dir = run_dir / "figures"
    data_dir = run_dir / "figure_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir / f"{name}.csv", index=False)
    metadata = {"figure": name, **metadata}
    (data_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar(run_dir: Path, df: pd.DataFrame, metric: str, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    if df.empty or metric not in df:
        ax.text(0.5, 0.5, "No completed data", ha="center", va="center")
        plot_df = pd.DataFrame()
    else:
        plot_df = df.dropna(subset=[metric]).copy()
        grouped = plot_df.groupby("model", as_index=False)[metric].mean()
        ax.bar(grouped["model"], grouped[metric], color="#4c78a8")
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
    ax.set_title(title)
    save_figure(fig, run_dir, name, plot_df if "plot_df" in locals() else pd.DataFrame(), {"metric": metric, "title": title})


def plot_heatmap(run_dir: Path, matrix: np.ndarray, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("sample")
    save_figure(fig, run_dir, name, pd.DataFrame(matrix), {"title": title, "axes": ["sample", "sample"]})


def generate_all_plots(run_dir: Path) -> list[str]:
    metrics_path = run_dir / "metrics" / "primary_metrics.csv"
    df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    figures = []
    figure_specs = [
        ("P1_F03_accuracy_vs_feature_count", "accuracy", "Problem 1: Accuracy by model"),
        ("P2_F01_gradient_variance_vs_depth", "gradient_variance", "Problem 2: Gradient variance"),
        ("P3_F02_rotation_robustness", "robust_accuracy", "Problem 3: Spatial robustness"),
        ("P4_F01_flip_rate_vs_shots", "flip_rate", "Problem 4: Finite-shot flip rate"),
        ("P5_F02_accuracy_vs_wall_time", "wall_time_seconds", "Problem 5: Wall-time comparison"),
        ("UF_F01_five_problem_dashboard", "failure_indicator", "Unified failure dashboard"),
    ]
    for name, metric, title in figure_specs:
        plot_metric_bar(run_dir, df, metric if metric in df else "accuracy", name, title)
        figures.append(name)
    fidelity = run_dir / "processed" / "fidelity_matrix.npy"
    if fidelity.exists():
        plot_heatmap(run_dir, np.load(fidelity), "P1_F04_fidelity_matrix", "Problem 1: Quantum representation fidelity")
        figures.append("P1_F04_fidelity_matrix")
    return figures
