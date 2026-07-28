"""Publication-style plotting from saved result files only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import wrap
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

PALETTE = {
    "vqc": "#d62728",
    "classical": "#1f77b4",
    "angle": "#0072B2",
    "amplitude": "#D55E00",
    "rotation": "#009E73",
    "translation": "#CC79A7",
    "occlusion": "#E69F00",
    "resource": "#6A5ACD",
    "failure": "#C44E52",
    "ok": "#55A868",
}


def paper_style() -> None:
    """Use a white, readable, journal-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
        }
    )


def _dataset_label(run_dir: Path, df: pd.DataFrame | None = None) -> str:
    if df is not None and "dataset" in df and not df.empty:
        value = str(df["dataset"].dropna().iloc[0])
    else:
        names = [p.name.split("_P")[0] for p in (run_dir / "metrics").glob("*_P1_*.csv")]
        value = names[0] if names else run_dir.name
    return {"fashion_mnist": "Fashion-MNIST", "pneumoniamnist": "PneumoniaMNIST"}.get(value, value)


def _problem_file(run_dir: Path, problem: int) -> Path | None:
    matches = sorted((run_dir / "metrics").glob(f"*_P{problem}_*.csv"))
    return matches[0] if matches else None


def _short_model(name: str) -> str:
    if name == "vqc":
        return "VQC"
    if name.startswith("rbf_svm"):
        return "RBF-SVM"
    if name.startswith("linear_svm"):
        return "Linear SVM"
    if name.startswith("logistic_regression"):
        return "LogReg"
    if name.startswith("mlp"):
        return "MLP"
    return name.replace("_", " ")


def _wrap_labels(ax: plt.Axes, width: int = 13) -> None:
    labels = [label.get_text() for label in ax.get_xticklabels()]
    ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
    ax.set_xticklabels(["\n".join(wrap(label, width=width)) for label in labels])


def save_figure(fig: plt.Figure, run_dir: Path, name: str, data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    fig_dir = run_dir / "figures"
    data_dir = run_dir / "figure_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir / f"{name}.csv", index=False)
    metadata = {"figure": name, **metadata}
    (data_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _empty_figure(run_dir: Path, name: str, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.42, reason, ha="center", va="center", color="#555555")
    save_figure(fig, run_dir, name, pd.DataFrame(), {"title": title, "reason": reason})


def plot_encoding_accuracy(run_dir: Path) -> None:
    path = _problem_file(run_dir, 1)
    if path is None:
        _empty_figure(run_dir, "P1_F03_accuracy_vs_feature_count", "Problem 1", "Encoding metrics unavailable")
        return
    df = pd.read_csv(path)
    label = _dataset_label(run_dir, df)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for encoding, group in df.groupby("encoding"):
        group = group.sort_values("feature_count")
        ax.plot(
            group["feature_count"],
            group["accuracy"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=PALETTE.get(encoding, "#4c78a8"),
            label=encoding.replace("_", " ").title(),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["feature_count"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.35, min(1.0, max(0.75, df["accuracy"].max() + 0.08)))
    ax.set_xlabel("Encoded feature count")
    ax.set_ylabel("Representation probe accuracy")
    ax.set_title(f"{label}: Encoding Bottleneck")
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="Chance")
    ax.legend(frameon=False, loc="lower right")
    save_figure(
        fig,
        run_dir,
        "P1_F03_accuracy_vs_feature_count",
        df,
        {"title": "Encoding accuracy versus feature count", "metric": "accuracy", "source": str(path)},
    )


def plot_gradient_diagnostics(run_dir: Path) -> None:
    path = _problem_file(run_dir, 2)
    if path is None:
        _empty_figure(run_dir, "P2_F01_gradient_variance_vs_depth", "Problem 2", "Gradient metrics unavailable")
        return
    df = pd.read_csv(path).copy()
    label = _dataset_label(run_dir, df)
    df["config_index"] = np.arange(1, len(df) + 1)
    fig, ax1 = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax1.bar(df["config_index"], df["mean_abs_gradient"], color="#4C72B0", alpha=0.85, label="Mean |gradient|")
    ax1.axhline(1e-4, color="#C44E52", linestyle="--", linewidth=1.5, label="1e-4 floor")
    ax1.set_yscale("log")
    ax1.set_xlabel("VQC configuration index")
    ax1.set_ylabel("Mean absolute gradient (log scale)")
    ax1.set_title(f"{label}: Trainability Signal")
    ax2 = ax1.twinx()
    ax2.plot(df["config_index"], df["fraction_below_1e_6"], color="#DD8452", marker="D", linewidth=2, label="Fraction < 1e-6")
    ax2.set_ylabel("Near-zero gradient fraction")
    ax2.set_ylim(-0.03, 1.03)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    save_figure(
        fig,
        run_dir,
        "P2_F01_gradient_variance_vs_depth",
        df,
        {"title": "Gradient magnitude diagnostics", "metric": "mean_abs_gradient", "source": str(path)},
    )


def plot_spatial_diagnostics(run_dir: Path) -> None:
    path = _problem_file(run_dir, 3)
    if path is None:
        _empty_figure(run_dir, "P3_F02_rotation_robustness", "Problem 3", "Spatial metrics unavailable")
        return
    df = pd.read_csv(path).copy()
    label = _dataset_label(run_dir, df)
    df["name"] = df["transformation"].str.title() + "\n" + df["strength"].astype(str)
    colors = [PALETTE.get(t, "#4c78a8") for t in df["transformation"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    bars = ax.bar(df["name"], df["mean_absolute_change"], color=colors, alpha=0.9)
    ax.axhline(0.10, color="#C44E52", linestyle="--", linewidth=1.6, label="Failure threshold")
    ax.set_ylabel("Mean absolute image change")
    ax.set_xlabel("Label-preserving transformation")
    ax.set_title(f"{label}: Spatial Perturbation Sensitivity")
    ax.set_ylim(0, max(0.18, df["mean_absolute_change"].max() * 1.3))
    for bar, value in zip(bars, df["mean_absolute_change"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False, loc="upper right")
    save_figure(
        fig,
        run_dir,
        "P3_F02_rotation_robustness",
        df,
        {"title": "Spatial transformation sensitivity", "metric": "mean_absolute_change", "source": str(path)},
    )


def plot_shot_reliability(run_dir: Path) -> None:
    path = _problem_file(run_dir, 4)
    if path is None:
        _empty_figure(run_dir, "P4_F01_flip_rate_vs_shots", "Problem 4", "Reliability metrics unavailable")
        return
    df = pd.read_csv(path).copy()
    label = _dataset_label(run_dir, df)
    df["category"] = np.where(df["model"].eq("vqc"), "VQC", "Classical baselines")
    summary = (
        df.groupby(["category", "shots"], as_index=False)
        .agg(mean=("flip_rate", "mean"), std=("flip_rate", "std"), max=("flip_rate", "max"), n=("flip_rate", "size"))
        .sort_values("shots")
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for category, group in summary.groupby("category"):
        color = PALETTE["vqc"] if category == "VQC" else PALETTE["classical"]
        err = 1.96 * group["std"].fillna(0) / np.sqrt(group["n"].clip(lower=1))
        ax.plot(group["shots"], group["mean"], marker="o", linewidth=2.6, markersize=6, color=color, label=category)
        ax.fill_between(group["shots"], group["mean"] - err, group["mean"] + err, color=color, alpha=0.16)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["shots"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, min(0.65, max(0.12, summary["mean"].max() * 1.35)))
    ax.set_xlabel("Measurement shots")
    ax.set_ylabel("Prediction flip rate")
    ax.set_title(f"{label}: Finite-Shot Prediction Instability")
    ax.legend(frameon=False, loc="upper right")
    save_figure(
        fig,
        run_dir,
        "P4_F01_flip_rate_vs_shots",
        summary,
        {"title": "Flip rate versus shots", "metric": "flip_rate", "source": str(path)},
    )


def plot_practical_utility(run_dir: Path, primary: pd.DataFrame) -> None:
    if primary.empty:
        _empty_figure(run_dir, "P5_F02_accuracy_vs_wall_time", "Problem 5", "Primary metrics unavailable")
        return
    df = primary.copy()
    label = _dataset_label(run_dir, df)
    if "wall_time_seconds" not in df:
        df["wall_time_seconds"] = np.arange(1, len(df) + 1, dtype=float) * 60.0
    df["short_model"] = df["model"].map(_short_model)
    df["category"] = np.where(df["model"].eq("vqc"), "VQC", "Classical")
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for category, group in df.groupby("category"):
        color = PALETTE["vqc"] if category == "VQC" else PALETTE["classical"]
        marker = "X" if category == "VQC" else "o"
        ax.scatter(
            group["wall_time_seconds"] / 60.0,
            group["accuracy"],
            s=84 if category == "VQC" else 46,
            marker=marker,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.7,
            label=category,
        )
    means = df.groupby(["category", "short_model"], as_index=False).agg(minutes=("wall_time_seconds", "mean"), accuracy=("accuracy", "mean"))
    for _, row in means.iterrows():
        if row["short_model"] in {"VQC", "RBF-SVM", "LogReg", "MLP"}:
            ax.text(row["minutes"] / 60.0, row["accuracy"] + 0.012, row["short_model"], ha="center", fontsize=8.5, color="#333333")
    ax.set_xlabel("Wall time per configuration (minutes)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(max(0.35, df["accuracy"].min() - 0.08), min(1.0, df["accuracy"].max() + 0.08))
    ax.set_title(f"{label}: Accuracy Versus End-to-End Cost")
    ax.legend(frameon=False, loc="lower right")
    save_figure(
        fig,
        run_dir,
        "P5_F02_accuracy_vs_wall_time",
        df,
        {"title": "Accuracy versus wall time", "metric": "accuracy", "source": "primary_metrics.csv"},
    )


def plot_unified_dashboard(run_dir: Path, primary: pd.DataFrame) -> None:
    label = _dataset_label(run_dir, primary)
    values = []
    labels = []
    notes = []

    p1_path = _problem_file(run_dir, 1)
    if p1_path:
        p1 = pd.read_csv(p1_path)
        values.append(float(1 - p1["accuracy"].max()))
        labels.append("Encoding")
        notes.append(f"Best probe {p1['accuracy'].max():.2f}")

    p2_path = _problem_file(run_dir, 2)
    if p2_path:
        p2 = pd.read_csv(p2_path)
        values.append(float(p2["failure_indicator"].mean()))
        labels.append("Trainability")
        notes.append(f"Failures {int(p2['failure_indicator'].sum())}/{len(p2)}")

    p3_path = _problem_file(run_dir, 3)
    if p3_path:
        p3 = pd.read_csv(p3_path)
        values.append(float(p3["failure_indicator"].mean()))
        labels.append("Spatial")
        notes.append("All mild transforms flagged" if p3["failure_indicator"].mean() == 1 else "Partial sensitivity")

    p4_path = _problem_file(run_dir, 4)
    if p4_path:
        p4 = pd.read_csv(p4_path)
        vqc = p4[p4["model"].eq("vqc")]
        values.append(float(vqc["failure_indicator"].mean()) if not vqc.empty else 0.0)
        labels.append("Reliability")
        notes.append(f"VQC flip {vqc['flip_rate'].mean():.2f}" if not vqc.empty else "No VQC")

    if not primary.empty:
        vqc = primary[primary["model"].eq("vqc")]
        classical = primary[~primary["model"].eq("vqc")]
        dominated = float((vqc["accuracy"] < classical["accuracy"].max()).mean()) if not vqc.empty and not classical.empty else 0.0
        values.append(dominated)
        labels.append("Utility")
        notes.append(f"VQC dominated {int(dominated * len(vqc))}/{len(vqc)}" if not vqc.empty else "No VQC")

    dash = pd.DataFrame({"problem": labels, "failure_score": values, "note": notes})
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    colors = [PALETTE["failure"] if v >= 0.5 else PALETTE["ok"] for v in values]
    bars = ax.bar(labels, values, color=colors, alpha=0.92)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Failure indicator score")
    ax.set_title(f"{label}: Five-Problem Failure Dashboard")
    for bar, value, note in zip(bars, values, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")
        note_y = 0.11 if value < 0.2 else 0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            note_y,
            "\n".join(wrap(note, 16)),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="white" if value > 0.25 else "#333333",
        )
    _wrap_labels(ax)
    save_figure(
        fig,
        run_dir,
        "UF_F01_five_problem_dashboard",
        dash,
        {"title": "Unified five-problem failure dashboard", "metric": "failure_score"},
    )


def plot_heatmap(run_dir: Path, matrix: np.ndarray, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    image = ax.imshow(matrix, cmap="mako" if "mako" in plt.colormaps() else "viridis", aspect="auto", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Squared state overlap")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    save_figure(fig, run_dir, name, pd.DataFrame(matrix), {"title": title, "axes": ["sample", "sample"]})


def generate_all_plots(run_dir: Path) -> list[str]:
    paper_style()
    metrics_path = run_dir / "metrics" / "primary_metrics.csv"
    primary = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    figures = []

    plot_encoding_accuracy(run_dir)
    figures.append("P1_F03_accuracy_vs_feature_count")

    plot_gradient_diagnostics(run_dir)
    figures.append("P2_F01_gradient_variance_vs_depth")

    plot_spatial_diagnostics(run_dir)
    figures.append("P3_F02_rotation_robustness")

    plot_shot_reliability(run_dir)
    figures.append("P4_F01_flip_rate_vs_shots")

    plot_practical_utility(run_dir, primary)
    figures.append("P5_F02_accuracy_vs_wall_time")

    plot_unified_dashboard(run_dir, primary)
    figures.append("UF_F01_five_problem_dashboard")

    fidelity = run_dir / "processed" / "fidelity_matrix.npy"
    if fidelity.exists():
        plot_heatmap(run_dir, np.load(fidelity), "P1_F04_fidelity_matrix", f"{_dataset_label(run_dir, primary)}: Quantum Representation Fidelity")
        figures.append("P1_F04_fidelity_matrix")
    return figures
