"""V2 add-ons for spatial robustness, measurement reliability, and utility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import configuration_id
from ..evaluation import calibration_errors
from ..preprocessing import fit_transform_variant
from ..robustness import central_occlusion, rotate_images, translate_images
from ..sampling import nested_balanced_indices

PROTOCOL = "v2_failure_isolation"


def _models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed))]),
        "rbf_svm": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed))]),
        "matched_mlp": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=seed, early_stopping=True))]),
    }


def _prob(estimator: Any, x: np.ndarray) -> np.ndarray:
    return estimator.predict_proba(x)[:, 1]


def _metric_row(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = prob >= 0.5
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "brier_score": float(brier_score_loss(y, prob)),
        **calibration_errors(y, prob),
    }


def run_spatial_v2(run_dir: Path, dataset: str, bundle: Any, config: dict[str, Any], seed: int) -> pd.DataFrame:
    out = run_dir / "problem3"
    out.mkdir(parents=True, exist_ok=True)
    p3 = config.get("v2", {}).get("problem3", {})
    feature_sets = {
        "flat_pca_vqc_control": fit_transform_variant("pca_4", bundle.x_train, bundle.x_val, bundle.x_test, out / "preprocessing"),
        "patch_vqc_control": fit_transform_variant("patch_2x2", bundle.x_train, bundle.x_val, bundle.x_test, out / "preprocessing"),
    }
    trained = {}
    for feature_name, fs in feature_sets.items():
        for model_name, estimator in _models(seed).items():
            estimator.fit(fs.x_train, bundle.y_train)
            trained[(feature_name, model_name)] = (estimator, fs)
    rows = []
    pred_rows = []
    transformations = [("rotation", v, rotate_images(bundle.x_test, v)) for v in p3.get("rotations", [0, 10])]
    transformations += [("translation_x", v, translate_images(bundle.x_test, dx=int(v), dy=0)) for v in p3.get("translations", [0, 2])]
    transformations += [("translation_y", v, translate_images(bundle.x_test, dx=0, dy=int(v))) for v in p3.get("translations", [0, 2])]
    transformations += [("occlusion", v, central_occlusion(bundle.x_test, float(v))) for v in p3.get("occlusions", [0, 20])]
    for transform, strength, x_img in transformations:
        for feature_name, fs_ref in feature_sets.items():
            fs_t = fit_transform_variant(fs_ref.name, bundle.x_train, bundle.x_val, x_img, out / "preprocessing")
            for model_name, (estimator, _) in trained.items():
                if model_name[0] != feature_name:
                    continue
                original_prob = _prob(estimator, fs_ref.x_test)
                transformed_prob = _prob(estimator, fs_t.x_test)
                originally_correct = (original_prob >= 0.5) == bundle.y_test
                cid = configuration_id({"problem": 3, "dataset": dataset, "seed": seed, "feature": feature_name, "model": model_name[1], "transform": transform, "strength": strength, "protocol": PROTOCOL})
                metrics = _metric_row(bundle.y_test, transformed_prob)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "configuration_id": cid,
                        "protocol_version": PROTOCOL,
                        "model": model_name[1],
                        "representation": feature_name,
                        "transformation": transform,
                        "strength": strength,
                        "prediction_flip_rate_all": float(np.mean((original_prob >= 0.5) != (transformed_prob >= 0.5))),
                        "prediction_flip_rate_correct_only": float(np.mean(((original_prob >= 0.5) != (transformed_prob >= 0.5))[originally_correct])) if originally_correct.any() else np.nan,
                        "mean_probability_change": float(np.mean(np.abs(original_prob - transformed_prob))),
                        "n_originally_correct": int(originally_correct.sum()),
                        **metrics,
                    }
                )
                pred_rows.append(pd.DataFrame({"dataset": dataset, "seed": seed, "configuration_id": cid, "protocol_version": PROTOCOL, "model": model_name[1], "representation": feature_name, "sample_id": bundle.test_indices, "label": bundle.y_test, "transformation": transform, "strength": strength, "original_probability": original_prob, "transformed_probability": transformed_prob, "prediction_flip": (original_prob >= 0.5) != (transformed_prob >= 0.5)}))
    metrics_df = pd.DataFrame(rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics_df.to_parquet(out / "transformation_metrics.parquet", index=False)
    preds.to_parquet(out / "transformed_predictions.parquet", index=False)
    pd.DataFrame(columns=["dataset", "model", "transformation", "threshold", "boundary", "CI_low", "CI_high", "protocol_version"]).to_parquet(out / "failure_boundaries.parquet", index=False)
    (out / "selected_visual_samples.json").write_text("{}\n", encoding="utf-8")
    metrics_df.to_csv(run_dir / "metrics" / f"{dataset}_P3_v2_spatial_robustness.csv", index=False)
    return metrics_df


def run_reliability_v2(run_dir: Path, dataset: str, config: dict[str, Any], seed: int) -> pd.DataFrame:
    out = run_dir / "problem4"
    out.mkdir(parents=True, exist_ok=True)
    p4 = config.get("v2", {}).get("problem4", {})
    shots = [int(s) for s in p4.get("shots", [64, 256])]
    mode = config.get("project", {}).get("mode", "pilot")
    repeats = int(p4.get("repeats", {}).get(mode, 5))
    pred_files = sorted((run_dir / "predictions").glob("*_predictions.csv"))
    rows = []
    shot_frames = []
    rng = np.random.default_rng(seed)
    for path in pred_files:
        pred = pd.read_csv(path)
        if pred.empty or pred["dataset"].iloc[0] != dataset or pred["model"].iloc[0] != "vqc":
            continue
        y = pred["label"].to_numpy()
        p = pred["probability"].to_numpy()
        analytic_correct = (p >= 0.5) == y
        margin = np.abs(p - 0.5)
        for shot in shots:
            repeated = rng.binomial(shot, np.clip(p, 0, 1), size=(repeats, len(p))) / shot
            flip = (repeated >= 0.5) != (p >= 0.5)[None, :]
            sample_flip = flip.mean(axis=0)
            rows.append({"dataset": dataset, "seed": seed, "configuration_id": pred["configuration_id"].iloc[0], "protocol_version": PROTOCOL, "model": "vqc", "shots": shot, "repeats": repeats, "Flip_all": float(flip.mean()), "Flip_correct": float(flip[:, analytic_correct].mean()) if analytic_correct.any() else np.nan, "n_analytic_correct": int(analytic_correct.sum()), "probability_std": float(np.mean(np.std(repeated, axis=0))), **calibration_errors(y, p)})
            shot_frames.append(pd.DataFrame({"dataset": dataset, "seed": seed, "configuration_id": pred["configuration_id"].iloc[0], "protocol_version": PROTOCOL, "sample_id": pred["sample_id"], "label": y, "analytic_probability": p, "analytic_margin": margin, "analytic_correct": analytic_correct, "shots": shot, "sample_flip_rate": sample_flip}))
    summary = pd.DataFrame(rows)
    shot_df = pd.concat(shot_frames, ignore_index=True) if shot_frames else pd.DataFrame()
    margin_bins = _margin_bins(shot_df)
    boundary = _shot_boundary(summary)
    summary.to_parquet(out / "flip_correct_summary.parquet", index=False)
    shot_df.to_parquet(out / "shot_predictions.parquet", index=False)
    margin_bins.to_parquet(out / "margin_bins.parquet", index=False)
    boundary.to_parquet(out / "shot_stability_boundary.parquet", index=False)
    pd.DataFrame(columns=["dataset", "seed", "noise_channel", "noise_strength", "flip_rate", "accuracy", "protocol_version"]).to_parquet(out / "noise_results.parquet", index=False)
    summary.to_csv(run_dir / "metrics" / f"{dataset}_P4_v2_margin_reliability.csv", index=False)
    return summary


def _margin_bins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50]
    out = df.copy()
    out["margin_bin"] = pd.cut(out["analytic_margin"], bins=bins, right=False, include_lowest=True).astype(str)
    grouped = out.groupby(["dataset", "shots", "margin_bin"], observed=True).agg(flip_rate=("sample_flip_rate", "mean"), n=("sample_id", "size"), margin_mean=("analytic_margin", "mean")).reset_index()
    corrs = []
    for (dataset, shots), group in out.groupby(["dataset", "shots"]):
        unstable = group["sample_flip_rate"] >= 0.10
        corr = spearmanr(group["analytic_margin"], group["sample_flip_rate"])
        corrs.append({"dataset": dataset, "shots": shots, "spearman_margin_flip": corr.statistic, "spearman_p": corr.pvalue, "margin_unstable_auroc": roc_auc_score(unstable, -group["analytic_margin"]) if unstable.nunique() == 2 else np.nan, "margin_unstable_average_precision": average_precision_score(unstable, -group["analytic_margin"]) if unstable.nunique() == 2 else np.nan})
    return grouped.merge(pd.DataFrame(corrs), on=["dataset", "shots"], how="left")


def _shot_boundary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    for (dataset, cid), group in summary.groupby(["dataset", "configuration_id"]):
        ok = group[(group["Flip_correct"] < 0.05) & (group["expected_calibration_error"] <= 0.10)].sort_values("shots")
        rows.append({"dataset": dataset, "configuration_id": cid, "protocol_version": PROTOCOL, "S_star": int(ok.iloc[0]["shots"]) if not ok.empty else np.nan, "notes": "not reached within tested shot budget" if ok.empty else "boundary reached"})
    return pd.DataFrame(rows)


def run_utility_v2(run_dir: Path, dataset: str, bundle: Any, config: dict[str, Any], seed: int) -> pd.DataFrame:
    out = run_dir / "problem5"
    out.mkdir(parents=True, exist_ok=True)
    p5 = config.get("v2", {}).get("problem5", {})
    sizes = [int(s) for s in p5.get("train_sizes_per_class", [25, 50])]
    subsets = nested_balanced_indices(bundle.y_train, sizes, seed)
    rows = []
    fs = fit_transform_variant("pca_4", bundle.x_train, bundle.x_val, bundle.x_test, out / "preprocessing")
    for size, idx in subsets.items():
        for model_name, estimator in _models(seed).items():
            estimator.fit(fs.x_train[idx], bundle.y_train[idx])
            prob = _prob(estimator, fs.x_test)
            rows.append({"dataset": dataset, "seed": seed, "configuration_id": configuration_id({"problem": 5, "dataset": dataset, "seed": seed, "size": size, "model": model_name, "protocol": PROTOCOL}), "protocol_version": PROTOCOL, "comparison_regime": "matched_information", "train_samples_per_class": size, "model": model_name, **_metric_row(bundle.y_test, prob)})
    df = pd.DataFrame(rows)
    primary_path = run_dir / "metrics" / "primary_metrics.csv"
    if primary_path.exists():
        primary = pd.read_csv(primary_path)
        for _, row in primary[(primary.get("dataset") == dataset) & (primary.get("seed") == seed) & (primary.get("model") == "vqc") & (primary.get("preprocessing").isin(["pca_4", "pca_8"]))].iterrows():
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "configuration_id": row.get("configuration_id"),
                                "protocol_version": PROTOCOL,
                                "comparison_regime": "matched_information",
                                "train_samples_per_class": np.nan,
                                "model": "vqc",
                                "accuracy": row.get("accuracy"),
                                "balanced_accuracy": row.get("balanced_accuracy"),
                                "macro_f1": row.get("macro_f1"),
                                "auroc": row.get("auroc"),
                                "brier_score": row.get("brier_score"),
                                "expected_calibration_error": row.get("expected_calibration_error"),
                                "maximum_calibration_error": row.get("maximum_calibration_error"),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    df.to_parquet(out / "low_data_results.parquet", index=False)
    df.to_parquet(out / "matched_information_results.parquet", index=False)
    resources = df[["dataset", "seed", "configuration_id", "model", "protocol_version"]].copy()
    resources["trainable_parameters"] = np.nan
    resources["circuit_evaluations"] = np.where(resources["model"].eq("vqc"), np.nan, 0)
    resources.to_parquet(out / "resource_metrics.parquet", index=False)
    pareto = _pareto(df)
    pareto.to_parquet(out / "pareto_results.parquet", index=False)
    pd.DataFrame().to_parquet(out / "dominance_matrix.parquet", index=False)
    df.to_csv(run_dir / "metrics" / f"{dataset}_P5_v2_low_data_utility.csv", index=False)
    return df


def _pareto(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    latest = df.groupby("model", as_index=False).agg(error=("accuracy", lambda s: 1 - s.mean()), brier_score=("brier_score", "mean"))
    dominated = []
    for _, row in latest.iterrows():
        others = latest[latest["model"] != row["model"]]
        is_dom = bool(((others["error"] <= row["error"]) & (others["brier_score"] <= row["brier_score"]) & ((others["error"] < row["error"]) | (others["brier_score"] < row["brier_score"]))).any())
        dominated.append(is_dom)
    latest["pareto_frontier"] = ~pd.Series(dominated)
    latest["protocol_version"] = PROTOCOL
    return latest
