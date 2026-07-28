"""Problem 1 v2: separate compression loss from quantum utilisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, pairwise_distances, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import configuration_id
from ..encodings import EncodingSpec, fidelity_matrix, quantum_feature_matrix
from ..evaluation import binary_metrics
from ..preprocessing import flatten_images, reconstruction_metrics

PROTOCOL = "v2_failure_isolation"


def _probe_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0))]),
        "rbf_svm": Pipeline([("scaler", StandardScaler()), ("model", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=0))]),
    }


def _dist_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    d = pairwise_distances(x)
    same = y[:, None] == y[None, :]
    tri = np.triu(np.ones_like(same, dtype=bool), 1)
    intra = d[same & tri]
    inter = d[~same & tri]
    return {
        "mean_intra_class_distance": float(np.mean(intra)) if intra.size else np.nan,
        "mean_inter_class_distance": float(np.mean(inter)) if inter.size else np.nan,
        "inter_intra_distance_ratio": float(np.mean(inter) / (np.mean(intra) + 1e-12)) if intra.size and inter.size else np.nan,
    }


def _fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    vals = np.unique(y)
    if len(vals) != 2:
        return np.nan
    a, b = x[y == vals[0]], x[y == vals[1]]
    return float(np.sum((a.mean(0) - b.mean(0)) ** 2) / (a.var(0).mean() + b.var(0).mean() + 1e-12))


def _prob(estimator: Any, x: np.ndarray) -> np.ndarray:
    return estimator.predict_proba(x)[:, 1]


def run_problem1_v2(run_dir: Path, dataset: str, bundle: Any, config: dict[str, Any], seed: int) -> dict[str, pd.DataFrame]:
    out = run_dir / "problem1"
    out.mkdir(parents=True, exist_ok=True)
    dims = [int(d) for d in config.get("v2", {}).get("pca_dimensions", [64, 32, 16, 8, 4, 2])]
    quantum_dims = [int(d) for d in config.get("v2", {}).get("quantum_dimensions", [4, 8])]
    xtr, xva, xte = map(flatten_images, [bundle.x_train, bundle.x_val, bundle.x_test])
    compression_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    transformed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]] = {}

    for dim in dims:
        cid = configuration_id({"problem": 1, "stage": "compression", "dataset": dataset, "seed": seed, "dim": dim, "protocol": PROTOCOL})
        scaler = StandardScaler().fit(xtr)
        max_dim = min(dim, xtr.shape[0], xtr.shape[1])
        pca = PCA(n_components=max_dim, random_state=seed).fit(scaler.transform(xtr))
        ztr, zva, zte = [pca.transform(scaler.transform(x)) for x in [xtr, xva, xte]]
        transformed[dim] = (ztr, zva, zte, pca, scaler)
        joblib.dump({"scaler": scaler, "pca": pca}, out / f"{dataset}_seed{seed}_pca{dim}.joblib")
        np.savez_compressed(out / f"{dataset}_seed{seed}_pca{dim}_features.npz", train=ztr, val=zva, test=zte)
        rec = scaler.inverse_transform(pca.inverse_transform(zte)).reshape(bundle.x_test.shape)
        np.savez_compressed(out / f"{dataset}_seed{seed}_pca{dim}_reconstructions.npz", reconstructed=rec)
        rec_metrics = reconstruction_metrics(bundle.x_test, rec)
        compression_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "configuration_id": cid,
                "protocol_version": PROTOCOL,
                "pca_dimension": dim,
                "explained_variance_ratio": json.dumps(pca.explained_variance_ratio_.tolist()),
                "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                "fisher_discriminant_ratio": _fisher_ratio(ztr, bundle.y_train),
                **_dist_metrics(ztr, bundle.y_train),
                **rec_metrics,
            }
        )
        for model_name, estimator in _probe_models().items():
            estimator.fit(ztr, bundle.y_train)
            val_prob = _prob(estimator, zva)
            test_prob = _prob(estimator, zte)
            pred = test_prob >= 0.5
            probe_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "configuration_id": configuration_id({"problem": 1, "stage": "probe", "dataset": dataset, "seed": seed, "dim": dim, "model": model_name, "protocol": PROTOCOL}),
                    "protocol_version": PROTOCOL,
                    "pca_dimension": dim,
                    "model": model_name,
                    "validation_accuracy": accuracy_score(bundle.y_val, val_prob >= 0.5),
                    "accuracy": accuracy_score(bundle.y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(bundle.y_test, pred),
                    "macro_f1": f1_score(bundle.y_test, pred, average="macro", zero_division=0),
                    "auroc": roc_auc_score(bundle.y_test, test_prob) if len(np.unique(bundle.y_test)) == 2 else np.nan,
                }
            )

    comp = pd.DataFrame(compression_rows)
    probes = pd.DataFrame(probe_rows)
    comp.to_parquet(out / "compression_metrics.parquet", index=False)
    probes.to_parquet(out / "probe_predictions.parquet", index=False)
    probes.to_csv(run_dir / "metrics" / f"{dataset}_P1_v2_probe_accuracy.csv", index=False)

    boundary_rows = []
    drops = config.get("v2", {}).get("compression_accuracy_drop_thresholds", [0.025, 0.05, 0.075, 0.10])
    variances = config.get("v2", {}).get("compression_variance_thresholds", [0.70, 0.80, 0.90])
    ref = probes.groupby("pca_dimension")["validation_accuracy"].max().sort_index(ascending=False).cummax().max()
    for drop in drops:
        for variance in variances:
            merged = probes.groupby("pca_dimension", as_index=False)["validation_accuracy"].max().merge(comp[["pca_dimension", "cumulative_explained_variance"]], on="pca_dimension")
            merged["insufficient"] = ((ref - merged["validation_accuracy"]) > drop) | (merged["cumulative_explained_variance"] < variance)
            failing = merged[merged["insufficient"]].sort_values("pca_dimension")
            boundary_rows.append({"dataset": dataset, "seed": seed, "protocol_version": PROTOCOL, "accuracy_drop_threshold": drop, "variance_threshold": variance, "failure_boundary_dimension": int(failing.iloc[0]["pca_dimension"]) if not failing.empty else np.nan})
    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_parquet(out / "compression_failure_boundary.parquet", index=False)

    gap_rows = []
    primary_path = run_dir / "metrics" / "primary_metrics.csv"
    primary = pd.read_csv(primary_path) if primary_path.exists() else pd.DataFrame()
    for dim in quantum_dims:
        if dim not in transformed:
            continue
        ztr, zva, zte, _, _ = transformed[dim]
        model_metrics: dict[str, dict[str, float]] = {}
        for model_name, estimator in {
            **_probe_models(),
            "compact_mlp": Pipeline([("scaler", StandardScaler()), ("model", __import__("sklearn.neural_network").neural_network.MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=seed, early_stopping=True))]),
        }.items():
            estimator.fit(ztr, bundle.y_train)
            model_metrics[model_name] = binary_metrics(bundle.y_test, _prob(estimator, zte))  # type: ignore[assignment]
        prep_name = f"pca_{dim}"
        vqc_match = primary[(primary.get("dataset") == dataset) & (primary.get("seed") == seed) & (primary.get("model") == "vqc") & (primary.get("preprocessing") == prep_name)] if not primary.empty else pd.DataFrame()
        if not vqc_match.empty:
            row = vqc_match.iloc[-1]
            vqc_metrics = {"accuracy": row.get("accuracy", np.nan), "macro_f1": row.get("macro_f1", np.nan), "auroc": row.get("auroc", np.nan), "brier_score": row.get("brier_score", np.nan), "status": "completed"}
        else:
            vqc_metrics = {"accuracy": np.nan, "macro_f1": np.nan, "auroc": np.nan, "brier_score": np.nan, "status": "requires_vqc_training"}
        best_classical = max(v["accuracy"] for v in model_metrics.values())
        gap_rows.append({"dataset": dataset, "seed": seed, "protocol_version": PROTOCOL, "pca_dimension": dim, "comparison_regime": "matched_information", "best_classical_accuracy": best_classical, "vqc_accuracy": vqc_metrics["accuracy"], "G_util_accuracy": best_classical - vqc_metrics["accuracy"] if np.isfinite(vqc_metrics["accuracy"]) else np.nan, "vqc_status": vqc_metrics["status"]})
        for name, metrics in model_metrics.items():
            gap_rows.append({"dataset": dataset, "seed": seed, "protocol_version": PROTOCOL, "pca_dimension": dim, "comparison_regime": "matched_information", "model": name, **{k: metrics.get(k) for k in ["accuracy", "macro_f1", "auroc", "brier_score"]}})
        if vqc_metrics["status"] == "completed":
            gap_rows.append({"dataset": dataset, "seed": seed, "protocol_version": PROTOCOL, "pca_dimension": dim, "comparison_regime": "matched_information", "model": "vqc", **{k: vqc_metrics.get(k) for k in ["accuracy", "macro_f1", "auroc", "brier_score"]}})
    gaps = pd.DataFrame(gap_rows)
    gaps.to_parquet(out / "utilisation_gap.parquet", index=False)

    fidelity_rows = []
    rng = np.random.default_rng(seed)
    per_class = int(config.get("v2", {}).get("fidelity_samples_per_class", 25))
    sample_ids = np.concatenate([rng.choice(np.flatnonzero(bundle.y_test == cls), size=min(per_class, np.sum(bundle.y_test == cls)), replace=False) for cls in np.unique(bundle.y_test)])
    for dim in quantum_dims:
        if dim not in transformed:
            continue
        z = transformed[dim][2][sample_ids]
        y = bundle.y_test[sample_ids]
        for enc in ["angle", "amplitude"]:
            qz, _ = quantum_feature_matrix(z, EncodingSpec(enc, dim, dim if enc == "angle" else int(np.ceil(np.log2(dim)))))
            fid = fidelity_matrix(qz)
            same = y[:, None] == y[None, :]
            tri = np.triu(np.ones_like(same, dtype=bool), 1)
            within, between = fid[same & tri], fid[~same & tri]
            delta = float(np.mean(within) - np.mean(between))
            stat = mannwhitneyu(within, between, alternative="two-sided") if within.size and between.size else None
            fidelity_rows.append({"dataset": dataset, "seed": seed, "protocol_version": PROTOCOL, "pca_dimension": dim, "encoding": enc, "mean_within_fidelity": float(np.mean(within)), "mean_between_fidelity": float(np.mean(between)), "delta_F": delta, "mannwhitney_u": float(stat.statistic) if stat else np.nan, "p_value": float(stat.pvalue) if stat else np.nan, "n_samples": len(sample_ids)})
            np.savez_compressed(out / f"{dataset}_seed{seed}_pca{dim}_{enc}_fidelity_values.npz", fidelity=fid, labels=y, sample_ids=sample_ids)
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_parquet(out / "fidelity_summary.parquet", index=False)
    return {"compression": comp, "probes": probes, "boundaries": boundaries, "gaps": gaps, "fidelity": fidelity}
