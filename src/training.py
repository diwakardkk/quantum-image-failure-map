"""Training orchestration for one reference configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .classical_models import fit_classical_baseline, trainable_parameter_count
from .config import configuration_id
from .evaluation import binary_metrics, prediction_frame
from .quantum_models import PennyLaneVQC, train_vqc


def train_reference_models(
    run_dir: Path,
    dataset: str,
    seed: int,
    features: Any,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
    config: dict[str, Any],
    backend: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg_payload = {"dataset": dataset, "seed": seed, "feature": features.name}

    for model_name in ["logistic_regression", "linear_svm", "rbf_svm", "mlp"]:
        cid = configuration_id({**cfg_payload, "model": model_name})
        fitted = fit_classical_baseline(model_name, features.x_train, y_train, features.x_val, y_val)
        joblib.dump(fitted.estimator, run_dir / "checkpoints" / f"{cid}_{fitted.name}.joblib")
        prob = fitted.predict_proba(features.x_test)
        pred = prediction_frame(y_test, prob, sample_ids=test_indices, seed=seed, model=fitted.name, dataset=dataset, split="test", configuration_id=cid)
        pred.to_csv(run_dir / "predictions" / f"{cid}_predictions.csv", index=False)
        metrics = binary_metrics(y_test, prob)
        metrics.update(
            {
                "configuration_id": cid,
                "dataset": dataset,
                "seed": seed,
                "model": fitted.name,
                "preprocessing": features.name,
                "trainable_parameters": trainable_parameter_count(fitted.estimator),
            }
        )
        pd.DataFrame(fitted.validation_results).to_csv(run_dir / "metrics" / f"{cid}_validation_grid.csv", index=False)
        rows.append(metrics)

    qcfg = config.get("quantum_models", {}).get("reference", {})
    selected_device = backend.get("selected")
    if selected_device:
        n_features = min(int(qcfg.get("features", 4)), features.x_train.shape[1])
        n_qubits = int(qcfg.get("qubits", n_features))
        cid = configuration_id({**cfg_payload, "model": "vqc", "qubits": n_qubits, "layers": qcfg.get("layers", 2)})
        try:
            result = train_vqc(
                features.x_train[:, :n_features],
                y_train,
                features.x_val[:, :n_features],
                y_val,
                n_features=n_features,
                n_qubits=n_qubits,
                layers=int(qcfg.get("layers", 2)),
                entanglement=str(qcfg.get("entanglement", "ring")),
                device_name=selected_device,
                epochs=int(config.get("training", {}).get("epochs", {}).get(config.get("project", {}).get("mode", "smoke"), 2)),
                learning_rate=float(config.get("training", {}).get("learning_rate", 0.01)),
                seed=seed,
            )
            np.save(run_dir / "checkpoints" / f"{cid}_vqc_parameters.npy", result.parameters)
            pd.DataFrame(result.history).to_csv(run_dir / "metrics" / f"{cid}_loss_history.csv", index=False)
            np.save(run_dir / "gradients" / f"{cid}_initial_gradients.npy", result.gradients)
            pd.DataFrame([result.resources]).to_csv(run_dir / "resources" / f"{cid}_resources.csv", index=False)
            model = PennyLaneVQC(n_features, n_qubits, int(qcfg.get("layers", 2)), str(qcfg.get("entanglement", "ring")), device_name=selected_device)
            prob = model.predict_proba(features.x_test[:, :n_features], result.parameters)
            pred = prediction_frame(y_test, prob, sample_ids=test_indices, seed=seed, model="vqc", dataset=dataset, split="test", configuration_id=cid)
            pred.to_csv(run_dir / "predictions" / f"{cid}_predictions.csv", index=False)
            metrics = binary_metrics(y_test, prob)
            metrics.update({"configuration_id": cid, "dataset": dataset, "seed": seed, "model": "vqc", "preprocessing": features.name, **result.resources})
            rows.append(metrics)
        except Exception as exc:
            rows.append({"configuration_id": cid, "dataset": dataset, "seed": seed, "model": "vqc", "status": "failed", "exception": str(exc)})
    return rows

