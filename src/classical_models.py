"""Classical baselines used as fair comparators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class FittedClassicalModel:
    name: str
    estimator: Any
    validation_results: list[dict[str, Any]]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x)[:, 1]
        scores = self.estimator.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))


def candidate_models(name: str) -> list[tuple[str, Any]]:
    if name == "logistic_regression":
        return [
            (
                f"logistic_regression_C{c}",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=0)),
                    ]
                ),
            )
            for c in [0.1, 1.0, 10.0]
        ]
    if name == "linear_svm":
        return [
            (
                f"linear_svm_C{c}",
                Pipeline([("scaler", StandardScaler()), ("model", SVC(C=c, kernel="linear", probability=True, class_weight="balanced", random_state=0))]),
            )
            for c in [0.1, 1.0, 10.0]
        ]
    if name == "rbf_svm":
        return [
            (
                f"rbf_svm_C{c}",
                Pipeline([("scaler", StandardScaler()), ("model", SVC(C=c, kernel="rbf", probability=True, class_weight="balanced", random_state=0))]),
            )
            for c in [0.1, 1.0, 10.0]
        ]
    if name == "random_forest":
        return [("random_forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0))]
    if name == "mlp":
        return [
            (
                "mlp_32",
                Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0, early_stopping=True))]),
            )
        ]
    raise ValueError(f"Unknown classical model: {name}")


def fit_classical_baseline(
    name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> FittedClassicalModel:
    results: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_name = name
    best_model: Any = None
    for candidate_name, estimator in candidate_models(name):
        estimator.fit(x_train, y_train)
        prob = np.clip(estimator.predict_proba(x_val)[:, 1], 1e-7, 1 - 1e-7)
        loss = float(log_loss(y_val, np.column_stack([1 - prob, prob]), labels=[0, 1]))
        results.append({"candidate": candidate_name, "validation_log_loss": loss})
        if loss < best_loss:
            best_loss = loss
            best_name = candidate_name
            best_model = estimator
    return FittedClassicalModel(best_name, best_model, results)


def trainable_parameter_count(estimator: Any) -> int:
    model = estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model", estimator)
    if hasattr(model, "coef_"):
        return int(np.prod(model.coef_.shape) + np.prod(getattr(model, "intercept_", np.array([])).shape))
    if hasattr(model, "coefs_"):
        return int(sum(np.prod(w.shape) for w in model.coefs_) + sum(np.prod(b.shape) for b in model.intercepts_))
    return int(getattr(model, "n_features_in_", 0))

