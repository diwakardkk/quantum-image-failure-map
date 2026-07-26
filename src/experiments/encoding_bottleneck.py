"""Problem 1: encoding bottleneck diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

from ..encodings import EncodingSpec, fidelity_matrix, quantum_feature_matrix
from ..preprocessing import FeatureSet, reconstruction_metrics


def fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) != 2:
        return float("nan")
    m0, m1 = x[y == classes[0]].mean(axis=0), x[y == classes[1]].mean(axis=0)
    v0, v1 = x[y == classes[0]].var(axis=0).mean(), x[y == classes[1]].var(axis=0).mean()
    return float(np.sum((m0 - m1) ** 2) / (v0 + v1 + 1e-12))


def run_encoding_diagnostics(run_dir: Path, dataset: str, features: FeatureSet, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for n_features in sorted(set([2, 4, min(8, features.x_train.shape[1]), min(features.x_train.shape[1], 16)])):
        for encoding in ["angle", "amplitude"]:
            n_qubits = n_features if encoding == "angle" else int(np.ceil(np.log2(max(2, n_features))))
            spec = EncodingSpec(encoding, int(n_features), int(n_qubits))
            z_train, resources = quantum_feature_matrix(features.x_train, spec)
            z_test, _ = quantum_feature_matrix(features.x_test, spec)
            fid = fidelity_matrix(z_test[: min(40, len(z_test))])
            np.save(run_dir / "processed" / "fidelity_matrix.npy", fid)
            knn = KNeighborsClassifier(n_neighbors=min(3, len(z_train)))
            knn.fit(z_train, y_train)
            rows.append(
                {
                    "problem": 1,
                    "dataset": dataset,
                    "model": "representation_probe",
                    "preprocessing": features.name,
                    "encoding": encoding,
                    "feature_count": n_features,
                    "accuracy": float(knn.score(z_test, y_test)),
                    "fisher_ratio": fisher_ratio(z_train, y_train),
                    "intra_inter_distance_gap": float(pairwise_distances(z_train[y_train == 0]).mean() - pairwise_distances(z_train[y_train == 1]).mean()) if len(np.unique(y_train)) == 2 else np.nan,
                    **resources,
                    **reconstruction_metrics(np.zeros((0, 1, 1, 1)), None),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / f"{dataset}_P1_encoding_metrics.csv", index=False)
    return df

