"""Classical-to-quantum encoding helpers and diagnostics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncodingSpec:
    name: str
    n_features: int
    n_qubits: int
    repetitions: int = 1


def scale_angle_features(x: np.ndarray, n_features: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] < n_features:
        x = np.pad(x, ((0, 0), (0, n_features - x.shape[1])))
    x = x[:, :n_features]
    mins = x.min(axis=0, keepdims=True)
    maxs = x.max(axis=0, keepdims=True)
    return ((x - mins) / (maxs - mins + 1e-12) * math.pi).astype(np.float32)


def amplitude_vectors(x: np.ndarray, n_features: int) -> np.ndarray:
    n_qubits = int(math.ceil(math.log2(n_features)))
    dim = 2**n_qubits
    clipped = x[:, :n_features]
    if clipped.shape[1] < dim:
        clipped = np.pad(clipped, ((0, 0), (0, dim - clipped.shape[1])))
    norm = np.linalg.norm(clipped, axis=1, keepdims=True)
    return (clipped / (norm + 1e-12)).astype(np.float32)


def encoding_resources(spec: EncodingSpec) -> dict[str, Any]:
    if spec.name == "angle":
        one = spec.n_features * spec.repetitions
        two = 0
        depth = spec.repetitions
    elif spec.name == "amplitude":
        one = 2 ** spec.n_qubits
        two = max(0, 2 ** spec.n_qubits - 2)
        depth = spec.n_qubits * 2
    elif spec.name == "reuploading":
        one = spec.n_features * spec.repetitions
        two = spec.n_qubits * spec.repetitions
        depth = spec.repetitions * 3
    else:
        one = spec.n_features
        two = max(0, spec.n_qubits - 1)
        depth = 2
    return {
        "encoding": spec.name,
        "n_features": spec.n_features,
        "n_qubits": spec.n_qubits,
        "encoding_repetitions": spec.repetitions,
        "one_qubit_gates": one,
        "two_qubit_gates": two,
        "total_gates": one + two,
        "circuit_depth": depth,
        "estimated_circuit_executions": 1,
    }


def quantum_feature_matrix(x: np.ndarray, spec: EncodingSpec) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    if spec.name == "amplitude":
        z = amplitude_vectors(x, spec.n_features)
    else:
        z = scale_angle_features(x, spec.n_features)
    meta = encoding_resources(spec)
    meta["state_preparation_time_seconds"] = time.perf_counter() - start
    return z, meta


def fidelity_matrix(representations: np.ndarray) -> np.ndarray:
    reps = np.asarray(representations, dtype=np.float64)
    reps = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-12)
    return np.square(np.clip(reps @ reps.T, -1.0, 1.0))

