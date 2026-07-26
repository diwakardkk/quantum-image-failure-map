"""PennyLane quantum model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VQCResult:
    parameters: np.ndarray
    history: list[dict[str, float]]
    gradients: np.ndarray
    resources: dict[str, Any]


def initialize_parameters(layers: int, qubits: int, seed: int, method: str = "xavier") -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (layers, qubits, 2)
    if method == "near_zero":
        return rng.normal(0, 0.01, size=shape)
    if method == "uniform":
        return rng.uniform(-np.pi, np.pi, size=shape)
    scale = np.sqrt(2.0 / max(1, qubits))
    return rng.normal(0, scale, size=shape)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _require_qml():
    try:
        import pennylane as qml

        return qml
    except Exception as exc:
        raise RuntimeError(f"PennyLane is required for quantum models: {exc}") from exc


class PennyLaneVQC:
    def __init__(
        self,
        n_features: int,
        n_qubits: int,
        layers: int,
        entanglement: str = "ring",
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int = 0,
    ) -> None:
        self.qml = _require_qml()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.device_name = device_name or "default.qubit"
        self.shots = shots
        self.seed = seed
        self.dev = self.qml.device(self.device_name, wires=n_qubits, shots=shots)

        @self.qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def circuit(features, weights):
            for wire in range(n_qubits):
                self.qml.RY(features[wire % len(features)], wires=wire)
            for layer in range(layers):
                for wire in range(n_qubits):
                    self.qml.RY(weights[layer, wire, 0], wires=wire)
                    self.qml.RZ(weights[layer, wire, 1], wires=wire)
                if entanglement in {"linear", "ring"}:
                    for wire in range(n_qubits - 1):
                        self.qml.CNOT(wires=[wire, wire + 1])
                if entanglement == "ring" and n_qubits > 2:
                    self.qml.CNOT(wires=[n_qubits - 1, 0])
            return self.qml.expval(self.qml.PauliZ(0))

        self.circuit = circuit

    def expectation(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return np.asarray([self.circuit(row[: self.n_features], parameters) for row in x], dtype=float)

    def predict_proba(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        expvals = self.expectation(x, parameters)
        return np.asarray((1.0 - expvals) / 2.0, dtype=float)

    def finite_shot_probabilities(self, analytic_prob: np.ndarray, shots: int, repeats: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = rng.binomial(shots, np.clip(analytic_prob, 0, 1), size=(repeats, len(analytic_prob)))
        return counts / shots

    def resources(self) -> dict[str, Any]:
        two_qubit = (self.n_qubits - 1 + int(self.entanglement == "ring" and self.n_qubits > 2)) * self.layers
        one_qubit = self.n_qubits + 2 * self.n_qubits * self.layers
        return {
            "model": "vqc",
            "qubits": self.n_qubits,
            "layers": self.layers,
            "entanglement": self.entanglement,
            "one_qubit_gates": one_qubit,
            "two_qubit_gates": two_qubit,
            "total_gates": one_qubit + two_qubit,
            "circuit_depth": 1 + self.layers * (2 + int(two_qubit > 0)),
            "trainable_parameters": int(self.layers * self.n_qubits * 2),
            "shots": self.shots,
            "device": self.device_name,
        }


def train_vqc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_features: int,
    n_qubits: int,
    layers: int,
    entanglement: str,
    device_name: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    init_method: str = "xavier",
) -> VQCResult:
    qml = _require_qml()
    pnp = qml.numpy
    model = PennyLaneVQC(n_features, n_qubits, layers, entanglement, device_name=device_name, seed=seed)
    weights = pnp.array(initialize_parameters(layers, n_qubits, seed, init_method), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    history: list[dict[str, float]] = []

    def loss_fn(w):
        probs = pnp.array([(1 - model.circuit(row[:n_features], w)) / 2 for row in x_train])
        probs = pnp.clip(probs, 1e-7, 1 - 1e-7)
        return -pnp.mean(y_train * pnp.log(probs) + (1 - y_train) * pnp.log(1 - probs))

    grad_fn = qml.grad(loss_fn)
    init_grad = np.asarray(grad_fn(weights), dtype=float)
    for epoch in range(epochs):
        weights, train_loss = opt.step_and_cost(loss_fn, weights)
        val_prob = model.predict_proba(x_val, np.asarray(weights, dtype=float))
        val_loss = float(-np.mean(y_val * np.log(np.clip(val_prob, 1e-7, 1)) + (1 - y_val) * np.log(np.clip(1 - val_prob, 1e-7, 1))))
        history.append({"epoch": epoch, "train_loss": float(train_loss), "validation_loss": val_loss})
    final_params = np.asarray(weights, dtype=float)
    return VQCResult(final_params, history, init_grad, model.resources())

