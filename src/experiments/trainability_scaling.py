"""Problem 2 v2: dedicated VQC gradient-scaling stress test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import configuration_id
from ..encodings import scale_angle_features
from ..preprocessing import flatten_images
from ..quantum_models import initialize_parameters

PROTOCOL = "v2_failure_isolation"


def _balanced_subset(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        chosen.append(idx[: min(per_class, len(idx))])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def _gradient_summary(grad: np.ndarray) -> dict[str, float]:
    flat = np.abs(np.asarray(grad, dtype=float).reshape(-1))
    variance = float(np.var(flat))
    return {
        "mean_abs_gradient": float(np.mean(flat)),
        "median_abs_gradient": float(np.median(flat)),
        "gradient_variance": variance,
        "log10_gradient_variance": float(np.log10(variance + 1e-18)),
        "gradient_std": float(np.std(flat)),
        "gradient_norm_L2": float(np.linalg.norm(flat, ord=2)),
        "gradient_norm_L1": float(np.linalg.norm(flat, ord=1)),
        "near_zero_fraction_1e_4": float(np.mean(flat < 1e-4)),
        "near_zero_fraction_1e_6": float(np.mean(flat < 1e-6)),
        "near_zero_fraction_1e_8": float(np.mean(flat < 1e-8)),
    }


def _qml_gradient(x: np.ndarray, y: np.ndarray, qubits: int, layers: int, cost_type: str, init_seed: int, init_method: str, device_name: str) -> np.ndarray:
    import pennylane as qml

    pnp = qml.numpy
    dev = qml.device(device_name or "default.qubit", wires=qubits, shots=None)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def circuit(features, weights):
        for wire in range(qubits):
            qml.RY(features[wire % len(features)], wires=wire)
        for layer in range(layers):
            for wire in range(qubits):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            for wire in range(qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            if qubits > 2:
                qml.CNOT(wires=[qubits - 1, 0])
        if cost_type == "local":
            return [qml.expval(qml.PauliZ(w)) for w in range(min(2, qubits))]
        return qml.expval(qml.PauliZ(0))

    weights = pnp.array(initialize_parameters(layers, qubits, init_seed, init_method), requires_grad=True)

    def loss_fn(w):
        values = []
        for row, label in zip(x, y):
            raw = circuit(row[:qubits], w)
            expval = pnp.mean(pnp.array(raw)) if cost_type == "local" else raw
            prob = pnp.clip((1 - expval) / 2, 1e-7, 1 - 1e-7)
            values.append(-(label * pnp.log(prob) + (1 - label) * pnp.log(1 - prob)))
        return pnp.mean(pnp.array(values))

    return np.asarray(qml.grad(loss_fn)(weights), dtype=float)


def run_trainability_scaling_v2(run_dir: Path, dataset: str, bundle: Any, config: dict[str, Any], backend: dict[str, Any], seed: int) -> pd.DataFrame:
    out = run_dir / "problem2"
    raw = out / "raw_gradients"
    raw.mkdir(parents=True, exist_ok=True)
    mode = config.get("project", {}).get("mode", "pilot")
    p2 = config.get("v2", {}).get("problem2", {})
    qubits_grid = [int(q) for q in p2.get("qubits", [4, 6])]
    layers_grid = [int(l) for l in p2.get("layers", [1, 2])]
    replicates = int(p2.get("initialization_replicates", {}).get(mode, 3))
    per_class = int(p2.get("subset_per_class", 16))
    cost_types = p2.get("cost_types", ["global"])
    init_controls = p2.get("initialization_controls", ["uniform"])
    subset = _balanced_subset(bundle.y_train, per_class, seed)
    np.savez_compressed(out / f"{dataset}_seed{seed}_subset_ids.npz", subset_indices=subset)
    x = scale_angle_features(flatten_images(bundle.x_train[subset]), max(qubits_grid))
    y = bundle.y_train[subset]
    rows = []
    device_name = backend.get("selected") or "default.qubit"
    try:
        import pennylane  # noqa: F401

        qml_available = True
    except Exception:
        qml_available = False
    for qubits in qubits_grid:
        for layers in layers_grid:
            for cost_type in cost_types:
                for init_method in init_controls:
                    for rep in range(replicates):
                        init_seed = int(seed * 10_000 + qubits * 1_000 + layers * 100 + rep)
                        cid = configuration_id({"problem": 2, "dataset": dataset, "seed": seed, "q": qubits, "l": layers, "cost": cost_type, "init": init_method, "rep": rep, "protocol": PROTOCOL})
                        path = raw / f"{cid}.npz"
                        if path.exists():
                            grad = np.load(path)["gradient"]
                            status = "completed"
                            error = ""
                        elif qml_available:
                            try:
                                grad = _qml_gradient(x[:, :qubits], y, qubits, layers, cost_type, init_seed, init_method, device_name)
                                np.savez_compressed(path, gradient=grad, configuration_id=cid)
                                status = "completed"
                                error = ""
                            except Exception as exc:
                                grad = np.array([])
                                status = "failed"
                                error = str(exc)
                        else:
                            grad = np.array([])
                            status = "skipped"
                            error = "PennyLane unavailable"
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "configuration_id": cid,
                            "protocol_version": PROTOCOL,
                            "qubits": qubits,
                            "layers": layers,
                            "cost_type": cost_type,
                            "initialization": init_method,
                            "initialization_seed": init_seed,
                            "replicate": rep,
                            "status": status,
                            "gradient_file": str(path.relative_to(run_dir)) if path.exists() else "",
                            "differentiation_method": "parameter-shift",
                            "exception": error,
                        }
                        row.update(_gradient_summary(grad) if grad.size else {})
                        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "gradient_summary.parquet", index=False)
    df.to_csv(run_dir / "metrics" / f"{dataset}_P2_v2_gradient_scaling.csv", index=False)
    fits = _fit_scaling(df)
    fits.to_parquet(out / "scaling_fit_results.parquet", index=False)
    pd.DataFrame(columns=["dataset", "seed", "configuration_id", "training_success", "validation_loss_reduction"]).to_parquet(out / "selected_training_runs.parquet", index=False)
    return df


def _fit_scaling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[(df["status"] == "completed") & np.isfinite(df.get("gradient_variance", np.nan))]
    if valid.empty:
        return pd.DataFrame(columns=["dataset", "layers", "cost_type", "fit_model", "aic", "bic", "r2", "notes"])
    grouped = valid.groupby(["dataset", "layers", "cost_type"], dropna=False)
    for (dataset, layers, cost), group in grouped:
        agg = group.groupby("qubits", as_index=False)["gradient_variance"].mean()
        if len(agg) < 3:
            continue
        x = agg["qubits"].to_numpy(dtype=float)
        y = np.clip(agg["gradient_variance"].to_numpy(dtype=float), 1e-18, None)
        for model in ["exponential", "power_law"]:
            tx = x if model == "exponential" else np.log(x)
            ty = np.log(y)
            coef = np.polyfit(tx, ty, 1)
            pred = np.polyval(coef, tx)
            rss = float(np.sum((ty - pred) ** 2))
            n = len(y)
            k = 2
            tss = float(np.sum((ty - ty.mean()) ** 2))
            rows.append({"dataset": dataset, "layers": layers, "cost_type": cost, "fit_model": model, "slope": float(coef[0]), "aic": n * np.log(rss / n + 1e-18) + 2 * k, "bic": n * np.log(rss / n + 1e-18) + k * np.log(n), "r2": 1 - rss / (tss + 1e-18), "notes": "log-scale fit"})
    return pd.DataFrame(rows)
