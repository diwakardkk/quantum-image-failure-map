"""Hardware and backend discovery."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def detect_torch() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "cuda_available": False}
    try:
        import torch

        info["available"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        info["error"] = str(exc)
    return info


def select_pennylane_device(preferred: str = "auto", wires: int = 4, shots: int | None = None) -> dict[str, Any]:
    try:
        import pennylane as qml
    except Exception as exc:
        return {"framework": "pennylane", "selected": None, "available": False, "error": str(exc)}

    candidates = [preferred] if preferred and preferred != "auto" else [
        "lightning.gpu",
        "lightning.qubit",
        "default.qubit",
    ]
    attempted: list[dict[str, str]] = []
    for name in candidates:
        try:
            dev = qml.device(name, wires=wires, shots=shots)
            # A tiny execution catches installed-but-nonfunctional lightning.gpu backends.
            @qml.qnode(dev)
            def circuit():
                return qml.expval(qml.PauliZ(0))

            _ = circuit()
            return {
                "framework": "pennylane",
                "available": True,
                "selected": name,
                "wires": wires,
                "shots": shots,
                "fallback_attempts": attempted,
            }
        except Exception as exc:  # pragma: no cover - backend dependent
            attempted.append({"device": name, "error": str(exc)})
    return {
        "framework": "pennylane",
        "available": False,
        "selected": None,
        "fallback_attempts": attempted,
    }


def collect_environment(selected_backend: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "torch": detect_torch(),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "scikit-image",
                "torch",
                "torchvision",
                "pennylane",
                "pennylane-lightning",
                "medmnist",
                "matplotlib",
                "seaborn",
            ]
        ),
        "selected_quantum_backend": selected_backend,
    }


def write_git_commit(path: Path) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:
        commit = f"unavailable: {exc}"
    path.write_text(commit + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
