"""Run-directory and manifest persistence."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import dump_config
from .hardware import collect_environment, select_pennylane_device, write_git_commit, write_json

RUN_SUBDIRS = [
    "checkpoints",
    "raw",
    "processed",
    "metrics",
    "predictions",
    "gradients",
    "resources",
    "tables",
    "figures",
    "figure_data",
    "artifacts",
]


@dataclass
class RunContext:
    run_dir: Path
    config: dict[str, Any]
    start_time: float
    backend: dict[str, Any]


def create_run_dir(output_root: Path, experiment_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"{stamp}_{experiment_name}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stamp}_{experiment_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    for subdir in RUN_SUBDIRS:
        (candidate / subdir).mkdir(parents=True, exist_ok=True)
    return candidate


def init_run(config: dict[str, Any], experiment_name: str, command: str | None = None, run_dir: str | Path | None = None) -> RunContext:
    root = Path(config.get("project", {}).get("output_root", "outputs"))
    if run_dir:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        for subdir in RUN_SUBDIRS:
            (path / subdir).mkdir(parents=True, exist_ok=True)
    else:
        path = create_run_dir(root, experiment_name)
    backend = select_pennylane_device(
        preferred=config.get("quantum_backend", {}).get("preferred", "auto"),
        wires=int(config.get("quantum_models", {}).get("reference", {}).get("qubits", 4)),
        shots=config.get("quantum_backend", {}).get("analytic_shots"),
    )
    dump_config(config, path / "config_resolved.yaml")
    (path / "command.txt").write_text(command or " ".join(sys.argv), encoding="utf-8")
    write_git_commit(path / "git_commit.txt")
    write_json(path / "environment.json", collect_environment(backend))
    write_json(path / "hardware.json", collect_environment(backend))
    return RunContext(run_dir=path, config=config, start_time=time.time(), backend=backend)


def finalize_run(ctx: RunContext, status: str = "completed", extra: dict[str, Any] | None = None) -> None:
    end = time.time()
    summary = {
        "status": status,
        "start_timestamp": datetime.fromtimestamp(ctx.start_time).isoformat(),
        "end_timestamp": datetime.fromtimestamp(end).isoformat(),
        "runtime_seconds": end - ctx.start_time,
        "random_seeds": ctx.config.get("subsets", {}).get("seeds", []),
        "selected_backend": ctx.backend,
    }
    if extra:
        summary.update(extra)
    write_json(ctx.run_dir / "run_summary.json", summary)


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    shutil.move(str(tmp), str(path))


def append_manifest(run_dir: Path, row: dict[str, Any]) -> None:
    path = run_dir / "metrics" / "experiment_manifest.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    atomic_write_dataframe(df, path)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    shutil.move(str(tmp), str(path))

