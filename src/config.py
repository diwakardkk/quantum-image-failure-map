"""Configuration loading and command-line plumbing."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    default = load_yaml(root / "configs" / "default.yaml")
    if path is None:
        return default
    supplied = load_yaml(Path(path))
    return deep_update(default, supplied)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if getattr(args, "mode", None):
        cfg.setdefault("project", {})["mode"] = args.mode
    if getattr(args, "datasets", None):
        cfg.setdefault("datasets", {})["names"] = args.datasets.split(",")
    if getattr(args, "seeds", None):
        cfg.setdefault("subsets", {})["seeds"] = [int(x) for x in args.seeds.split(",")]
    if getattr(args, "num_workers", None) is not None:
        cfg.setdefault("training", {})["num_workers"] = args.num_workers
    if getattr(args, "device", None):
        cfg.setdefault("quantum_backend", {})["preferred"] = args.device
    if getattr(args, "max_configs", None) is not None:
        cfg.setdefault("project", {})["max_configs"] = args.max_configs
    return cfg


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default=None)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names.")
    parser.add_argument("--problems", default=None, help="Comma-separated problem numbers, e.g. 1,3,5.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    return parser


def dump_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def configuration_id(payload: dict[str, Any]) -> str:
    import hashlib

    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

