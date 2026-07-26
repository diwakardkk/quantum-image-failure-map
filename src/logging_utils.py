"""Logging setup for runs."""

from __future__ import annotations

import json
import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        record = {"event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def setup_logging(run_dir: Path, level: str = "INFO") -> tuple[logging.Logger, JsonlEventLogger]:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quantum_image_failure_map")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(run_dir / "run.log", maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    warning_handler = RotatingFileHandler(run_dir / "warnings.log", maxBytes=2_000_000, backupCount=2)
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    logger.addHandler(warning_handler)

    def _showwarning(message, category, filename, lineno, file=None, line=None):  # type: ignore[no-untyped-def]
        logger.warning("%s:%s: %s: %s", filename, lineno, category.__name__, message)

    warnings.showwarning = _showwarning
    return logger, JsonlEventLogger(run_dir / "raw" / "events.jsonl")

