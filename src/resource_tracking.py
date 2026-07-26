"""Lightweight resource measurement helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import psutil


@dataclass
class ResourceSnapshot:
    wall_time_seconds: float
    cpu_time_seconds: float
    peak_host_ram_bytes: int


@contextmanager
def track_resources() -> Iterator[dict[str, ResourceSnapshot | None]]:
    process = psutil.Process()
    start_wall = time.perf_counter()
    start_cpu = sum(process.cpu_times()[:2])
    holder: dict[str, ResourceSnapshot | None] = {"snapshot": None}
    try:
        yield holder
    finally:
        cpu = sum(process.cpu_times()[:2]) - start_cpu
        holder["snapshot"] = ResourceSnapshot(
            wall_time_seconds=time.perf_counter() - start_wall,
            cpu_time_seconds=cpu,
            peak_host_ram_bytes=process.memory_info().rss,
        )

