"""Top-level experiment pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import aggregate_metric_files
from .config import apply_cli_overrides, load_config
from .datasets import load_dataset, save_dataset_artifacts
from .experiments.encoding_bottleneck import run_encoding_diagnostics
from .experiments.measurement_reliability import run_reliability_diagnostics
from .experiments.practical_utility import run_utility_summary
from .experiments.spatial_structure import run_spatial_diagnostics
from .experiments.trainability import summarize_gradients
from .logging_utils import setup_logging
from .plotting import generate_all_plots
from .preprocessing import fit_transform_variant, save_features
from .reproducibility import seed_everything
from .resource_tracking import track_resources
from .result_store import append_manifest, finalize_run, init_run
from .tables import generate_tables
from .training import train_reference_models
from .verification import verify_run


def _selected_problems(args: Any | None) -> set[int]:
    value = getattr(args, "problems", None) if args is not None else None
    if not value:
        return {1, 2, 3, 4, 5}
    return {int(x) for x in value.split(",")}


def run_pipeline(args: Any, experiment_name: str = "all_experiments") -> Path:
    config = apply_cli_overrides(load_config(getattr(args, "config", None)), args)
    mode = config.get("project", {}).get("mode", "smoke")
    ctx = init_run(config, experiment_name, run_dir=getattr(args, "run_dir", None))
    logger, events = setup_logging(ctx.run_dir, config.get("logging", {}).get("level", "INFO"))
    logger.info("Run directory: %s", ctx.run_dir)
    logger.info("Selected quantum backend: %s", ctx.backend.get("selected"))
    events.write("run_start", run_dir=str(ctx.run_dir), mode=mode)
    problems = _selected_problems(args)
    all_primary_rows: list[dict[str, Any]] = []
    all_problem_rows: list[pd.DataFrame] = []

    try:
        if getattr(args, "plot_only", False):
            aggregate_metric_files(ctx.run_dir)
            generate_all_plots(ctx.run_dir)
            generate_tables(ctx.run_dir)
            verify_run(ctx.run_dir)
            finalize_run(ctx)
            return ctx.run_dir

        datasets = config.get("datasets", {}).get("names", ["fashion_mnist"])
        seeds = config.get("subsets", {}).get("seeds", [11])
        data_root = Path(config.get("project", {}).get("data_root", "data"))
        max_configs = config.get("project", {}).get("max_configs")
        completed_configs = 0
        for dataset_name in datasets:
            for seed in seeds:
                if max_configs is not None and completed_configs >= int(max_configs):
                    break
                seed_state = seed_everything(int(seed))
                logger.info("Loading dataset=%s seed=%s", dataset_name, seed)
                bundle = load_dataset(dataset_name, data_root, config, mode, int(seed))
                save_dataset_artifacts(bundle, ctx.run_dir)
                variants = config.get("preprocessing", {}).get("variants", ["resize_4"])
                for variant in variants:
                    if max_configs is not None and completed_configs >= int(max_configs):
                        break
                    start = time.time()
                    status = "completed"
                    exception = ""
                    try:
                        logger.info("Preprocessing dataset=%s variant=%s", dataset_name, variant)
                        features = fit_transform_variant(
                            variant,
                            bundle.x_train,
                            bundle.x_val,
                            bundle.x_test,
                            ctx.run_dir / "artifacts" / "preprocessing",
                        )
                        save_features(features, ctx.run_dir, dataset_name)
                        rows: list[dict[str, Any]] = []
                        with track_resources() as resources:
                            rows = train_reference_models(
                                ctx.run_dir,
                                dataset_name,
                                int(seed),
                                features,
                                bundle.y_train,
                                bundle.y_val,
                                bundle.y_test,
                                bundle.test_indices,
                                config,
                                ctx.backend,
                            )
                        snapshot = resources["snapshot"]
                        for row in rows:
                            row["seed_notes"] = "; ".join(seed_state.notes)
                            if snapshot:
                                row["wall_time_seconds"] = snapshot.wall_time_seconds
                                row["cpu_time_seconds"] = snapshot.cpu_time_seconds
                                row["peak_host_ram_bytes"] = snapshot.peak_host_ram_bytes
                        all_primary_rows.extend(rows)
                        pd.DataFrame(all_primary_rows).to_csv(ctx.run_dir / "metrics" / "primary_metrics.csv", index=False)
                        if 1 in problems:
                            all_problem_rows.append(run_encoding_diagnostics(ctx.run_dir, dataset_name, features, bundle.y_train, bundle.y_test))
                        if 2 in problems:
                            all_problem_rows.append(summarize_gradients(ctx.run_dir, dataset_name))
                        if 3 in problems:
                            all_problem_rows.append(run_spatial_diagnostics(ctx.run_dir, dataset_name, bundle.x_test, bundle.y_test))
                        if 4 in problems:
                            repeats = int(config.get("reliability_experiment", {}).get("repeats", {}).get(mode, 5))
                            shots = config.get("reliability_experiment", {}).get("shots", [64, 256])
                            all_problem_rows.append(run_reliability_diagnostics(ctx.run_dir, dataset_name, int(seed), repeats, shots))
                        if 5 in problems:
                            all_problem_rows.append(run_utility_summary(ctx.run_dir, dataset_name))
                    except Exception as exc:
                        status = "failed"
                        exception = str(exc)
                        logger.exception("Configuration failed")
                    finally:
                        append_manifest(
                            ctx.run_dir,
                            {
                                "configuration_id": f"{dataset_name}_{seed}_{variant}",
                                "experiment": experiment_name,
                                "problem_number": ",".join(map(str, sorted(problems))),
                                "dataset": dataset_name,
                                "model": "reference_suite",
                                "preprocessing": variant,
                                "encoding": config.get("quantum_models", {}).get("reference", {}).get("encoding", "angle"),
                                "qubits": config.get("quantum_models", {}).get("reference", {}).get("qubits", 4),
                                "layers": config.get("quantum_models", {}).get("reference", {}).get("layers", 2),
                                "shots": None,
                                "noise": None,
                                "seed": seed,
                                "status": status,
                                "start_time": start,
                                "end_time": time.time(),
                                "runtime": time.time() - start,
                                "output_paths": str(ctx.run_dir),
                                "exception_message": exception,
                            },
                        )
                        completed_configs += 1
        if all_problem_rows:
            pd.concat(all_problem_rows, ignore_index=True, sort=False).to_csv(ctx.run_dir / "metrics" / "problem_metrics.csv", index=False)
        aggregate_metric_files(ctx.run_dir)
        generate_tables(ctx.run_dir)
        if not getattr(args, "skip_plots", False):
            generate_all_plots(ctx.run_dir)
        report = verify_run(ctx.run_dir)
        finalize_run(ctx, extra={"verification_failed": report["failed"]})
        return ctx.run_dir
    except Exception:
        finalize_run(ctx, status="failed")
        raise

