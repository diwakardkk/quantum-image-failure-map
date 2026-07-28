"""Top-level pipeline for the v2 failure-isolation protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregate import aggregate_metric_files
from .config import apply_cli_overrides, load_config
from .datasets import load_dataset, save_dataset_artifacts
from .experiments.hypothesis_summary import build_hypothesis_summary
from .experiments.problem1_v2 import run_problem1_v2
from .experiments.problem345_v2 import run_reliability_v2, run_spatial_v2, run_utility_v2
from .experiments.trainability_scaling import run_trainability_scaling_v2
from .logging_utils import setup_logging
from .plotting import generate_all_plots
from .preprocessing import fit_transform_variant, save_features
from .reproducibility import seed_everything
from .result_store import append_manifest, finalize_run, init_run
from .tables import generate_tables
from .training import train_reference_models
from .verification import verify_run

PROTOCOL = "v2_failure_isolation"


def _selected_problems(args: Any | None) -> set[int]:
    value = getattr(args, "problems", None) if args is not None else None
    return {1, 2, 3, 4, 5} if not value else {int(x) for x in value.split(",")}


def run_pipeline_v2(args: Any, experiment_name: str = "all_experiments_v2") -> Path:
    config = apply_cli_overrides(load_config(getattr(args, "config", None)), args)
    config.setdefault("project", {})["experiment_protocol_version"] = PROTOCOL
    mode = config.get("project", {}).get("mode", "pilot")
    ctx = init_run(config, experiment_name, run_dir=getattr(args, "run_dir", None))
    logger, events = setup_logging(ctx.run_dir, config.get("logging", {}).get("level", "INFO"))
    logger.info("Run directory: %s", ctx.run_dir)
    logger.info("Protocol: %s", PROTOCOL)
    logger.info("Selected quantum backend: %s", ctx.backend.get("selected"))
    events.write("run_start", run_dir=str(ctx.run_dir), mode=mode, protocol=PROTOCOL)
    problems = _selected_problems(args)
    primary_rows: list[dict[str, Any]] = []
    problem_frames: list[pd.DataFrame] = []
    try:
        datasets = config.get("datasets", {}).get("names", ["fashion_mnist"])
        seeds = config.get("subsets", {}).get("seeds", [11])
        data_root = Path(config.get("project", {}).get("data_root", "data"))
        max_configs = config.get("project", {}).get("max_configs")
        completed = 0
        for dataset_name in datasets:
            for seed in seeds:
                if max_configs is not None and completed >= int(max_configs):
                    break
                seed_everything(int(seed))
                logger.info("Loading dataset=%s seed=%s", dataset_name, seed)
                bundle = load_dataset(dataset_name, data_root, config, mode, int(seed))
                save_dataset_artifacts(bundle, ctx.run_dir)
                variants = config.get("preprocessing", {}).get("variants", ["pca_4"])
                for variant in variants:
                    if max_configs is not None and completed >= int(max_configs):
                        break
                    start = time.time()
                    status = "completed"
                    exception = ""
                    try:
                        logger.info("Reference models v2 dataset=%s seed=%s variant=%s", dataset_name, seed, variant)
                        features = fit_transform_variant(variant, bundle.x_train, bundle.x_val, bundle.x_test, ctx.run_dir / "artifacts" / "preprocessing")
                        save_features(features, ctx.run_dir, dataset_name)
                        rows = train_reference_models(ctx.run_dir, dataset_name, int(seed), features, bundle.y_train, bundle.y_val, bundle.y_test, bundle.test_indices, config, ctx.backend)
                        for row in rows:
                            row["protocol_version"] = PROTOCOL
                        primary_rows.extend(rows)
                        pd.DataFrame(primary_rows).to_csv(ctx.run_dir / "metrics" / "primary_metrics.csv", index=False)
                    except Exception as exc:
                        status = "failed"
                        exception = str(exc)
                        logger.exception("Reference configuration failed")
                    finally:
                        append_manifest(
                            ctx.run_dir,
                            {
                                "configuration_id": f"{dataset_name}_{seed}_{variant}_{PROTOCOL}",
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
                                "protocol_version": PROTOCOL,
                            },
                        )
                        completed += 1
                if 1 in problems:
                    logger.info("Problem 1 v2 dataset=%s seed=%s", dataset_name, seed)
                    p1 = run_problem1_v2(ctx.run_dir, dataset_name, bundle, config, int(seed))
                    problem_frames.extend([v for v in p1.values() if isinstance(v, pd.DataFrame)])
                if 2 in problems:
                    logger.info("Problem 2 v2 dataset=%s seed=%s", dataset_name, seed)
                    problem_frames.append(run_trainability_scaling_v2(ctx.run_dir, dataset_name, bundle, config, ctx.backend, int(seed)))
                if 3 in problems:
                    logger.info("Problem 3 v2 dataset=%s seed=%s", dataset_name, seed)
                    problem_frames.append(run_spatial_v2(ctx.run_dir, dataset_name, bundle, config, int(seed)))
                if 4 in problems:
                    logger.info("Problem 4 v2 dataset=%s seed=%s", dataset_name, seed)
                    problem_frames.append(run_reliability_v2(ctx.run_dir, dataset_name, config, int(seed)))
                if 5 in problems:
                    logger.info("Problem 5 v2 dataset=%s seed=%s", dataset_name, seed)
                    problem_frames.append(run_utility_v2(ctx.run_dir, dataset_name, bundle, config, int(seed)))
        if problem_frames:
            pd.concat(problem_frames, ignore_index=True, sort=False).to_csv(ctx.run_dir / "metrics" / "problem_metrics_v2.csv", index=False)
        build_hypothesis_summary(ctx.run_dir)
        _empty_stats(ctx.run_dir)
        aggregate_metric_files(ctx.run_dir)
        generate_tables(ctx.run_dir)
        if not getattr(args, "skip_plots", False):
            generate_all_plots(ctx.run_dir)
        report = verify_run(ctx.run_dir)
        finalize_run(ctx, extra={"verification_failed": report["failed"], "experiment_protocol_version": PROTOCOL})
        return ctx.run_dir
    except Exception:
        finalize_run(ctx, status="failed", extra={"experiment_protocol_version": PROTOCOL})
        raise


def _empty_stats(run_dir: Path) -> None:
    cols = ["problem", "dataset", "comparison", "metric", "test", "statistic", "p_value", "adjusted_p", "effect_size", "CI_low", "CI_high", "n", "notes"]
    path = run_dir / "metrics" / "statistical_tests_v2.parquet"
    if not path.exists():
        pd.DataFrame(columns=cols).to_parquet(path, index=False)
