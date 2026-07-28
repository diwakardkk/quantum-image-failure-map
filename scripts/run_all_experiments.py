#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import common_parser, load_config
from src.pipeline import run_pipeline
from src.pipeline_v2 import run_pipeline_v2


if __name__ == "__main__":
    args = common_parser("Run all quantum image failure-map experiments.").parse_args()
    cfg = load_config(args.config)
    if cfg.get("project", {}).get("experiment_protocol_version") == "v2_failure_isolation":
        run_pipeline_v2(args, "all_experiments_v2")
    else:
        run_pipeline(args, "all_experiments")
