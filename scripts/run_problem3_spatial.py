#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import common_parser, load_config
from src.pipeline import run_pipeline
from src.pipeline_v2 import run_pipeline_v2

if __name__ == "__main__":
    parser = common_parser("Run Problem 3 spatial robustness experiments.")
    parser.set_defaults(problems="3")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("project", {}).get("experiment_protocol_version") == "v2_failure_isolation":
        run_pipeline_v2(args, "problem3_spatial_v2")
    else:
        run_pipeline(args, "problem3_spatial")
