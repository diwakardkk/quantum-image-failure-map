#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import common_parser
from src.pipeline_v2 import run_pipeline_v2


if __name__ == "__main__":
    parser = common_parser("Run Problem 2 v2 trainability scaling stress test.")
    parser.set_defaults(config="configs/pilot_experiment_v2.yaml", problems="2")
    run_pipeline_v2(parser.parse_args(), "problem2_scaling")

