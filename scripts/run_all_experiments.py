#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import common_parser
from src.pipeline import run_pipeline


if __name__ == "__main__":
    args = common_parser("Run all quantum image failure-map experiments.").parse_args()
    run_pipeline(args, "all_experiments")

