#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import common_parser
from src.pipeline import run_pipeline


if __name__ == "__main__":
    parser = common_parser("Run the end-to-end smoke test.")
    parser.set_defaults(config="configs/smoke_test.yaml", mode="smoke")
    args = parser.parse_args()
    run_pipeline(args, "smoke_test")

