#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import common_parser
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = common_parser("Run Problem 2 trainability experiments.")
    parser.set_defaults(problems="2")
    run_pipeline(parser.parse_args(), "problem2_trainability")

