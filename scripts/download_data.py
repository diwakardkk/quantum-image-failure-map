#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="smoke")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.get("project", {}).get("data_root", "data"))
    for name in cfg.get("datasets", {}).get("names", ["fashion_mnist", "pneumoniamnist"]):
        load_dataset(name, root, cfg, args.mode, cfg.get("subsets", {}).get("seeds", [11])[0])
        print(f"Downloaded and validated {name}")

