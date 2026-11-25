from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.training import benchmark_models

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    benchmark_models(load_config(args.config))
