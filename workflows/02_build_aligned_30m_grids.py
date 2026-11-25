from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.preparation import build_aligned_grids

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    build_aligned_grids(load_config(args.config))
