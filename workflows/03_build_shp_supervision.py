from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.preparation import build_supervision

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    build_supervision(load_config(args.config))
