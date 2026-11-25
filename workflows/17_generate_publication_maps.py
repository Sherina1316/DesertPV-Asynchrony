from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.mapping import generate_maps

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    generate_maps(load_config(args.config))
