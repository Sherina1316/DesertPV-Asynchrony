from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.uncertainty import quantify_uncertainty

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    quantify_uncertainty(load_config(args.config))
