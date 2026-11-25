from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.ecology import extract_dynamic_ecology

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    extract_dynamic_ecology(load_config(args.config))
