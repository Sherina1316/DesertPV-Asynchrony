from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.response import analyze_response_dynamics

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    analyze_response_dynamics(load_config(args.config))
