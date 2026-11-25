from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.dynamics import summarize_pv_dynamics

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    summarize_pv_dynamics(load_config(args.config))
