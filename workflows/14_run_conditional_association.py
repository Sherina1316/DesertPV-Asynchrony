from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.association import run_conditional_association

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    run_conditional_association(load_config(args.config))
