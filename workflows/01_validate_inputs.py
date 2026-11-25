from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.preparation import validate_inputs

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    validate_inputs(load_config(args.config))
