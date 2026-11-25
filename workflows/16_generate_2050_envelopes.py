from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.future import generate_future_envelopes

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    generate_future_envelopes(load_config(args.config))
