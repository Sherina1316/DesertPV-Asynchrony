from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.events import detect_pv_expansion_events

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    detect_pv_expansion_events(load_config(args.config))
