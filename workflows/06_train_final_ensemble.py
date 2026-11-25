from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.training import train_final_ensemble

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    train_final_ensemble(load_config(args.config))
