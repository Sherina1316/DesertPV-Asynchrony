from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.validation import validate_segmentation_and_change

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    validate_segmentation_and_change(load_config(args.config))
