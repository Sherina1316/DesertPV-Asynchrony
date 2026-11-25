from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.quality import build_unit_quality_and_endmembers
from pvchange.preparation import build_patch_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_unit_quality_and_endmembers(cfg)
    build_patch_dataset(cfg)
