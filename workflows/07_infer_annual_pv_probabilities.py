from __future__ import annotations

import argparse

from pvchange.calibration import calibrate_annual_probabilities
from pvchange.config import load_config
from pvchange.inference import infer_annual_probabilities
from pvchange.authoritative_reference import reconstruct_authoritative_reference_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    infer_annual_probabilities(cfg)
    calibrate_annual_probabilities(cfg)
    reconstruct_authoritative_reference_probabilities(cfg)
