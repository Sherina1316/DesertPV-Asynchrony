from __future__ import annotations

import argparse

from pvchange.config import load_config
from pvchange.authoritative_temporal import decode_authoritative_reference_change


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    decode_authoritative_reference_change(load_config(args.config))
