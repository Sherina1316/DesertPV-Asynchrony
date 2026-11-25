from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.audit import audit_outputs

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    audit_outputs(load_config(args.config))
