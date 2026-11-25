from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.report import generate_chinese_report
from pvchange.tech_route import generate_chinese_technical_route

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args=p.parse_args()
    cfg = load_config(args.config)
    generate_chinese_technical_route(cfg)
    generate_chinese_report(cfg)
