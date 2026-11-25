from __future__ import annotations
import argparse
from pvchange.config import load_config
from pvchange.research_metrics import compute_research_claim_metrics
from pvchange.figures import generate_figures

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); args=p.parse_args()
    cfg=load_config(args.config)
    compute_research_claim_metrics(cfg)
    generate_figures(cfg)
