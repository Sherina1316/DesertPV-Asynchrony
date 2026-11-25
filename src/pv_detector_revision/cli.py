from __future__ import annotations

import argparse
from pathlib import Path

from .atlas import generate_atlas
from .pipeline import apply_detector_revision, load_config, preflight


def parse_units(value: str) -> list[int]:
    units = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not units:
        raise argparse.ArgumentTypeError("At least one unit ID is required")
    return units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v24.22 annual PV detector revision v1.1.1")
    parser.add_argument("command", choices=["preflight", "apply", "atlas", "run"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--units", type=parse_units, default=[35, 107])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        result = preflight(args.output_root, args.units, config)
        print(result.to_string(index=False))
        return 0
    if args.command in {"apply", "run"}:
        annual, units = apply_detector_revision(args.output_root, args.units, config)
        print(units.to_string(index=False))
        print(f"Annual audit rows: {len(annual)}")
    if args.command in {"atlas", "run"}:
        for unit_id in args.units:
            path = generate_atlas(args.output_root, unit_id, config)
            print(f"Atlas: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
