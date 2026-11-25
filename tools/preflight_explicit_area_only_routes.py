from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

EXPECTED_EXPLICIT_AREA_ONLY = {83, 104, 141}
PIXEL_MODES = {"direct_pixel", "anchor_guided_candidate"}
TERMINAL_COLUMNS = {
    "terminal_2025_expansion_source_path",
    "terminal_2025_probability_gain_path",
    "terminal_2025_site_likelihood_path",
    "terminal_2025_component_similarity_path",
    "terminal_2025_multicue_change_path",
    "terminal_2025_front_proximity_path",
    "terminal_2025_remote_census_likelihood_path",
    "terminal_2025_remote_change_support_path",
    "terminal_2025_remote_candidate_density_path",
    "terminal_2025_absolute_darkness_path",
}


def _path_exists(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() != "nan" and Path(text).is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["output_root"])
    start = int(cfg["study"]["start_year"])
    end = int(cfg["study"]["end_year"])
    reference_year = int(cfg["study"]["reference_year"])
    expected_years = end - start + 1
    expected_units = len(cfg["study"].get("unit_ids", list(range(1, 143))))
    expected_records = expected_units * expected_years

    registry_path = root / "07_probability_calibration" / "calibrated_inference_registry.csv"
    if not registry_path.is_file():
        raise SystemExit(f"Missing Stage-7 registry: {registry_path}")
    registry = pd.read_csv(registry_path)
    required = {"Id", "year", "inference_mode"} | TERMINAL_COLUMNS
    missing = sorted(required - set(registry.columns))
    if missing:
        raise SystemExit(f"Stage-7 registry missing columns: {missing}")

    ids = pd.to_numeric(registry["Id"], errors="coerce")
    years = pd.to_numeric(registry["year"], errors="coerce")
    if ids.isna().any() or years.isna().any():
        raise SystemExit("Stage-7 registry contains invalid Id/year values")
    registry = registry.assign(Id=ids.astype(int), year=years.astype(int))
    if len(registry) != expected_records or registry["Id"].nunique() != expected_units:
        raise SystemExit(
            f"Stage-7 coverage is incomplete: {len(registry)}/{expected_records} records, "
            f"{registry['Id'].nunique()}/{expected_units} units"
        )
    counts = registry.groupby("Id")["year"].nunique()
    bad_counts = counts[counts.ne(expected_years)].to_dict()
    if bad_counts:
        raise SystemExit(f"Units without {expected_years} annual rows: {bad_counts}")
    if registry.duplicated(["Id", "year"]).any():
        raise SystemExit("Stage-7 registry contains duplicate Id-year rows")

    area_only = registry["inference_mode"].astype(str).eq("area_only_fallback")
    area_only_ids = set(registry.loc[area_only, "Id"].unique().tolist())
    if not EXPECTED_EXPLICIT_AREA_ONLY.issubset(area_only_ids):
        raise SystemExit(
            "Explicit area-only routing is incomplete: "
            f"missing {sorted(EXPECTED_EXPLICIT_AREA_ONLY - area_only_ids)}"
        )

    reference = registry.loc[registry["year"].eq(reference_year)]
    pixel_reference = reference.loc[
        reference["inference_mode"].astype(str).isin(PIXEL_MODES)
    ]
    area_reference = reference.loc[
        reference["inference_mode"].astype(str).eq("area_only_fallback")
    ]
    unknown = reference.loc[
        ~reference["inference_mode"].astype(str).isin(PIXEL_MODES | {"area_only_fallback"})
    ]
    if len(unknown):
        raise SystemExit(
            "Unknown Stage-7 inference modes: "
            f"{sorted(unknown['inference_mode'].astype(str).unique().tolist())}"
        )

    missing_pixel: dict[str, list[int]] = {}
    fabricated_area_only: dict[str, list[int]] = {}
    for column in sorted(TERMINAL_COLUMNS):
        missing_mask = ~pixel_reference[column].map(_path_exists)
        if missing_mask.any():
            missing_pixel[column] = sorted(
                pixel_reference.loc[missing_mask, "Id"].astype(int).unique().tolist()
            )
        fabricated_mask = area_reference[column].map(_path_exists)
        if fabricated_mask.any():
            fabricated_area_only[column] = sorted(
                area_reference.loc[fabricated_mask, "Id"].astype(int).unique().tolist()
            )
    if missing_pixel:
        raise SystemExit(f"Pixel-route Stage-7 diagnostics missing: {missing_pixel}")
    if fabricated_area_only:
        raise SystemExit(f"Area-only Stage-7 rasters were fabricated: {fabricated_area_only}")

    quality_path = root / "04_unit_quality" / "unit_quality_registry.csv"
    supervision_path = root / "03_supervision" / "supervision_registry.csv"
    quality = pd.read_csv(quality_path)
    supervision = pd.read_csv(supervision_path)
    for name, frame in [("quality", quality), ("supervision", supervision)]:
        missing_ids = EXPECTED_EXPLICIT_AREA_ONLY - set(
            pd.to_numeric(frame["Id"], errors="coerce").dropna().astype(int).tolist()
        )
        if missing_ids:
            raise SystemExit(f"{name} registry misses units {sorted(missing_ids)}")

    preview = {
        "status": "passed",
        "records": int(len(registry)),
        "units": int(registry["Id"].nunique()),
        "pixel_units": int(registry.loc[~area_only, "Id"].nunique()),
        "area_only_units": sorted(area_only_ids),
        "explicit_area_only_units": sorted(EXPECTED_EXPLICIT_AREA_ONLY),
        "expected_stage8_records": expected_records,
        "downstream_policy": (
            "area-only units retain 15 annual area records and uncertainty, "
            "but no pixel masks or 2050 candidate rasters are fabricated"
        ),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
