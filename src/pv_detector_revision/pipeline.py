from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import shutil

import numpy as np
import pandas as pd
import yaml

from .detector import (
    build_state_products,
    calibrate_thresholds,
    classify_annual_state,
    combine_evidence,
    refine_reference_year,
    temporal_regularize,
)
from .raster_io import (
    PATH_COLUMNS,
    GridIdentity,
    atomic_write_csv,
    choose_template_path,
    existing_path,
    find_registry,
    load_reference_anchor,
    pixel_area_km2,
    read_bool,
    read_float,
    require_same_grid,
    same_grid,
    write_raster_atomic,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid configuration: {config_path}")
    return data


def _scale_evidence(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = array.astype(np.float32, copy=True)
    values = out[valid & np.isfinite(out)]
    if values.size == 0:
        return out
    q99 = float(np.nanquantile(values, 0.99))
    q01 = float(np.nanquantile(values, 0.01))
    if q99 <= 1.5 and q01 >= -0.2:
        return np.clip(out, 0.0, 1.0)
    if q99 <= 100.0 and q01 >= 0.0:
        return np.clip(out / 100.0, 0.0, 1.0)
    if q99 <= 10000.0 and q01 >= 0.0:
        return np.clip(out / 10000.0, 0.0, 1.0)
    lo, hi = np.nanquantile(values, [0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(out, dtype=np.float32)
    return np.clip((out - float(lo)) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def _read_aligned_float(
    row: pd.Series,
    candidates: tuple[str, ...],
    template_profile: Mapping[str, object],
    label: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    path = existing_path(row, candidates)
    if path is None:
        return None, None, None
    array, profile, valid = read_float(path)
    require_same_grid(template_profile, profile, label)
    return _scale_evidence(array, valid), valid, str(path)


def _read_aligned_bool(
    row: pd.Series,
    candidates: tuple[str, ...],
    template_profile: Mapping[str, object],
    label: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    path = existing_path(row, candidates)
    if path is None:
        return None, None, None
    array, profile, valid = read_bool(path)
    require_same_grid(template_profile, profile, label)
    return array, valid, str(path)


def _unit_rows(frame: pd.DataFrame, unit_id: int, years: list[int]) -> pd.DataFrame:
    local = frame.copy()
    local["Id"] = pd.to_numeric(local["Id"], errors="coerce")
    local["year"] = pd.to_numeric(local["year"], errors="coerce")
    local = local.loc[local["Id"].eq(unit_id) & local["year"].isin(years)].copy()
    local = local.sort_values("year")
    duplicates = local["year"].duplicated(keep=False)
    if duplicates.any():
        duplicate_years = sorted(local.loc[duplicates, "year"].astype(int).unique().tolist())
        raise ValueError(f"Unit {unit_id} has duplicate annual registry rows: {duplicate_years}")
    found = sorted(local["year"].dropna().astype(int).tolist())
    if found != years:
        missing = sorted(set(years) - set(found))
        raise ValueError(f"Unit {unit_id} is missing Stage-7 years: {missing}")
    return local.set_index("year", drop=False)


def _area_only_reason(rows: pd.DataFrame) -> str | None:
    for column in ("inference_mode", "estimation_route", "skip_reason"):
        if column not in rows.columns:
            continue
        values = rows[column].fillna("").astype(str)
        bad = values.str.contains("area_only|anchor_analog_area|predeclared_area_only", case=False, regex=True)
        if bad.any():
            return f"{column}={values.loc[bad].iloc[0]}"
    return None


def preflight(
    output_root: str | Path,
    unit_ids: Iterable[int],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    root = Path(output_root).resolve()
    inputs = config["inputs"]
    years = [int(value) for value in config["study"]["years"]]
    stage7_path = find_registry(root, str(inputs["stage7_registry"]))
    stage8_path = find_registry(root, str(inputs["stage8_registry"]))
    stage7 = pd.read_csv(stage7_path)
    stage8 = pd.read_csv(stage8_path)
    records: list[dict[str, Any]] = []

    for unit_id in sorted({int(value) for value in unit_ids}):
        record: dict[str, Any] = {"Id": unit_id, "status": "failed"}
        try:
            rows7 = _unit_rows(stage7, unit_id, years)
            rows8 = _unit_rows(stage8, unit_id, years)
            reason = _area_only_reason(rows7)
            if reason and bool(config["validation"].get("require_non_area_only_stage7", True)):
                raise RuntimeError(
                    f"Unit {unit_id} has no canonical annual pixel route ({reason}). "
                    "Use the original full v24.22 output, not the four-unit v1.1.0 area-only test output."
                )
            row2024 = rows7.loc[int(config["study"]["reference_year"])]
            template_path = choose_template_path(row2024)
            _, template_profile, _ = read_float(template_path)
            grid = GridIdentity.from_profile(template_profile)
            anchor, anchor_source = load_reference_anchor(
                row2024,
                template_profile,
                Path(str(inputs["reference_shp"])),
            )
            raster_years = 0
            for year in years:
                row = rows7.loc[year]
                annual_path = choose_template_path(row)
                _, profile, _ = read_float(annual_path)
                require_same_grid(template_profile, profile, f"Unit {unit_id}, year {year}")
                raster_years += 1
            record.update(
                status="passed",
                annual_registry_rows=len(rows7),
                annual_pixel_raster_years=raster_years,
                stage8_rows=len(rows8),
                template_path=str(template_path),
                grid_crs=grid.crs,
                grid_width=grid.width,
                grid_height=grid.height,
                grid_transform="|".join(f"{value:.12g}" for value in grid.transform),
                anchor_source=anchor_source,
                anchor_pixels=int(anchor.sum()),
            )
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    result = pd.DataFrame(records)
    audit_dir = root / str(config["outputs"]["audit_subdir"])
    atomic_write_csv(result, audit_dir / "preflight_units.csv")
    if not result["status"].eq("passed").all():
        failures = result.loc[~result["status"].eq("passed"), ["Id", "reason"]]
        raise RuntimeError("Detector revision preflight failed:\n" + failures.to_string(index=False))
    return result


def _ensure_registry_columns(registry: pd.DataFrame) -> pd.DataFrame:
    result = registry.copy()
    columns: dict[str, Any] = {
        "active_mask_path": "",
        "observed_active_mask_path": "",
        "cumulative_mask_path": "",
        "cumulative_installed_mask_path": "",
        "newly_installed_mask_path": "",
        "inactive_or_retired_mask_path": "",
        "annual_state_mask_path": "",
        "first_installed_year_path": "",
        "observed_active_area_km2": np.nan,
        "cumulative_installed_area_km2": np.nan,
        "newly_installed_area_km2": np.nan,
        "annual_installed_area_increment_km2": np.nan,
        "inactive_or_retired_area_km2": np.nan,
        "pv_area_km2": np.nan,
        "estimation_mode": "",
        "detector_revision": "",
        "detector_revision_score_path": "",
    }
    for column, default in columns.items():
        if column not in result.columns:
            result[column] = default
    return result


def apply_detector_revision(
    output_root: str | Path,
    unit_ids: Iterable[int],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_root).resolve()
    unit_ids = sorted({int(value) for value in unit_ids})
    preflight(root, unit_ids, config)

    inputs = config["inputs"]
    years = [int(value) for value in config["study"]["years"]]
    reference_year = int(config["study"]["reference_year"])
    stage7_path = find_registry(root, str(inputs["stage7_registry"]))
    stage8_path = find_registry(root, str(inputs["stage8_registry"]))
    stage7 = pd.read_csv(stage7_path)
    stage8_original = pd.read_csv(stage8_path)
    stage8 = _ensure_registry_columns(stage8_original)
    stage8["Id"] = pd.to_numeric(stage8["Id"], errors="coerce")
    stage8["year"] = pd.to_numeric(stage8["year"], errors="coerce")

    if bool(config["outputs"].get("backup_canonical_stage8_registry", True)):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = stage8_path.with_name(stage8_path.stem + f".pre_detector_revision_v111_{stamp}.csv")
        shutil.copy2(stage8_path, backup_path)

    revision_root = root / str(config["outputs"]["revision_subdir"])
    audit_root = root / str(config["outputs"]["audit_subdir"])
    audit_root.mkdir(parents=True, exist_ok=True)
    annual_audits: list[dict[str, Any]] = []
    unit_audits: list[dict[str, Any]] = []

    for unit_id in unit_ids:
        rows7 = _unit_rows(stage7, unit_id, years)
        rows8 = _unit_rows(stage8_original, unit_id, years)
        row2024 = rows7.loc[reference_year]
        template_path = choose_template_path(row2024)
        _, template_profile, template_valid = read_float(template_path)
        anchor, anchor_source = load_reference_anchor(
            row2024,
            template_profile,
            Path(str(inputs["reference_shp"])),
        )
        if not anchor.any():
            raise RuntimeError(f"Unit {unit_id}: 2024 reference anchor is empty")

        score_arrays: list[np.ndarray] = []
        raw_masks: list[np.ndarray] = []
        legacy_masks: list[np.ndarray] = []
        terminal_expansions: list[np.ndarray] = []
        valid_arrays: list[np.ndarray] = []
        evidence_path_records: dict[int, dict[str, str | None]] = {}

        for year in years:
            row7 = rows7.loc[year]
            row8 = rows8.loc[year]
            evidence: dict[str, np.ndarray | None] = {}
            path_record: dict[str, str | None] = {}
            annual_valid = template_valid.copy()

            for name in (
                "calibrated_probability",
                "raw_probability",
                "anchor_similarity",
                "lsma_fraction",
                "candidate_score",
                "multiband_break_support",
            ):
                array, valid, path_value = _read_aligned_float(
                    row7,
                    PATH_COLUMNS[name],
                    template_profile,
                    f"Unit {unit_id}, year {year}, {name}",
                )
                evidence[name] = array
                path_record[name] = path_value
                if valid is not None:
                    annual_valid &= valid

            legacy, legacy_valid, legacy_path = _read_aligned_bool(
                row8,
                PATH_COLUMNS["legacy_active"],
                template_profile,
                f"Unit {unit_id}, year {year}, legacy active mask",
            )
            if legacy is None:
                legacy = np.zeros(anchor.shape, dtype=bool)
            else:
                annual_valid &= legacy_valid
            evidence["legacy_positive_mask"] = legacy.astype(np.float32)
            path_record["legacy_positive_mask"] = legacy_path

            expansion, expansion_valid, expansion_path = _read_aligned_bool(
                row7,
                PATH_COLUMNS["terminal_expansion"],
                template_profile,
                f"Unit {unit_id}, year {year}, terminal expansion",
            )
            if expansion is None:
                expansion = np.zeros(anchor.shape, dtype=bool)
            else:
                annual_valid &= expansion_valid
            path_record["terminal_expansion"] = expansion_path

            score, denominator = combine_evidence(
                evidence,
                annual_valid,
                config["annual_state"]["evidence_weights"],
                float(config["annual_state"]["minimum_available_weight"]),
            )
            score_arrays.append(score)
            valid_arrays.append(annual_valid & np.isfinite(score))
            legacy_masks.append(legacy)
            terminal_expansions.append(expansion)
            raw_masks.append(np.zeros(anchor.shape, dtype=bool))
            path_record["available_weight_mean"] = str(float(np.nanmean(denominator[annual_valid])))
            evidence_path_records[year] = path_record

        score_stack = np.stack(score_arrays)
        valid_stack = np.stack(valid_arrays)
        legacy_stack = np.stack(legacy_masks)
        terminal_stack = np.stack(terminal_expansions)
        reference_index = years.index(reference_year)
        thresholds = calibrate_thresholds(
            score_stack[reference_index],
            anchor,
            valid_stack[reference_index],
            config["annual_state"],
        )

        for index, year in enumerate(years):
            mask, audit = classify_annual_state(
                score_stack[index],
                score_stack[reference_index],
                anchor,
                valid_stack[index],
                thresholds,
                config["annual_state"],
            )
            # Preserve the existing 2025 detector as the terminal positive state.
            # Earlier years use the same annual state logic but are not required
            # to satisfy the 2025 novelty gate.
            if year == int(config["study"]["terminal_year"]):
                mask |= legacy_stack[index] | terminal_stack[index]
            raw_masks[index] = mask
            annual_audits.append(
                {
                    "Id": unit_id,
                    "year": year,
                    **audit,
                    "raw_active_pixels": int(mask.sum()),
                    "valid_pixels": int(valid_stack[index].sum()),
                    **{f"path_{key}": value for key, value in evidence_path_records[year].items()},
                }
            )

        active = temporal_regularize(
            np.stack(raw_masks),
            score_stack,
            anchor,
            years,
            thresholds,
            config["temporal"],
            legacy_positive_stack=legacy_stack,
        )

        idx23 = years.index(2023) if 2023 in years else max(reference_index - 1, 0)
        idx25 = years.index(2025) if 2025 in years else min(reference_index + 1, len(years) - 1)
        refined_2024, boundary_audit = refine_reference_year(
            active[reference_index],
            anchor,
            score_stack[idx23],
            score_stack[reference_index],
            score_stack[idx25],
            valid_stack[reference_index],
            config["reference_2024"],
        )
        active[reference_index] = refined_2024

        products = build_state_products(active, years)
        area_per_pixel = pixel_area_km2(template_profile)
        unit_dir = revision_root / f"unit_{unit_id:03d}"
        score_dir = unit_dir / "annual_state_score_30m"
        active_dir = unit_dir / "observed_active_pv_masks_30m"
        cumulative_dir = unit_dir / "cumulative_installed_pv_masks_30m"
        new_dir = unit_dir / "newly_installed_pv_masks_30m"
        inactive_dir = unit_dir / "inactive_or_retired_pv_masks_30m"
        state_dir = unit_dir / "annual_state_masks_30m"
        first_path = unit_dir / "first_installed_year_30m" / f"unit_{unit_id:03d}_first_installed_year_v111.tif"
        write_raster_atomic(first_path, products["first_year"], template_profile, dtype="uint16", nodata=0)

        for index, year in enumerate(years):
            score_path = score_dir / f"unit_{unit_id:03d}_{year}_annual_state_score_v111.tif"
            active_path = active_dir / f"unit_{unit_id:03d}_{year}_observed_active_v111.tif"
            cumulative_path = cumulative_dir / f"unit_{unit_id:03d}_{year}_cumulative_v111.tif"
            new_path = new_dir / f"unit_{unit_id:03d}_{year}_newly_installed_v111.tif"
            inactive_path = inactive_dir / f"unit_{unit_id:03d}_{year}_inactive_v111.tif"
            state_path = state_dir / f"unit_{unit_id:03d}_{year}_annual_state_v111.tif"
            write_raster_atomic(score_path, np.nan_to_num(score_stack[index], nan=-9999.0), template_profile, dtype="float32", nodata=-9999.0)
            write_raster_atomic(active_path, products["active"][index].astype(np.uint8), template_profile, dtype="uint8", nodata=0)
            write_raster_atomic(cumulative_path, products["cumulative"][index].astype(np.uint8), template_profile, dtype="uint8", nodata=0)
            write_raster_atomic(new_path, products["newly"][index].astype(np.uint8), template_profile, dtype="uint8", nodata=0)
            write_raster_atomic(inactive_path, products["inactive"][index].astype(np.uint8), template_profile, dtype="uint8", nodata=0)
            write_raster_atomic(state_path, products["state"][index].astype(np.uint8), template_profile, dtype="uint8", nodata=0)

            row_mask = stage8["Id"].eq(unit_id) & stage8["year"].eq(year)
            if int(row_mask.sum()) != 1:
                raise RuntimeError(f"Unit {unit_id}, year {year}: Stage-8 row count is {int(row_mask.sum())}, expected 1")
            active_area = float(products["active"][index].sum() * area_per_pixel)
            cumulative_area = float(products["cumulative"][index].sum() * area_per_pixel)
            new_area = float(products["newly"][index].sum() * area_per_pixel)
            inactive_area = float(products["inactive"][index].sum() * area_per_pixel)
            updates = {
                "active_mask_path": str(active_path),
                "observed_active_mask_path": str(active_path),
                "cumulative_mask_path": str(cumulative_path),
                "cumulative_installed_mask_path": str(cumulative_path),
                "newly_installed_mask_path": str(new_path),
                "inactive_or_retired_mask_path": str(inactive_path),
                "annual_state_mask_path": str(state_path),
                "first_installed_year_path": str(first_path),
                "observed_active_area_km2": active_area,
                "cumulative_installed_area_km2": cumulative_area,
                "newly_installed_area_km2": new_area,
                "annual_installed_area_increment_km2": new_area,
                "inactive_or_retired_area_km2": inactive_area,
                "pv_area_km2": active_area,
                "estimation_mode": "pixel_annual_state_detector_v111",
                "detector_revision": "v1.1.1_annual_state_same_logic_all_years",
                "detector_revision_score_path": str(score_path),
            }
            for column, value in updates.items():
                stage8.loc[row_mask, column] = value

            annual_record = next(
                row for row in reversed(annual_audits)
                if row["Id"] == unit_id and row["year"] == year
            )
            annual_record.update(
                final_active_pixels=int(products["active"][index].sum()),
                cumulative_pixels=int(products["cumulative"][index].sum()),
                new_pixels=int(products["newly"][index].sum()),
                active_area_km2=active_area,
                cumulative_area_km2=cumulative_area,
            )

        # Reopen one product to prove that no CRS, transform, extent or tile shift occurred.
        _, output_profile, _ = read_float(
            revision_root / f"unit_{unit_id:03d}" / "observed_active_pv_masks_30m" /
            f"unit_{unit_id:03d}_{reference_year}_observed_active_v111.tif"
        )
        if not same_grid(template_profile, output_profile):
            raise RuntimeError(f"Unit {unit_id}: detector output grid identity changed")
        retention = float(boundary_audit["anchor_retention_fraction"])
        minimum_retention = float(config["validation"]["require_2024_anchor_retention_fraction"])
        if retention + 1e-12 < minimum_retention:
            raise RuntimeError(
                f"Unit {unit_id}: 2024 anchor retention {retention:.6f} is below {minimum_retention:.6f}"
            )

        unit_audits.append(
            {
                "Id": unit_id,
                "status": "passed",
                "template_path": str(template_path),
                "grid": json.dumps(asdict(GridIdentity.from_profile(template_profile)), ensure_ascii=False),
                "anchor_source": anchor_source,
                "strong_threshold": thresholds.strong,
                "weak_threshold": thresholds.weak,
                "active_2011_pixels": int(products["active"][0].sum()),
                "active_2023_pixels": int(products["active"][years.index(2023)].sum()),
                "active_2024_pixels": int(products["active"][reference_index].sum()),
                "active_2025_pixels": int(products["active"][years.index(2025)].sum()),
                **boundary_audit,
                "grid_identity_check": "passed",
            }
        )

    annual_df = pd.DataFrame(annual_audits).sort_values(["Id", "year"])
    unit_df = pd.DataFrame(unit_audits).sort_values("Id")
    atomic_write_csv(annual_df, audit_root / "annual_detection_audit_v111.csv")
    atomic_write_csv(unit_df, audit_root / "unit_detection_audit_v111.csv")
    revised_registry_path = audit_root / "temporal_mask_registry_detector_revision_v111.csv"
    atomic_write_csv(stage8, revised_registry_path)
    if bool(config["outputs"].get("overwrite_canonical_stage8_registry", True)):
        atomic_write_csv(stage8, stage8_path)

    status = {
        "release": config["release"]["name"],
        "version": config["release"]["version"],
        "status": "passed",
        "units": unit_ids,
        "years": years,
        "canonical_stage8_registry": str(stage8_path),
        "revised_registry_snapshot": str(revised_registry_path),
        "algorithm_contract": (
            "The original v24.22 grids, unit extents, annual Stage-7 evidence, "
            "Stage-8 registry schema and downstream workflow are retained. "
            "Only annual PV state classification and conservative temporal cleanup are revised."
        ),
    }
    (audit_root / "detector_revision_status_v111.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return annual_df, unit_df
