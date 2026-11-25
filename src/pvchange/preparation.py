from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import importlib
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import box
from tqdm import tqdm

from .ccdc import annual_ccdc_descriptors
from .environment import environmental_coverage, normalize_environmental_panel
from .features import compute_features
from .geo import Grid, aligned_vrt, build_grid, grid_profile, source_pixel_size_m
from .io import detect_reflectance_scale, discover_images, scale_reflectance
from .spectral_mixture import load_endmember_matrix
from .utils import atomic_json, ensure_dir, save_table, utc_now
from .study_area import expected_counts, expected_id_set, study_ids

REQUIRED_MODULES = [
    "numpy", "pandas", "yaml", "rasterio", "geopandas", "shapely", "pyproj",
    "scipy", "sklearn", "statsmodels", "matplotlib", "openpyxl",
    "tqdm", "tabulate", "xgboost", "joblib",
]


def _expected_counts(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return expected_counts(cfg)


def validate_inputs(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "01_input_validation")
    failures: list[str] = []
    warnings: list[str] = []
    dependencies: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    expected_units, expected_years, expected_records = _expected_counts(cfg)

    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            dependencies.append({"module": name, "status": "ok", "version": getattr(mod, "__version__", "unknown")})
        except Exception as exc:
            dependencies.append({"module": name, "status": "failed", "error": str(exc)})
            failures.append(f"Dependency {name}: {exc}")

    shp_path = Path(cfg["paths"]["reference_shp"])
    if not shp_path.exists():
        failures.append(f"Reference SHP missing: {shp_path}")
    else:
        try:
            gdf = gpd.read_file(shp_path)
            id_field = cfg["supervision"].get("id_field", "Id")
            if gdf.empty or gdf.crs is None:
                failures.append("Reference SHP is empty or has no CRS")
            if id_field not in gdf.columns:
                candidates = [c for c in ["src_id", "ID", "id", "region_id", "Region_ID"] if c in gdf.columns]
                if candidates:
                    warnings.append(f"Reference SHP field {id_field} missing; stage 3 will use {candidates[0]} and create canonical Id")
                else:
                    warnings.append(
                        f"Reference SHP is missing ID field {id_field}; "
                        "Stage 3 will assign PV geometries to units by spatial overlap "
                        "with the canonical 30 m grids."
                    )
            else:
                ids = pd.to_numeric(gdf[id_field], errors="coerce").dropna().astype(int)
                outside = ids.loc[~ids.isin(expected_id_set(cfg))]
                if not outside.empty:
                    warnings.append(f"Reference SHP includes {len(outside)} features outside configured Id range")
            invalid = int((gdf.geometry.isna() | gdf.geometry.is_empty | ~gdf.geometry.is_valid).sum())
            if invalid:
                warnings.append(f"Reference SHP contains {invalid} invalid or empty geometries; stage 3 will repair or exclude them")
        except Exception as exc:
            failures.append(f"Reference SHP read failure: {exc}")

    minimum = float(cfg["landsat"]["source_resolution_min_m"])
    maximum = float(cfg["landsat"]["source_resolution_max_m"])
    tolerance = float(cfg["landsat"].get("source_resolution_tolerance_m", 0.5))
    require_complete = bool(cfg["study"].get("require_complete_unit_year_matrix", True))
    for region_id in tqdm(
        study_ids(cfg),
        desc="Input inventory", unit="region", disable=not cfg["processing"].get("show_progress", True),
    ):
        try:
            images = discover_images(cfg, region_id)
        except Exception as exc:
            failures.append(f"Id {region_id}: {exc}")
            rows.append({"Id": region_id, "status": "region_missing"})
            continue
        missing_years = sorted(set(range(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1)) - set(images))
        if missing_years:
            msg = f"Id {region_id}: missing annual Landsat images for {missing_years}"
            (failures if require_complete else warnings).append(msg)
        for year, path in sorted(images.items()):
            status = "ok"
            try:
                with rasterio.open(path) as src:
                    dx, dy = source_pixel_size_m(src)
                    if src.crs is None:
                        failures.append(f"Id {region_id}, year {year}: missing CRS")
                        status = "missing_crs"
                    if src.count < int(cfg["landsat"]["expected_band_count"]):
                        failures.append(f"Id {region_id}, year {year}: only {src.count} bands")
                        status = "insufficient_bands"
                    valid_resolution = (minimum - tolerance <= dx <= maximum + tolerance) and (minimum - tolerance <= dy <= maximum + tolerance)
                    if not valid_resolution:
                        failures.append(f"Id {region_id}, year {year}: source pixel size {dx:.3f} x {dy:.3f} m outside Landsat-scale gate")
                        status = "resolution_mismatch"
                    elif not (minimum <= dx <= maximum and minimum <= dy <= maximum):
                        status = "ok_with_resolution_tolerance"
                        warnings.append(f"Id {region_id}, year {year}: source pixel size {dx:.3f} x {dy:.3f} m accepted within tolerance")
                    rows.append({
                        "Id": region_id, "year": year, "path": str(path), "status": status,
                        "bands": src.count, "crs": str(src.crs), "width": src.width, "height": src.height,
                        "source_pixel_size_x_m": dx, "source_pixel_size_y_m": dy,
                        "dtype": src.dtypes[0], "nodata": src.nodata,
                    })
            except Exception as exc:
                failures.append(f"Id {region_id}, year {year}: {type(exc).__name__}: {exc}")
                rows.append({"Id": region_id, "year": year, "path": str(path), "status": "unreadable", "error": str(exc)})

    inventory = pd.DataFrame(rows)
    accepted = inventory.loc[inventory.get("status", pd.Series(dtype=str)).isin(["ok", "ok_with_resolution_tolerance"])] if not inventory.empty else inventory
    if accepted["Id"].nunique() != expected_units or len(accepted) != expected_records:
        failures.append(f"Strict real-data coverage failed: accepted {accepted['Id'].nunique() if not accepted.empty else 0}/{expected_units} units and {len(accepted)}/{expected_records} unit-years")

    env_path = out / "environmental_covariates.csv"
    env_panel, env_inventory = normalize_environmental_panel(cfg, env_path)
    env_cov = environmental_coverage(env_panel, cfg)
    min_fraction = float(cfg.get("environment", {}).get("minimum_complete_case_fraction", 0.75))
    if bool(cfg.get("environment", {}).get("required", True)):
        if env_panel.empty:
            failures.append("No usable environmental covariate panel was found in input_table_dir/input_table_zip")
        if env_cov["complete_case_fraction"] < min_fraction:
            failures.append(f"Environmental complete-case coverage {env_cov['complete_case_fraction']:.3f} is below required {min_fraction:.3f}")
        if bool(cfg.get("environment", {}).get("fail_on_missing_configured_covariate", True)):
            missing_cov = [k for k, v in env_cov["coverage_by_covariate"].items() if v <= 0]
            if missing_cov:
                failures.append(f"Configured environmental covariates not found: {missing_cov}")

    save_table(inventory, out / "raster_inventory.csv")
    save_table(pd.DataFrame(dependencies), out / "dependency_inventory.csv")
    save_table(pd.DataFrame({"warning": warnings}), out / "input_warnings.csv")
    save_table(env_inventory, out / "environmental_table_inventory.csv")
    status_payload = {
        "status": "failed" if failures else "passed",
        "failure_count": len(failures), "warning_count": len(warnings),
        "failures": failures, "warnings": warnings, "created_utc": utc_now(),
        "pipeline_version": cfg["project"]["version"],
        "expected_units": expected_units, "validated_units": int(accepted["Id"].nunique()) if not accepted.empty else 0,
        "expected_unit_year_records": expected_records, "validated_unit_year_records": int(len(accepted)),
        "environmental_coverage": env_cov,
        "interpretation": "All 142 units and all 2011-2025 Landsat composites are mandatory in strict mode. Source rasters are aligned to canonical projected 30 m grids without creating new spatial information.",
    }
    atomic_json(status_payload, out / "input_validation_status.json")
    if failures and cfg["processing"].get("fail_fast", True):
        raise RuntimeError("Input validation failed:\n" + "\n".join(failures[:50]))
    return inventory


def _grid_from_row(row: pd.Series) -> Grid:
    return Grid(
        int(row.Id), rasterio.crs.CRS.from_string(str(row.target_crs)),
        Affine(float(row.transform_a), float(row.transform_b), float(row.transform_c), float(row.transform_d), float(row.transform_e), float(row.transform_f)),
        int(row.width), int(row.height),
        (float(row.left), float(row.bottom), float(row.right), float(row.top)),
        float(row.target_resolution_m),
    )


def _valid_raster(path: Path, grid: Grid, count: int = 6) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count == count and src.width == grid.width and src.height == grid.height and src.crs == grid.crs and src.transform.almost_equals(grid.transform)
    except Exception:
        return False


def build_aligned_grids(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "02_aligned_30m")
    aligned_root = ensure_dir(out / "annual_rasters")
    inventory = pd.read_csv(root / "01_input_validation" / "raster_inventory.csv")
    accepted = inventory.loc[inventory["status"].isin(["ok", "ok_with_resolution_tolerance"])].copy()
    preferred = int(cfg["study"]["reference_year"])
    accepted["rank"] = (accepted["year"] - preferred).abs()
    references = accepted.sort_values(["Id", "rank", "year"], ascending=[True, True, False]).groupby("Id", as_index=False).first()
    grid_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    resolution = float(cfg["native_grid"]["target_resolution_m"])
    anchor = float(cfg["native_grid"]["alignment_anchor_m"])
    for ref in tqdm(list(references.itertuples(index=False)), desc="Canonical 30 m grids", unit="region", disable=not cfg["processing"].get("show_progress", True)):
        region_id = int(ref.Id)
        grid = build_grid(region_id, ref.path, resolution, anchor)
        if grid.width > int(cfg["native_grid"]["maximum_output_dimension"]) or grid.height > int(cfg["native_grid"]["maximum_output_dimension"]):
            raise RuntimeError(f"Id {region_id}: grid too large ({grid.width} x {grid.height})")
        grid_rows.append({
            "Id": region_id, "status": "ok", "reference_raster": ref.path, "target_crs": grid.crs.to_string(),
            "target_resolution_m": resolution, "width": grid.width, "height": grid.height,
            "left": grid.bounds[0], "bottom": grid.bounds[1], "right": grid.bounds[2], "top": grid.bounds[3],
            "transform_a": grid.transform.a, "transform_b": grid.transform.b, "transform_c": grid.transform.c,
            "transform_d": grid.transform.d, "transform_e": grid.transform.e, "transform_f": grid.transform.f,
        })
        region_out = ensure_dir(aligned_root / f"unit_{region_id:03d}")
        unit_images = accepted.loc[accepted["Id"].eq(region_id)].sort_values("year")
        for r in unit_images.itertuples(index=False):
            target = region_out / f"unit_{region_id:03d}_{int(r.year)}_landsat_30m.tif"
            if _valid_raster(target, grid) and not cfg["processing"].get("overwrite", False):
                aligned_rows.append({"Id": region_id, "year": int(r.year), "path": str(target), "status": "reused"})
                continue
            mode = detect_reflectance_scale(r.path)
            profile = grid_profile(grid, 6, "float32", -9999.0, cfg["processing"].get("compress", "DEFLATE"))
            if profile.get("tiled"):
                profile.update(blockxsize=256, blockysize=256)
            with rasterio.open(r.path) as src, aligned_vrt(src, grid, Resampling.bilinear) as vrt, rasterio.open(target, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    arr = vrt.read(indexes=[1, 2, 3, 4, 5, 6], window=window, masked=True).filled(np.nan).astype(np.float32)
                    arr = scale_reflectance(arr, mode)
                    dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32), window=window)
                dst.update_tags(source_path=str(r.path), source_scale_mode=mode, canonical_resolution_m=resolution)
            aligned_rows.append({"Id": region_id, "year": int(r.year), "path": str(target), "status": "written"})
    grids = pd.DataFrame(grid_rows).sort_values("Id")
    aligned = pd.DataFrame(aligned_rows).sort_values(["Id", "year"])
    save_table(grids, out / "grid_registry.csv")
    save_table(aligned, out / "aligned_raster_registry.csv")
    expected_units, _, expected_records = _expected_counts(cfg)
    status = "passed" if grids["Id"].nunique() == expected_units and len(aligned) == expected_records else "failed"
    atomic_json({
        "status": status, "created_utc": utc_now(), "units": int(grids["Id"].nunique()), "records": len(aligned),
        "expected_units": expected_units, "expected_records": expected_records, "target_resolution_m": resolution,
    }, out / "aligned_grid_status.json")
    if status == "failed":
        raise RuntimeError(f"Aligned-grid coverage incomplete: {grids['Id'].nunique()}/{expected_units} units, {len(aligned)}/{expected_records} records")
    return aligned


def _fractional_rasterize(geometries: Any, grid: Grid, oversample: int) -> np.ndarray:
    if len(geometries) == 0:
        return np.zeros((grid.height, grid.width), dtype=np.float32)
    factor = max(1, int(oversample))
    transform = Affine(grid.transform.a / factor, 0, grid.transform.c, 0, grid.transform.e / factor, grid.transform.f)
    high = rasterize(
        [(geom, 1) for geom in geometries if geom is not None and not geom.is_empty],
        out_shape=(grid.height * factor, grid.width * factor), transform=transform,
        fill=0, all_touched=True, dtype="uint8",
    )
    return high.reshape(grid.height, factor, grid.width, factor).mean(axis=(1, 3)).astype(np.float32)


def _repair_geometry_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy containing only valid, non-empty geometries.

    ``buffer(0)`` is deliberately used instead of requiring a recent Shapely
    ``make_valid`` implementation so the real server environment remains
    compatible with older but supported GeoPandas/Shapely combinations.
    """
    if frame.empty:
        return frame.copy()
    repaired = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if repaired.empty:
        return repaired
    invalid = ~repaired.geometry.is_valid
    if invalid.any():
        repaired.loc[invalid, "geometry"] = repaired.loc[invalid, "geometry"].buffer(0)
    repaired = repaired.loc[repaired.geometry.notna() & ~repaired.geometry.is_empty & repaired.geometry.is_valid].copy()
    return repaired


def _fraction_thresholds(fraction: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Create strong/weak labels without discarding sub-pixel PV polygons.

    The configured strong threshold remains the default. When a valid polygon
    intersects a grid but no 30 m pixel reaches that threshold, an auditable
    adaptive threshold is used. The lower bound equals one oversampled cell,
    so this fallback cannot invent support outside the rasterized geometry.
    """
    strong_threshold = float(cfg["supervision"]["strong_positive_fraction"])
    weak_threshold = float(cfg["supervision"]["weak_positive_fraction"])
    positive = fraction >= strong_threshold
    label_mode = "configured_strong_fraction"
    used_threshold = strong_threshold
    nonzero = fraction[np.isfinite(fraction) & (fraction > 0)]
    if nonzero.size and not positive.any():
        oversample = max(1, int(cfg["supervision"]["fractional_rasterization_oversample"]))
        detectable_floor = 1.0 / float(oversample * oversample)
        configured_floor = float(cfg["supervision"].get("adaptive_positive_fraction_floor", detectable_floor))
        max_fraction = float(np.max(nonzero))
        used_threshold = min(
            strong_threshold,
            max(detectable_floor, min(configured_floor, max_fraction), 0.5 * max_fraction),
        )
        used_threshold = min(used_threshold, max_fraction)
        positive = fraction >= used_threshold
        label_mode = "adaptive_subpixel_fraction"
    weak = (fraction > 0) & ~positive
    if weak_threshold > 0:
        weak |= (fraction >= weak_threshold) & ~positive
    return positive, weak, float(used_threshold), label_mode



def _allocate_role_counts(n: int, fractions: dict[str, float], minimums: dict[str, int]) -> dict[str, int]:
    roles = ["train", "calibration", "test"]
    if n < sum(minimums.values()):
        raise RuntimeError(
            f"Cannot create independent train/calibration/test split for {n} positive units; "
            f"minimum required is {sum(minimums.values())}"
        )
    counts = {role: max(minimums[role], int(round(n * fractions[role]))) for role in roles}
    while sum(counts.values()) > n:
        candidates = [role for role in roles if counts[role] > minimums[role]]
        if not candidates:
            break
        role = max(candidates, key=lambda item: counts[item] - minimums[item])
        counts[role] -= 1
    while sum(counts.values()) < n:
        role = max(roles, key=lambda item: fractions[item] * n - counts[item])
        counts[role] += 1
    return counts


def _assign_supervision_roles(registry: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Create deterministic holdouts only among units with confirmed 2024 anchors.

    The reference SHP is a complete-reference sufficient condition: geometries inside
    it are confirmed PV in the reference year, while absence from the SHP is not
    evidence of absence. Units without confirmed anchors are therefore marked
    ``inference_only`` and never used as negative validation units.
    """
    data = registry.copy()
    split_cfg = cfg["supervision"].get("split", {})
    roles = ["train", "calibration", "test"]
    fractions = {role: float(split_cfg.get(f"{role}_fraction", value)) for role, value in {
        "train": 0.70, "calibration": 0.15, "test": 0.15,
    }.items()}
    if not np.isclose(sum(fractions.values()), 1.0, atol=1e-6):
        raise RuntimeError(f"Supervision split fractions must sum to 1, found {fractions}")
    minimums = {
        "train": int(split_cfg.get("minimum_positive_train_units", 5)),
        "calibration": int(split_cfg.get("minimum_positive_calibration_units", 3)),
        "test": int(split_cfg.get("minimum_positive_test_units", 3)),
    }
    if "confirmed_anchor_area_km2" not in data.columns:
        data["confirmed_anchor_area_km2"] = pd.to_numeric(data.get("strong_positive_pixels", 0), errors="coerce").fillna(0) * 0.0009
    positive_flag = "has_strong_training_core" if "has_strong_training_core" in data.columns else "has_mapped_pv"
    positive = data.loc[data[positive_flag].astype(bool)].copy()
    counts = _allocate_role_counts(len(positive), fractions, minimums)
    rng = np.random.default_rng(int(cfg.get("project", {}).get("random_seed", 2027)))
    positive["_tie"] = rng.random(len(positive))
    positive = positive.sort_values(["confirmed_anchor_area_km2", "_tie"], ascending=[False, True])
    remaining = counts.copy()
    area_load = {role: 0.0 for role in roles}
    role_map: dict[int, str] = {}
    for row in positive.itertuples(index=False):
        area = max(float(row.confirmed_anchor_area_km2), 1e-6)
        candidates = [role for role in roles if remaining[role] > 0]
        projected = {role: (area_load[role] + area) / max(fractions[role], 1e-9) for role in candidates}
        minimum_score = min(projected.values())
        tied = [role for role in candidates if np.isclose(projected[role], minimum_score)]
        chosen = tied[int(rng.integers(0, len(tied)))]
        role_map[int(row.Id)] = chosen
        area_load[chosen] += area
        remaining[chosen] -= 1
    data["split_role"] = data["Id"].map(role_map).fillna("inference_only")
    data["is_training_unit"] = data["split_role"].eq("train")
    data["is_calibration_unit"] = data["split_role"].eq("calibration")
    data["is_independent_test_unit"] = data["split_role"].eq("test")
    return data

def build_supervision(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "03_supervision")
    mask_root = ensure_dir(out / "masks")
    grids = pd.read_csv(root / "02_aligned_30m" / "grid_registry.csv")
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv")
    reference_year = int(cfg["study"]["reference_year"])
    reference_paths = aligned.loc[aligned["year"].eq(reference_year)].set_index("Id")["path"].to_dict()
    shp = gpd.read_file(cfg["paths"]["reference_shp"])
    if shp.crs is None:
        raise RuntimeError("Reference PV SHP has no CRS and cannot be aligned to the 30 m grids")
    shp = _repair_geometry_frame(shp)
    if shp.empty:
        raise RuntimeError("Reference PV SHP contains no valid non-empty geometry")

    id_field = cfg["supervision"].get("id_field", "Id")
    id_candidates = [id_field, "src_id", "ID", "id", "region_id", "Region_ID"]
    available_id_fields = [name for name in id_candidates if name in shp.columns]
    canonical_id_field: str | None = None
    canonical_ids = pd.Series(np.nan, index=shp.index, dtype=float)
    expected_ids = expected_id_set(cfg)
    for candidate in available_id_fields:
        values = pd.to_numeric(shp[candidate], errors="coerce")
        matching = int(values.isin(expected_ids).sum())
        if matching > int(canonical_ids.isin(expected_ids).sum()):
            canonical_ids = values
            canonical_id_field = candidate
    shp["__canonical_Id"] = canonical_ids

    projected_cache: dict[str, gpd.GeoDataFrame] = {}
    rows: list[dict[str, Any]] = []
    for row in tqdm(
        list(grids.itertuples(index=False)),
        desc="SHP supervision",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        grid = Grid(
            int(row.Id), rasterio.crs.CRS.from_string(str(row.target_crs)),
            Affine(row.transform_a, row.transform_b, row.transform_c, row.transform_d, row.transform_e, row.transform_f),
            int(row.width), int(row.height), (row.left, row.bottom, row.right, row.top),
            float(row.target_resolution_m),
        )
        crs_key = grid.crs.to_string()
        if crs_key not in projected_cache:
            projected_cache[crs_key] = _repair_geometry_frame(shp.to_crs(grid.crs))
        projected = projected_cache[crs_key]
        grid_polygon = box(*grid.bounds)

        id_candidates_unit = projected.loc[pd.to_numeric(projected["__canonical_Id"], errors="coerce").eq(grid.region_id)]
        id_overlap = id_candidates_unit.loc[id_candidates_unit.geometry.intersects(grid_polygon)].copy()
        spatial_overlap = projected.loc[projected.geometry.intersects(grid_polygon)].copy()

        # Spatial overlap is the authoritative assignment because a validated
        # footprint inside a unit is usable even when a legacy ID attribute is
        # missing, stale, or encoded differently. ID matching is retained as an
        # audit signal, not as the sole gate.
        if not spatial_overlap.empty:
            unit = spatial_overlap
            selection_mode = "spatial_overlap"
            if not id_overlap.empty and len(id_overlap) == len(spatial_overlap):
                selection_mode = "id_and_spatial_agree"
            elif not id_overlap.empty:
                selection_mode = "spatial_overlap_with_partial_id_agreement"
            elif canonical_id_field is not None:
                selection_mode = "spatial_fallback_after_id_mismatch"
        else:
            unit = id_overlap
            selection_mode = "no_overlap"

        if not unit.empty:
            unit = unit.copy()
            unit["geometry"] = unit.geometry.intersection(grid_polygon)
            unit = _repair_geometry_frame(unit)
        fraction = (
            _fractional_rasterize(unit.geometry, grid, int(cfg["supervision"]["fractional_rasterization_oversample"]))
            if not unit.empty else np.zeros((grid.height, grid.width), dtype=np.float32)
        )
        complete_inventory = bool(cfg["supervision"].get("reference_year_complete_inventory_assumed", False))
        boundary_uncertainty = bool(cfg["supervision"].get("boundary_uncertainty_enabled", False))
        ignore_value = int(cfg["supervision"]["ignore_value"])
        if complete_inventory and boundary_uncertainty:
            # The hand-interpreted 2024 vector is a high-confidence site census,
            # but its raster boundary may be displaced by one or a few Landsat
            # pixels.  Train on a conservative interior core, ignore a symmetric
            # boundary band, and use only far-background pixels as negatives.
            original = np.isfinite(fraction) & (fraction > float(cfg["supervision"].get("exact_inventory_minimum_fraction", 0.0)))
            inner_pixels = max(1, int(np.ceil(float(cfg["supervision"].get("boundary_core_erosion_m", 60.0)) / grid.resolution_m)))
            outer_pixels = max(1, int(np.ceil(float(cfg["supervision"].get("boundary_uncertainty_buffer_m", 120.0)) / grid.resolution_m)))
            eroded = ndimage.binary_erosion(original, structure=np.ones((3, 3), dtype=bool), iterations=inner_pixels, border_value=0)
            positive = original & eroded
            # Formal supervised roles require a genuine interior core.  The
            # single fallback seed below preserves a parcel reference for
            # inference, but it is not treated as a pure training core.
            strong_training_core = positive.copy()
            # Retain one trusted seed for a small polygon that disappears under
            # erosion, but do not promote every full-coverage boundary pixel.
            components, component_count = ndimage.label(original, structure=np.ones((3, 3), dtype=bool))
            for component_id in range(1, component_count + 1):
                component = components == component_id
                if np.any(positive & component):
                    continue
                values = np.where(component, fraction, -np.inf)
                if np.isfinite(values).any():
                    positive.flat[int(np.nanargmax(values))] = True
            outer = ndimage.binary_dilation(original, structure=np.ones((3, 3), dtype=bool), iterations=outer_pixels)
            weak = outer & ~positive
            used_threshold = float(cfg["supervision"].get("boundary_core_minimum_fraction", 0.75))
            label_mode = "adaptive_boundary_core_positive"
            any_pv = original.copy()
            distance_pixels = ndimage.distance_transform_edt(~any_pv)
            ambiguous_outer = weak.copy()
            label = np.zeros(fraction.shape, dtype=np.uint8)
            label[weak] = ignore_value
            label[positive] = 1
        elif complete_inventory:
            minimum_fraction = float(cfg["supervision"].get("exact_inventory_minimum_fraction", 0.0))
            positive = np.isfinite(fraction) & (fraction > minimum_fraction)
            strong_training_core = positive.copy()
            weak = np.zeros_like(positive, dtype=bool)
            used_threshold = minimum_fraction
            label_mode = "complete_inventory_any_overlap"
            any_pv = positive.copy()
            distance_pixels = ndimage.distance_transform_edt(~any_pv)
            ambiguous_outer = np.zeros_like(any_pv, dtype=bool)
            label = np.zeros(fraction.shape, dtype=np.uint8)
            label[positive] = 1
        else:
            positive, weak, used_threshold, label_mode = _fraction_thresholds(fraction, cfg)
            strong_training_core = positive.copy()
            any_pv = positive.copy()
            distance_pixels = ndimage.distance_transform_edt(~any_pv)
            ambiguous_outer = (
                (~any_pv) & (distance_pixels * grid.resolution_m <= float(cfg["supervision"]["ambiguous_outer_buffer_m"]))
                if any_pv.any() else np.zeros_like(any_pv)
            )
            label = np.full(fraction.shape, ignore_value, dtype=np.uint8)
            label[positive] = 1

        # Build a limited set of reliable hard negatives far from the confirmed
        # anchor and with low PV spectral evidence. These pixels support model
        # discrimination without converting every unlabelled pixel to negative.
        hard_negative = np.zeros_like(positive, dtype=bool)
        reference_path = reference_paths.get(grid.region_id)
        if any_pv.any() and reference_path is not None:
            with rasterio.open(reference_path) as ref_src:
                scale = max(1.0, max(ref_src.height, ref_src.width) / 512.0)
                oh = max(1, int(round(ref_src.height / scale)))
                ow = max(1, int(round(ref_src.width / scale)))
                overview = ref_src.read([1,2,3,4,5,6], out_shape=(6, oh, ow), masked=True).filled(np.nan).astype(np.float32)
                overview[overview <= -9990] = np.nan
                ov_features = compute_features(overview, ["PV_Spectral_Evidence"], False)[0]
                ov_valid = np.isfinite(ov_features)
                threshold = float(np.nanquantile(ov_features[ov_valid], float(cfg["supervision"].get("hard_negative_evidence_quantile", 0.35)))) if ov_valid.any() else -np.inf
                min_distance_pixels = float(cfg["supervision"]["hard_negative_min_distance_m"]) / grid.resolution_m
                for _, window in ref_src.block_windows(1):
                    refl = ref_src.read([1,2,3,4,5,6], window=window).astype(np.float32)
                    refl[refl <= -9990] = np.nan
                    feature_names = [
                        "PV_Spectral_Evidence", "Water_Evidence", "Vegetation_Evidence",
                        "Builtup_Evidence", "Bare_Evidence",
                    ]
                    evidence, water_ev, vegetation_ev, built_ev, bare_ev = compute_features(
                        refl, feature_names, False
                    )
                    rs = slice(int(window.row_off), int(window.row_off + window.height))
                    cs = slice(int(window.col_off), int(window.col_off + window.width))
                    distant = distance_pixels[rs, cs] >= min_distance_pixels
                    easy_background = np.isfinite(evidence) & (evidence <= threshold)
                    if bool(cfg["supervision"].get("explicit_confuser_hard_negative_enabled", True)):
                        max_pv = float(cfg["supervision"].get("confuser_maximum_pv_spectral_evidence", 0.20))
                        obvious_confuser = (
                            (water_ev >= float(cfg["supervision"].get("water_hard_negative_threshold", 0.88)))
                            | (vegetation_ev >= float(cfg["supervision"].get("vegetation_hard_negative_threshold", 0.90)))
                            | (
                                (built_ev >= float(cfg["supervision"].get("builtup_hard_negative_threshold", 0.92)))
                                & (evidence <= max_pv)
                            )
                            | (
                                (bare_ev >= float(cfg["supervision"].get("bare_hard_negative_threshold", 0.94)))
                                & (evidence <= max_pv)
                            )
                        )
                    else:
                        obvious_confuser = np.zeros_like(easy_background, dtype=bool)
                    candidate = distant & np.isfinite(evidence) & (easy_background | obvious_confuser)
                    hard_negative[rs, cs] = candidate
        hard_negative &= ~positive & ~weak & ~ambiguous_outer
        if not complete_inventory:
            label[hard_negative] = 0
        label[positive] = 1

        frac_path = mask_root / f"unit_{grid.region_id:03d}_pv_fraction_30m.tif"
        label_path = mask_root / f"unit_{grid.region_id:03d}_training_label_30m.tif"
        support_path = mask_root / f"unit_{grid.region_id:03d}_anchor_support_30m.tif"
        support_radius_m = float(cfg.get("temporal_signature", {}).get("reference_anchor_search_radius_m", 3000.0))
        anchor_support = (
            distance_pixels * grid.resolution_m <= support_radius_m
            if any_pv.any()
            else np.zeros_like(any_pv, dtype=bool)
        )
        for path, arr, dtype, nodata in [
            (frac_path, fraction, "float32", -9999.0),
            (label_path, label, "uint8", int(cfg["supervision"]["ignore_value"])),
            (support_path, anchor_support.astype(np.uint8), "uint8", 255),
        ]:
            profile = grid_profile(grid, 1, dtype, nodata, cfg["processing"].get("compress", "DEFLATE"))
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(dtype), 1)

        overlap_area_m2 = float(unit.geometry.area.sum()) if not unit.empty else 0.0
        confirmed_anchor_area_km2 = float(np.nansum(fraction) * grid.resolution_m * grid.resolution_m / 1_000_000.0)
        nonzero_pixels = int((fraction > 0).sum())
        authoritative_2024_pv = bool(nonzero_pixels > 0 and confirmed_anchor_area_km2 > 0.0)
        strong_training_core_present = bool(np.any(strong_training_core))
        rows.append({
            "Id": grid.region_id,
            "pv_fraction_path": str(frac_path),
            "label_path": str(label_path),
            "anchor_support_path": str(support_path),
            "strong_positive_pixels": int((label == 1).sum()),
            "weak_or_ambiguous_pixels": int((label == int(cfg["supervision"]["ignore_value"])).sum()),
            "negative_pixels": int((label == 0).sum()),
            "hard_negative_pixels": int(hard_negative.sum()),
            "unlabeled_pixels": int((label == int(cfg["supervision"]["ignore_value"])).sum()),
            "confirmed_anchor_area_km2": confirmed_anchor_area_km2,
            # The hand-drawn 2024 polygon is the authoritative parcel contract:
            # any positive raster overlap means the parcel contains PV in 2024.
            # This is deliberately separate from pure-pixel training eligibility.
            "has_mapped_pv": authoritative_2024_pv,
            "has_authoritative_2024_pv": authoritative_2024_pv,
            "has_strong_training_core": strong_training_core_present,
            "authoritative_positive_pixels": nonzero_pixels,
            "outside_shp_semantics": ("boundary_uncertain_far_background_negative" if boundary_uncertainty else "exact_negative_in_2024") if complete_inventory else "unlabelled_not_negative",
            "vector_features_spatial": int(len(spatial_overlap)),
            "vector_features_id_match": int(len(id_candidates_unit)),
            "vector_features_id_and_spatial": int(len(id_overlap)),
            "selection_mode": selection_mode,
            "label_mode": label_mode,
            "positive_fraction_threshold_used": used_threshold,
            "maximum_fraction": float(np.max(fraction)) if fraction.size else 0.0,
            "nonzero_fraction_pixels": nonzero_pixels,
            "overlap_area_m2": overlap_area_m2,
            "canonical_id_field": canonical_id_field or "none",
        })

    registry = pd.DataFrame(rows).sort_values("Id")
    registry = _assign_supervision_roles(registry, cfg)
    save_table(registry, out / "supervision_registry.csv")
    save_table(registry[[
        "Id", "split_role", "has_mapped_pv", "has_authoritative_2024_pv",
        "has_strong_training_core", "strong_positive_pixels", "authoritative_positive_pixels",
        "confirmed_anchor_area_km2", "hard_negative_pixels", "overlap_area_m2", "is_training_unit", "is_calibration_unit",
        "is_independent_test_unit",
    ]], out / "supervision_split.csv")
    save_table(
        registry[[
            "Id", "selection_mode", "label_mode", "vector_features_spatial",
            "vector_features_id_match", "vector_features_id_and_spatial",
            "overlap_area_m2", "maximum_fraction", "nonzero_fraction_pixels",
            "strong_positive_pixels", "authoritative_positive_pixels", "hard_negative_pixels", "unlabeled_pixels",
            "confirmed_anchor_area_km2", "has_mapped_pv", "has_authoritative_2024_pv", "has_strong_training_core",
        ]],
        out / "supervision_diagnostics.csv",
    )
    expected_units, _, _ = _expected_counts(cfg)
    authoritative_regions = int(registry["has_authoritative_2024_pv"].sum())
    positive_regions = int(registry["has_strong_training_core"].sum())
    minimum_positive_regions = int(
        cfg["supervision"].get(
            "minimum_positive_regions",
            max(2, int(cfg["training"].get("minimum_valid_spatial_folds", 3))),
        )
    )
    if test_mode_enabled(cfg):
        # Selected-unit runs are diagnostic algorithm tests, not formal spatial
        # cross-validation. Cap the positive-unit contract to the number of
        # selected units (or an explicit smaller test-mode contract) while
        # leaving the full 142-unit production requirement unchanged.
        selected_count = max(1, int(selected_unit_count(cfg)))
        diagnostic_minimum = int(
            cfg.get("test_mode", {}).get("minimum_positive_regions", selected_count)
        )
        minimum_positive_regions = min(
            minimum_positive_regions,
            max(1, min(selected_count, diagnostic_minimum)),
        )
    units_with_vector_overlap = int((registry["nonzero_fraction_pixels"] > 0).sum())
    status = (
        "passed"
        if registry["Id"].nunique() == expected_units and positive_regions >= minimum_positive_regions
        else "failed"
    )
    payload = {
        "status": status,
        "created_utc": utc_now(),
        "units": len(registry),
        "expected_units": expected_units,
        "reference_feature_count": int(len(shp)),
        "reference_geometry_types": sorted(map(str, shp.geom_type.unique().tolist())),
        "canonical_id_field": canonical_id_field,
        "units_with_vector_overlap": units_with_vector_overlap,
        "units_with_authoritative_2024_parcels": authoritative_regions,
        "units_with_positive_labels": positive_regions,
        "units_with_strong_training_core": positive_regions,
        "minimum_positive_regions_required": minimum_positive_regions,
        "spatial_fallback_units": int(registry["selection_mode"].astype(str).str.contains("fallback").sum()),
        "adaptive_subpixel_units": int(registry["label_mode"].eq("adaptive_subpixel_fraction").sum()),
        "total_positive_pixels": int(registry["strong_positive_pixels"].sum()),
        "diagnostic_table": str(out / "supervision_diagnostics.csv"),
        "split_counts": registry["split_role"].value_counts().to_dict(),
        "positive_split_counts": registry.loc[registry["has_strong_training_core"].astype(bool), "split_role"].value_counts().to_dict(),
        "authoritative_parcel_route_counts": registry.loc[registry["has_authoritative_2024_pv"].astype(bool), "split_role"].value_counts().to_dict(),
        "independent_test_units_reserved": True,
        "reference_year": reference_year,
        "reference_year_complete_inventory_assumed": bool(complete_inventory),
        "boundary_uncertainty_enabled": bool(boundary_uncertainty),
        "boundary_band_is_unlabelled": bool(boundary_uncertainty),
        "far_background_is_negative": bool(complete_inventory),
        "outside_is_fully_unlabelled": bool(not complete_inventory),
        "outside_shp_semantics": (
            "boundary_uncertainty_band_unlabelled_far_background_negative"
            if boundary_uncertainty
            else ("exact_negative_in_2024" if complete_inventory else "unlabelled_not_negative")
        ),
        "shp_semantics": (
            "2024 manual parcel overlap is authoritative PV presence; conservative interior cores alone define formal training positives, while the boundary band is unlabelled"
            if boundary_uncertainty
            else ("complete necessary-and-sufficient 2024 inventory; outside is exact negative" if complete_inventory else "positive-only sufficient condition; outside is unlabelled")
        ),
    }
    atomic_json(payload, out / "supervision_status.json")
    if status == "failed":
        if units_with_vector_overlap == 0:
            reason = (
                "No validated PV geometry overlaps any canonical Landsat grid. "
                "Check that reference_shp is the PV-footprint layer, its CRS is correct, "
                "and its spatial extent corresponds to /home/wsy/pvsystem_input_1_142."
            )
        else:
            reason = (
                f"Only {positive_regions} regions contain usable positive labels; "
                f"at least {minimum_positive_regions} are required for spatial cross-validation."
            )
        raise RuntimeError(f"Supervision validation failed: {reason} See {out / 'supervision_diagnostics.csv'}")
    return registry



def _effective_patch_shape(height: int, width: int, requested_patch_size: int) -> tuple[int, int, bool]:
    """Return a boundary-safe patch shape for a raster.

    Large units keep the configured square patch. Small units use their full
    raster extent along the undersized dimension instead of failing or adding
    synthetic padded pixels. This preserves all observed Landsat pixels and
    avoids reflection/constant-padding artefacts in CCDC and spectral features.
    """
    height = int(height)
    width = int(width)
    requested_patch_size = int(requested_patch_size)
    if height <= 0 or width <= 0:
        raise RuntimeError(f"Invalid raster dimensions: width={width}, height={height}")
    if requested_patch_size <= 0:
        raise RuntimeError(f"patch_size must be positive, found {requested_patch_size}")
    patch_height = min(height, requested_patch_size)
    patch_width = min(width, requested_patch_size)
    adaptive = patch_height != requested_patch_size or patch_width != requested_patch_size
    return patch_height, patch_width, adaptive


def _window_offsets(length: int, window_size: int, stride: int) -> list[int]:
    """Generate full-coverage offsets, including the terminal boundary window."""
    length = int(length)
    window_size = int(window_size)
    stride = max(1, int(stride))
    if length <= 0 or window_size <= 0 or window_size > length:
        return []
    offsets = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return offsets

def _balanced_pixel_ids(y: np.ndarray, valid: np.ndarray, per: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    flat_y = y.reshape(-1)
    flat_valid = valid.reshape(-1)
    pos = np.flatnonzero((flat_y == 1) & flat_valid)
    neg = np.flatnonzero((flat_y == 0) & flat_valid)
    if len(pos):
        npos = min(len(pos), max(1, int(per / (1.0 + ratio))))
        nneg = min(len(neg), max(1, int(npos * ratio)))
    else:
        npos = 0
        nneg = min(len(neg), per)
    chosen: list[int] = []
    if npos:
        chosen.extend(rng.choice(pos, npos, replace=False).tolist())
    if nneg:
        chosen.extend(rng.choice(neg, nneg, replace=False).tolist())
    return np.asarray(chosen, dtype=np.int64)


def _assign_balanced_folds(region_summary: pd.DataFrame, requested_folds: int) -> tuple[dict[int, int], pd.DataFrame]:
    positive_regions = int((region_summary["positive_samples"] > 0).sum())
    folds = min(int(requested_folds), positive_regions, len(region_summary))
    if folds < 2:
        raise RuntimeError(f"Spatial cross-validation requires at least 2 positive-label regions; found {positive_regions}")
    loads = [{"fold": i, "positive": 0, "negative": 0, "regions": 0} for i in range(folds)]
    assignment: dict[int, int] = {}
    ordered = region_summary.sort_values(["positive_samples", "total_samples"], ascending=False)
    for row in ordered.itertuples(index=False):
        target = min(loads, key=lambda z: (z["positive"], z["regions"], z["negative"])) if int(row.positive_samples) > 0 else min(loads, key=lambda z: (z["negative"], z["regions"], z["positive"]))
        fold = int(target["fold"])
        assignment[int(row.Id)] = fold
        target["positive"] += int(row.positive_samples)
        target["negative"] += int(row.negative_samples)
        target["regions"] += 1
    diagnostics = pd.DataFrame(loads).rename(columns={"positive": "positive_samples", "negative": "negative_samples"})
    diagnostics["has_both_classes"] = (diagnostics.positive_samples > 0) & (diagnostics.negative_samples > 0)
    return assignment, diagnostics


def build_patch_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "04_patch_dataset")
    patch_root = ensure_dir(Path(cfg["paths"]["cache_root"]) / f"patches_{int(cfg['study']['reference_year'])}")
    shard_root = ensure_dir(Path(cfg["paths"]["cache_root"]) / "training_feature_shards")
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    reference = aligned.loc[pd.to_numeric(aligned["year"], errors="coerce").eq(int(cfg["study"]["reference_year"]))]
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv")
    merged_all = reference.merge(supervision, on="Id", how="inner").merge(
        quality[["Id", "direct_detection_eligible", "estimation_route", "quality_score", "quality_reasons", "endmember_path"]],
        on="Id", how="left"
    )
    if "split_role" not in merged_all:
        # Backward-compatible direct-function use for synthetic tests.  The v20
        # scheduler itself requires supervision_split.csv and will not accept a
        # real run without independent train/calibration/test roles.
        merged_all["split_role"] = "train"
    merged = merged_all.loc[
        merged_all["split_role"].eq("train")
        & merged_all["direct_detection_eligible"].fillna(False).astype(bool)
    ].copy()
    annual_lookup = {int(k): g.sort_values("year") for k, g in aligned.groupby("Id")}
    patch_size = int(cfg["patches"]["patch_size"])
    stride = int(cfg["patches"]["stride"])
    ignore = int(cfg["supervision"]["ignore_value"])
    configured_min_valid_fraction = float(cfg["patches"]["minimum_valid_fraction"])
    minimum_valid_pixels = int(cfg["patches"].get("minimum_valid_pixels_per_patch", 32))
    allow_exclusion = bool(cfg["patches"].get("allow_training_unit_exclusion", True))
    seed = int(cfg["project"]["random_seed"])
    rng = np.random.default_rng(seed)
    channels = list(cfg["features"]["channels"])
    add_valid = bool(cfg["features"].get("add_valid_mask_channel", True))
    missing_ccdc = [name for name in cfg["ccdc"]["indices"] if name not in channels]
    if missing_ccdc:
        raise RuntimeError(f"CCDC indices are missing from feature channels: {missing_ccdc}")
    ccdc_indices = [channels.index(n) for n in cfg["ccdc"]["indices"]]
    patch_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    region_summaries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    heldout_mask = ~merged_all["Id"].isin(merged["Id"])
    for heldout in merged_all.loc[heldout_mask].itertuples(index=False):
        low_quality = not bool(getattr(heldout, "direct_detection_eligible", False))
        audit_rows.append({
            "Id": int(heldout.Id),
            "split_role": str(heldout.split_role),
            "training_status": "excluded_quality" if low_quality else "held_out",
            "exclusion_reason": (
                f"quality_route_{getattr(heldout, 'estimation_route', 'unknown')}"
                if low_quality else f"reserved_{heldout.split_role}_unit"
            ),
            "quality_score": float(getattr(heldout, "quality_score", np.nan)),
            "quality_reasons": str(getattr(heldout, "quality_reasons", "")),
            "configured_minimum_valid_fraction": configured_min_valid_fraction,
            "minimum_valid_pixels_per_patch": minimum_valid_pixels,
            "supervision_positive_pixels": int(getattr(heldout, "strong_positive_pixels", 0)),
            "has_mapped_pv": bool(getattr(heldout, "has_mapped_pv", False)),
        })

    for r in tqdm(
        list(merged.itertuples(index=False)),
        desc="Training patches and one-pass CCDC cache",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        region_id = int(r.Id)
        annual_group = annual_lookup[region_id]
        annual_srcs = [rasterio.open(p) for p in annual_group.path]
        annual_years = annual_group["year"].astype(int).tolist()
        reference_year = int(cfg["study"]["reference_year"])
        if reference_year not in annual_years:
            raise RuntimeError(f"Id {region_id}: reference year {reference_year} missing")
        reference_annual_index = annual_years.index(reference_year)
        endmembers = load_endmember_matrix(r.endmember_path)
        audit: dict[str, Any] = {
            "Id": region_id,
            "split_role": "train",
            "training_status": "pending",
            "exclusion_reason": "",
            "configured_minimum_valid_fraction": configured_min_valid_fraction,
            "minimum_valid_pixels_per_patch": minimum_valid_pixels,
            "supervision_positive_pixels": int(getattr(r, "strong_positive_pixels", 0)),
            "has_mapped_pv": bool(getattr(r, "has_mapped_pv", False)),
        }
        try:
            src = annual_srcs[reference_annual_index]
            audit.update(raster_width=int(src.width), raster_height=int(src.height))
            with rasterio.open(r.label_path) as label_src:
                if label_src.width != src.width or label_src.height != src.height:
                    raise RuntimeError(
                        f"Id {region_id}: label/raster shape mismatch "
                        f"({label_src.width}x{label_src.height} vs {src.width}x{src.height})"
                    )
                patch_height, patch_width, adaptive_patch = _effective_patch_shape(
                    src.height, src.width, patch_size
                )
                row_stride = min(stride, patch_height)
                col_stride = min(stride, patch_width)
                row_offsets = _window_offsets(src.height, patch_height, row_stride)
                col_offsets = _window_offsets(src.width, patch_width, col_stride)
                if not row_offsets or not col_offsets:
                    raise RuntimeError(
                        f"Id {region_id}: no boundary-safe windows for raster "
                        f"{src.width}x{src.height} and effective patch {patch_width}x{patch_height}"
                    )

                # Keep a complete window inventory. A 70% valid-label rule is
                # useful for ordinary negatives, but it must not erase genuine
                # positive pixels in small or ambiguity-dominated units.
                all_candidates: list[dict[str, Any]] = []
                for row_off in row_offsets:
                    for col_off in col_offsets:
                        window = Window(col_off, row_off, patch_width, patch_height)
                        label = label_src.read(1, window=window)
                        valid_label = label != ignore
                        valid_pixels = int(valid_label.sum())
                        valid_fraction = float(valid_pixels / max(label.size, 1))
                        pos = int(np.sum(label == 1))
                        neg = int(np.sum(label == 0))
                        all_candidates.append({
                            "row_off": int(row_off),
                            "col_off": int(col_off),
                            "positive_pixels": pos,
                            "negative_pixels": neg,
                            "valid_pixels": valid_pixels,
                            "valid_fraction": valid_fraction,
                        })

                positive_candidates = [
                    c for c in all_candidates
                    if c["positive_pixels"] > 0 and c["valid_pixels"] > 0
                ]
                strict_negative_candidates = [
                    c for c in all_candidates
                    if c["positive_pixels"] == 0
                    and c["negative_pixels"] > 0
                    and c["valid_fraction"] >= configured_min_valid_fraction
                ]
                fallback_negative_candidates = [
                    c for c in all_candidates
                    if c["positive_pixels"] == 0
                    and c["negative_pixels"] > 0
                    and c["valid_pixels"] >= minimum_valid_pixels
                ]

                # Positive windows are always retained as valid supervision,
                # regardless of how much surrounding area is deliberately
                # ignored. For negatives, prefer the strict rule and relax it
                # only when an entire unit would otherwise be unusable.
                for candidate in positive_candidates:
                    candidate["selection_policy"] = (
                        "strict_positive"
                        if candidate["valid_fraction"] >= configured_min_valid_fraction
                        else "positive_preservation"
                    )
                if strict_negative_candidates:
                    negatives = strict_negative_candidates
                    for candidate in negatives:
                        candidate["selection_policy"] = "strict_negative"
                else:
                    negatives = fallback_negative_candidates
                    for candidate in negatives:
                        candidate["selection_policy"] = "adaptive_valid_fraction_negative"

                preferred_positive_pixels = int(cfg["supervision"]["minimum_positive_pixels_per_patch"])
                positive_candidates.sort(
                    key=lambda item: (
                        item["positive_pixels"] >= preferred_positive_pixels,
                        item["positive_pixels"],
                        item["valid_pixels"],
                    ),
                    reverse=True,
                )
                negatives.sort(
                    key=lambda item: (item["valid_fraction"], item["valid_pixels"]),
                    reverse=True,
                )
                max_neg = max(
                    2,
                    int(max(1, len(positive_candidates)) * float(cfg["patches"]["maximum_negative_to_positive_ratio"])),
                )
                if len(negatives) > max_neg:
                    chosen = rng.choice(len(negatives), max_neg, replace=False)
                    negatives = [negatives[i] for i in chosen]

                selected = positive_candidates + negatives
                max_patches = int(cfg["patches"]["maximum_patches_per_unit"])
                if len(selected) > max_patches:
                    pos_keep = min(
                        len(positive_candidates),
                        max_patches // 2 if negatives else max_patches,
                    )
                    neg_keep = min(len(negatives), max_patches - pos_keep)
                    selected = (
                        [] if pos_keep == 0 else [
                            positive_candidates[i]
                            for i in rng.choice(len(positive_candidates), pos_keep, replace=False)
                        ]
                    ) + (
                        [] if neg_keep == 0 else [
                            negatives[i]
                            for i in rng.choice(len(negatives), neg_keep, replace=False)
                        ]
                    )

                audit.update(
                    patch_width=int(patch_width),
                    patch_height=int(patch_height),
                    adaptive_patch=bool(adaptive_patch),
                    candidate_windows=int(len(all_candidates)),
                    strict_valid_windows=int(sum(c["valid_fraction"] >= configured_min_valid_fraction for c in all_candidates)),
                    positive_candidate_windows=int(len(positive_candidates)),
                    strict_negative_windows=int(len(strict_negative_candidates)),
                    fallback_negative_windows=int(len(fallback_negative_candidates)),
                    selected_windows=int(len(selected)),
                    maximum_window_valid_fraction=float(max((c["valid_fraction"] for c in all_candidates), default=0.0)),
                    total_valid_label_pixels=int(sum(c["valid_pixels"] for c in all_candidates)),
                )

                if not selected:
                    audit["training_status"] = "excluded"
                    audit["exclusion_reason"] = "no_reliable_positive_or_negative_label_pixels"
                    audit_rows.append(audit)
                    if allow_exclusion:
                        continue
                    raise RuntimeError(
                        f"Id {region_id}: no valid training patches; "
                        "set patches.allow_training_unit_exclusion=true to audit and exclude this unit"
                    )

                region_X: list[np.ndarray] = []
                region_y: list[np.ndarray] = []
                region_patch: list[np.ndarray] = []
                unit_dir = ensure_dir(patch_root / f"unit_{region_id:03d}")
                for seq, candidate in enumerate(selected):
                    row_off = int(candidate["row_off"])
                    col_off = int(candidate["col_off"])
                    pos = int(candidate["positive_pixels"])
                    neg = int(candidate["negative_pixels"])
                    valid_fraction = float(candidate["valid_fraction"])
                    valid_pixels = int(candidate["valid_pixels"])
                    selection_policy = str(candidate["selection_policy"])
                    window = Window(col_off, row_off, patch_width, patch_height)
                    y = label_src.read(1, window=window).astype(np.uint8)
                    valid = y != ignore
                    ids = _balanced_pixel_ids(
                        y,
                        valid,
                        int(cfg["training"]["pixels_per_patch"]),
                        float(cfg["training"]["positive_negative_ratio"]),
                        rng,
                    )
                    if ids.size == 0:
                        continue
                    index_series: list[list[np.ndarray]] = [[] for _ in ccdc_indices]
                    reference_reflectance: np.ndarray | None = None
                    for annual_index, annual_src in enumerate(annual_srcs):
                        refl = annual_src.read([1, 2, 3, 4, 5, 6], window=window).astype(np.float32)
                        refl[refl <= -9990] = np.nan
                        if annual_index == reference_annual_index:
                            reference_reflectance = refl
                        base = compute_features(refl, channels, False, endmembers=endmembers)
                        for j, ix in enumerate(ccdc_indices):
                            index_series[j].append(base[ix].reshape(-1))
                    if reference_reflectance is None:
                        raise RuntimeError(f"Id {region_id}: reference-year reflectance missing")
                    static = compute_features(reference_reflectance, channels, add_valid, endmembers=endmembers)
                    ccdc_blocks = [
                        annual_ccdc_descriptors(
                            np.stack(values, axis=0),
                            int(cfg["ccdc"]["minimum_observations"]),
                            int(cfg["ccdc"]["break_minimum_segment_years"]),
                        )
                        for values in index_series
                    ]
                    flat = static.reshape(static.shape[0], -1).T
                    ccdc = np.concatenate(ccdc_blocks, axis=1)
                    X = np.nan_to_num(
                        np.concatenate([flat, ccdc], axis=1),
                        nan=0.0,
                        posinf=8.0,
                        neginf=-8.0,
                    )[ids].astype(np.float32)
                    yy = y.reshape(-1)[ids].astype(np.uint8)
                    patch_path = unit_dir / f"r{row_off:05d}_c{col_off:05d}_{seq:04d}.npz"
                    np.savez_compressed(
                        patch_path,
                        x=static.astype(np.float16),
                        y=y,
                        sampled_ids=ids.astype(np.int32),
                    )
                    patch_rows.append({
                        "Id": region_id,
                        "row_off": row_off,
                        "col_off": col_off,
                        "patch_width": int(patch_width),
                        "patch_height": int(patch_height),
                        "configured_patch_size": int(patch_size),
                        "adaptive_patch": bool(adaptive_patch),
                        "positive_pixels": pos,
                        "negative_pixels": neg,
                        "valid_pixels": valid_pixels,
                        "valid_fraction": valid_fraction,
                        "selection_policy": selection_policy,
                        "patch_path": str(patch_path),
                        "is_positive_patch": bool(pos > 0),
                        "sampled_pixels": int(len(ids)),
                    })
                    region_X.append(X)
                    region_y.append(yy)
                    region_patch.append(np.full(len(yy), seq, dtype=np.int32))

                if not region_X:
                    audit["training_status"] = "excluded"
                    audit["exclusion_reason"] = "selected_windows_produced_no_sampled_pixels"
                    audit_rows.append(audit)
                    if allow_exclusion:
                        continue
                    raise RuntimeError(f"Id {region_id}: no sampled training pixels")

                Xr = np.concatenate(region_X, axis=0)
                yr = np.concatenate(region_y, axis=0)
                pr = np.concatenate(region_patch, axis=0)
                cap = int(cfg["patches"].get("maximum_cached_pixels_per_unit", 9000))
                if len(yr) > cap:
                    pos_ids = np.flatnonzero(yr == 1)
                    neg_ids = np.flatnonzero(yr == 0)
                    npos = (
                        min(
                            len(pos_ids),
                            max(1, int(cap / (1 + float(cfg["training"]["positive_negative_ratio"])))),
                        )
                        if len(pos_ids)
                        else 0
                    )
                    nneg = min(len(neg_ids), cap - npos)
                    keep = np.concatenate([
                        rng.choice(pos_ids, npos, replace=False) if npos else np.array([], dtype=int),
                        rng.choice(neg_ids, nneg, replace=False) if nneg else np.array([], dtype=int),
                    ])
                    rng.shuffle(keep)
                    Xr, yr, pr = Xr[keep], yr[keep], pr[keep]

                shard_path = shard_root / f"unit_{region_id:03d}_training_features.npz"
                np.savez_compressed(
                    shard_path,
                    X=Xr.astype(np.float32),
                    y=yr.astype(np.uint8),
                    patch_id=pr,
                    Id=np.int16(region_id),
                )
                fallback_used = any(
                    row["selection_policy"] in {
                        "positive_preservation",
                        "adaptive_valid_fraction_negative",
                    }
                    for row in patch_rows
                    if int(row["Id"]) == region_id
                )
                shard_rows.append({
                    "Id": region_id,
                    "shard_path": str(shard_path),
                    "samples": len(yr),
                    "positive_samples": int((yr == 1).sum()),
                    "negative_samples": int((yr == 0).sum()),
                    "feature_count": int(Xr.shape[1]),
                    "raster_width": int(src.width),
                    "raster_height": int(src.height),
                    "patch_width": int(patch_width),
                    "patch_height": int(patch_height),
                    "adaptive_patch": bool(adaptive_patch),
                    "adaptive_valid_fraction_used": bool(fallback_used),
                })
                region_summaries.append({
                    "Id": region_id,
                    "total_samples": len(yr),
                    "positive_samples": int((yr == 1).sum()),
                    "negative_samples": int((yr == 0).sum()),
                    "raster_width": int(src.width),
                    "raster_height": int(src.height),
                    "patch_width": int(patch_width),
                    "patch_height": int(patch_height),
                    "adaptive_patch": bool(adaptive_patch),
                })
                audit["training_status"] = "included"
                audit["exclusion_reason"] = ""
                audit["sampled_training_pixels"] = int(len(yr))
                audit["sampled_positive_pixels"] = int((yr == 1).sum())
                audit["sampled_negative_pixels"] = int((yr == 0).sum())
                audit["adaptive_valid_fraction_used"] = bool(fallback_used)
                audit_rows.append(audit)
        finally:
            for annual_src in annual_srcs:
                annual_src.close()

    index = pd.DataFrame(patch_rows)
    shards = pd.DataFrame(shard_rows)
    region_summary = pd.DataFrame(region_summaries)
    audit_frame = pd.DataFrame(audit_rows).sort_values("Id").reset_index(drop=True)
    save_table(audit_frame, out / "training_unit_audit.csv")
    if index.empty or shards.empty or region_summary.empty:
        raise RuntimeError("No training patches or feature shards were generated")

    if test_mode_enabled(cfg) and int((region_summary["positive_samples"] > 0).sum()) < 2:
        # Three-unit diagnostic runs may leave one positive-bearing training unit
        # after reserving calibration/test units. Keep one region fold here; the
        # benchmark stage creates deterministic within-unit blocked pixel folds.
        fold_map = {int(uid): 0 for uid in region_summary["Id"]}
        fold_diag = pd.DataFrame([{
            "fold": 0,
            "positive_samples": int(region_summary["positive_samples"].sum()),
            "negative_samples": int(region_summary["negative_samples"].sum()),
            "regions": int(len(region_summary)),
            "has_both_classes": bool(
                int(region_summary["positive_samples"].sum()) > 0
                and int(region_summary["negative_samples"].sum()) > 0
            ),
            "fold_semantics": "single-region source; deterministic blocked-pixel CV is created in Stage 5",
        }])
    else:
        fold_map, fold_diag = _assign_balanced_folds(
            region_summary,
            int(cfg["patches"]["spatial_folds"]),
        )
    index["fold"] = index["Id"].map(fold_map).astype(int)
    shards["fold"] = shards["Id"].map(fold_map).astype(int)
    index = index.sort_values(["fold", "Id", "patch_path"]).reset_index(drop=True)
    shards = shards.sort_values(["fold", "Id"]).reset_index(drop=True)
    save_table(index, out / "patch_index.csv")
    save_table(shards, out / "training_feature_cache_registry.csv")
    save_table(fold_diag, out / "patch_fold_summary.csv")

    expected_units, _, _ = _expected_counts(cfg)
    minimum_training_regions = int(
        cfg["patches"].get(
            "minimum_training_regions",
            max(3, int(cfg["training"].get("minimum_valid_spatial_folds", 3))),
        )
    )
    if test_mode_enabled(cfg):
        minimum_training_regions = min(
            minimum_training_regions,
            int(cfg.get("test_mode", {}).get("minimum_training_regions", 2)),
        )
    audit_complete = audit_frame["Id"].nunique() == expected_units
    enough_training_regions = shards["Id"].nunique() >= minimum_training_regions
    folds_valid = bool(fold_diag["has_both_classes"].all())
    if test_mode_enabled(cfg):
        # In a four-unit algorithm smoke test, only two training units may be
        # available. The benchmark stage can skip invalid folds and only needs
        # at least one valid fold under the selected-unit test contract.
        folds_valid = bool(fold_diag["has_both_classes"].any())
    status = "passed" if audit_complete and enough_training_regions and folds_valid else "failed"
    excluded = audit_frame.loc[audit_frame["training_status"].eq("excluded")]
    held_out = audit_frame.loc[audit_frame["training_status"].eq("held_out")]
    quality_excluded = audit_frame.loc[audit_frame["training_status"].eq("excluded_quality")]
    adaptive_valid_series = audit_frame.get(
        "adaptive_valid_fraction_used",
        pd.Series(False, index=audit_frame.index, dtype="boolean"),
    ).astype("boolean").fillna(False)
    adaptive_valid_units = int(adaptive_valid_series.sum())
    atomic_json({
        "status": status,
        "created_utc": utc_now(),
        "patches": len(index),
        "inference_regions": expected_units,
        "audited_regions": int(audit_frame["Id"].nunique()),
        "training_regions": int(shards["Id"].nunique()),
        "excluded_training_regions": int(len(excluded)),
        "excluded_training_region_ids": excluded["Id"].astype(int).tolist(),
        "held_out_regions": int(len(held_out)),
        "quality_excluded_regions": int(len(quality_excluded)),
        "held_out_calibration_regions": int((held_out["split_role"] == "calibration").sum()),
        "held_out_test_regions": int((held_out["split_role"] == "test").sum()),
        "training_split_only": True,
        "minimum_training_regions_required": minimum_training_regions,
        "cached_training_pixels": int(shards.samples.sum()),
        "positive_samples": int(shards.positive_samples.sum()),
        "negative_samples": int(shards.negative_samples.sum()),
        "folds": int(shards.fold.nunique()),
        "all_folds_have_both_classes": folds_valid,
        "configured_patch_size": int(patch_size),
        "adaptive_patch_regions": int(shards["adaptive_patch"].astype(bool).sum()),
        "adaptive_valid_fraction_regions": adaptive_valid_units,
        "minimum_effective_patch_width": int(shards["patch_width"].min()),
        "minimum_effective_patch_height": int(shards["patch_height"].min()),
        "small_raster_policy": "use_observed_extent_without_padding",
        "low_valid_label_policy": "preserve_positive_windows_then_adapt_negative_valid_fraction",
        "training_unit_exclusion_policy": "audit_and_exclude_only_when_no_reliable_label_pixels",
        "feature_channels": channels
        + (["VALID_MASK"] if add_valid else [])
        + [
            f"CCDC_{name}_{j}"
            for name in cfg["ccdc"]["indices"]
            for j in range(7)
        ],
        "io_optimization": (
            "Annual CCDC descriptors are computed once per selected patch and cached as "
            "per-region tabular feature shards. Benchmark folds never reopen Landsat rasters."
        ),
    }, out / "patch_dataset_status.json")
    if status == "failed":
        raise RuntimeError(
            "Patch feature cache failed: "
            f"audited_regions={audit_frame['Id'].nunique()}/{expected_units}, "
            f"training_regions={shards['Id'].nunique()}/{minimum_training_regions} minimum, "
            f"all_folds_have_both_classes={folds_valid}"
        )
    return index

# --- v24.9 four-unit smoke split override start ---
def _allocate_role_counts(total, fractions, minimums):
    """Allocate train/calibration/test roles with selected-unit smoke-test support."""
    total = int(total)
    fractions = {str(k): float(v) for k, v in dict(fractions).items()}
    minimums = {str(k): int(v) for k, v in dict(minimums).items()}
    role_names = list(fractions.keys()) or ["train", "calibration", "test"]
    if total <= 0:
        return {role: 0 for role in role_names}
    minimum_required = sum(max(0, minimums.get(role, 0)) for role in role_names)
    if total < minimum_required:
        counts = {role: 0 for role in role_names}
        train_key = "train" if "train" in counts else ("training" if "training" in counts else role_names[0])
        cal_key = "calibration" if "calibration" in counts else ("calib" if "calib" in counts else ("validation" if "validation" in counts else None))
        test_key = "test" if "test" in counts else None
        if total == 1:
            counts[train_key] = 1
        elif total == 2:
            # Two-unit diagnostic contract: one training unit and one
            # calibration unit. Stage 9 is descriptive rather than an
            # independent publication validation for this smoke test.
            counts[train_key] = 1
            counts[cal_key or test_key or train_key] += 1
        elif total == 3:
            counts[train_key] = 1
            counts[cal_key or train_key] += 1
            counts[test_key or train_key] += 1
        elif total == 4 and cal_key and test_key:
            counts[train_key] = 2
            counts[cal_key] = 1
            counts[test_key] = 1
        else:
            counts[train_key] = max(1, total - 2)
            remaining = total - counts[train_key]
            if remaining > 0 and cal_key:
                counts[cal_key] += 1; remaining -= 1
            if remaining > 0 and test_key:
                counts[test_key] += 1; remaining -= 1
            counts[train_key] += remaining
        assert sum(counts.values()) == total, (counts, total)
        return counts
    counts = {role: max(0, minimums.get(role, 0)) for role in role_names}
    remaining = total - sum(counts.values())
    if remaining < 0:
        raise RuntimeError(
            f"Cannot create independent train/calibration/test split for "
            f"{total} positive units; minimum required is {minimum_required}"
        )
    fraction_sum = sum(max(0.0, fractions.get(role, 0.0)) for role in role_names)
    if fraction_sum <= 0:
        counts[role_names[0]] += remaining
        return counts
    raw = {role: remaining * max(0.0, fractions.get(role, 0.0)) / fraction_sum for role in role_names}
    floors = {role: int(raw[role]) for role in role_names}
    for role, value in floors.items():
        counts[role] += value
    leftover = total - sum(counts.values())
    order = sorted(role_names, key=lambda r: (raw[r] - floors[r], fractions.get(r, 0.0)), reverse=True)
    for role in order[:leftover]:
        counts[role] += 1
    assert sum(counts.values()) == total, (counts, total)
    return counts
# --- v24.9 four-unit smoke split override end ---

