from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from tqdm import tqdm

from .config import load_config
from .features import compute_features
from .spectral_mixture import load_endmember_matrix
from .temporal_signature import (
    annual_active_masks_from_evidence,
    annual_latent_masks_from_onset,
    detect_temporal_signature,
    compute_anchor_reference_spectral_similarity,
    filter_transition_components,
    propagate_anchor_backcast_within_objects,
    rescue_anchor_internal_onset_from_trajectory,
    refine_anchor_onset_with_change_boundary_coherence,
    refine_anchor_onset_with_landsat_object_pavg,
    enforce_spatiotemporal_topology,
)
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _safe_nanmedian(array: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(array, axis=axis)


def _safe_nanmax(array: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(array, axis=axis)


def _safe_nanmin(array: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(array, axis=axis)


def _release_native_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _core_windows(width: int, height: int, size: int) -> Iterable[Window]:
    size = max(16, int(size))
    for row in range(0, int(height), size):
        for col in range(0, int(width), size):
            yield Window(
                col_off=col,
                row_off=row,
                width=min(size, int(width) - col),
                height=min(size, int(height) - row),
            )


def _expanded_window(core: Window, width: int, height: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    halo = max(0, int(halo))
    left = max(0, int(core.col_off) - halo)
    top = max(0, int(core.row_off) - halo)
    right = min(int(width), int(core.col_off + core.width) + halo)
    bottom = min(int(height), int(core.row_off + core.height) + halo)
    expanded = Window(left, top, right - left, bottom - top)
    row0 = int(core.row_off) - top
    col0 = int(core.col_off) - left
    crop = (
        slice(row0, row0 + int(core.height)),
        slice(col0, col0 + int(core.width)),
    )
    return expanded, crop


def _read_probability_cube(paths: list[Path], window: Window) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    for path in paths:
        with rasterio.open(path, sharing=False) as src:
            array = src.read(1, window=window).astype(np.float32, copy=False)
        arrays.append(array)
    cube = np.stack(arrays, axis=0)
    valid = np.isfinite(cube) & (cube > -9990.0)
    cube[~valid] = np.nan
    return cube, valid


def _append_bounded(existing: np.ndarray, values: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return existing
    combined = values if existing.size == 0 else np.concatenate([existing, values])
    if combined.size <= maximum:
        return combined
    ids = rng.choice(combined.size, size=maximum, replace=False)
    return combined[ids]


def _estimate_unit_biases(
    probability_paths: list[Path],
    width: int,
    height: int,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, int]:
    settings = cfg.get("temporal_signature", {})
    tile_size = int(settings.get("harmonization_tile_size", 256))
    maximum_samples = int(settings.get("harmonization_maximum_samples_per_year", 100000))
    per_tile = int(settings.get("harmonization_maximum_samples_per_tile", 2500))
    ceiling = float(settings.get("stable_background_probability_ceiling", 0.22))
    max_range = float(settings.get("stable_background_maximum_range", 0.15))
    minimum = int(settings.get("harmonization_minimum_pixels", 500))
    minimum_years = int(settings.get("harmonization_minimum_valid_years", max(6, len(probability_paths) // 2)))
    rng = np.random.default_rng(int(seed))
    reservoirs = [np.empty(0, dtype=np.float32) for _ in probability_paths]
    stable_total = 0

    for window in _core_windows(width, height, tile_size):
        probability, valid = _read_probability_cube(probability_paths, window)
        median = _safe_nanmedian(probability, axis=0)
        maximum = _safe_nanmax(probability, axis=0)
        minimum_value = _safe_nanmin(probability, axis=0)
        value_range = maximum - minimum_value
        enough = np.sum(valid, axis=0) >= minimum_years
        stable = enough & np.isfinite(median) & (median <= ceiling) & np.isfinite(value_range) & (value_range <= max_range)
        ids = np.flatnonzero(stable.ravel())
        stable_total += int(ids.size)
        if ids.size:
            if ids.size > per_tile:
                ids = rng.choice(ids, size=per_tile, replace=False)
            clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
            logits = np.log(clipped / (1.0 - clipped))
            for index in range(len(probability_paths)):
                flat = logits[index].ravel()
                selected = flat[ids]
                selected = selected[np.isfinite(selected)]
                reservoirs[index] = _append_bounded(
                    reservoirs[index], selected, maximum_samples, rng
                )
        del probability, valid, median, maximum, minimum_value, value_range, stable

    if stable_total < minimum:
        return np.zeros(len(probability_paths), dtype=np.float32), stable_total

    medians = np.asarray(
        [float(np.median(values)) if values.size else np.nan for values in reservoirs],
        dtype=np.float32,
    )
    target = float(np.nanmedian(medians)) if np.isfinite(medians).any() else 0.0
    biases = np.where(np.isfinite(medians), medians - target, 0.0).astype(np.float32)
    return biases, stable_total


def _apply_biases(probability: np.ndarray, valid: np.ndarray, biases: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped))
    adjusted = 1.0 / (1.0 + np.exp(-np.clip(logits - biases[:, None, None], -20.0, 20.0)))
    adjusted = adjusted.astype(np.float32)
    adjusted[~valid] = np.nan
    return adjusted


def _ensure_anchor_support(
    fraction_path: Path,
    support_path: Path,
    radius_m: float,
    compress: str,
) -> Path:
    if support_path.exists() and support_path.stat().st_size > 0:
        return support_path
    ensure_dir(support_path.parent)
    with rasterio.open(fraction_path, sharing=False) as src:
        fraction = src.read(1).astype(np.float32)
        valid = np.isfinite(fraction) & (fraction > -9990.0)
        anchor = valid & (fraction > 0.0)
        pixel_size = math.sqrt(abs(float(src.transform.a * src.transform.e)))
        maximum_distance_pixels = max(0.0, float(radius_m) / max(pixel_size, 1e-6))
        if anchor.any():
            distance = ndimage.distance_transform_edt(~anchor)
            support = distance <= maximum_distance_pixels
        else:
            support = np.zeros(anchor.shape, dtype=bool)
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress=compress,
            BIGTIFF="IF_SAFER",
        )
    temporary = support_path.with_suffix(support_path.suffix + ".tmp")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(support.astype(np.uint8), 1)
    os.replace(temporary, support_path)
    del fraction, anchor, support
    _release_native_memory()
    return support_path


def _valid_existing_unit(checkpoint: Path, registry_path: Path, expected_years: int) -> bool:
    if not checkpoint.exists() or not registry_path.exists():
        return False
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        registry = pd.read_csv(registry_path)
        required_columns = {
            "mask_path",
            "cumulative_mask_path",
            "model_only_cumulative_mask_path",
            "anchor_backcast_path",
            "anchor_forced_reference_path",
            "anchor_backcast_object_fill_path",
            "anchor_reference_path",
        }
        return (
            payload.get("status") == "passed"
            and payload.get("schema_version") == "24.15.0"
            and len(registry) == expected_years
            and required_columns.issubset(registry.columns)
            and all(Path(value).exists() for value in registry["mask_path"].astype(str))
            and all(Path(value).exists() for value in registry["cumulative_mask_path"].astype(str))
            and all(Path(value).exists() for value in registry["model_only_cumulative_mask_path"].astype(str))
            and all(Path(value).exists() for value in registry["anchor_backcast_path"].astype(str))
            and all(Path(value).exists() for value in registry["anchor_forced_reference_path"].astype(str))
            and all(Path(value).exists() for value in registry["anchor_backcast_object_fill_path"].astype(str))
            and all(Path(value).exists() for value in registry["anchor_reference_path"].astype(str))
        )
    except Exception:
        return False


def process_unit(cfg: dict[str, Any], region_id: int, *, overwrite: bool = False) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "08_temporal_pv_change")
    years = np.arange(
        int(cfg["study"]["start_year"]),
        int(cfg["study"]["end_year"]) + 1,
        dtype=int,
    )
    mask_nodata = int(cfg.get("temporal_decoding", {}).get("mask_nodata", 255))

    inference = pd.read_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv")
    group = inference.loc[inference["Id"].eq(int(region_id))].sort_values("year")
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv")
    aligned_group = aligned.loc[aligned["Id"].eq(int(region_id))].sort_values("year")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")

    if len(group) != len(years) or len(aligned_group) != len(years):
        raise RuntimeError(
            f"Id {region_id}: expected {len(years)} annual probability and Landsat records; "
            f"found {len(group)} and {len(aligned_group)}"
        )

    mode = str(group.iloc[0].get("inference_mode", "direct_pixel"))
    route = str(group.iloc[0].get("estimation_route", mode))
    if mode not in {"direct_pixel", "anchor_guided_candidate"}:
        raise RuntimeError(f"Id {region_id}: {mode} is not a pixel-mapping route")

    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    unit_registry_root = ensure_dir(out / "unit_registries")
    checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
    registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
    if not overwrite and _valid_existing_unit(checkpoint, registry_path, len(years)):
        return pd.read_csv(registry_path)

    probability_paths = [Path(value) for value in group["calibrated_probability_path"]]
    landsat_paths = [Path(value) for value in aligned_group["path"]]
    fraction_path = Path(str(supervision.loc[region_id, "pv_fraction_path"]))
    configured_support = supervision.loc[region_id].get("anchor_support_path", "")
    support_path = (
        Path(str(configured_support))
        if isinstance(configured_support, str) and configured_support
        else root / "03_supervision" / "masks" / f"unit_{region_id:03d}_anchor_support_30m.tif"
    )
    endmembers = load_endmember_matrix(quality.loc[region_id, "endmember_path"])

    posterior_root = ensure_dir(out / "harmonized_probability_30m" / f"unit_{region_id:03d}")
    active_root = ensure_dir(out / "observed_active_pv_masks_30m" / f"unit_{region_id:03d}")
    cumulative_root = ensure_dir(out / "cumulative_installed_pv_masks_30m" / f"unit_{region_id:03d}")
    model_only_cumulative_root = ensure_dir(out / "model_only_cumulative_installed_pv_masks_30m" / f"unit_{region_id:03d}")
    onset_root = ensure_dir(out / "first_installed_year_30m" / f"unit_{region_id:03d}")
    model_only_onset_root = ensure_dir(out / "model_only_first_installed_year_30m" / f"unit_{region_id:03d}")
    retirement_root = ensure_dir(out / "first_retirement_year_30m" / f"unit_{region_id:03d}")
    score_root = ensure_dir(out / "transition_score_30m" / f"unit_{region_id:03d}")
    confuser_root = ensure_dir(out / "confuser_penalty_30m" / f"unit_{region_id:03d}")
    diagnostic_root = ensure_dir(out / "decision_diagnostics_30m" / f"unit_{region_id:03d}")
    anchor_audit_root = ensure_dir(out / "anchor_constraint_diagnostics_30m" / f"unit_{region_id:03d}")

    probability_out = [
        posterior_root / f"unit_{region_id:03d}_{year}_pv_probability_harmonized_30m.tif"
        for year in years
    ]
    active_out = [
        active_root / f"unit_{region_id:03d}_{year}_pv_active_footprint_30m.tif"
        for year in years
    ]
    cumulative_out = [
        cumulative_root / f"unit_{region_id:03d}_{year}_pv_cumulative_installed_30m.tif"
        for year in years
    ]
    model_only_cumulative_out = [
        model_only_cumulative_root / f"unit_{region_id:03d}_{year}_pv_model_only_cumulative_installed_30m.tif"
        for year in years
    ]
    onset_path = onset_root / f"unit_{region_id:03d}_first_installed_year_30m.tif"
    model_only_onset_path = model_only_onset_root / f"unit_{region_id:03d}_model_only_first_installed_year_30m.tif"
    retirement_path = retirement_root / f"unit_{region_id:03d}_first_retirement_year_30m.tif"
    score_path = score_root / f"unit_{region_id:03d}_transition_score_30m.tif"
    confuser_path = confuser_root / f"unit_{region_id:03d}_confuser_penalty_30m.tif"
    candidate_path = diagnostic_root / f"unit_{region_id:03d}_candidate_before_morphology_30m.tif"
    morphology_removed_path = diagnostic_root / f"unit_{region_id:03d}_morphology_removed_30m.tif"
    valid_count_path = diagnostic_root / f"unit_{region_id:03d}_valid_year_count_30m.tif"
    exclusion_path = diagnostic_root / f"unit_{region_id:03d}_temporal_exclusion_code_30m.tif"
    anchor_backcast_path = anchor_audit_root / f"unit_{region_id:03d}_anchor_backcast_supported_30m.tif"
    anchor_forced_reference_path = anchor_audit_root / f"unit_{region_id:03d}_anchor_forced_reference_only_30m.tif"
    anchor_backcast_object_fill_path = anchor_audit_root / f"unit_{region_id:03d}_anchor_backcast_object_consensus_fill_30m.tif"
    anchor_reference_path = anchor_audit_root / f"unit_{region_id:03d}_reference_anchor_inclusion_30m.tif"

    temporal_settings = cfg.get("temporal_signature", {})
    processing_tile_size = int(temporal_settings.get("processing_tile_size", 128))
    processing_halo = int(temporal_settings.get("processing_halo_pixels", 24))
    compress = str(cfg.get("processing", {}).get("compress", "DEFLATE"))
    anchor_constraint = cfg.get("anchor_constraint", {})
    anchor_inclusion_fraction = float(anchor_constraint.get("reference_inclusion_fraction", 0.05))

    support_path = _ensure_anchor_support(
        fraction_path,
        support_path,
        float(temporal_settings.get("reference_anchor_search_radius_m", 3000.0)),
        compress,
    )

    with rasterio.open(probability_paths[0], sharing=False) as reference:
        width = int(reference.width)
        height = int(reference.height)
        transform = reference.transform
        pixel_area_km2 = abs(float(transform.a * transform.e)) / 1_000_000.0
        unit_total_pixels = int(width * height)
        float_profile = reference.profile.copy()
        float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=compress, BIGTIFF="IF_SAFER")
        mask_profile = reference.profile.copy()
        mask_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=compress, BIGTIFF="IF_SAFER")
        byte_profile = reference.profile.copy()
        byte_profile.update(count=1, dtype="uint8", nodata=255, compress=compress, BIGTIFF="IF_SAFER")
        int_profile = reference.profile.copy()
        int_profile.update(count=1, dtype="int16", nodata=-32768, compress=compress, BIGTIFF="IF_SAFER")

    biases, stable_pixels = _estimate_unit_biases(
        probability_paths,
        width,
        height,
        cfg,
        int(cfg.get("project", {}).get("random_seed", 2027)) + int(region_id),
    )

    all_paths = [
        *probability_out,
        *active_out,
        *cumulative_out,
        *model_only_cumulative_out,
        onset_path,
        model_only_onset_path,
        retirement_path,
        score_path,
        confuser_path,
        candidate_path,
        morphology_removed_path,
        valid_count_path,
        exclusion_path,
        anchor_backcast_path,
        anchor_forced_reference_path,
        anchor_backcast_object_fill_path,
        anchor_reference_path,
    ]
    for output_path in all_paths:
        output_path.unlink(missing_ok=True)

    active_pixels = np.zeros(len(years), dtype=np.int64)
    cumulative_pixels = np.zeros(len(years), dtype=np.int64)
    model_only_cumulative_pixels = np.zeros(len(years), dtype=np.int64)
    annual_valid = np.zeros(len(years), dtype=np.int64)
    transition_pixels = 0
    baseline_pixels = 0
    recent_pixels = 0
    morphology_removed_pixels = 0
    anchor_reference_pixels = 0
    anchor_backcast_pixels = 0
    anchor_forced_reference_pixels = 0
    anchor_backcast_object_fill_pixels = 0

    feature_channels = [
        "PV_Fraction_Evidence",
        "Water_Evidence",
        "Vegetation_Evidence",
        "Builtup_Evidence",
        "Bare_Evidence",
        "PV_Confuser_Penalty",
    ]
    reference_index = int(np.clip(int(cfg.get("study", {}).get("reference_year", years[-2])) - int(years[0]), 0, len(years) - 1))

    gdal_cache = int(cfg.get("resource_control", {}).get("gdal_cache_mb", 32))
    with rasterio.Env(GDAL_CACHEMAX=gdal_cache):
        with ExitStack() as outputs:
            probability_dsts = [outputs.enter_context(rasterio.open(p, "w", **float_profile)) for p in probability_out]
            active_dsts = [outputs.enter_context(rasterio.open(p, "w", **mask_profile)) for p in active_out]
            cumulative_dsts = [outputs.enter_context(rasterio.open(p, "w", **mask_profile)) for p in cumulative_out]
            model_only_cumulative_dsts = [outputs.enter_context(rasterio.open(p, "w", **mask_profile)) for p in model_only_cumulative_out]
            onset_dst = outputs.enter_context(rasterio.open(onset_path, "w", **int_profile))
            model_only_onset_dst = outputs.enter_context(rasterio.open(model_only_onset_path, "w", **int_profile))
            retirement_dst = outputs.enter_context(rasterio.open(retirement_path, "w", **int_profile))
            score_dst = outputs.enter_context(rasterio.open(score_path, "w", **float_profile))
            confuser_dst = outputs.enter_context(rasterio.open(confuser_path, "w", **float_profile))
            candidate_dst = outputs.enter_context(rasterio.open(candidate_path, "w", **byte_profile))
            morphology_dst = outputs.enter_context(rasterio.open(morphology_removed_path, "w", **byte_profile))
            valid_count_dst = outputs.enter_context(rasterio.open(valid_count_path, "w", **byte_profile))
            exclusion_dst = outputs.enter_context(rasterio.open(exclusion_path, "w", **byte_profile))
            anchor_backcast_dst = outputs.enter_context(rasterio.open(anchor_backcast_path, "w", **byte_profile))
            anchor_forced_reference_dst = outputs.enter_context(rasterio.open(anchor_forced_reference_path, "w", **byte_profile))
            anchor_backcast_object_fill_dst = outputs.enter_context(rasterio.open(anchor_backcast_object_fill_path, "w", **byte_profile))
            anchor_reference_dst = outputs.enter_context(rasterio.open(anchor_reference_path, "w", **byte_profile))

            core_windows = list(_core_windows(width, height, processing_tile_size))
            for core_window in tqdm(
                core_windows,
                desc=f"Unit {region_id:03d} temporal tiles",
                unit="tile",
                leave=False,
                disable=not cfg.get("processing", {}).get("show_progress", True),
            ):
                window, crop = _expanded_window(core_window, width, height, processing_halo)
                probability, valid = _read_probability_cube(probability_paths, window)
                harmonized = _apply_biases(probability, valid, biases)
                fraction_evidence = np.full_like(harmonized, np.nan, dtype=np.float32)
                confuser = np.full_like(harmonized, np.nan, dtype=np.float32)
                reflectance_cube = np.full((len(years), 6, *harmonized.shape[1:]), np.nan, dtype=np.float32)

                with rasterio.open(fraction_path, sharing=False) as fraction_src:
                    anchor_fraction = fraction_src.read(1, window=window).astype(np.float32)

                for index, landsat_path in enumerate(landsat_paths):
                    with rasterio.open(landsat_path, sharing=False) as src:
                        reflectance = src.read([1, 2, 3, 4, 5, 6], window=window).astype(np.float32)
                    reflectance[reflectance <= -9990.0] = np.nan
                    reflectance_cube[index] = reflectance
                    features = compute_features(reflectance, feature_channels, False, endmembers=endmembers)
                    fraction_evidence[index] = features[0]
                    explicit = _safe_nanmax(features[1:5], axis=0)
                    confuser[index] = np.clip(0.65 * explicit + 0.35 * features[5], 0.0, 1.0)
                    del reflectance, features, explicit
                # All materially intersected SHP pixels are certain PV in 2024.
                # Near-anchor threshold relaxation is restricted to a local
                # topology envelope instead of the former broad support mask.
                anchor_reference = np.isfinite(anchor_fraction) & (anchor_fraction >= anchor_inclusion_fraction)
                spectral_similarity = compute_anchor_reference_spectral_similarity(
                    reflectance_cube, anchor_reference, valid, reference_index, cfg
                )
                near_radius_m = float(cfg.get("spatial_topology", {}).get("near_anchor_radius_m", 600.0))
                pixel_size_m = float(cfg.get("native_grid", {}).get("target_resolution_m", 30.0))
                near_radius_pixels = max(0, int(math.ceil(near_radius_m / max(pixel_size_m, 1e-6))))
                if np.any(anchor_reference) and near_radius_pixels > 0:
                    near_anchor = ndimage.binary_dilation(
                        anchor_reference,
                        structure=np.ones((3, 3), dtype=np.uint8),
                        iterations=near_radius_pixels,
                    )
                else:
                    near_anchor = anchor_reference.copy()

                result = detect_temporal_signature(
                    harmonized,
                    fraction_evidence,
                    confuser,
                    valid,
                    near_anchor,
                    cfg,
                    anchor_core=anchor_reference,
                )
                confuser_median_full = _safe_nanmedian(confuser, axis=0)
                model_only_onset, anchor_backcast_combined, anchor_backcast_object_fill = (
                    propagate_anchor_backcast_within_objects(
                        result.onset_index,
                        result.anchor_backcast,
                        anchor_reference,
                        result.transition_score,
                        cfg,
                    )
                )
                model_candidate_raw = result.qualified | anchor_backcast_object_fill
                initial_component_filter = filter_transition_components(
                    model_candidate_raw,
                    near_anchor,
                    cfg,
                    transition_score=result.transition_score,
                    reference_probability=harmonized[reference_index],
                    confuser_penalty=confuser_median_full,
                    anchor_core=anchor_reference,
                )
                component_removed = model_candidate_raw & ~initial_component_filter & ~anchor_reference

                model_only_onset, topology_qualified, topology_removed = enforce_spatiotemporal_topology(
                    model_only_onset,
                    initial_component_filter,
                    anchor_reference,
                    near_anchor,
                    result.transition_score,
                    harmonized,
                    fraction_evidence,
                    confuser_median_full,
                    cfg,
                )

                model_only_onset, model_only_qualified, anchor_internal_rescue = (
                    rescue_anchor_internal_onset_from_trajectory(
                        model_only_onset,
                        topology_qualified,
                        anchor_reference,
                        harmonized,
                        fraction_evidence,
                        confuser,
                        valid,
                        cfg,
                        spectral_similarity=spectral_similarity,
                    )
                )
                model_only_onset, model_only_qualified, cbc_anchor_rescue, cbc_anchor_score = (
                    refine_anchor_onset_with_change_boundary_coherence(
                        model_only_onset,
                        model_only_qualified,
                        anchor_reference,
                        harmonized,
                        fraction_evidence,
                        confuser,
                        valid,
                        reflectance_cube,
                        cfg,
                        spectral_similarity=spectral_similarity,
                    )
                )
                anchor_internal_rescue |= cbc_anchor_rescue
                result.transition_score[:] = np.maximum(result.transition_score, cbc_anchor_score)
                model_only_onset, model_only_qualified, object_anchor_rescue, object_anchor_score = (
                    refine_anchor_onset_with_landsat_object_pavg(
                        model_only_onset,
                        model_only_qualified,
                        anchor_reference,
                        harmonized,
                        fraction_evidence,
                        confuser,
                        valid,
                        reflectance_cube,
                        cfg,
                        spectral_similarity=spectral_similarity,
                    )
                )
                anchor_internal_rescue |= object_anchor_rescue
                result.transition_score[:] = np.maximum(result.transition_score, object_anchor_score)
                morphology_removed = (component_removed | topology_removed) & ~anchor_reference
                model_only_cumulative = annual_latent_masks_from_onset(
                    model_only_onset, model_only_qualified, len(years)
                )

                final_onset = model_only_onset.copy()
                anchor_with_change_evidence = (
                    anchor_reference
                    & model_only_qualified
                    & (model_only_onset >= 0)
                    & (model_only_onset <= reference_index)
                )
                anchor_forced_reference = anchor_reference & ~anchor_with_change_evidence
                final_onset[anchor_forced_reference] = reference_index
                final_qualified = model_only_qualified | anchor_reference

                cumulative = annual_latent_masks_from_onset(final_onset, final_qualified, len(years))
                # Annual activity uses the final installed state. The 2024 SHP
                # is the reference seed for the 2025 observation, while actual
                # 2025 probability, LSMA, confuser and spectral similarity are
                # evaluated. One post-reference year cannot imply retirement.
                latent_active, observed_active, retirement_index = annual_active_masks_from_evidence(
                    final_onset,
                    final_qualified,
                    harmonized,
                    fraction_evidence,
                    valid,
                    cfg,
                    anchor_reference=anchor_reference,
                    spectral_similarity=spectral_similarity,
                )
                if bool(anchor_constraint.get("force_reference_anchor_into_active", True)):
                    latent_active[reference_index] |= anchor_reference
                    observed_active[reference_index] |= anchor_reference

                if np.any(anchor_reference & ~cumulative[reference_index]):
                    raise RuntimeError(
                        f"Id {region_id}: reference anchor is missing from the reference-year cumulative installed footprint"
                    )
                if np.any(latent_active & ~cumulative) or np.any(observed_active & ~cumulative):
                    raise RuntimeError(
                        f"Id {region_id}: active PV state exceeds cumulative installed state"
                    )

                core_valid = valid[(slice(None),) + crop]
                core_harmonized = harmonized[(slice(None),) + crop]
                core_cumulative = cumulative[(slice(None),) + crop]
                core_model_only_cumulative = model_only_cumulative[(slice(None),) + crop]
                core_active = latent_active[(slice(None),) + crop]
                core_observed_active = observed_active[(slice(None),) + crop]
                core_onset = final_onset[crop]
                core_model_only_onset = model_only_onset[crop]
                core_retirement = retirement_index[crop]
                core_qualified = final_qualified[crop]
                core_model_only_qualified = model_only_qualified[crop]
                core_baseline = result.baseline_preexisting[crop]
                core_recent = result.recent_right_censored[crop]
                core_anchor_reference = anchor_reference[crop]
                core_anchor_backcast = (anchor_reference & anchor_backcast_combined)[crop]
                core_anchor_backcast_object_fill = anchor_backcast_object_fill[crop]
                core_anchor_forced_reference = anchor_forced_reference[crop]
                core_ever_valid = np.any(core_valid, axis=0) | core_anchor_reference

                onset_year = np.zeros(core_qualified.shape, dtype=np.int16)
                model_only_onset_year = np.zeros(core_qualified.shape, dtype=np.int16)
                retirement_year = np.zeros(core_qualified.shape, dtype=np.int16)
                for index, year in enumerate(years):
                    onset_year[(core_onset == index) & core_qualified] = int(year)
                    model_only_onset_year[(core_model_only_onset == index) & core_model_only_qualified] = int(year)
                    retirement_year[core_retirement == index] = int(year)
                onset_encoded = np.where(core_ever_valid, onset_year, -32768).astype(np.int16)
                model_only_onset_encoded = np.where(core_ever_valid, model_only_onset_year, -32768).astype(np.int16)
                retirement_encoded = np.where(core_ever_valid, retirement_year, -32768).astype(np.int16)
                score_encoded = np.where(core_ever_valid, result.transition_score[crop], -9999.0).astype(np.float32)
                confuser_encoded = np.where(core_ever_valid, confuser_median_full[crop], -9999.0).astype(np.float32)
                candidate_encoded = np.where(core_ever_valid, model_candidate_raw[crop].astype(np.uint8), 255).astype(np.uint8)
                morphology_encoded = np.where(core_ever_valid, morphology_removed[crop].astype(np.uint8), 255).astype(np.uint8)
                valid_count_encoded = np.where(core_ever_valid, result.valid_year_count[crop], 255).astype(np.uint8)
                anchor_backcast_encoded = np.where(core_ever_valid, core_anchor_backcast.astype(np.uint8), 255).astype(np.uint8)
                anchor_forced_encoded = np.where(core_ever_valid, core_anchor_forced_reference.astype(np.uint8), 255).astype(np.uint8)
                anchor_backcast_object_fill_encoded = np.where(core_ever_valid, core_anchor_backcast_object_fill.astype(np.uint8), 255).astype(np.uint8)
                anchor_reference_encoded = np.where(core_ever_valid, core_anchor_reference.astype(np.uint8), 255).astype(np.uint8)
                exclusion_core = result.exclusion_code[crop].copy()
                exclusion_core[morphology_removed[crop]] = 13
                exclusion_encoded = np.where(core_ever_valid, exclusion_core, 255).astype(np.uint8)

                onset_dst.write(onset_encoded, 1, window=core_window)
                model_only_onset_dst.write(model_only_onset_encoded, 1, window=core_window)
                retirement_dst.write(retirement_encoded, 1, window=core_window)
                score_dst.write(score_encoded, 1, window=core_window)
                confuser_dst.write(confuser_encoded, 1, window=core_window)
                candidate_dst.write(candidate_encoded, 1, window=core_window)
                morphology_dst.write(morphology_encoded, 1, window=core_window)
                valid_count_dst.write(valid_count_encoded, 1, window=core_window)
                exclusion_dst.write(exclusion_encoded, 1, window=core_window)
                anchor_backcast_dst.write(anchor_backcast_encoded, 1, window=core_window)
                anchor_forced_reference_dst.write(anchor_forced_encoded, 1, window=core_window)
                anchor_backcast_object_fill_dst.write(anchor_backcast_object_fill_encoded, 1, window=core_window)
                anchor_reference_dst.write(anchor_reference_encoded, 1, window=core_window)

                transition_pixels += int(np.count_nonzero(core_model_only_qualified & ~core_baseline))
                baseline_pixels += int(np.count_nonzero(core_qualified & core_baseline))
                recent_pixels += int(np.count_nonzero(core_qualified & core_recent))
                morphology_removed_pixels += int(np.count_nonzero(morphology_removed[crop]))
                anchor_reference_pixels += int(np.count_nonzero(core_anchor_reference))
                anchor_backcast_pixels += int(np.count_nonzero(core_anchor_backcast))
                anchor_forced_reference_pixels += int(np.count_nonzero(core_anchor_forced_reference))
                anchor_backcast_object_fill_pixels += int(np.count_nonzero(core_anchor_backcast_object_fill))

                for index in range(len(years)):
                    probability_encoded = np.where(core_valid[index], core_harmonized[index], -9999.0).astype(np.float32)
                    probability_dsts[index].write(probability_encoded, 1, window=core_window)

                    active_encoded = np.full(core_qualified.shape, mask_nodata, dtype=np.uint8)
                    active_write_support = core_valid[index] | (
                        core_anchor_reference if index == reference_index else False
                    )
                    active_encoded[active_write_support] = core_active[index][active_write_support].astype(np.uint8)
                    active_dsts[index].write(active_encoded, 1, window=core_window)

                    cumulative_encoded = np.full(core_qualified.shape, mask_nodata, dtype=np.uint8)
                    cumulative_encoded[core_ever_valid] = core_cumulative[index][core_ever_valid].astype(np.uint8)
                    cumulative_dsts[index].write(cumulative_encoded, 1, window=core_window)

                    model_only_cumulative_encoded = np.full(core_qualified.shape, mask_nodata, dtype=np.uint8)
                    model_only_cumulative_encoded[core_ever_valid] = core_model_only_cumulative[index][core_ever_valid].astype(np.uint8)
                    model_only_cumulative_dsts[index].write(model_only_cumulative_encoded, 1, window=core_window)

                    active_pixels[index] += int(np.count_nonzero(core_active[index]))
                    cumulative_pixels[index] += int(np.count_nonzero(core_cumulative[index]))
                    model_only_cumulative_pixels[index] += int(np.count_nonzero(core_model_only_cumulative[index]))
                    annual_valid[index] += int(np.count_nonzero(core_valid[index] | (core_anchor_reference if index == reference_index else False)))

                del (
                    probability, valid, harmonized, fraction_evidence, confuser, reflectance_cube, spectral_similarity,
                    anchor_fraction, anchor_reference, near_anchor, result, model_only_onset,
                    anchor_backcast_combined, anchor_backcast_object_fill, model_candidate_raw, initial_component_filter, topology_qualified, topology_removed, anchor_internal_rescue, cbc_anchor_rescue, cbc_anchor_score, model_only_qualified,
                    anchor_with_change_evidence, anchor_forced_reference, final_onset, final_qualified,
                    morphology_removed, model_only_cumulative, cumulative, latent_active, observed_active,
                    retirement_index, core_valid, core_harmonized, core_cumulative, core_model_only_cumulative,
                    core_active, core_observed_active, core_onset, core_model_only_onset, core_retirement,
                    core_qualified, core_model_only_qualified, core_baseline, core_recent,
                    core_anchor_reference, core_anchor_backcast, core_anchor_backcast_object_fill,
                    core_anchor_forced_reference, core_ever_valid,
                    onset_year, model_only_onset_year, retirement_year, onset_encoded,
                    model_only_onset_encoded, retirement_encoded, score_encoded, confuser_encoded, candidate_encoded,
                    morphology_encoded, valid_count_encoded, anchor_backcast_encoded, anchor_forced_encoded,
                    anchor_backcast_object_fill_encoded, anchor_reference_encoded, exclusion_encoded, exclusion_core, confuser_median_full,
                    cumulative_encoded, model_only_cumulative_encoded, active_encoded, active_write_support, probability_encoded,
                )

    inferred_deployment_year = (
        int(years[np.argmax(np.diff(np.r_[0, cumulative_pixels]))])
        if cumulative_pixels.max(initial=0) > 0
        else np.nan
    )
    rows: list[dict[str, Any]] = []
    for index, year in enumerate(years):
        active_area = float(active_pixels[index] * pixel_area_km2)
        cumulative_area = float(cumulative_pixels[index] * pixel_area_km2)
        model_only_cumulative_area = float(model_only_cumulative_pixels[index] * pixel_area_km2)
        rows.append({
            "Id": int(region_id),
            "year": int(year),
            "posterior_path": str(probability_out[index]),
            "mask_path": str(active_out[index]),
            "active_mask_path": str(active_out[index]),
            "cumulative_mask_path": str(cumulative_out[index]),
            "model_only_cumulative_mask_path": str(model_only_cumulative_out[index]),
            "pv_pixels": int(active_pixels[index]),
            "active_pv_pixels": int(active_pixels[index]),
            "cumulative_installed_pixels": int(cumulative_pixels[index]),
            "model_only_cumulative_installed_pixels": int(model_only_cumulative_pixels[index]),
            "valid_pixels": int(annual_valid[index]),
            "pv_area_km2": active_area,
            "observed_active_area_km2": active_area,
            "cumulative_installed_area_km2": cumulative_area,
            "model_only_cumulative_installed_area_km2": model_only_cumulative_area,
            "anchor_constrained_added_area_km2": max(cumulative_area - model_only_cumulative_area, 0.0),
            "anchor_constrained_removed_area_km2": max(model_only_cumulative_area - cumulative_area, 0.0),
            "pv_area_latent_persistent_km2": cumulative_area,
            "inactive_or_retired_area_km2": max(cumulative_area - active_area, 0.0),
            "observation_coverage_fraction": float(annual_valid[index] / max(unit_total_pixels, 1)),
            "change_year_path": str(onset_path),
            "model_only_change_year_path": str(model_only_onset_path),
            "retirement_year_path": str(retirement_path),
            "transition_score_path": str(score_path),
            "confuser_penalty_path": str(confuser_path),
            "temporal_candidate_path": str(candidate_path),
            "morphology_removed_path": str(morphology_removed_path),
            "valid_year_count_path": str(valid_count_path),
            "temporal_exclusion_code_path": str(exclusion_path),
            "anchor_backcast_path": str(anchor_backcast_path),
            "anchor_forced_reference_path": str(anchor_forced_reference_path),
            "anchor_backcast_object_fill_path": str(anchor_backcast_object_fill_path),
            "anchor_reference_path": str(anchor_reference_path),
            "anchor_constraint_applied": True,
            "calibrated_threshold": float(group.iloc[index].get("calibrated_threshold", np.nan)),
            "posterior_threshold": float(temporal_settings.get("minimum_transition_score", 0.58)),
            "estimation_mode": mode,
            "estimation_route": route,
            "pv_area_ci_low_km2": np.nan,
            "pv_area_ci_high_km2": np.nan,
            "confirmed_anchor_area_km2": float(supervision.loc[region_id, "confirmed_anchor_area_km2"]),
            "inferred_deployment_year": inferred_deployment_year,
            "annual_probability_bias": float(biases[index]),
            "area_state_definition": "observed_active",
        })

    frame = pd.DataFrame(rows)
    save_table(frame, registry_path)
    peak_rss_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mb = peak_rss_raw / 1024.0 if os.name == "posix" else peak_rss_raw / (1024.0**2)
    atomic_json({
        "status": "passed",
        "schema_version": "24.15.0",
        "Id": int(region_id),
        "records": int(len(frame)),
        "transition_pixels": int(transition_pixels),
        "baseline_preexisting_pixels": int(baseline_pixels),
        "recent_right_censored_pixels": int(recent_pixels),
        "morphology_removed_pixels": int(morphology_removed_pixels),
        "anchor_reference_pixels": int(anchor_reference_pixels),
        "anchor_backcast_pixels": int(anchor_backcast_pixels),
        "anchor_forced_reference_pixels": int(anchor_forced_reference_pixels),
        "anchor_backcast_object_fill_pixels": int(anchor_backcast_object_fill_pixels),
        "anchor_backcast_fraction": float(anchor_backcast_pixels / max(anchor_reference_pixels, 1)),
        "anchor_forced_reference_fraction": float(anchor_forced_reference_pixels / max(anchor_reference_pixels, 1)),
        "harmonization_stable_pixels": int(stable_pixels),
        "annual_probability_bias": biases.tolist(),
        "processing_tile_size": int(processing_tile_size),
        "processing_halo_pixels": int(processing_halo),
        "peak_worker_rss_mb": float(peak_rss_mb),
        "anchor_support_path": str(support_path),
        "reference_shp_loaded_in_stage8": False,
        "reference_anchor_forced_in_reference_year": True,
        "reference_anchor_copied_to_historical_years_without_change_evidence": False,
        "model_only_diagnostic_product_retained": True,
        "area_accounting": "anchor-constrained cumulative installed footprint plus observed active footprint; model-only product retained",
        "created_utc": utc_now(),
    }, checkpoint)
    _release_native_memory()
    return frame

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--unit-id", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process_unit(load_config(args.config), int(args.unit_id), overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
