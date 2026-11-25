from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator

import joblib
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy import ndimage
from tqdm import tqdm

from .anchor_guided import (
    anchor_mask_for_window,
    anchor_similarity_matrix,
    buffered_support_mask,
    build_anchor_prototype,
    classify_anchor_guided,
    select_anchor_geometry,
)
from .ccdc import causal_ccdc_features
from .features import compute_features
from .spectral_mixture import load_endmember_matrix
from .routes import PIXEL_MODES, is_pixel_mode
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _cleanup(mask: np.ndarray, min_component: int, max_hole: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= int(min_component)
        keep[0] = False
        mask = keep[labels]
    if max_hole > 0:
        labels, count = ndimage.label(~mask)
        if count:
            sizes = np.bincount(labels.ravel())
            fill = sizes <= int(max_hole)
            fill[0] = False
            mask = mask | fill[labels]
    return mask


def _core_windows(width: int, height: int, tile_size: int) -> Iterator[Window]:
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            yield Window(col_off, row_off, min(tile_size, width - col_off), min(tile_size, height - row_off))


def _expanded_window(core: Window, width: int, height: int, overlap: int) -> tuple[Window, tuple[slice, slice]]:
    col0 = max(0, int(core.col_off) - overlap)
    row0 = max(0, int(core.row_off) - overlap)
    col1 = min(width, int(core.col_off + core.width) + overlap)
    row1 = min(height, int(core.row_off + core.height) + overlap)
    expanded = Window(col0, row0, col1 - col0, row1 - row0)
    r0 = int(core.row_off) - row0
    c0 = int(core.col_off) - col0
    crop = (slice(r0, r0 + int(core.height)), slice(c0, c0 + int(core.width)))
    return expanded, crop


def _effective_tile_size(cfg: dict[str, Any], years: int, static_features: int, ccdc_features: int) -> int:
    requested = int(cfg["inference"]["tile_size"])
    overlap = int(cfg["inference"]["tile_overlap"])
    budget = float(cfg["processing"].get("maximum_working_memory_gb", 3.0)) * (1024 ** 3)
    # Conservative estimate: static features, CCDC descriptors, source reflectance, model matrix and predictions.
    bytes_per_pixel = years * (static_features + ccdc_features + 6 + 4) * 4 * 1.35
    maximum_full_side = int(math.sqrt(max(budget / max(bytes_per_pixel, 1.0), 64 ** 2)))
    effective = min(requested, max(64, maximum_full_side - 2 * overlap))
    return max(32, effective)


def _valid_region_checkpoint(checkpoint: Path, expected_years: int) -> pd.DataFrame | None:
    if not checkpoint.exists():
        return None
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        registry_path = Path(payload["registry_path"])
        if payload.get("status") != "passed" or not registry_path.exists():
            return None
        registry = pd.read_csv(registry_path)
        if len(registry) != expected_years:
            return None
        if "inference_mode" in registry and registry["inference_mode"].astype(str).eq("area_only_fallback").all():
            return registry
        required = ["probability_path", "epistemic_std_path", "raw_mask_path"]
        if not all(Path(str(path)).exists() and Path(str(path)).stat().st_size > 0 for col in required for path in registry[col]):
            return None
        return registry
    except Exception:
        return None



def _configure_model_threads(models: list[Any], requested_threads: int) -> int:
    """Set estimator prediction threads without changing fitted parameters."""
    threads = max(1, int(requested_threads))
    for model in models:
        try:
            params = model.get_params(deep=False)
        except Exception:
            params = {}
        if "n_jobs" in params:
            try:
                model.set_params(n_jobs=threads)
            except Exception:
                try:
                    model.n_jobs = threads
                except Exception:
                    pass
    return threads


def _predict_probability_chunked(model: Any, matrix: np.ndarray, batch_rows: int) -> np.ndarray:
    """Predict positive-class probability in bounded row batches."""
    row_count = int(matrix.shape[0])
    if row_count == 0:
        return np.empty(0, dtype=np.float32)
    batch_rows = max(1, int(batch_rows))
    if row_count <= batch_rows:
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float32)
    output = np.empty(row_count, dtype=np.float32)
    for start in range(0, row_count, batch_rows):
        stop = min(row_count, start + batch_rows)
        output[start:stop] = np.asarray(model.predict_proba(matrix[start:stop])[:, 1], dtype=np.float32)
    return output


def _predict_weighted_ensemble(
    models: list[Any],
    weights: np.ndarray,
    matrix: np.ndarray,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted ensemble mean/std without retaining a member-by-row cube."""
    row_count = int(matrix.shape[0])
    if row_count == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty
    mean = np.zeros(row_count, dtype=np.float64)
    second = np.zeros(row_count, dtype=np.float64)
    for model, weight in zip(models, weights, strict=True):
        probability = _predict_probability_chunked(model, matrix, batch_rows).astype(np.float64, copy=False)
        mean += float(weight) * probability
        second += float(weight) * probability * probability
    variance = np.maximum(second - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def infer_annual_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "07_annual_pv_inference")
    prob_root = ensure_dir(out / "probability_30m")
    std_root = ensure_dir(out / "epistemic_std_30m")
    mask_root = ensure_dir(out / "raw_masks_30m")
    similarity_root = ensure_dir(out / "anchor_similarity_30m")
    candidate_score_root = ensure_dir(out / "anchor_candidate_score_30m")
    candidate_class_root = ensure_dir(out / "anchor_candidate_class_30m")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    unit_registry_root = ensure_dir(out / "unit_registries")

    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    reference_shape = gpd.read_file(cfg["paths"]["reference_shp"])
    manifest = pd.read_csv(root / "06_final_models" / "final_ensemble_manifest.csv")
    models = [joblib.load(path) for path in manifest.model_path]
    weights = manifest["weight"].to_numpy(dtype=np.float64)
    weights = weights / weights.sum()
    inference_cfg = cfg.get("inference", {})
    configured_threads = int(inference_cfg.get("inference_threads", cfg["processing"].get("workers", 4)))
    inference_threads = _configure_model_threads(models, configured_threads)
    prediction_batch_rows = int(inference_cfg.get("prediction_batch_rows", 1_500_000))
    skip_invalid_pixels = bool(inference_cfg.get("skip_invalid_pixels", True))

    final_status = json.loads((root / "06_final_models" / "final_model_status.json").read_text(encoding="utf-8"))
    threshold = float(final_status["threshold"])
    expected_feature_count = int(final_status["feature_count"])
    channels = list(cfg["features"]["channels"])
    add_valid = bool(cfg["features"].get("add_valid_mask_channel", True))
    ccdc_names = list(cfg["ccdc"]["indices"])
    ccdc_indices = [channels.index(name) for name in ccdc_names]
    expected_years = int(cfg["study"]["end_year"]) - int(cfg["study"]["start_year"]) + 1
    all_rows: list[pd.DataFrame] = []

    grouped = list(aligned.groupby("Id"))
    for region_id, group in tqdm(
        grouped,
        desc="Bounded-memory annual PV inference",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        region_started = time.perf_counter()
        region_id = int(region_id)
        group = group.sort_values("year")
        if len(group) != expected_years:
            raise RuntimeError(f"Id {region_id}: inference requires {expected_years} years, found {len(group)}")
        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        if not cfg["processing"].get("overwrite", False):
            existing = _valid_region_checkpoint(checkpoint, expected_years)
            if existing is not None:
                all_rows.append(existing)
                continue

        years = group.year.astype(int).to_numpy()
        paths = [Path(path) for path in group.path]
        unit_rows: list[dict[str, Any]] = []
        if region_id not in quality.index:
            raise RuntimeError(f"Id {region_id}: unit quality route missing")
        route = str(quality.loc[region_id, "estimation_route"])
        pixel_eligible = route in PIXEL_MODES
        guided = route == "anchor_guided_candidate"
        endmembers = load_endmember_matrix(quality.loc[region_id, "endmember_path"])
        prototype = None
        if guided:
            reference_year = int(cfg["study"]["reference_year"])
            reference_rows = group.loc[group.year.astype(int).eq(reference_year)]
            if reference_rows.empty or region_id not in supervision.index:
                pixel_eligible = False
                route = "anchor_analog_area"
            else:
                prototype = build_anchor_prototype(
                    Path(str(reference_rows.iloc[0].path)),
                    Path(str(supervision.loc[region_id, "pv_fraction_path"])),
                    Path(str(quality.loc[region_id, "endmember_path"])),
                    cfg,
                )
                if prototype is None:
                    allow_reference_internal = bool(
                        cfg.get("anchor_guided", {}).get(
                            "allow_reference_internal_analysis_without_prototype", True
                        )
                    )
                    route_reason = str(quality.loc[region_id].get("anchor_guided_route_reason", ""))
                    if not (allow_reference_internal and route_reason == "reference_internal_change_analysis"):
                        pixel_eligible = False
                        route = "anchor_analog_area"
        if not pixel_eligible:
            for year in years:
                unit_rows.append({
                    "Id": region_id, "year": int(year),
                    "probability_path": "", "epistemic_std_path": "", "raw_mask_path": "",
                    "threshold": threshold, "ensemble_members": len(models),
                    "predicted_pixels": 0, "predicted_area_km2": np.nan,
                    "ccdc_causal_years_used": int(year - years[0] + 1),
                    "effective_tile_size": 0, "tile_overlap": 0, "mask_nodata": 255,
                    "sparse_valid_pixel_inference": True,
                    "region_valid_pixel_year_fraction": float(quality.loc[region_id, "median_valid_fraction"]),
                    "region_elapsed_seconds": 0.0, "region_tile_count": 0,
                    "inference_threads": inference_threads,
                    "inference_mode": "area_only_fallback",
                    "estimation_route": route,
                    "anchor_similarity_path": "", "candidate_score_path": "", "candidate_class_path": "",
                    "quality_score": float(quality.loc[region_id, "quality_score"]),
                })
            unit_registry = pd.DataFrame(unit_rows).sort_values("year")
            unit_registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
            save_table(unit_registry, unit_registry_path)
            atomic_json({
                "status": "passed", "Id": region_id, "records": len(unit_registry),
                "registry_path": str(unit_registry_path), "created_utc": utc_now(),
                "bounded_memory": True, "inference_mode": "area_only_fallback",
                "estimation_route": route,
            }, checkpoint)
            all_rows.append(unit_registry)
            continue
        region_total_pixel_years = 0
        region_predicted_pixel_years = 0
        region_tiles = 0
        with ExitStack() as stack:
            srcs = [stack.enter_context(rasterio.open(path)) for path in paths]
            ref = srcs[0]
            height, width = ref.height, ref.width
            fraction_src = None
            anchor_geometry = None
            if guided:
                fraction_src = stack.enter_context(rasterio.open(str(supervision.loc[region_id, "pv_fraction_path"])))
                anchor_geometry = select_anchor_geometry(reference_shape, region_id, ref.crs, ref.bounds, cfg)
            if any(
                src.width != width
                or src.height != height
                or src.crs != ref.crs
                or not src.transform.almost_equals(ref.transform)
                for src in srcs[1:]
            ):
                raise RuntimeError(f"Id {region_id}: aligned annual rasters do not share a canonical grid")
            tile_size = _effective_tile_size(cfg, len(years), len(channels) + int(add_valid), len(ccdc_names) * 7)
            overlap = int(cfg["inference"]["tile_overlap"])
            profile_float = ref.profile.copy()
            profile_float.update(
                count=1,
                dtype="float32",
                nodata=-9999.0,
                compress=cfg["processing"].get("compress", "DEFLATE"),
                BIGTIFF="IF_SAFER",
            )
            mask_nodata = int(cfg.get("calibration", {}).get("mask_nodata", 255))
            profile_mask = ref.profile.copy()
            profile_mask.update(
                count=1,
                dtype="uint8",
                nodata=mask_nodata,
                compress=cfg["processing"].get("compress", "DEFLATE"),
                BIGTIFF="IF_SAFER",
            )
            probability_paths = [
                ensure_dir(prob_root / f"unit_{region_id:03d}")
                / f"unit_{region_id:03d}_{year}_pv_probability_30m.tif"
                for year in years
            ]
            std_paths = [
                ensure_dir(std_root / f"unit_{region_id:03d}")
                / f"unit_{region_id:03d}_{year}_pv_epistemic_std_30m.tif"
                for year in years
            ]
            mask_paths = [
                ensure_dir(mask_root / f"unit_{region_id:03d}")
                / f"unit_{region_id:03d}_{year}_pv_raw_mask_30m.tif"
                for year in years
            ]
            similarity_paths = [
                ensure_dir(similarity_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_anchor_similarity_30m.tif"
                for year in years
            ] if guided else [Path("") for _ in years]
            candidate_score_paths = [
                ensure_dir(candidate_score_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_anchor_candidate_score_30m.tif"
                for year in years
            ] if guided else [Path("") for _ in years]
            candidate_class_paths = [
                ensure_dir(candidate_class_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_anchor_candidate_class_30m.tif"
                for year in years
            ] if guided else [Path("") for _ in years]
            prob_dsts = [stack.enter_context(rasterio.open(path, "w", **profile_float)) for path in probability_paths]
            std_dsts = [stack.enter_context(rasterio.open(path, "w", **profile_float)) for path in std_paths]
            mask_dsts = [stack.enter_context(rasterio.open(path, "w", **profile_mask)) for path in mask_paths]
            similarity_dsts = [stack.enter_context(rasterio.open(path, "w", **profile_float)) for path in similarity_paths] if guided else []
            candidate_score_dsts = [stack.enter_context(rasterio.open(path, "w", **profile_float)) for path in candidate_score_paths] if guided else []
            candidate_class_dsts = [stack.enter_context(rasterio.open(path, "w", **profile_mask)) for path in candidate_class_paths] if guided else []
            predicted_pixels = np.zeros(len(years), dtype=np.int64)
            medium_pixels = np.zeros(len(years), dtype=np.int64)

            windows = list(_core_windows(width, height, tile_size))
            for core in tqdm(
                windows,
                desc=f"Id {region_id:03d} tiles",
                unit="tile",
                leave=False,
                disable=not cfg["processing"].get("show_progress", True),
            ):
                region_tiles += 1
                expanded, crop = _expanded_window(core, width, height, overlap)
                flattened_static: list[np.ndarray] = []
                valid_flats: list[np.ndarray] = []
                index_series: list[list[np.ndarray]] = [[] for _ in ccdc_indices]
                for src in srcs:
                    reflectance = src.read([1, 2, 3, 4, 5, 6], window=expanded).astype(np.float32)
                    reflectance[reflectance <= -9990] = np.nan
                    valid = np.isfinite(reflectance).all(axis=0)
                    # Compute the full feature stack once. The first len(channels)
                    # planes are also the CCDC index source, so the former duplicate
                    # compute_features call is unnecessary.
                    static = compute_features(reflectance, channels, add_valid, endmembers=endmembers)
                    static_matrix = static.reshape(static.shape[0], -1).T
                    flattened_static.append(static_matrix)
                    valid_flat = valid.reshape(-1)
                    valid_flats.append(valid_flat)
                    for j, index in enumerate(ccdc_indices):
                        index_series[j].append(static[index].reshape(-1))

                expanded_shape = (int(expanded.height), int(expanded.width))
                pixels_per_year = int(expanded.width * expanded.height)
                region_total_pixel_years += pixels_per_year * len(years)
                if skip_invalid_pixels:
                    active_mask = np.logical_or.reduce(valid_flats)
                    active_indices = np.flatnonzero(active_mask)
                else:
                    active_indices = np.arange(pixels_per_year, dtype=np.int64)

                mean_cube = np.full((len(years), *expanded_shape), np.nan, dtype=np.float32)
                std_cube = np.full((len(years), *expanded_shape), np.nan, dtype=np.float32)
                if active_indices.size:
                    causal_blocks = [
                        causal_ccdc_features(
                            np.stack(values, axis=0)[:, active_indices],
                            int(cfg["ccdc"]["minimum_observations"]),
                            int(cfg["ccdc"]["break_minimum_segment_years"]),
                        )
                        for values in index_series
                    ]
                    causal = np.concatenate(causal_blocks, axis=2)
                    matrices: list[np.ndarray] = []
                    positions_by_year: list[np.ndarray] = []
                    counts_by_year: list[int] = []
                    for year_index in range(len(years)):
                        if skip_invalid_pixels:
                            active_valid = valid_flats[year_index][active_indices]
                            positions = active_indices[active_valid]
                            static_part = flattened_static[year_index][positions]
                            causal_part = causal[year_index][active_valid]
                        else:
                            positions = active_indices
                            static_part = flattened_static[year_index][positions]
                            causal_part = causal[year_index]
                        matrix = np.nan_to_num(
                            np.concatenate([static_part, causal_part], axis=1),
                            nan=0.0,
                            posinf=8.0,
                            neginf=-8.0,
                        ).astype(np.float32, copy=False)
                        if matrix.shape[1] != expected_feature_count:
                            raise RuntimeError(
                                f"Id {region_id}: inference feature count {matrix.shape[1]} != trained {expected_feature_count}"
                            )
                        matrices.append(matrix)
                        positions_by_year.append(positions)
                        counts_by_year.append(int(matrix.shape[0]))

                    nonempty = [matrix for matrix in matrices if matrix.shape[0] > 0]
                    if nonempty:
                        batched_matrix = np.concatenate(nonempty, axis=0)
                        region_predicted_pixel_years += int(batched_matrix.shape[0])
                        weighted_mean, weighted_std = _predict_weighted_ensemble(
                            models,
                            weights,
                            batched_matrix,
                            prediction_batch_rows,
                        )
                        cursor = 0
                        for year_index, count in enumerate(counts_by_year):
                            if count == 0:
                                continue
                            stop = cursor + count
                            mean_flat = mean_cube[year_index].reshape(-1)
                            std_flat = std_cube[year_index].reshape(-1)
                            positions = positions_by_year[year_index]
                            mean_flat[positions] = weighted_mean[cursor:stop]
                            std_flat[positions] = weighted_std[cursor:stop]
                            cursor = stop
                        del batched_matrix, weighted_mean, weighted_std
                    del causal

                if guided:
                    tile_transform = window_transform(expanded, ref.transform)
                    near_anchor = buffered_support_mask(anchor_geometry, tile_transform, expanded_shape, cfg)
                    reference_anchor = anchor_mask_for_window(fraction_src, expanded, cfg) if fraction_src is not None else np.zeros(expanded_shape, dtype=bool)
                else:
                    near_anchor = np.zeros(expanded_shape, dtype=bool)
                    reference_anchor = np.zeros(expanded_shape, dtype=bool)

                for year_index in range(len(years)):
                    mean = mean_cube[year_index]
                    std = std_cube[year_index]
                    valid = valid_flats[year_index].reshape(expanded_shape)
                    mean[~valid] = np.nan
                    std[~valid] = np.nan
                    if guided and prototype is not None:
                        similarity = anchor_similarity_matrix(flattened_static[year_index], channels, prototype).reshape(expanded_shape)
                        high, medium, candidate_score = classify_anchor_guided(
                            np.nan_to_num(mean, nan=0.0), similarity, np.nan_to_num(std, nan=1.0),
                            near_anchor, valid, reference_anchor,
                            int(years[year_index]) == int(cfg["study"]["reference_year"]), cfg,
                        )
                        cleaned = high
                    else:
                        similarity = np.zeros(expanded_shape, dtype=np.float32)
                        medium = np.zeros(expanded_shape, dtype=bool)
                        candidate_score = np.nan_to_num(mean, nan=0.0).astype(np.float32)
                        cleaned = _cleanup(
                            np.nan_to_num(mean, nan=0.0) >= threshold,
                            int(cfg["inference"]["minimum_component_pixels"]),
                            int(cfg["inference"]["maximum_hole_pixels"]),
                        )
                    mean_core = mean[crop]
                    std_core = std[crop]
                    mask_core = cleaned[crop]
                    medium_core = medium[crop]
                    valid_core = valid[crop]
                    prob_dsts[year_index].write(
                        np.where(np.isfinite(mean_core), mean_core, -9999.0).astype(np.float32),
                        1,
                        window=core,
                    )
                    std_dsts[year_index].write(
                        np.where(np.isfinite(std_core), std_core, -9999.0).astype(np.float32),
                        1,
                        window=core,
                    )
                    encoded_mask = np.full(mask_core.shape, mask_nodata, dtype=np.uint8)
                    encoded_mask[valid_core] = mask_core[valid_core].astype(np.uint8)
                    mask_dsts[year_index].write(encoded_mask, 1, window=core)
                    if guided:
                        similarity_core = similarity[crop]
                        score_core = candidate_score[crop]
                        similarity_dsts[year_index].write(np.where(valid_core, similarity_core, -9999.0).astype(np.float32), 1, window=core)
                        candidate_score_dsts[year_index].write(np.where(valid_core, score_core, -9999.0).astype(np.float32), 1, window=core)
                        classes = np.full(mask_core.shape, mask_nodata, dtype=np.uint8)
                        classes[valid_core] = 0
                        classes[medium_core & valid_core] = 3
                        classes[mask_core & valid_core] = 2
                        if int(years[year_index]) == int(cfg["study"]["reference_year"]):
                            anchor_core = reference_anchor[crop]
                            classes[anchor_core & valid_core] = 1
                        candidate_class_dsts[year_index].write(classes, 1, window=core)
                    predicted_pixels[year_index] += int(np.count_nonzero(mask_core & valid_core))
                    medium_pixels[year_index] += int(np.count_nonzero(medium_core & valid_core))
                del mean_cube, std_cube, flattened_static, valid_flats, index_series

            elapsed_seconds = float(time.perf_counter() - region_started)
            valid_fraction = (
                float(region_predicted_pixel_years / region_total_pixel_years)
                if region_total_pixel_years > 0
                else 0.0
            )
            for i, year in enumerate(years):
                prob_dsts[i].update_tags(
                    ccdc_mode="causal annual-composite harmonic break",
                    tile_size=tile_size,
                    tile_overlap=overlap,
                    sparse_valid_pixel_inference=str(skip_invalid_pixels).lower(),
                    inference_threads=inference_threads,
                )
                unit_rows.append(
                    {
                        "Id": region_id,
                        "year": int(year),
                        "probability_path": str(probability_paths[i]),
                        "epistemic_std_path": str(std_paths[i]),
                        "raw_mask_path": str(mask_paths[i]),
                        "threshold": threshold,
                        "ensemble_members": len(models),
                        "predicted_pixels": int(predicted_pixels[i]),
                        "predicted_area_km2": float(predicted_pixels[i] * 0.0009),
                        "ccdc_causal_years_used": int(i + 1),
                        "effective_tile_size": tile_size,
                        "tile_overlap": overlap,
                        "mask_nodata": mask_nodata,
                        "sparse_valid_pixel_inference": skip_invalid_pixels,
                        "region_valid_pixel_year_fraction": valid_fraction,
                        "region_elapsed_seconds": elapsed_seconds,
                        "region_tile_count": region_tiles,
                        "inference_threads": inference_threads,
                        "inference_mode": route,
                        "estimation_route": route,
                        "quality_score": float(quality.loc[region_id, "quality_score"]),
                        "anchor_similarity_path": str(similarity_paths[i]) if guided else "",
                        "candidate_score_path": str(candidate_score_paths[i]) if guided else "",
                        "candidate_class_path": str(candidate_class_paths[i]) if guided else "",
                        "anchor_guided_high_pixels": int(predicted_pixels[i]) if guided else 0,
                        "anchor_guided_medium_pixels": int(medium_pixels[i]) if guided else 0,
                        "anchor_prototype_pixels": int(prototype.anchor_pixels) if guided and prototype is not None else 0,
                        "anchor_prototype_available": bool(prototype is not None) if guided else False,
                        "reference_internal_analysis_without_prototype": bool(
                            guided
                            and prototype is None
                            and str(quality.loc[region_id].get("anchor_guided_route_reason", ""))
                            == "reference_internal_change_analysis"
                        ),
                    }
                )

        unit_registry = pd.DataFrame(unit_rows).sort_values("year")
        unit_registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
        save_table(unit_registry, unit_registry_path)
        atomic_json(
            {
                "status": "passed",
                "Id": region_id,
                "records": len(unit_registry),
                "registry_path": str(unit_registry_path),
                "created_utc": utc_now(),
                "bounded_memory": True,
                "sparse_valid_pixel_inference": skip_invalid_pixels,
                "inference_threads": inference_threads,
                "prediction_batch_rows": prediction_batch_rows,
                "elapsed_seconds": float(time.perf_counter() - region_started),
                "total_pixel_years": int(region_total_pixel_years),
                "predicted_pixel_years": int(region_predicted_pixel_years),
            },
            checkpoint,
        )
        all_rows.append(unit_registry)

    registry = pd.concat(all_rows, ignore_index=True).sort_values(["Id", "year"])
    save_table(registry, out / "annual_inference_registry.csv")
    expected_units = expected_counts(cfg)[0]
    expected_records = expected_units * expected_years
    status = "passed" if registry.Id.nunique() == expected_units and len(registry) == expected_records else "failed"
    atomic_json(
        {
            "status": status,
            "created_utc": utc_now(),
            "records": len(registry),
            "units": int(registry.Id.nunique()),
            "expected_records": expected_records,
            "expected_units": expected_units,
            "threshold": threshold,
            "backend": "sparse-valid-pixel bounded-memory CPU ensemble",
            "ccdc_mode": "causal annual-composite harmonic break",
            "resolution_m": 30,
            "whole_region_arrays_retained": False,
            "per_region_checkpoints": True,
            "sparse_valid_pixel_inference": skip_invalid_pixels,
            "inference_threads": inference_threads,
            "prediction_batch_rows": prediction_batch_rows,
            "direct_pixel_units": int(registry.groupby("Id")["inference_mode"].first().eq("direct_pixel").sum()),
            "anchor_guided_candidate_units": int(registry.groupby("Id")["inference_mode"].first().eq("anchor_guided_candidate").sum()),
            "pixel_mapping_units": int(registry.groupby("Id")["inference_mode"].first().isin(PIXEL_MODES).sum()),
            "area_only_units": int(registry.groupby("Id")["inference_mode"].first().eq("area_only_fallback").sum()),
        },
        out / "annual_inference_status.json",
    )
    if status == "failed":
        raise RuntimeError("Annual inference coverage incomplete")
    return registry

