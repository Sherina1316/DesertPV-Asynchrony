from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from tqdm import tqdm

from .features import compute_features
from .spectral_mixture import load_endmember_matrix
from .study_area import expected_counts, test_mode_enabled
from .utils import atomic_json, ensure_dir, save_table, utc_now


EPS = 1e-6


def _core_windows(width: int, height: int, size: int) -> Iterator[Window]:
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def _expanded(core: Window, width: int, height: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    c0 = max(0, int(core.col_off) - halo)
    r0 = max(0, int(core.row_off) - halo)
    c1 = min(width, int(core.col_off + core.width) + halo)
    r1 = min(height, int(core.row_off + core.height) + halo)
    expanded = Window(c0, r0, c1 - c0, r1 - r0)
    crop = (
        slice(int(core.row_off) - r0, int(core.row_off) - r0 + int(core.height)),
        slice(int(core.col_off) - c0, int(core.col_off) - c0 + int(core.width)),
    )
    return expanded, crop


def _robust01(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(arr)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    vals = arr[mask]
    if vals.size < 20:
        return np.clip(np.nan_to_num(arr, nan=0.5), 0.0, 1.0).astype(np.float32)
    lo, hi = np.nanquantile(vals, [0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        out = np.full(arr.shape, 0.5, dtype=np.float32)
        out[~mask] = 0.0
        return out
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _environment_reliability(panel: pd.DataFrame, years: np.ndarray, region_id: int) -> dict[int, float]:
    if panel.empty or "Id" not in panel or "year" not in panel:
        return {int(y): 1.0 for y in years}
    unit = panel.loc[pd.to_numeric(panel["Id"], errors="coerce").eq(region_id)].copy()
    if unit.empty:
        return {int(y): 1.0 for y in years}
    climate = [
        c for c in [
            "precipitation", "temperature", "potential_evapotranspiration",
            "soil_moisture", "radiation", "wind_speed",
        ] if c in unit.columns
    ]
    if not climate:
        return {int(y): 1.0 for y in years}
    z_parts: list[np.ndarray] = []
    for col in climate:
        x = pd.to_numeric(unit[col], errors="coerce").to_numpy(float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = max(1.4826 * mad, 1e-6)
        z_parts.append(np.abs((x - med) / scale))
    extremeness = np.nanmean(np.vstack(z_parts), axis=0)
    # Climate is a reliability modifier, never direct evidence of PV presence.
    reliability = np.clip(np.exp(-0.10 * np.nan_to_num(extremeness, nan=0.0)), 0.72, 1.0)
    return {
        int(y): float(r)
        for y, r in zip(pd.to_numeric(unit["year"], errors="coerce").fillna(-1).astype(int), reliability)
    } | {int(y): 1.0 for y in years if int(y) not in set(pd.to_numeric(unit["year"], errors="coerce").dropna().astype(int))}


def _constraint_sources(stack: ExitStack, cfg: dict[str, Any], region_id: int, ref: rasterio.DatasetReader) -> dict[str, rasterio.DatasetReader]:
    root = Path(cfg["paths"].get("future_constraint_root", ""))
    patterns = cfg.get("future_2050", {}).get("constraint_rasters", {})
    sources: dict[str, rasterio.DatasetReader] = {}
    if not root.exists():
        return sources
    for name, pattern in patterns.items():
        path = root / str(pattern).format(Id=region_id)
        if not path.exists():
            continue
        src = stack.enter_context(rasterio.open(path))
        if src.shape == ref.shape and src.crs == ref.crs and src.transform.almost_equals(ref.transform):
            sources[str(name)] = src
    return sources


def _spatial_suitability(constraints: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    if not constraints:
        return np.full(valid.shape, 0.5, dtype=np.float32)
    parts: list[tuple[float, np.ndarray]] = []
    if "solar_irradiance" in constraints:
        parts.append((0.30, _robust01(constraints["solar_irradiance"], valid)))
    if "slope" in constraints:
        slope = np.asarray(constraints["slope"], dtype=np.float32)
        parts.append((0.22, np.exp(-np.clip(slope, 0, 90) / 12.0).astype(np.float32)))
    if "grid_distance" in constraints:
        distance = np.maximum(np.asarray(constraints["grid_distance"], dtype=np.float32), 0.0)
        parts.append((0.17, np.exp(-distance / 50_000.0).astype(np.float32)))
    if "road_distance" in constraints:
        distance = np.maximum(np.asarray(constraints["road_distance"], dtype=np.float32), 0.0)
        parts.append((0.11, np.exp(-distance / 30_000.0).astype(np.float32)))
    if "water_stress" in constraints:
        # Water stress is not a hard exclusion; it weakly favours lower additional water demand.
        parts.append((0.08, _robust01(constraints["water_stress"], valid)))
    if "protected_area" in constraints:
        protected = np.asarray(constraints["protected_area"], dtype=np.float32)
        parts.append((0.12, (protected <= 0).astype(np.float32)))
    if not parts:
        return np.full(valid.shape, 0.5, dtype=np.float32)
    total = sum(weight for weight, _ in parts)
    score = sum(weight * np.nan_to_num(arr, nan=0.5) for weight, arr in parts) / max(total, EPS)
    score[~valid] = 0.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _temporal_evidence(prob: np.ndarray, spectral: np.ndarray, valid: np.ndarray, low_probability: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = prob.shape[0]
    change = np.zeros_like(prob, dtype=np.float32)
    persistence = np.zeros_like(prob, dtype=np.float32)
    spectral_change = np.zeros_like(prob, dtype=np.float32)
    for t in range(years):
        prior_start = max(0, t - 3)
        prior_prob = np.nanmedian(np.where(valid[prior_start:t], prob[prior_start:t], np.nan), axis=0) if t > 0 else prob[t]
        prior_spec = np.nanmedian(np.where(valid[prior_start:t], spectral[prior_start:t], np.nan), axis=0) if t > 0 else spectral[t]
        if t == 0:
            change[t] = np.clip(prob[t], 0.0, 1.0)
            spectral_change[t] = np.clip(spectral[t], 0.0, 1.0)
        else:
            change[t] = _sigmoid((prob[t] - np.nan_to_num(prior_prob, nan=prob[t])) / 0.09)
            spectral_change[t] = _sigmoid((spectral[t] - np.nan_to_num(prior_spec, nan=spectral[t])) / 0.10)
        lo = max(0, t - 1)
        hi = min(years, t + 3)
        neighbourhood = valid[lo:hi] & (prob[lo:hi] >= low_probability)
        denom = np.maximum(valid[lo:hi].sum(axis=0), 1)
        persistence[t] = neighbourhood.sum(axis=0) / denom
    return change, persistence.astype(np.float32), spectral_change


def _object_filter(
    candidate: np.ndarray,
    score: np.ndarray,
    votes: np.ndarray,
    persistence: np.ndarray,
    anchor: np.ndarray,
    near_anchor: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    settings = cfg.get("multisource_refinement", {})
    minimum = int(settings.get("minimum_component_pixels", 6))
    remote_minimum = int(settings.get("remote_minimum_component_pixels", 18))
    minimum_fill = float(settings.get("minimum_bbox_fill", 0.10))
    maximum_elongation = float(settings.get("maximum_elongation", 12.0))
    minimum_object_score = float(settings.get("minimum_object_mean_score", 0.58))
    minimum_object_votes = float(settings.get("minimum_object_mean_votes", 2.2))
    minimum_object_persistence = float(settings.get("minimum_object_mean_persistence", 0.40))

    working = ndimage.binary_closing(candidate, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, count = ndimage.label(working)
    keep = np.zeros(count + 1, dtype=bool)
    rejection = np.zeros(candidate.shape, dtype=np.uint8)
    if count == 0:
        return candidate.astype(bool), rejection
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        component = labels[slc] == label_id
        size = int(sizes[label_id])
        rows = int(slc[0].stop - slc[0].start)
        cols = int(slc[1].stop - slc[1].start)
        bbox_area = max(rows * cols, 1)
        fill = size / bbox_area
        elongation = max(rows, cols) / max(min(rows, cols), 1)
        anchor_hit = bool(np.any(anchor[slc] & component))
        near_hit = bool(np.any(near_anchor[slc] & component))
        mean_score = float(np.mean(score[slc][component]))
        mean_votes = float(np.mean(votes[slc][component]))
        mean_persistence = float(np.mean(persistence[slc][component]))
        required_size = minimum if near_hit else remote_minimum
        accepted = anchor_hit or (
            size >= required_size
            and mean_score >= minimum_object_score
            and mean_votes >= minimum_object_votes
            and mean_persistence >= minimum_object_persistence
            and (near_hit or (fill >= minimum_fill and elongation <= maximum_elongation))
        )
        if accepted:
            keep[label_id] = True
        else:
            rejection_slice = rejection[slc]
            rejection_slice[component] = 4
            rejection[slc] = rejection_slice
    return keep[labels], rejection


def _read_mask(src: rasterio.DatasetReader, window: Window, threshold: float = 0.5) -> np.ndarray:
    arr = src.read(1, window=window).astype(np.float32)
    return np.isfinite(arr) & (arr >= threshold)


def refine_multisource_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("multisource_refinement", {})
    if not bool(settings.get("enabled", True)):
        return pd.read_csv(Path(cfg["paths"]["output_root"]) / "07_probability_calibration" / "calibrated_inference_registry.csv")

    root = Path(cfg["paths"]["output_root"])
    calibration_dir = root / "07_probability_calibration"
    source_registry_path = calibration_dir / "calibrated_inference_registry.csv"
    registry = pd.read_csv(source_registry_path).sort_values(["Id", "year"])
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    env_path = root / "01_input_validation" / "environmental_covariates.csv"
    environment = pd.read_csv(env_path) if env_path.exists() else pd.DataFrame()

    out = ensure_dir(root / "07_multisource_refinement")
    prob_root = ensure_dir(out / "refined_probability_30m")
    mask_root = ensure_dir(out / "refined_masks_30m")
    vote_root = ensure_dir(out / "support_votes_30m")
    reject_root = ensure_dir(out / "rejection_code_30m")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    unit_registry_root = ensure_dir(out / "unit_registries")

    years = np.arange(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1)
    reference_year = int(cfg["study"]["reference_year"])
    tile_size = int(settings.get("tile_size", 256))
    halo = int(settings.get("halo_pixels", 24))
    low_probability = float(settings.get("low_probability", 0.48))
    high_score = float(settings.get("high_score_threshold", 0.66))
    low_score = float(settings.get("low_score_threshold", 0.54))
    minimum_votes = int(settings.get("minimum_support_votes", 3))
    hard_confuser = float(settings.get("hard_confuser_threshold", 0.82))
    mask_nodata = int(cfg.get("calibration", {}).get("mask_nodata", 255))
    diagnostics_units = {int(x) for x in settings.get("diagnostic_unit_ids", [1, 6, 28])}
    write_diagnostics = bool(settings.get("write_diagnostics", False)) or test_mode_enabled(cfg)

    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for region_id, group in tqdm(
        list(registry.groupby("Id")),
        desc="Multisource object-temporal PV refinement",
        unit="region",
        disable=not cfg.get("processing", {}).get("show_progress", True),
    ):
        region_id = int(region_id)
        group = group.sort_values("year")
        aligned_group = aligned.loc[aligned["Id"].eq(region_id)].sort_values("year")
        if len(group) != len(years) or len(aligned_group) != len(years):
            raise RuntimeError(f"Id {region_id}: multisource refinement requires {len(years)} annual records")
        if group["inference_mode"].astype(str).eq("area_only_fallback").all():
            passthrough = group.copy()
            passthrough["multisource_refinement"] = "not_applicable_area_only"
            all_rows.append(passthrough)
            continue

        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        unit_registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
        if checkpoint.exists() and unit_registry_path.exists() and not cfg.get("processing", {}).get("overwrite", False):
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing = pd.read_csv(unit_registry_path)
                if payload.get("status") == "passed" and len(existing) == len(years) and all(Path(p).exists() for p in existing["calibrated_probability_path"].astype(str)):
                    all_rows.append(existing)
                    summary_rows.append(payload)
                    continue
            except Exception:
                pass

        endmembers = load_endmember_matrix(quality.loc[region_id, "endmember_path"])
        env_reliability = _environment_reliability(environment, years, region_id)
        year_to_row = {int(row.year): row for row in group.itertuples(index=False)}
        year_to_landsat = {int(row.year): Path(str(row.path)) for row in aligned_group.itertuples(index=False)}

        refined_prob_paths = [ensure_dir(prob_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_pv_probability_multisource_30m.tif" for year in years]
        refined_mask_paths = [ensure_dir(mask_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_pv_mask_multisource_30m.tif" for year in years]
        vote_paths = [ensure_dir(vote_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_support_votes_30m.tif" for year in years]
        rejection_paths = [ensure_dir(reject_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{year}_rejection_code_30m.tif" for year in years]

        predicted_pixels = np.zeros(len(years), dtype=np.int64)
        valid_pixels = np.zeros(len(years), dtype=np.int64)
        anchor_reference_pixels = 0
        anchor_reference_detected = 0
        hard_negative_pixels = 0
        hard_negative_detected = 0
        pre_pixels = np.array([int(getattr(year_to_row[int(y)], "calibrated_predicted_pixels", 0) or 0) for y in years], dtype=np.int64)

        with ExitStack() as stack:
            probability_srcs = [stack.enter_context(rasterio.open(str(year_to_row[int(y)].calibrated_probability_path))) for y in years]
            uncertainty_srcs = [stack.enter_context(rasterio.open(str(year_to_row[int(y)].epistemic_std_path))) for y in years]
            landsat_srcs = [stack.enter_context(rasterio.open(year_to_landsat[int(y)])) for y in years]
            fraction_src = stack.enter_context(rasterio.open(str(supervision.loc[region_id, "pv_fraction_path"])))
            label_src = stack.enter_context(rasterio.open(str(supervision.loc[region_id, "label_path"])))
            support_src = stack.enter_context(rasterio.open(str(supervision.loc[region_id, "anchor_support_path"])))
            ref = probability_srcs[0]
            constraint_srcs = _constraint_sources(stack, cfg, region_id, ref)

            float_profile = ref.profile.copy()
            float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            byte_profile = ref.profile.copy()
            byte_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            prob_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in refined_prob_paths]
            mask_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in refined_mask_paths]
            diagnostic_on = write_diagnostics and region_id in diagnostics_units
            vote_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in vote_paths] if diagnostic_on else []
            reject_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in rejection_paths] if diagnostic_on else []

            for core in _core_windows(ref.width, ref.height, tile_size):
                expanded, crop = _expanded(core, ref.width, ref.height, halo)
                shape = (int(expanded.height), int(expanded.width))
                prob = np.zeros((len(years), *shape), dtype=np.float32)
                uncertainty = np.ones_like(prob, dtype=np.float32)
                valid = np.zeros_like(prob, dtype=bool)
                spectral = np.zeros_like(prob, dtype=np.float32)
                fraction_evidence = np.zeros_like(prob, dtype=np.float32)
                confuser = np.ones_like(prob, dtype=np.float32)

                for t, (psrc, usrc, lsrc) in enumerate(zip(probability_srcs, uncertainty_srcs, landsat_srcs)):
                    raw_p = psrc.read(1, window=expanded).astype(np.float32)
                    raw_u = usrc.read(1, window=expanded).astype(np.float32)
                    reflectance = lsrc.read([1, 2, 3, 4, 5, 6], window=expanded).astype(np.float32)
                    reflectance[reflectance <= -9990] = np.nan
                    valid[t] = np.isfinite(raw_p) & (raw_p > -9990) & np.isfinite(reflectance).all(axis=0)
                    prob[t] = np.where(valid[t], np.clip(raw_p, 0.0, 1.0), 0.0)
                    uncertainty[t] = np.where(valid[t], np.clip(np.nan_to_num(raw_u, nan=1.0), 0.0, 1.0), 1.0)
                    feats = compute_features(
                        reflectance,
                        ["PV_Spectral_Evidence", "PV_Fraction_Evidence", "PV_Confuser_Penalty"],
                        False,
                        endmembers=endmembers,
                    )
                    spectral[t] = np.clip(0.5 + 0.5 * np.nan_to_num(feats[0], nan=-1.0), 0.0, 1.0)
                    fraction_evidence[t] = np.clip(np.nan_to_num(feats[1], nan=0.0), 0.0, 1.0)
                    confuser[t] = np.clip(np.nan_to_num(feats[2], nan=1.0), 0.0, 1.0)

                fraction_anchor = _read_mask(fraction_src, expanded, float(cfg["supervision"].get("strong_positive_fraction", 0.7)))
                near_anchor = _read_mask(support_src, expanded, 0.5)
                training_label = label_src.read(1, window=expanded)
                hard_negative = training_label == 0
                constraints = {name: src.read(1, window=expanded).astype(np.float32) for name, src in constraint_srcs.items()}
                spatial = _spatial_suitability(constraints, np.logical_or.reduce(valid))
                change, persistence, spectral_change = _temporal_evidence(prob, spectral, valid, low_probability)

                for t, year in enumerate(years):
                    local_mean = ndimage.uniform_filter(prob[t], size=9, mode="nearest")
                    local_contrast = np.clip(0.5 + (prob[t] - local_mean) / 0.30, 0.0, 1.0)
                    env_rel = float(env_reliability.get(int(year), 1.0))
                    uncertainty_penalty = np.clip(uncertainty[t] / 0.35, 0.0, 1.0)
                    spectral_joint = np.clip(0.55 * spectral[t] + 0.45 * fraction_evidence[t], 0.0, 1.0)
                    temporal_joint = np.clip(0.45 * change[t] + 0.25 * spectral_change[t] + 0.30 * persistence[t], 0.0, 1.0)
                    score = (
                        0.42 * prob[t]
                        + 0.20 * spectral_joint
                        + 0.16 * temporal_joint
                        + 0.08 * local_contrast
                        + 0.06 * spatial
                        + 0.04 * persistence[t]
                        + 0.04 * env_rel
                        - 0.23 * confuser[t]
                        - 0.09 * uncertainty_penalty
                    )
                    score = np.clip(score, 0.0, 1.0) * valid[t]

                    model_vote = prob[t] >= float(settings.get("model_vote_threshold", 0.62))
                    spectral_vote = (spectral_joint >= float(settings.get("spectral_vote_threshold", 0.56)))
                    temporal_vote = temporal_joint >= float(settings.get("temporal_vote_threshold", 0.58))
                    persistence_vote = persistence[t] >= float(settings.get("persistence_vote_threshold", 0.50))
                    contrast_vote = local_contrast >= float(settings.get("contrast_vote_threshold", 0.56))
                    spatial_vote = spatial >= float(settings.get("spatial_vote_threshold", 0.45))
                    votes = (
                        model_vote.astype(np.uint8)
                        + spectral_vote.astype(np.uint8)
                        + temporal_vote.astype(np.uint8)
                        + persistence_vote.astype(np.uint8)
                        + contrast_vote.astype(np.uint8)
                        + spatial_vote.astype(np.uint8)
                    )
                    candidate = valid[t] & (
                        (score >= high_score)
                        | ((score >= low_score) & (votes >= minimum_votes) & (model_vote | spectral_vote) & temporal_vote)
                        | (near_anchor & (score >= low_score - 0.04) & (votes >= max(2, minimum_votes - 1)))
                    )
                    confuser_reject = (
                        confuser[t] >= hard_confuser
                    ) & ~(prob[t] >= 0.92) & ~(spectral_joint >= 0.82) & ~fraction_anchor
                    candidate &= ~confuser_reject

                    rejection = np.zeros(shape, dtype=np.uint8)
                    rejection[valid[t] & (score < low_score)] = 1
                    rejection[valid[t] & (score >= low_score) & (votes < minimum_votes)] = 2
                    rejection[confuser_reject] = 3
                    filtered, object_rejection = _object_filter(candidate, score, votes, persistence[t], fraction_anchor, near_anchor, cfg)
                    rejection = np.maximum(rejection, object_rejection)

                    if int(year) == reference_year:
                        filtered |= fraction_anchor & valid[t]
                        score = np.maximum(score, np.where(fraction_anchor & valid[t], 0.995, 0.0))

                    core_valid = valid[t][crop]
                    core_score = score[crop]
                    core_mask = filtered[crop]
                    encoded = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                    encoded[core_valid] = core_mask[core_valid].astype(np.uint8)
                    prob_dsts[t].write(np.where(core_valid, core_score, -9999.0).astype(np.float32), 1, window=core)
                    mask_dsts[t].write(encoded, 1, window=core)
                    if diagnostic_on:
                        vote_encoded = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        reject_encoded = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        vote_encoded[core_valid] = votes[crop][core_valid]
                        reject_encoded[core_valid] = rejection[crop][core_valid]
                        vote_dsts[t].write(vote_encoded, 1, window=core)
                        reject_dsts[t].write(reject_encoded, 1, window=core)

                    predicted_pixels[t] += int(np.count_nonzero(core_mask & core_valid))
                    valid_pixels[t] += int(np.count_nonzero(core_valid))
                    if int(year) == reference_year:
                        anchor_core = fraction_anchor[crop] & core_valid
                        anchor_reference_pixels += int(anchor_core.sum())
                        anchor_reference_detected += int(np.count_nonzero(core_mask & anchor_core))
                        negative_core = hard_negative[crop] & core_valid
                        hard_negative_pixels += int(negative_core.sum())
                        hard_negative_detected += int(np.count_nonzero(core_mask & negative_core))

        unit_rows: list[dict[str, Any]] = []
        for t, year in enumerate(years):
            original = year_to_row[int(year)]._asdict()
            unit_rows.append({
                **original,
                "pre_refinement_calibrated_probability_path": original.get("calibrated_probability_path", ""),
                "pre_refinement_calibrated_mask_path": original.get("calibrated_mask_path", ""),
                "pre_refinement_predicted_pixels": int(pre_pixels[t]),
                "calibrated_probability_path": str(refined_prob_paths[t]),
                "calibrated_mask_path": str(refined_mask_paths[t]),
                "calibrated_predicted_pixels": int(predicted_pixels[t]),
                "calibrated_predicted_area_km2": float(predicted_pixels[t] * 0.0009),
                "multisource_refinement": "object_temporal_environmental_evidence_fusion_v24_13",
                "multisource_support_votes_path": str(vote_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "multisource_rejection_code_path": str(rejection_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "environment_reliability": float(env_reliability.get(int(year), 1.0)),
                "available_spatial_constraints": ";".join(sorted(constraint_srcs.keys())),
            })
        unit_registry = pd.DataFrame(unit_rows).sort_values("year")
        save_table(unit_registry, unit_registry_path)
        payload = {
            "status": "passed",
            "created_utc": utc_now(),
            "Id": region_id,
            "records": len(unit_registry),
            "registry_path": str(unit_registry_path),
            "pre_refinement_pixel_years": int(pre_pixels.sum()),
            "post_refinement_pixel_years": int(predicted_pixels.sum()),
            "relative_pixel_year_change": float((predicted_pixels.sum() - pre_pixels.sum()) / max(pre_pixels.sum(), 1)),
            "reference_anchor_recall": float(anchor_reference_detected / max(anchor_reference_pixels, 1)),
            "reference_hard_negative_fpr": float(hard_negative_detected / max(hard_negative_pixels, 1)),
            "available_spatial_constraints": sorted(constraint_srcs.keys()),
            "diagnostic_rasters_written": bool(write_diagnostics and region_id in diagnostics_units),
        }
        atomic_json(payload, checkpoint)
        summary_rows.append(payload)
        all_rows.append(unit_registry)

    refined = pd.concat(all_rows, ignore_index=True, sort=False).sort_values(["Id", "year"])
    backup = calibration_dir / "calibrated_inference_registry_pre_multisource_refinement.csv"
    if not backup.exists():
        save_table(registry, backup)
    save_table(refined, out / "refined_inference_registry.csv")
    # Downstream v24 modules consume this canonical path. The original is retained above.
    save_table(refined, source_registry_path)
    summary = pd.DataFrame(summary_rows)
    save_table(summary, out / "multisource_refinement_by_unit.csv")

    expected_units, _, expected_records = expected_counts(cfg)
    coverage_ok = len(refined) == expected_records and refined["Id"].nunique() == expected_units
    anchor_recall = float(pd.to_numeric(summary.get("reference_anchor_recall"), errors="coerce").mean()) if not summary.empty else float("nan")
    hard_negative_fpr = float(pd.to_numeric(summary.get("reference_hard_negative_fpr"), errors="coerce").mean()) if not summary.empty else float("nan")
    status = {
        "status": "passed" if coverage_ok else "failed",
        "created_utc": utc_now(),
        "records": len(refined),
        "expected_records": expected_records,
        "units": int(refined["Id"].nunique()),
        "expected_units": expected_units,
        "method": "calibrated model + LSMA/spectral evidence + local contrast + retrospective temporal change/persistence + weak environmental/spatial plausibility + object geometry",
        "historical_detection_rule": "environmental variables modify reliability and plausibility only; they cannot create a PV detection without spectral/model and temporal support",
        "reference_2024_rule": "SHP anchor is forced positive only in 2024; historical backcast still requires image-derived temporal evidence",
        "mean_reference_anchor_recall": anchor_recall,
        "mean_reference_hard_negative_fpr": hard_negative_fpr,
        "diagnostic_units": sorted(diagnostics_units),
        "canonical_registry_replaced": str(source_registry_path),
        "pre_refinement_registry_backup": str(backup),
    }
    atomic_json(status, out / "multisource_refinement_status.json")
    if not coverage_ok:
        raise RuntimeError("Multisource refinement coverage incomplete")
    return refined
