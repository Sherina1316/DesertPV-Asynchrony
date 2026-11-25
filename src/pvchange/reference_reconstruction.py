from __future__ import annotations

"""Reference-constrained annual PV evidence reconstruction (v24.16).

The 2024 reference vector has one and only one hard semantic meaning:
all valid pixels inside the rasterised vector are PV in 2024.  It is not copied
into any earlier year.  Pixels outside the vector remain unlabelled and may be
added to the 2024 inventory when independent image evidence supports them.

Historical maps are reconstructed retrospectively from annual Landsat evidence,
not by back-propagating the 2024 geometry.  Environmental covariates are used to
reduce confidence in spectrally atypical climate years; they never create a PV
label.  The resulting annual evidence is passed to the v24.16 change-point
stage, which estimates installation time separately for each pixel/object.
"""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from tqdm import tqdm

from .object_graph_refinement import (
    FEATURE_CHANNELS,
    PrototypeBundle,
    _build_unit_prototypes,
    _expanded,
    _feature_cube,
    _prototype_similarity,
    _read_reflectance,
    _robust01,
    _texture_features,
    _windows,
)
from .spectral_mixture import load_endmember_matrix
from .study_area import test_mode_enabled
from .utils import atomic_json, ensure_dir, save_table, utc_now

EPS = 1e-6

# Evidence-mask classes written by Stage 7.
EVIDENCE_BACKGROUND = 0
EVIDENCE_ANNUAL_IMAGE = 1
EVIDENCE_2024_VECTOR = 2
EVIDENCE_2024_OUTSIDE_SUPPLEMENT = 3

# Rejection codes written for diagnostic units.
REJECT_NONE = 0
REJECT_STABLE_SAND = 61
REJECT_WATER_VEGETATION = 62
REJECT_FRAGMENTED_OBJECT = 63
REJECT_NO_INSTALLATION_CHANGE = 64
REJECT_LOW_REFERENCE_SUPPORT = 65


@dataclass(frozen=True)
class ObjectStats:
    pixels: int
    bbox_fill: float
    solidity: float
    perimeter_density: float
    mean_score: float
    mean_change: float
    mean_sand: float
    mean_reference: float


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.clip(np.asarray(x, dtype=np.float32), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _safe_mean(x: np.ndarray, axis: int = 0, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    total = np.where(finite, arr, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    out = np.full(total.shape, fill, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _safe_median(x: np.ndarray, axis: int = 0, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis)
    work = np.where(valid, arr, np.nan)
    # Avoid RuntimeWarning on all-NaN slices by inserting a harmless value only
    # where every element along the reduction axis is invalid.
    if axis == 0 and np.any(count == 0):
        work = work.copy()
        rr, cc = np.where(count == 0)
        work[0, rr, cc] = fill
    with np.errstate(all="ignore"):
        out = np.nanmedian(work, axis=axis)
    return np.where(count > 0, out, fill).astype(np.float32)


def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float(default)
    return float(np.nanquantile(x, q))


def _environment_reliability(
    environment: pd.DataFrame,
    region_id: int,
    years: np.ndarray,
    reference_year: int,
    minimum: float,
) -> np.ndarray:
    """Return a per-year reliability multiplier; never a positive prior."""
    if environment.empty:
        return np.ones(len(years), dtype=np.float32)
    block = environment.loc[environment["Id"].eq(int(region_id))].copy()
    if block.empty:
        return np.ones(len(years), dtype=np.float32)
    covariates = [c for c in block.columns if c not in {"Id", "year"}]
    if not covariates:
        return np.ones(len(years), dtype=np.float32)
    block = block.set_index("year").reindex(years)
    z = np.zeros((len(years), len(covariates)), dtype=np.float32)
    for j, name in enumerate(covariates):
        values = pd.to_numeric(block[name], errors="coerce").to_numpy(dtype=float)
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        scale = max(float(mad) * 1.4826, float(np.nanstd(values)), 1e-6)
        z[:, j] = np.nan_to_num((values - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    ref_idx = int(np.where(years == int(reference_year))[0][0])
    distance = np.sqrt(np.mean((z - z[ref_idx]) ** 2, axis=1))
    # Extreme climate years can change vegetation/soil appearance, so image
    # evidence is downweighted.  Reliability cannot increase evidence above 1.
    reliability = np.exp(-0.16 * distance)
    return np.clip(reliability, float(minimum), 1.0).astype(np.float32)


def _spectral_layers(
    reflectance: np.ndarray,
    cube: np.ndarray,
    valid: np.ndarray,
    positive_similarity: np.ndarray,
    sand_similarity: np.ndarray,
    model_probability: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = {name: FEATURE_CHANNELS.index(name) for name in FEATURE_CHANNELS}
    blue, green, red = cube[idx["BLUE"]], cube[idx["GREEN"]], cube[idx["RED"]]
    nir, swir1, swir2 = cube[idx["NIR"]], cube[idx["SWIR1"]], cube[idx["SWIR2"]]
    ndvi = cube[idx["NDVI"]]
    mndwi = cube[idx["MNDWI"]]
    bsi = cube[idx["BSI"]]
    brightness = cube[idx["TC_Brightness"]]
    pv_fraction = np.clip(np.nan_to_num(cube[idx["PV_Fraction_Evidence"]]), 0.0, 1.0)

    # FEATURE_CHANNELS is intentionally compact because the same cube is used
    # for prototype fitting. Water_Evidence and Vegetation_Evidence are
    # optional confuser channels and are not present in the current prototype
    # cube. Derive conservative equivalents from MNDWI and NDVI when absent
    # instead of assuming those optional channels exist.
    if "Water_Evidence" in idx:
        water = np.clip(np.nan_to_num(cube[idx["Water_Evidence"]]), 0.0, 1.0)
    else:
        water = np.clip((np.nan_to_num(mndwi, nan=-1.0) - 0.05) / 0.35, 0.0, 1.0).astype(np.float32)

    if "Vegetation_Evidence" in idx:
        vegetation = np.clip(np.nan_to_num(cube[idx["Vegetation_Evidence"]]), 0.0, 1.0)
    else:
        vegetation = np.clip((np.nan_to_num(ndvi, nan=-1.0) - 0.18) / 0.45, 0.0, 1.0).astype(np.float32)

    if "Bare_Evidence" in idx:
        bare = np.clip(np.nan_to_num(cube[idx["Bare_Evidence"]]), 0.0, 1.0)
    else:
        bare = np.clip(
            ((np.nan_to_num(bsi, nan=-1.0) - 0.05) / 0.50)
            * ((np.nan_to_num(brightness, nan=-1.0) - 0.08) / 0.50),
            0.0,
            1.0,
        ).astype(np.float32)

    mean_reflectance = _safe_mean(reflectance[:6], axis=0, fill=0.0)
    visible = (blue + green + red) / 3.0
    nirswir = (nir + swir1 + swir2) / 3.0
    low_visible = 1.0 - _robust01(visible, valid, 0.01, 0.98)
    low_nirswir = 1.0 - _robust01(nirswir, valid, 0.01, 0.98)
    low_brightness = 1.0 - _robust01(brightness, valid, 0.01, 0.98)
    low_raw = 1.0 - _robust01(mean_reflectance, valid, 0.01, 0.98)

    fill = float(np.nanmedian(mean_reflectance[valid])) if np.any(valid) else 0.0
    local = ndimage.median_filter(np.where(valid, mean_reflectance, fill), size=17, mode="nearest")
    broad = ndimage.median_filter(np.where(valid, mean_reflectance, fill), size=45, mode="nearest")
    local_dark = _robust01(np.maximum(local - mean_reflectance, broad - mean_reflectance), valid, 0.01, 0.995)

    texture, coherence, _, _ = _texture_features(reflectance, valid)
    confuser = np.clip(0.56 * water + 0.38 * vegetation + 0.10 * np.clip(mndwi, 0.0, 1.0), 0.0, 1.0)
    sand = np.clip(
        0.42 * sand_similarity
        + 0.24 * bare
        + 0.20 * coherence
        + 0.12 * texture
        + 0.08 * np.clip(bsi, 0.0, 1.0)
        - 0.16 * pv_fraction,
        0.0,
        1.0,
    )
    uncertainty_penalty = np.clip(np.nan_to_num(uncertainty, nan=1.0), 0.0, 1.0)

    # Evidence is deliberately multi-source.  No single dark-pixel term can
    # dominate, which prevents stable dune texture from becoming PV merely
    # because it is locally dark.
    score = np.clip(
        0.28 * np.clip(model_probability, 0.0, 1.0)
        + 0.20 * positive_similarity
        + 0.18 * pv_fraction
        + 0.10 * low_visible
        + 0.08 * low_nirswir
        + 0.07 * low_brightness
        + 0.05 * low_raw
        + 0.10 * local_dark
        - 0.30 * sand
        - 0.36 * confuser
        - 0.08 * uncertainty_penalty,
        0.0,
        1.0,
    )
    for arr in (score, sand, confuser, mean_reflectance, local_dark, pv_fraction):
        arr[~valid] = 0.0
    return {
        "score": score.astype(np.float32),
        "sand": sand.astype(np.float32),
        "confuser": confuser.astype(np.float32),
        "brightness": mean_reflectance.astype(np.float32),
        "local_dark": local_dark.astype(np.float32),
        "pv_fraction": pv_fraction.astype(np.float32),
    }


def _temporal_change_support(
    score: np.ndarray,
    brightness: np.ndarray,
    valid: np.ndarray,
    reference_index: int,
    minimum_baseline_years: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute retrospective installation evidence without copying the anchor."""
    n = score.shape[0]
    onset = np.zeros_like(score, dtype=np.float32)
    persistence = np.zeros_like(score, dtype=np.float32)
    for t in range(n):
        pre_start = max(0, t - 5)
        pre = slice(pre_start, t)
        post = slice(t, min(n, t + 4))
        if t >= minimum_baseline_years:
            pre_score = _safe_median(np.where(valid[pre], score[pre], np.nan), axis=0)
            pre_brightness = _safe_median(np.where(valid[pre], brightness[pre], np.nan), axis=0)
        else:
            pre_score = np.zeros(score.shape[1:], dtype=np.float32)
            pre_brightness = brightness[t]
        post_score = _safe_median(np.where(valid[post], score[post], np.nan), axis=0)
        post_valid = valid[post].sum(axis=0)
        high_count = ((score[post] >= 0.45) & valid[post]).sum(axis=0)
        future = np.zeros(score.shape[1:], dtype=np.float32)
        np.divide(high_count, np.maximum(post_valid, 1), out=future, where=post_valid > 0)
        gain = _sigmoid((post_score - pre_score - 0.05) / 0.09)
        darkening = _sigmoid((pre_brightness - brightness[t] - 0.006) / 0.022)
        onset[t] = np.clip(0.50 * gain + 0.32 * darkening + 0.18 * future, 0.0, 1.0)
        persistence[t] = future

    # A 2024 inventory pixel installed in 2020 must still have strong historical
    # change support in 2024.  Retain the best onset evidence up to 2024.
    max_change_to_reference = np.max(onset[: reference_index + 1], axis=0)
    reference_persistence = _safe_median(
        np.where(valid[max(0, reference_index - 2) : min(n, reference_index + 2)],
                 score[max(0, reference_index - 2) : min(n, reference_index + 2)], np.nan),
        axis=0,
    )
    temporal_range = np.max(np.where(valid, score, -np.inf), axis=0) - np.min(
        np.where(valid, score, np.inf), axis=0
    )
    temporal_range[~np.any(valid, axis=0)] = 0.0
    return onset, persistence, max_change_to_reference.astype(np.float32), np.clip(reference_persistence, 0.0, 1.0)


def _object_stats(
    component: np.ndarray,
    score: np.ndarray,
    change: np.ndarray,
    sand: np.ndarray,
    reference_support: np.ndarray,
) -> ObjectStats:
    pixels = int(component.sum())
    if pixels == 0:
        return ObjectStats(0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    rr, cc = np.where(component)
    bbox = max((int(rr.max()) - int(rr.min()) + 1) * (int(cc.max()) - int(cc.min()) + 1), 1)
    bbox_fill = float(pixels / bbox)
    filled = ndimage.binary_fill_holes(component)
    solidity = float(pixels / max(int(filled.sum()), 1))
    eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool))
    perimeter_density = float(np.count_nonzero(component & ~eroded) / max(pixels, 1))
    return ObjectStats(
        pixels=pixels,
        bbox_fill=bbox_fill,
        solidity=solidity,
        perimeter_density=perimeter_density,
        mean_score=float(np.mean(score[component])),
        mean_change=float(np.mean(change[component])),
        mean_sand=float(np.mean(sand[component])),
        mean_reference=float(np.mean(reference_support[component])),
    )


def _filter_reference_inventory(
    anchor: np.ndarray,
    raw_candidate: np.ndarray,
    reference_score: np.ndarray,
    change_support: np.ndarray,
    sand_risk: np.ndarray,
    reference_support: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    settings = cfg.get("reference_reconstruction", {})
    structure = np.ones((3, 3), dtype=bool)
    closing_iterations = int(settings.get("reference_closing_iterations", 2))
    bridge_iterations = int(settings.get("reference_bridge_iterations", 1))
    grown = raw_candidate.copy()
    if closing_iterations > 0:
        grown = ndimage.binary_closing(grown, structure=structure, iterations=closing_iterations)
    if bridge_iterations > 0:
        grown = ndimage.binary_dilation(grown, structure=structure, iterations=bridge_iterations)
    labels, count = ndimage.label(grown, structure=structure)
    accepted = anchor.copy()
    rejected = np.zeros(anchor.shape, dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    min_pixels = int(settings.get("minimum_reference_object_pixels", 18))
    large_pixels = int(settings.get("large_reference_object_pixels", 60))
    for label_id in range(1, count + 1):
        obj = labels == label_id
        stats = _object_stats(obj, reference_score, change_support, sand_risk, reference_support)
        intersects_anchor = bool(np.any(obj & anchor))
        large_remote = (
            stats.pixels >= large_pixels
            and stats.mean_reference >= float(settings.get("large_object_minimum_reference_score", 0.46))
            and stats.mean_change >= float(settings.get("large_object_minimum_change_support", 0.42))
            and stats.mean_sand <= float(settings.get("large_object_maximum_sand_risk", 0.58))
        )
        ordinary = (
            stats.pixels >= min_pixels
            and stats.mean_reference >= float(settings.get("minimum_reference_object_score", 0.52))
            and stats.mean_change >= float(settings.get("minimum_reference_change_support", 0.34))
            and stats.mean_sand <= float(settings.get("maximum_reference_sand_risk", 0.50))
            and stats.solidity >= float(settings.get("minimum_reference_solidity", 0.40))
            and stats.perimeter_density <= float(settings.get("maximum_reference_perimeter_density", 0.90))
        )
        keep = intersects_anchor or large_remote or ordinary
        if keep:
            accepted |= obj
            reason = "anchor" if intersects_anchor else ("large_remote_change_site" if large_remote else "reference_outside_supplement")
        else:
            reason = "rejected"
            if stats.mean_sand > float(settings.get("maximum_reference_sand_risk", 0.50)):
                rejected[obj] = REJECT_STABLE_SAND
                reason = "stable_sand"
            elif stats.pixels < min_pixels or stats.perimeter_density > float(settings.get("maximum_reference_perimeter_density", 0.90)):
                rejected[obj] = REJECT_FRAGMENTED_OBJECT
                reason = "fragmented_or_edge"
            elif stats.mean_change < float(settings.get("minimum_reference_change_support", 0.34)):
                rejected[obj] = REJECT_NO_INSTALLATION_CHANGE
                reason = "no_installation_change"
            else:
                rejected[obj] = REJECT_LOW_REFERENCE_SUPPORT
                reason = "low_reference_support"
        rows.append({
            "object_id": int(label_id),
            "pixels": stats.pixels,
            "bbox_fill": stats.bbox_fill,
            "solidity": stats.solidity,
            "perimeter_density": stats.perimeter_density,
            "mean_reference_score": stats.mean_reference,
            "mean_change_support": stats.mean_change,
            "mean_sand_risk": stats.mean_sand,
            "intersects_2024_vector": intersects_anchor,
            "accepted": bool(keep),
            "decision": reason,
        })
    accepted[anchor] = True
    rejected[anchor] = REJECT_NONE
    return accepted, rejected, rows


def _filter_annual_evidence(
    candidate: np.ndarray,
    score: np.ndarray,
    onset: np.ndarray,
    persistence: np.ndarray,
    sand: np.ndarray,
    confuser: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    settings = cfg.get("reference_reconstruction", {})
    structure = np.ones((3, 3), dtype=bool)
    candidate = ndimage.binary_opening(candidate, structure=structure, iterations=1)
    candidate = ndimage.binary_closing(candidate, structure=structure, iterations=1)
    labels, count = ndimage.label(candidate, structure=structure)
    out = np.zeros(candidate.shape, dtype=bool)
    min_pixels = int(settings.get("minimum_annual_object_pixels", 8))
    for label_id in range(1, count + 1):
        obj = labels == label_id
        pixels = int(obj.sum())
        if pixels < min_pixels:
            continue
        rr, cc = np.where(obj)
        bbox = max((int(rr.max()) - int(rr.min()) + 1) * (int(cc.max()) - int(cc.min()) + 1), 1)
        fill = float(pixels / bbox)
        mean_score = float(np.mean(score[obj]))
        mean_onset = float(np.mean(onset[obj]))
        mean_persistence = float(np.mean(persistence[obj]))
        mean_sand = float(np.mean(sand[obj]))
        mean_confuser = float(np.mean(confuser[obj]))
        keep = (
            mean_score >= float(settings.get("minimum_annual_object_score", 0.48))
            and mean_sand <= float(settings.get("maximum_annual_sand_risk", 0.58))
            and mean_confuser <= float(settings.get("maximum_annual_confuser", 0.52))
            and fill >= float(settings.get("minimum_annual_bbox_fill", 0.08))
            and (
                mean_onset >= float(settings.get("minimum_annual_onset", 0.22))
                or mean_persistence >= float(settings.get("minimum_annual_persistence", 0.46))
                or mean_score >= float(settings.get("annual_strong_score_override", 0.72))
            )
        )
        if keep:
            out |= obj
    return out


def reconstruct_reference_constrained_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("reference_reconstruction", {})
    if not bool(settings.get("enabled", True)):
        root = Path(cfg["paths"]["output_root"])
        return pd.read_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv")

    root = Path(cfg["paths"]["output_root"])
    canonical = root / "07_probability_calibration" / "calibrated_inference_registry.csv"
    source_registry = pd.read_csv(canonical).sort_values(["Id", "year"])
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    env_path = root / "01_input_validation" / "environmental_covariates.csv"
    environment = pd.read_csv(env_path) if env_path.exists() else pd.DataFrame()

    out = ensure_dir(root / "07_reference_reconstruction")
    probability_root = ensure_dir(out / "annual_evidence_probability_30m")
    mask_root = ensure_dir(out / "annual_evidence_masks_30m")
    class_root = ensure_dir(out / "annual_evidence_source_30m")
    rejection_root = ensure_dir(out / "rejection_code_30m")
    anchor_root = ensure_dir(out / "reference_anchor_2024_30m")
    inventory_root = ensure_dir(out / "reference_inventory_2024_30m")
    supplement_root = ensure_dir(out / "reference_outside_supplement_2024_30m")
    change_root = ensure_dir(out / "installation_change_support_30m")
    sand_root = ensure_dir(out / "stable_sand_risk_30m")
    prototype_root = ensure_dir(out / "unit_prototypes")
    object_root = ensure_dir(out / "reference_object_tables")
    registry_root = ensure_dir(out / "unit_registries")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")

    years = np.arange(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1, dtype=int)
    reference_year = int(cfg["study"]["reference_year"])
    reference_index = int(np.where(years == reference_year)[0][0])
    tile_size = int(settings.get("tile_size", 384))
    halo = int(settings.get("halo_pixels", 96))
    mask_nodata = int(cfg.get("calibration", {}).get("mask_nodata", 255))
    diagnostics_units = {int(x) for x in settings.get("diagnostic_unit_ids", [6, 28])}
    diagnostic_on_global = bool(settings.get("write_diagnostics", False)) or test_mode_enabled(cfg)
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    prototype_cfg = dict(cfg)
    prototype_cfg["object_graph_refinement"] = {
        **cfg.get("object_graph_refinement", {}),
        "prototype_tile_size": int(settings.get("prototype_tile_size", 512)),
        "maximum_prototype_samples": int(settings.get("maximum_prototype_samples", 120000)),
        "positive_prototype_clusters": int(settings.get("positive_prototype_clusters", 4)),
        "sand_prototype_clusters": int(settings.get("sand_prototype_clusters", 6)),
    }

    iterator = tqdm(
        list(source_registry.groupby("Id")),
        desc="2024 reference inventory and annual evidence",
        unit="region",
        disable=not cfg.get("processing", {}).get("show_progress", True),
    )
    for region_id, group in iterator:
        region_id = int(region_id)
        group = group.sort_values("year")
        aligned_group = aligned.loc[aligned["Id"].eq(region_id)].sort_values("year")
        if len(group) != len(years) or len(aligned_group) != len(years):
            raise RuntimeError(f"Id {region_id}: expected {len(years)} annual records")
        if group["inference_mode"].astype(str).eq("area_only_fallback").all():
            passthrough = group.copy()
            passthrough["reference_reconstruction"] = "not_applicable_area_only"
            all_rows.append(passthrough)
            continue

        unit_registry_path = registry_root / f"unit_{region_id:03d}.csv"
        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        if checkpoint.exists() and unit_registry_path.exists() and not cfg.get("processing", {}).get("overwrite", False):
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing = pd.read_csv(unit_registry_path)
                if payload.get("status") == "passed" and payload.get("schema_version") == "24.18.0" and len(existing) == len(years):
                    if all(Path(p).exists() for p in existing["calibrated_probability_path"].astype(str)):
                        all_rows.append(existing)
                        summaries.append(payload)
                        continue
            except Exception:
                pass

        endmembers = load_endmember_matrix(quality.loc[region_id, "endmember_path"])
        year_rows = {int(row.year): row for row in group.itertuples(index=False)}
        landsat_paths = {int(row.year): Path(str(row.path)) for row in aligned_group.itertuples(index=False)}
        fraction_path = Path(str(supervision.loc[region_id, "pv_fraction_path"]))
        label_path = Path(str(supervision.loc[region_id, "label_path"]))
        support_path = Path(str(supervision.loc[region_id, "anchor_support_path"]))
        prototypes: PrototypeBundle = _build_unit_prototypes(
            prototype_cfg,
            region_id,
            year_rows[reference_year],
            landsat_paths[reference_year],
            fraction_path,
            label_path,
            support_path,
            endmembers,
            prototype_root / f"unit_{region_id:03d}_reference_prototypes.npz",
        )
        climate_reliability = _environment_reliability(
            environment,
            region_id,
            years,
            reference_year,
            float(settings.get("minimum_environment_reliability", 0.58)),
        )

        prob_paths = [ensure_dir(probability_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_annual_pv_evidence_30m.tif" for y in years]
        mask_paths = [ensure_dir(mask_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_annual_pv_evidence_mask_30m.tif" for y in years]
        class_paths = [ensure_dir(class_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_annual_evidence_source_30m.tif" for y in years]
        rejection_paths = [ensure_dir(rejection_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_rejection_code_30m.tif" for y in years]
        anchor_path = anchor_root / f"unit_{region_id:03d}_reference_anchor_2024_30m.tif"
        inventory_path = inventory_root / f"unit_{region_id:03d}_reference_inventory_2024_30m.tif"
        supplement_path = supplement_root / f"unit_{region_id:03d}_reference_outside_supplement_2024_30m.tif"
        change_path = change_root / f"unit_{region_id:03d}_maximum_installation_change_support_30m.tif"
        sand_path = sand_root / f"unit_{region_id:03d}_stable_sand_risk_30m.tif"

        annual_pixels = np.zeros(len(years), dtype=np.int64)
        anchor_pixels = 0
        supplement_pixels = 0
        object_rows: list[dict[str, Any]] = []
        with ExitStack() as stack:
            calibrated_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].calibrated_probability_path))) for y in years]
            uncertainty_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].epistemic_std_path))) for y in years]
            landsat_srcs = [stack.enter_context(rasterio.open(landsat_paths[int(y)])) for y in years]
            fraction_src = stack.enter_context(rasterio.open(fraction_path))
            ref = calibrated_srcs[0]
            width, height = int(ref.width), int(ref.height)
            float_profile = ref.profile.copy()
            float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            byte_profile = ref.profile.copy()
            byte_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            prob_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in prob_paths]
            mask_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in mask_paths]
            diagnostic_on = diagnostic_on_global and region_id in diagnostics_units
            class_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in class_paths] if diagnostic_on else []
            rejection_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in rejection_paths] if diagnostic_on else []
            anchor_dst = stack.enter_context(rasterio.open(anchor_path, "w", **byte_profile))
            inventory_dst = stack.enter_context(rasterio.open(inventory_path, "w", **byte_profile))
            supplement_dst = stack.enter_context(rasterio.open(supplement_path, "w", **byte_profile))
            change_dst = stack.enter_context(rasterio.open(change_path, "w", **float_profile))
            sand_dst = stack.enter_context(rasterio.open(sand_path, "w", **float_profile))

            for core in _windows(width, height, tile_size):
                expanded, crop = _expanded(core, width, height, halo)
                shape = (int(expanded.height), int(expanded.width))
                n_years = len(years)
                score = np.zeros((n_years, *shape), dtype=np.float32)
                sand = np.zeros_like(score)
                confuser = np.zeros_like(score)
                brightness = np.zeros_like(score)
                local_dark = np.zeros_like(score)
                pv_fraction = np.zeros_like(score)
                valid = np.zeros_like(score, dtype=bool)

                for t, year in enumerate(years):
                    reflectance = _read_reflectance(landsat_srcs[t], expanded)
                    valid[t] = np.all(np.isfinite(reflectance[:6]), axis=0)
                    cube = _feature_cube(reflectance, endmembers)
                    positive_similarity = _prototype_similarity(cube, prototypes.positive_centers, prototypes.positive_scales)
                    sand_similarity = _prototype_similarity(cube, prototypes.sand_centers, prototypes.sand_scales)
                    model = calibrated_srcs[t].read(1, window=expanded).astype(np.float32)
                    uncertainty = uncertainty_srcs[t].read(1, window=expanded).astype(np.float32)
                    layers = _spectral_layers(
                        reflectance,
                        cube,
                        valid[t],
                        positive_similarity,
                        sand_similarity,
                        np.where(np.isfinite(model), model, 0.0),
                        np.where(np.isfinite(uncertainty), uncertainty, 1.0),
                    )
                    reliability = float(climate_reliability[t])
                    # Climate can only reduce confidence.  It never raises a
                    # low image-evidence score into a positive detection.
                    score[t] = np.clip(layers["score"] * reliability, 0.0, 1.0)
                    sand[t] = layers["sand"]
                    confuser[t] = layers["confuser"]
                    brightness[t] = layers["brightness"]
                    local_dark[t] = layers["local_dark"]
                    pv_fraction[t] = layers["pv_fraction"]
                    del reflectance, cube, positive_similarity, sand_similarity, layers

                anchor_fraction = fraction_src.read(1, window=expanded).astype(np.float32)
                # The 2024 SHP/reference vector is a sufficient positive condition:
                # every pixel with non-trivial vector coverage is PV in the reference year.
                # It must not be gated by 2024 Landsat validity; cloudy/nodata pixels inside
                # the reference vector are still confirmed 2024 PV and are forced only in
                # the reference year downstream.
                reference_inclusion_fraction = float(
                    cfg.get("anchor_constraint", {}).get(
                        "reference_inclusion_fraction",
                        cfg.get("supervision", {}).get("strong_positive_fraction", 0.7),
                    )
                )
                anchor = np.isfinite(anchor_fraction) & (anchor_fraction >= reference_inclusion_fraction)
                onset, persistence, max_change, reference_persistence = _temporal_change_support(
                    score,
                    brightness,
                    valid,
                    reference_index,
                    int(settings.get("minimum_baseline_years", 2)),
                )
                temporal_variation = np.max(score, axis=0) - np.min(score, axis=0)
                stable_sand = np.clip(
                    0.58 * _safe_median(sand, axis=0)
                    + 0.25 * (1.0 - np.clip(temporal_variation / 0.25, 0.0, 1.0))
                    + 0.17 * (1.0 - max_change),
                    0.0,
                    1.0,
                )
                reference_support = np.clip(
                    0.46 * score[reference_index]
                    + 0.24 * reference_persistence
                    + 0.22 * max_change
                    + 0.08 * local_dark[reference_index]
                    - 0.22 * stable_sand
                    - 0.20 * confuser[reference_index],
                    0.0,
                    1.0,
                )
                raw_reference_candidate = (
                    valid[reference_index]
                    & ~anchor
                    & (confuser[reference_index] <= float(settings.get("reference_maximum_confuser", 0.52)))
                    & (stable_sand <= float(settings.get("reference_candidate_maximum_sand", 0.62)))
                    & (
                        (reference_support >= float(settings.get("reference_candidate_score", 0.50)))
                        | (
                            (max_change >= float(settings.get("reference_candidate_change_override", 0.60)))
                            & (score[reference_index] >= float(settings.get("reference_candidate_minimum_current_score", 0.34)))
                        )
                    )
                )
                inventory, reference_rejection, tile_object_rows = _filter_reference_inventory(
                    anchor,
                    raw_reference_candidate,
                    score[reference_index],
                    max_change,
                    stable_sand,
                    reference_support,
                    cfg,
                )
                supplement = inventory & ~anchor
                for row in tile_object_rows:
                    row.update({
                        "Id": region_id,
                        "tile_col": int(core.col_off),
                        "tile_row": int(core.row_off),
                    })
                object_rows.extend(tile_object_rows)

                annual_masks = np.zeros_like(valid, dtype=bool)
                annual_classes = np.zeros_like(valid, dtype=np.uint8)
                annual_rejection = np.zeros_like(valid, dtype=np.uint8)
                for t, year in enumerate(years):
                    raw = (
                        valid[t]
                        & (score[t] >= float(settings.get("annual_candidate_score", 0.44)))
                        & (sand[t] <= float(settings.get("annual_candidate_maximum_sand", 0.64)))
                        & (confuser[t] <= float(settings.get("maximum_annual_confuser", 0.52)))
                        & (
                            (onset[t] >= float(settings.get("annual_candidate_onset", 0.18)))
                            | (persistence[t] >= float(settings.get("annual_candidate_persistence", 0.42)))
                            | (score[t] >= float(settings.get("annual_candidate_strong_override", 0.72)))
                        )
                    )
                    annual_masks[t] = _filter_annual_evidence(
                        raw,
                        score[t],
                        onset[t],
                        persistence[t],
                        sand[t],
                        confuser[t],
                        cfg,
                    )
                    annual_classes[t][annual_masks[t]] = EVIDENCE_ANNUAL_IMAGE
                    annual_rejection[t][valid[t] & ~annual_masks[t] & (stable_sand >= float(settings.get("stable_sand_rejection_threshold", 0.66)))] = REJECT_STABLE_SAND
                    annual_rejection[t][valid[t] & ~annual_masks[t] & (confuser[t] >= float(settings.get("confuser_rejection_threshold", 0.70)))] = REJECT_WATER_VEGETATION

                # The reference vector is imposed in 2024 only.  This line is
                # intentionally after annual evidence filtering and there is no
                # analogous operation for any earlier year.
                annual_masks[reference_index][anchor] = True
                annual_classes[reference_index][anchor] = EVIDENCE_2024_VECTOR
                annual_rejection[reference_index][anchor] = REJECT_NONE
                annual_masks[reference_index][supplement] = True
                annual_classes[reference_index][supplement] = EVIDENCE_2024_OUTSIDE_SUPPLEMENT
                annual_rejection[reference_index][supplement] = REJECT_NONE

                # Annual posterior remains evidence based before 2024.  The
                # reference inventory raises only the reference-year posterior.
                posterior = np.clip(
                    0.70 * score + 0.18 * onset + 0.12 * persistence - 0.18 * stable_sand[None, ...],
                    0.0,
                    1.0,
                )
                posterior[~valid] = 0.0
                posterior[reference_index][anchor] = 1.0
                posterior[reference_index][supplement] = np.maximum(
                    posterior[reference_index][supplement],
                    float(settings.get("outside_reference_posterior_floor", 0.82)),
                )

                core_valid = valid[:, crop[0], crop[1]]
                core_anchor = anchor[crop]
                for t in range(n_years):
                    core_prob = posterior[t][crop]
                    core_mask = annual_masks[t][crop]
                    # Reference-year vector positives remain valid labels even where
                    # the Landsat stack is nodata.  Other years remain strictly image-evidence based.
                    write_valid = core_valid[t] | (core_anchor if t == reference_index else False)
                    encoded_prob = np.where(write_valid, core_prob, -9999.0).astype(np.float32)
                    encoded_mask = np.where(write_valid, core_mask.astype(np.uint8), mask_nodata).astype(np.uint8)
                    prob_dsts[t].write(encoded_prob, 1, window=core)
                    mask_dsts[t].write(encoded_mask, 1, window=core)
                    annual_pixels[t] += int(np.count_nonzero(core_mask & write_valid))
                    if diagnostic_on:
                        class_dsts[t].write(np.where(write_valid, annual_classes[t][crop], mask_nodata).astype(np.uint8), 1, window=core)
                        rejection_dsts[t].write(np.where(write_valid, annual_rejection[t][crop], mask_nodata).astype(np.uint8), 1, window=core)

                core_ref_valid = core_valid[reference_index]
                core_inventory = inventory[crop]
                core_supplement = supplement[crop]
                reference_domain = core_ref_valid | core_anchor | core_inventory | core_supplement
                anchor_domain = core_ref_valid | core_anchor
                anchor_dst.write(np.where(anchor_domain, core_anchor.astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                inventory_dst.write(np.where(reference_domain, core_inventory.astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                supplement_dst.write(np.where(reference_domain, core_supplement.astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                change_dst.write(np.where(np.any(core_valid, axis=0), max_change[crop], -9999.0).astype(np.float32), 1, window=core)
                sand_dst.write(np.where(np.any(core_valid, axis=0), stable_sand[crop], -9999.0).astype(np.float32), 1, window=core)
                anchor_pixels += int(np.count_nonzero(core_anchor))
                supplement_pixels += int(np.count_nonzero(core_supplement))

        if object_rows:
            save_table(pd.DataFrame(object_rows), object_root / f"unit_{region_id:03d}_reference_objects.csv")
        unit_rows: list[dict[str, Any]] = []
        for t, year in enumerate(years):
            original = group.iloc[t].to_dict()
            original.update({
                "pre_reference_reconstruction_probability_path": original.get("calibrated_probability_path", ""),
                "pre_reference_reconstruction_mask_path": original.get("calibrated_mask_path", ""),
                "calibrated_probability_path": str(prob_paths[t]),
                "calibrated_mask_path": str(mask_paths[t]),
                "reference_reconstruction": "v24_18_reference_inventory_and_annual_image_evidence",
                "reference_inventory_path": str(inventory_path),
                "reference_anchor_path": str(anchor_path),
                "reference_outside_supplement_path": str(supplement_path),
                "maximum_installation_change_support_path": str(change_path),
                "stable_sand_risk_path": str(sand_path),
                "environment_reliability": float(climate_reliability[t]),
                "annual_evidence_pixels": int(annual_pixels[t]),
                "reference_vector_forced_this_year": bool(int(year) == reference_year),
            })
            unit_rows.append(original)
        frame = pd.DataFrame(unit_rows)
        save_table(frame, unit_registry_path)
        payload = {
            "status": "passed",
            "schema_version": "24.18.0",
            "Id": region_id,
            "reference_year": reference_year,
            "reference_anchor_pixels": int(anchor_pixels),
            "reference_outside_supplement_pixels": int(supplement_pixels),
            "annual_evidence_pixels": {str(int(y)): int(annual_pixels[i]) for i, y in enumerate(years)},
            "historical_anchor_copy": False,
            "environment_creates_labels": False,
            "created_utc": utc_now(),
        }
        atomic_json(payload, checkpoint)
        all_rows.append(frame)
        summaries.append(payload)

    result = pd.concat(all_rows, ignore_index=True, sort=False).sort_values(["Id", "year"])
    backup = root / "07_probability_calibration" / "calibrated_inference_registry_pre_v24_18_reference_reconstruction.csv"
    if not backup.exists():
        source_registry.to_csv(backup, index=False)
    save_table(result, canonical)
    save_table(result, out / "reference_reconstruction_registry.csv")
    atomic_json({
        "status": "passed",
        "schema_version": "24.18.0",
        "units": int(result["Id"].nunique()),
        "records": int(len(result)),
        "reference_semantics": "inside vector is certain PV in 2024; outside remains unlabelled and can be supplemented",
        "historical_semantics": "no vector geometry is copied to 2011-2023; annual evidence and Stage-8 change-point reconstruction determine onset",
        "environment_semantics": "climate variables only downweight unreliable spectral years and never create a historical PV label",
        "created_utc": utc_now(),
    }, out / "reference_reconstruction_status.json")
    return result
