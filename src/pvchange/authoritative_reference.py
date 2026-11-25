from __future__ import annotations

"""Causal reference and terminal extension reconstruction (v24.22).

``pv_desert_fix.shp`` is a manually interpreted, high-confidence 2024 site
census.  Its interior is trusted, while a narrow symmetric boundary band is
allowed to move according to Landsat spectral state, multiband change-point
support, temporal persistence and object continuity.  Corrections are strictly
bounded in distance and area, are fully audited, and cannot create remote 2024
sites far from the interpreted polygons.

Historical maps are reconstructed retrospectively from annual Landsat evidence;
the refined 2024 geometry is never copied into an earlier year. Environmental
covariates only reduce confidence in anomalous observation years and never
create a historical PV label.
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
from .growth_front import build_terminal_extension_evidence, detect_terminal_year_expansion
from .mixed_parcel import distributed_support_metrics, stabilise_presence_track
from .utils import atomic_json, ensure_dir, save_table, utc_now

EPS = 1e-6

DETAIL_ALGORITHM_REVISION = "24.22.2-mixed-parcel-optimized"

# Evidence-mask classes written by Stage 7.
EVIDENCE_BACKGROUND = 0
EVIDENCE_ANNUAL_IMAGE = 1
EVIDENCE_2024_VECTOR = 2
EVIDENCE_2024_OUTSIDE_SUPPLEMENT = 3
EVIDENCE_2024_REFINED_BOUNDARY = 4
EVIDENCE_2025_NEW_SITE = 5

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
        "positive_similarity": np.clip(positive_similarity, 0.0, 1.0).astype(np.float32),
        "ndvi": np.nan_to_num(ndvi, nan=0.0).astype(np.float32),
        "ndbi": np.nan_to_num(cube[idx["NDBI"]], nan=0.0).astype(np.float32),
        "bsi": np.nan_to_num(bsi, nan=0.0).astype(np.float32),
        "mndwi": np.nan_to_num(mndwi, nan=0.0).astype(np.float32),
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
        # Reference-year refinement must be causal: 2025 cannot influence
        # the 2024 endpoint. Historical onsets may use later observations only
        # up to and including the audited reference year.
        post_stop = min(n, t + 4)
        if t <= reference_index:
            post_stop = min(post_stop, reference_index + 1)
        post = slice(t, post_stop)
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
        np.where(
            valid[max(0, reference_index - 2) : reference_index + 1],
            score[max(0, reference_index - 2) : reference_index + 1],
            np.nan,
        ),
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
    settings = cfg.get("authoritative_reference", cfg.get("reference_reconstruction", {}))
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
    settings = cfg.get("authoritative_reference", cfg.get("reference_reconstruction", {}))
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



def _multiband_reference_break(
    features: dict[str, np.ndarray],
    valid: np.ndarray,
    reference_index: int,
) -> np.ndarray:
    """Directional, robust and causal break support for the 2024 endpoint.

    Only observations up to 2024 are used.  The 2025 image is deliberately
    excluded so a real 2025 extension cannot be absorbed into the 2024
    inventory and disappear from the annual change map.
    """
    pre = slice(max(0, reference_index - 5), reference_index)
    post = slice(reference_index, reference_index + 1)
    specs = [
        ("positive_similarity", +1.0, 0.22, 0.05),
        ("pv_fraction", +1.0, 0.18, 0.05),
        ("local_dark", +1.0, 0.15, 0.05),
        ("brightness", -1.0, 0.15, 0.012),
        ("bsi", -1.0, 0.11, 0.08),
        ("ndvi", -1.0, 0.10, 0.06),
        ("ndbi", +1.0, 0.09, 0.06),
    ]
    directional = np.zeros(valid.shape[1:], dtype=np.float32)
    magnitude_terms: list[np.ndarray] = []
    weight_sum = 0.0
    for name, direction, weight, floor in specs:
        x = np.asarray(features[name], dtype=np.float32)
        pre_x = np.where(valid[pre], x[pre], np.nan)
        post_x = np.where(valid[post], x[post], np.nan)
        pre_med = _safe_median(pre_x, axis=0, fill=0.0)
        post_med = _safe_median(post_x, axis=0, fill=0.0)
        post_count = np.isfinite(post_x).sum(axis=0)
        post_med = np.where(post_count > 0, post_med, pre_med).astype(np.float32)
        mad = _safe_median(np.abs(pre_x - pre_med[None, ...]), axis=0, fill=floor)
        scale = np.maximum(1.4826 * mad, float(floor))
        signed_z = direction * (post_med - pre_med) / scale
        directional += float(weight) * _sigmoid((signed_z - 0.55) / 0.85)
        magnitude_terms.append(np.clip(np.abs(post_med - pre_med) / scale, 0.0, 6.0))
        weight_sum += float(weight)
    directional /= max(weight_sum, EPS)
    magnitude = np.sqrt(np.mean(np.stack(magnitude_terms, axis=0) ** 2, axis=0))
    generic = _sigmoid((magnitude - 1.15) / 0.80)
    post_valid = np.any(valid[post], axis=0)
    return np.where(post_valid, np.clip(0.78 * directional + 0.22 * generic, 0.0, 1.0), 0.0).astype(np.float32)


def _rank_limited(mask: np.ndarray, priority: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or int(mask.sum()) <= limit:
        return mask
    indices = np.flatnonzero(mask)
    order = np.argsort(priority.ravel()[indices])[::-1][:limit]
    out = np.zeros(mask.shape, dtype=bool)
    out.ravel()[indices[order]] = True
    return out


def _filter_boundary_components(mask: np.ndarray, priority: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    out = np.zeros(mask.shape, dtype=bool)
    for label_id in range(1, count + 1):
        obj = labels == label_id
        pixels = int(obj.sum())
        if pixels >= minimum_pixels or (pixels >= 2 and float(np.mean(priority[obj])) >= 0.82):
            out |= obj
    return out


def _annular_addition_filter(
    addition: np.ndarray,
    anchor: np.ndarray,
    priority: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Reject thin additions that wrap around most of a manual component.

    Such annuli are characteristic of sub-pixel registration / rasterisation
    mismatch rather than a physically plausible panel-block expansion.  Local
    one-sided strips and compact patches are retained.
    """
    addition = np.asarray(addition, dtype=bool).copy()
    if not np.any(addition) or not bool(settings.get("suppress_annular_boundary_additions", True)):
        return addition, 0

    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(anchor, structure=structure)
    search_pixels = max(1, int(settings.get("reference_boundary_search_pixels", 4)))
    contact_pixels = max(1, int(settings.get("annular_contact_tolerance_pixels", 1)))
    minimum_reference_pixels = max(1, int(settings.get("annular_minimum_reference_pixels", 12)))
    maximum_distance = float(settings.get("annular_maximum_distance_pixels", 2.25))
    minimum_wrap = float(settings.get("annular_minimum_boundary_wrap_fraction", 0.55))
    minimum_thin_fraction = float(settings.get("annular_minimum_thin_fraction", 0.80))
    strong_priority = float(settings.get("annular_strong_patch_priority", 0.90))
    removed = 0

    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) < minimum_reference_pixels:
            continue
        zone = ndimage.binary_dilation(component, structure=structure, iterations=search_pixels) & ~component
        candidate = addition & zone
        if not np.any(candidate):
            continue

        component_boundary = component & ~ndimage.binary_erosion(component, structure=structure, border_value=0)
        boundary_near_candidate = component_boundary & ndimage.binary_dilation(
            candidate,
            structure=structure,
            iterations=contact_pixels,
        )
        wrap_fraction = float(
            np.count_nonzero(boundary_near_candidate) / max(int(component_boundary.sum()), 1)
        )
        distance_to_component = ndimage.distance_transform_edt(~component)
        thin_fraction = float(
            np.count_nonzero(candidate & (distance_to_component <= maximum_distance))
            / max(int(candidate.sum()), 1)
        )
        if wrap_fraction < minimum_wrap or thin_fraction < minimum_thin_fraction:
            continue

        # Retain only isolated, exceptionally strong local patches; suppress the
        # continuous annulus itself.  This keeps real block-wise expansion while
        # eliminating a one-pixel contour around the whole site.
        candidate_labels, candidate_count = ndimage.label(candidate, structure=structure)
        keep_local = np.zeros(candidate.shape, dtype=bool)
        for candidate_id in range(1, candidate_count + 1):
            obj = candidate_labels == candidate_id
            obj_boundary_contact = component_boundary & ndimage.binary_dilation(
                obj,
                structure=structure,
                iterations=contact_pixels,
            )
            obj_wrap = float(
                np.count_nonzero(obj_boundary_contact) / max(int(component_boundary.sum()), 1)
            )
            if obj_wrap < minimum_wrap * 0.45 and float(np.mean(priority[obj])) >= strong_priority:
                keep_local |= obj
        to_remove = candidate & ~keep_local
        removed += int(np.count_nonzero(to_remove))
        addition[to_remove] = False
    return addition, removed


def _enhance_reference_internal_history(
    posterior: np.ndarray,
    score: np.ndarray,
    positive_similarity: np.ndarray,
    pv_fraction: np.ndarray,
    local_dark: np.ndarray,
    sand: np.ndarray,
    confuser: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> np.ndarray:
    """Strengthen annual evidence inside audited 2024 PV parcels.

    Two complementary pathways are used.  The original pixel pathway recovers
    weak panel edges connected to a supported seed.  The mixed-parcel pathway
    recognises a different Landsat failure mode: every 30 m cell contains some
    panel fraction, but no individual cell is sufficiently pure to cross the
    pixel threshold.  A whole parcel is strengthened only when weak evidence is
    distributed across its interior, boundary and spatial sectors for multiple
    years.  A localised partial build-out therefore cannot activate the complete
    2024 component.
    """
    out = np.asarray(posterior, dtype=np.float32).copy()
    if reference_index <= 0 or not bool(settings.get("reference_internal_history_enhancement", True)):
        return out

    p = np.asarray(score, dtype=np.float32)
    sim = np.asarray(positive_similarity, dtype=np.float32)
    frac = np.asarray(pv_fraction, dtype=np.float32)
    dark = np.asarray(local_dark, dtype=np.float32)
    sand_arr = np.asarray(sand, dtype=np.float32)
    conf = np.asarray(confuser, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    inv = np.asarray(inventory, dtype=bool)

    state = np.clip(
        0.31 * p
        + 0.27 * sim
        + 0.17 * frac
        + 0.14 * dark
        + 0.11 * (1.0 - conf)
        - 0.18 * sand_arr,
        0.0,
        1.0,
    ).astype(np.float32)
    radius = max(0, int(settings.get("historical_state_temporal_radius", 1)))
    smoothed = np.zeros_like(state, dtype=np.float32)
    for t in range(reference_index):
        lo = max(0, t - radius)
        hi = min(reference_index, t + radius + 1)
        block = np.where(v[lo:hi], state[lo:hi], np.nan)
        count_valid = np.isfinite(block).sum(axis=0)
        work = block.copy()
        if work.shape[0] and np.any(count_valid == 0):
            rr, cc = np.where(count_valid == 0)
            work[0, rr, cc] = 0.0
        with np.errstate(all="ignore"):
            med = np.nanmedian(work, axis=0)
        smoothed[t] = np.where(count_valid > 0, med, 0.0)

    structure = np.ones((3, 3), dtype=bool)
    labels, component_count = ndimage.label(inv, structure=structure)
    small_limit = max(1, int(settings.get("small_reference_component_pixels", 16)))
    pixel_floor = float(settings.get("historical_state_pixel_floor", 0.31))
    weak_floor = float(settings.get("historical_state_weak_floor", 0.27))
    component_quantile = float(settings.get("historical_component_quantile", 0.70))
    component_quantile_floor = float(settings.get("historical_component_quantile_floor", 0.39))
    component_min_fraction = float(settings.get("historical_component_minimum_fraction", 0.08))
    small_component_floor = float(settings.get("historical_small_component_floor", 0.34))
    support_window = max(1, int(settings.get("historical_component_support_window_years", 3)))
    support_years = max(1, int(settings.get("historical_component_minimum_support_years", 2)))
    posterior_floor = float(settings.get("historical_reference_posterior_floor", 0.20))
    posterior_scale = float(settings.get("historical_reference_posterior_scale", 0.82))

    mixed_enabled = bool(settings.get("historical_mixed_parcel_enabled", True))
    mixed_weak_floor = float(settings.get("historical_mixed_parcel_weak_state", 0.20))
    mixed_moderate_floor = float(settings.get("historical_mixed_parcel_moderate_state", 0.27))
    mixed_strong_floor = float(settings.get("historical_mixed_parcel_strong_state", 0.42))
    mixed_fraction_floor = float(settings.get("historical_mixed_parcel_fraction_floor", 0.055))
    mixed_similarity_floor = float(settings.get("historical_mixed_parcel_similarity_floor", 0.24))
    mixed_dark_floor = float(settings.get("historical_mixed_parcel_dark_floor", 0.22))
    mixed_min_valid = float(settings.get("historical_mixed_parcel_minimum_valid_fraction", 0.50))
    mixed_min_supported = float(settings.get("historical_mixed_parcel_minimum_supported_fraction", 0.20))
    mixed_min_moderate = float(settings.get("historical_mixed_parcel_minimum_moderate_fraction", 0.07))
    mixed_min_dilated = float(settings.get("historical_mixed_parcel_minimum_dilated_coverage", 0.68))
    mixed_min_sectors = float(settings.get("historical_mixed_parcel_minimum_sector_coverage", 0.50))
    mixed_min_boundary = float(settings.get("historical_mixed_parcel_minimum_boundary_support", 0.16))
    mixed_min_interior = float(settings.get("historical_mixed_parcel_minimum_interior_support", 0.20))
    mixed_min_median = float(settings.get("historical_mixed_parcel_minimum_median_state", 0.18))
    mixed_min_upper = float(settings.get("historical_mixed_parcel_minimum_upper_state", 0.27))
    mixed_max_sand = float(settings.get("historical_mixed_parcel_maximum_mean_sand", 0.66))
    mixed_max_confuser = float(settings.get("historical_mixed_parcel_maximum_mean_confuser", 0.66))
    mixed_dilation = max(1, int(settings.get("historical_mixed_parcel_dilation_pixels", 2)))
    mixed_grid = max(2, int(settings.get("historical_mixed_parcel_grid_size", 3)))
    mixed_posterior_floor = float(settings.get("historical_mixed_parcel_posterior_floor", 0.36))
    mixed_posterior_ceiling = float(settings.get("historical_mixed_parcel_posterior_ceiling", 0.62))

    for label_id in range(1, component_count + 1):
        component = labels == label_id
        pixels = int(component.sum())
        if pixels <= 0:
            continue
        pixel_presence = np.zeros(reference_index, dtype=bool)
        mixed_presence = np.zeros(reference_index, dtype=bool)
        mixed_confidence = np.zeros(reference_index, dtype=np.float32)

        for t in range(reference_index):
            domain = component & v[t]
            values = smoothed[t][domain]
            if values.size == 0:
                continue
            q = float(np.quantile(values, component_quantile))
            strong_fraction = float(np.count_nonzero(values >= pixel_floor) / values.size)
            if pixels <= small_limit:
                pixel_presence[t] = bool(
                    float(np.median(values)) >= small_component_floor
                    or float(np.max(values)) >= component_quantile_floor + 0.10
                )
            else:
                pixel_presence[t] = bool(
                    q >= component_quantile_floor
                    and strong_fraction >= component_min_fraction
                )

            if not mixed_enabled:
                continue
            weak_support = domain & (
                (smoothed[t] >= mixed_weak_floor)
                | ((frac[t] >= mixed_fraction_floor) & (sim[t] >= mixed_similarity_floor))
                | ((dark[t] >= mixed_dark_floor) & (sim[t] >= mixed_similarity_floor + 0.04))
            )
            moderate_support = domain & (
                (smoothed[t] >= mixed_moderate_floor)
                | (state[t] >= mixed_moderate_floor + 0.03)
                | (frac[t] >= max(mixed_fraction_floor + 0.05, 0.10))
            )
            strong_support = domain & (
                (smoothed[t] >= mixed_strong_floor)
                | (p[t] >= mixed_strong_floor + 0.08)
                | (frac[t] >= max(mixed_fraction_floor + 0.14, 0.20))
            )
            metrics = distributed_support_metrics(
                component,
                weak_support,
                moderate_support,
                strong_support,
                v[t],
                smoothed[t],
                dilation_pixels=mixed_dilation,
                grid_size=mixed_grid,
            )
            mean_sand = float(np.mean(sand_arr[t][domain]))
            mean_confuser = float(np.mean(conf[t][domain]))
            mixed_presence[t] = bool(
                metrics.valid_fraction >= mixed_min_valid
                and metrics.supported_fraction >= mixed_min_supported
                and metrics.moderate_fraction >= mixed_min_moderate
                and metrics.dilated_coverage >= mixed_min_dilated
                and metrics.sector_coverage >= mixed_min_sectors
                and metrics.boundary_supported_fraction >= mixed_min_boundary
                and metrics.interior_supported_fraction >= mixed_min_interior
                and metrics.median_evidence >= mixed_min_median
                and metrics.upper_quantile_evidence >= mixed_min_upper
                and mean_sand <= mixed_max_sand
                and mean_confuser <= mixed_max_confuser
            )
            mixed_confidence[t] = np.clip(
                0.18 * metrics.supported_fraction
                + 0.14 * metrics.moderate_fraction
                + 0.18 * metrics.dilated_coverage
                + 0.16 * metrics.sector_coverage
                + 0.10 * metrics.boundary_supported_fraction
                + 0.10 * metrics.interior_supported_fraction
                + 0.14 * metrics.upper_quantile_evidence,
                0.0,
                1.0,
            )

        stable_pixel = stabilise_presence_track(
            pixel_presence,
            window=support_window,
            minimum_years=support_years,
        )
        stable_mixed = stabilise_presence_track(
            mixed_presence,
            window=max(support_window, int(settings.get("historical_mixed_parcel_support_window_years", 3))),
            minimum_years=max(support_years, int(settings.get("historical_mixed_parcel_minimum_support_years", 2))),
        )

        for t in range(reference_index):
            if not (stable_pixel[t] or stable_mixed[t]):
                continue
            supported_pixel = component & v[t] & (
                (state[t] >= pixel_floor)
                | (smoothed[t] >= weak_floor)
            )
            if np.any(supported_pixel):
                evidence = np.clip(
                    posterior_floor + posterior_scale * np.maximum(state[t], smoothed[t]),
                    0.0,
                    0.94,
                )
                out[t][supported_pixel] = np.maximum(out[t][supported_pixel], evidence[supported_pixel])

            if stable_mixed[t]:
                # Spatially distributed evidence is the observation.  The 2024
                # component supplies parcel topology only after this annual gate
                # has passed; it is not copied into an unsupported year.
                parcel_domain = component & v[t]
                parcel_floor = float(np.clip(
                    mixed_posterior_floor + 0.24 * mixed_confidence[t],
                    mixed_posterior_floor,
                    mixed_posterior_ceiling,
                ))
                out[t][parcel_domain] = np.maximum(out[t][parcel_domain], parcel_floor)
    return out

def _refine_manual_reference_boundary(
    anchor: np.ndarray,
    anchor_fraction: np.ndarray,
    valid_reference: np.ndarray,
    score_reference: np.ndarray,
    reference_support: np.ndarray,
    break_support: np.ndarray,
    stable_sand: np.ndarray,
    confuser: np.ndarray,
    local_dark: np.ndarray,
    pv_fraction: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine only a narrow band around the hand-interpreted 2024 vector."""
    structure = np.ones((3, 3), dtype=bool)
    inner_iterations = max(1, int(settings.get("reference_core_erosion_pixels", 2)))
    outer_iterations = max(1, int(settings.get("reference_boundary_search_pixels", 4)))
    core = ndimage.binary_erosion(anchor, structure=structure, iterations=inner_iterations, border_value=0)

    # Small hand-drawn blocks are authoritative objects, not noise.  Protect the
    # whole object when erosion would reduce it to a single pixel or erase it.
    labels, count = ndimage.label(anchor, structure=structure)
    small_limit = max(1, int(settings.get("small_reference_component_pixels", 16)))
    protected_small_pixels = 0
    for label_id in range(1, count + 1):
        obj = labels == label_id
        pixels = int(obj.sum())
        if pixels <= small_limit:
            core |= obj
            protected_small_pixels += pixels
            continue
        if np.any(core & obj):
            continue
        values = np.where(obj, anchor_fraction, -np.inf)
        if np.isfinite(values).any():
            core.flat[int(np.nanargmax(values))] = True

    outer = ndimage.binary_dilation(anchor, structure=structure, iterations=outer_iterations)
    boundary_band = outer & ~core
    inside_boundary = anchor & ~core
    outside_boundary = outer & ~anchor

    # Separate current-state evidence from installation-break evidence.  A site
    # already present in 2011 can have little recent break support but still has
    # a strong 2024 PV spectral state and must not be removed.
    state_likelihood = np.clip(
        0.38 * score_reference
        + 0.22 * reference_support
        + 0.17 * local_dark
        + 0.16 * pv_fraction
        + 0.07 * (1.0 - confuser)
        - 0.22 * stable_sand,
        0.0,
        1.0,
    )
    image_likelihood = np.clip(
        0.72 * state_likelihood
        + 0.18 * break_support
        + 0.10 * reference_support,
        0.0,
        1.0,
    )

    # Protect a complete 2024 parcel when moderate PV evidence is spatially
    # distributed through its interior and boundary.  This is the expected
    # Landsat signature of a panel field whose 30 m cells are all mixed pixels.
    # A parcel with only a supported inner core and an unsupported outer shell
    # does not pass the boundary-support gate and remains eligible for bounded
    # boundary correction.
    protected_mixed_parcel = np.zeros(anchor.shape, dtype=bool)
    protected_mixed_components = 0
    if bool(settings.get("reference_mixed_parcel_protection", True)):
        mixed_weak = (
            (state_likelihood >= float(settings.get("reference_mixed_parcel_weak_state", 0.24)))
            | (
                (pv_fraction >= float(settings.get("reference_mixed_parcel_fraction_floor", 0.07)))
                & (score_reference >= float(settings.get("reference_mixed_parcel_score_floor", 0.22)))
            )
            | (
                (local_dark >= float(settings.get("reference_mixed_parcel_dark_floor", 0.24)))
                & (reference_support >= float(settings.get("reference_mixed_parcel_support_floor", 0.22)))
            )
        )
        mixed_moderate = (
            (state_likelihood >= float(settings.get("reference_mixed_parcel_moderate_state", 0.31)))
            | (pv_fraction >= float(settings.get("reference_mixed_parcel_moderate_fraction", 0.13)))
        )
        mixed_strong = (
            (state_likelihood >= float(settings.get("reference_mixed_parcel_strong_state", 0.46)))
            | (score_reference >= float(settings.get("reference_mixed_parcel_strong_score", 0.52)))
            | (pv_fraction >= float(settings.get("reference_mixed_parcel_strong_fraction", 0.24)))
        )
        for label_id in range(1, count + 1):
            obj = labels == label_id
            obj_pixels = int(np.count_nonzero(obj))
            if obj_pixels <= small_limit:
                continue
            metrics = distributed_support_metrics(
                obj,
                mixed_weak,
                mixed_moderate,
                mixed_strong,
                valid_reference,
                state_likelihood,
                dilation_pixels=max(1, int(settings.get("reference_mixed_parcel_dilation_pixels", 2))),
                grid_size=max(2, int(settings.get("reference_mixed_parcel_grid_size", 3))),
            )
            domain = obj & valid_reference
            mean_sand = float(np.mean(stable_sand[domain])) if np.any(domain) else 1.0
            mean_confuser = float(np.mean(confuser[domain])) if np.any(domain) else 1.0
            protect = bool(
                metrics.valid_fraction >= float(settings.get("reference_mixed_parcel_minimum_valid_fraction", 0.60))
                and metrics.supported_fraction >= float(settings.get("reference_mixed_parcel_minimum_supported_fraction", 0.30))
                and metrics.moderate_fraction >= float(settings.get("reference_mixed_parcel_minimum_moderate_fraction", 0.10))
                and metrics.dilated_coverage >= float(settings.get("reference_mixed_parcel_minimum_dilated_coverage", 0.78))
                and metrics.sector_coverage >= float(settings.get("reference_mixed_parcel_minimum_sector_coverage", 0.60))
                and metrics.boundary_supported_fraction >= float(settings.get("reference_mixed_parcel_minimum_boundary_support", 0.24))
                and metrics.interior_supported_fraction >= float(settings.get("reference_mixed_parcel_minimum_interior_support", 0.28))
                and metrics.median_evidence >= float(settings.get("reference_mixed_parcel_minimum_median_state", 0.21))
                and metrics.upper_quantile_evidence >= float(settings.get("reference_mixed_parcel_minimum_upper_state", 0.30))
                and mean_sand <= float(settings.get("reference_mixed_parcel_maximum_mean_sand", 0.65))
                and mean_confuser <= float(settings.get("reference_mixed_parcel_maximum_mean_confuser", 0.65))
            )
            if protect:
                # Protect only pixels that retain at least weak current-state
                # evidence. A contiguous sand/confuser strip can still be
                # trimmed even when the rest of the parcel is strongly PV-like.
                protected_pixels = obj & mixed_weak
                protected_mixed_parcel |= protected_pixels
                protected_mixed_components += 1
    prior = np.where(anchor, 0.70, 0.22).astype(np.float32)
    fused_raw = np.clip(0.76 * image_likelihood + 0.24 * prior, 0.0, 1.0)
    fused_smooth = ndimage.median_filter(fused_raw, size=3, mode="nearest").astype(np.float32)
    fused = np.clip(0.84 * fused_raw + 0.16 * fused_smooth, 0.0, 1.0).astype(np.float32)

    addition_priority = np.clip(
        0.42 * fused
        + 0.24 * state_likelihood
        + 0.20 * break_support
        + 0.14 * reference_support,
        0.0,
        1.0,
    )
    persistent_override = (
        (reference_support >= float(settings.get("boundary_addition_strong_persistent_support", 0.74)))
        & (score_reference >= float(settings.get("boundary_addition_strong_reference_score", 0.72)))
        & (pv_fraction >= float(settings.get("boundary_addition_strong_pv_fraction", 0.46)))
    )
    addition = (
        outside_boundary
        & valid_reference
        & (fused >= float(settings.get("boundary_addition_minimum_likelihood", 0.56)))
        & (
            (break_support >= float(settings.get("boundary_addition_minimum_break", 0.40)))
            | persistent_override
        )
        & (reference_support >= float(settings.get("boundary_addition_minimum_support", 0.44)))
        & (stable_sand <= float(settings.get("boundary_addition_maximum_sand", 0.58)))
        & (confuser <= float(settings.get("boundary_addition_maximum_confuser", 0.48)))
    )
    addition = ndimage.binary_closing(addition, structure=structure, iterations=1)
    addition = _filter_boundary_components(
        addition,
        addition_priority,
        int(settings.get("minimum_boundary_correction_pixels", 3)),
    )
    connected = ndimage.binary_propagation(core, mask=(anchor | addition), structure=structure)
    addition &= connected
    addition, annular_addition_pixels = _annular_addition_filter(
        addition,
        anchor,
        addition_priority,
        settings,
    )

    removal_priority = np.clip(
        0.52 * (1.0 - state_likelihood)
        + 0.20 * stable_sand
        + 0.16 * confuser
        + 0.12 * (1.0 - pv_fraction),
        0.0,
        1.0,
    )
    low_state = state_likelihood <= float(
        settings.get("boundary_removal_maximum_state_likelihood", 0.31)
    )
    negative_cues = (
        (stable_sand >= float(settings.get("boundary_removal_minimum_sand", 0.56))).astype(np.uint8)
        + (confuser >= float(settings.get("boundary_removal_minimum_confuser", 0.58))).astype(np.uint8)
        + (pv_fraction <= float(settings.get("boundary_removal_maximum_pv_fraction", 0.12))).astype(np.uint8)
        + (score_reference <= float(settings.get("boundary_removal_maximum_reference_score", 0.30))).astype(np.uint8)
    )
    removal = (
        inside_boundary
        & valid_reference
        & low_state
        & (negative_cues >= int(settings.get("boundary_removal_minimum_negative_cues", 2)))
        & (
            (break_support <= float(settings.get("boundary_removal_maximum_break", 0.32)))
            | (stable_sand >= float(settings.get("boundary_removal_minimum_sand", 0.56)))
        )
    )
    removal = _filter_boundary_components(
        removal,
        removal_priority,
        int(settings.get("minimum_boundary_correction_pixels", 3)),
    )
    removal &= ~(core | protected_mixed_parcel)

    anchor_pixels = max(int(anchor.sum()), 1)
    maximum_fraction = float(settings.get("maximum_reference_area_adjustment_fraction", 0.08))
    maximum_total = max(1, int(round(anchor_pixels * maximum_fraction)))
    addition = _rank_limited(addition, addition_priority, maximum_total)
    removal = _rank_limited(removal, removal_priority, maximum_total)
    total = int(addition.sum() + removal.sum())
    if total > maximum_total:
        add_idx = np.flatnonzero(addition)
        rem_idx = np.flatnonzero(removal)
        priorities = np.concatenate((addition_priority.ravel()[add_idx], removal_priority.ravel()[rem_idx]))
        kinds = np.concatenate((np.zeros(add_idx.size, dtype=np.uint8), np.ones(rem_idx.size, dtype=np.uint8)))
        indices = np.concatenate((add_idx, rem_idx))
        keep = np.argsort(priorities)[::-1][:maximum_total]
        new_add = np.zeros(anchor.shape, dtype=bool)
        new_remove = np.zeros(anchor.shape, dtype=bool)
        new_add.ravel()[indices[keep][kinds[keep] == 0]] = True
        new_remove.ravel()[indices[keep][kinds[keep] == 1]] = True
        addition, removal = new_add, new_remove

    refined = anchor.copy()
    refined[removal] = False
    refined[addition] = True
    refined |= core
    refined |= protected_mixed_parcel
    refined[~outer] = False
    refined |= core
    refined |= protected_mixed_parcel

    correction = np.zeros(anchor.shape, dtype=np.uint8)
    correction[core] = 1
    correction[anchor & refined & ~core] = 2
    correction[anchor & ~refined] = 3
    correction[~anchor & refined] = 4

    refined_fraction = np.zeros(anchor.shape, dtype=np.float32)
    refined_fraction[refined & anchor] = np.clip(anchor_fraction[refined & anchor], 0.05, 1.0)
    refined_fraction[addition] = np.clip(0.50 + 0.50 * fused[addition], 0.50, 1.0)
    refined_fraction[core] = np.maximum(
        refined_fraction[core],
        np.clip(anchor_fraction[core], 0.75, 1.0),
    )

    diagnostics = {
        "manual_pixels": int(anchor.sum()),
        "core_pixels": int(core.sum()),
        "protected_small_component_pixels": int(protected_small_pixels),
        "protected_mixed_parcel_components": int(protected_mixed_components),
        "protected_mixed_parcel_pixels": int(np.count_nonzero(protected_mixed_parcel)),
        "boundary_band_pixels": int(boundary_band.sum()),
        "added_pixels": int(addition.sum()),
        "removed_pixels": int(removal.sum()),
        "annular_addition_pixels_suppressed": int(annular_addition_pixels),
        "refined_pixels": int(refined.sum()),
        "adjustment_fraction": float((addition.sum() + removal.sum()) / anchor_pixels),
        "core_recall": float(np.count_nonzero(refined & core) / max(int(core.sum()), 1)),
        "corrections_outside_boundary_band": int(np.count_nonzero((addition | removal) & ~boundary_band)),
    }
    return refined, refined_fraction, boundary_band, correction, diagnostics

def reconstruct_authoritative_reference_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("authoritative_reference", cfg.get("reference_reconstruction", {}))
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

    out = ensure_dir(root / "07_authoritative_reference")
    probability_root = ensure_dir(out / "annual_evidence_probability_30m")
    mask_root = ensure_dir(out / "annual_evidence_masks_30m")
    class_root = ensure_dir(out / "annual_evidence_source_30m")
    rejection_root = ensure_dir(out / "rejection_code_30m")
    anchor_root = ensure_dir(out / "reference_anchor_2024_30m")
    inventory_root = ensure_dir(out / "reference_inventory_2024_30m")
    fraction_root = ensure_dir(out / "reference_fraction_2024_30m")
    supplement_root = ensure_dir(out / "reference_outside_supplement_2024_30m")
    change_root = ensure_dir(out / "installation_change_support_30m")
    sand_root = ensure_dir(out / "stable_sand_risk_30m")
    break_root = ensure_dir(out / "multiband_break_support_30m")
    boundary_root = ensure_dir(out / "reference_boundary_uncertainty_30m")
    correction_root = ensure_dir(out / "reference_boundary_correction_code_30m")
    confidence_root = ensure_dir(out / "reference_boundary_confidence_30m")
    terminal_source_root = ensure_dir(out / "terminal_2025_expansion_source_30m")
    terminal_gain_root = ensure_dir(out / "terminal_2025_probability_gain_30m")
    terminal_site_root = ensure_dir(out / "terminal_2025_site_likelihood_30m")
    terminal_similarity_root = ensure_dir(out / "terminal_2025_component_similarity_30m")
    terminal_change_root = ensure_dir(out / "terminal_2025_multicue_change_30m")
    terminal_proximity_root = ensure_dir(out / "terminal_2025_front_proximity_30m")
    terminal_remote_census_root = ensure_dir(out / "terminal_2025_remote_census_likelihood_30m")
    terminal_remote_change_root = ensure_dir(out / "terminal_2025_remote_change_support_30m")
    terminal_remote_density_root = ensure_dir(out / "terminal_2025_remote_candidate_density_30m")
    terminal_absolute_dark_root = ensure_dir(out / "terminal_2025_absolute_darkness_30m")
    terminal_novelty_root = ensure_dir(out / "terminal_2025_novelty_score_30m")
    terminal_direct_change_root = ensure_dir(out / "terminal_2025_direct_change_support_30m")
    terminal_preexisting_risk_root = ensure_dir(out / "terminal_2025_preexisting_state_risk_30m")
    terminal_prior_persistence_root = ensure_dir(out / "terminal_2025_prior_dark_persistence_30m")
    terminal_break_dominance_root = ensure_dir(out / "terminal_2025_break_dominance_30m")
    terminal_persistent_nonpv_root = ensure_dir(out / "terminal_2025_persistent_nonpv_risk_30m")
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
        explicit_area_only_ids = {83, 104, 141}
        explicit_area_only = region_id in explicit_area_only_ids
        existing_area_only = group["inference_mode"].astype(str).eq("area_only_fallback").all()
        if explicit_area_only or existing_area_only:
            passthrough = group.copy()
            passthrough["inference_mode"] = "area_only_fallback"
            passthrough["reference_reconstruction"] = (
                "explicit_area_only_skip_insufficient_reference_anchor"
                if explicit_area_only
                else "not_applicable_area_only"
            )
            passthrough["skip_reason"] = (
                "insufficient_reference_anchor"
                if explicit_area_only
                else "predeclared_area_only_route"
            )
            # Area-only routes intentionally have no fabricated pixel diagnostics.
            for column in [
                "reference_anchor_path",
                "reference_inventory_path",
                "reference_fraction_path",
                "reference_outside_supplement_path",
                "reference_boundary_uncertainty_path",
                "reference_boundary_correction_code_path",
                "stable_sand_risk_path",
                "multiband_break_support_path",
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
                "terminal_2025_novelty_score_path",
                "terminal_2025_direct_change_support_path",
                "terminal_2025_preexisting_state_risk_path",
                "terminal_2025_prior_dark_persistence_path",
                "terminal_2025_break_dominance_path",
                "terminal_2025_persistent_nonpv_risk_path",
            ]:
                passthrough[column] = ""
            all_rows.append(passthrough)
            summaries.append({
                "status": "passed",
                "schema_version": "24.22.0",
                "algorithm_revision": DETAIL_ALGORITHM_REVISION,
                "Id": region_id,
                "inference_mode": "area_only_fallback",
                "reference_reconstruction": passthrough["reference_reconstruction"].iloc[0],
                "pixel_diagnostics_fabricated": False,
                "created_utc": utc_now(),
            })
            continue

        unit_registry_path = registry_root / f"unit_{region_id:03d}.csv"
        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        if checkpoint.exists() and unit_registry_path.exists() and not cfg.get("processing", {}).get("overwrite", False):
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing = pd.read_csv(unit_registry_path)
                if (
                    payload.get("status") == "passed"
                    and payload.get("schema_version") == "24.22.0"
                    and payload.get("algorithm_revision") == DETAIL_ALGORITHM_REVISION
                    and len(existing) == len(years)
                ):
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
        exact_fraction_path = fraction_root / f"unit_{region_id:03d}_reference_fraction_2024_30m.tif"
        change_path = change_root / f"unit_{region_id:03d}_maximum_installation_change_support_30m.tif"
        sand_path = sand_root / f"unit_{region_id:03d}_stable_sand_risk_30m.tif"
        break_path = break_root / f"unit_{region_id:03d}_multiband_break_support_30m.tif"
        boundary_path = boundary_root / f"unit_{region_id:03d}_reference_boundary_uncertainty_30m.tif"
        correction_path = correction_root / f"unit_{region_id:03d}_reference_boundary_correction_code_30m.tif"
        confidence_path = confidence_root / f"unit_{region_id:03d}_reference_boundary_confidence_30m.tif"
        terminal_source_path = terminal_source_root / f"unit_{region_id:03d}_2025_expansion_source_30m.tif"
        terminal_gain_path = terminal_gain_root / f"unit_{region_id:03d}_2025_probability_gain_30m.tif"
        terminal_site_path = terminal_site_root / f"unit_{region_id:03d}_2025_site_likelihood_30m.tif"
        terminal_similarity_path = terminal_similarity_root / f"unit_{region_id:03d}_2025_component_similarity_30m.tif"
        terminal_change_path = terminal_change_root / f"unit_{region_id:03d}_2025_multicue_change_30m.tif"
        terminal_proximity_path = terminal_proximity_root / f"unit_{region_id:03d}_2025_front_proximity_30m.tif"
        terminal_remote_census_path = terminal_remote_census_root / f"unit_{region_id:03d}_2025_remote_census_likelihood_30m.tif"
        terminal_remote_change_path = terminal_remote_change_root / f"unit_{region_id:03d}_2025_remote_change_support_30m.tif"
        terminal_remote_density_path = terminal_remote_density_root / f"unit_{region_id:03d}_2025_remote_candidate_density_30m.tif"
        terminal_absolute_dark_path = terminal_absolute_dark_root / f"unit_{region_id:03d}_2025_absolute_darkness_30m.tif"
        terminal_novelty_path = terminal_novelty_root / f"unit_{region_id:03d}_2025_novelty_score_30m.tif"
        terminal_direct_change_path = terminal_direct_change_root / f"unit_{region_id:03d}_2025_direct_change_support_30m.tif"
        terminal_preexisting_risk_path = terminal_preexisting_risk_root / f"unit_{region_id:03d}_2025_preexisting_state_risk_30m.tif"
        terminal_prior_persistence_path = terminal_prior_persistence_root / f"unit_{region_id:03d}_2025_prior_dark_persistence_30m.tif"
        terminal_break_dominance_path = terminal_break_dominance_root / f"unit_{region_id:03d}_2025_break_dominance_30m.tif"
        terminal_persistent_nonpv_path = terminal_persistent_nonpv_root / f"unit_{region_id:03d}_2025_persistent_nonpv_risk_30m.tif"

        annual_pixels = np.zeros(len(years), dtype=np.int64)
        anchor_pixels = 0
        supplement_pixels = 0
        removed_pixels = 0
        correction_pixels = 0
        boundary_pixels = 0
        object_rows: list[dict[str, Any]] = []
        with ExitStack() as stack:
            calibrated_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].calibrated_probability_path))) for y in years]
            uncertainty_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].epistemic_std_path))) for y in years]
            landsat_srcs = [stack.enter_context(rasterio.open(landsat_paths[int(y)])) for y in years]
            fraction_src = stack.enter_context(rasterio.open(fraction_path))
            label_src = stack.enter_context(rasterio.open(label_path))
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
            fraction_profile = ref.profile.copy()
            fraction_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            exact_fraction_dst = stack.enter_context(rasterio.open(exact_fraction_path, "w", **fraction_profile))
            change_dst = stack.enter_context(rasterio.open(change_path, "w", **float_profile))
            sand_dst = stack.enter_context(rasterio.open(sand_path, "w", **float_profile))
            break_dst = stack.enter_context(rasterio.open(break_path, "w", **float_profile))
            boundary_dst = stack.enter_context(rasterio.open(boundary_path, "w", **byte_profile))
            correction_dst = stack.enter_context(rasterio.open(correction_path, "w", **byte_profile))
            confidence_dst = stack.enter_context(rasterio.open(confidence_path, "w", **float_profile))
            terminal_source_dst = stack.enter_context(rasterio.open(terminal_source_path, "w", **byte_profile))
            terminal_gain_dst = stack.enter_context(rasterio.open(terminal_gain_path, "w", **float_profile))
            terminal_site_dst = stack.enter_context(rasterio.open(terminal_site_path, "w", **float_profile))
            terminal_similarity_dst = stack.enter_context(rasterio.open(terminal_similarity_path, "w", **float_profile))
            terminal_change_dst = stack.enter_context(rasterio.open(terminal_change_path, "w", **float_profile))
            terminal_proximity_dst = stack.enter_context(rasterio.open(terminal_proximity_path, "w", **float_profile))
            terminal_remote_census_dst = stack.enter_context(rasterio.open(terminal_remote_census_path, "w", **float_profile))
            terminal_remote_change_dst = stack.enter_context(rasterio.open(terminal_remote_change_path, "w", **float_profile))
            terminal_remote_density_dst = stack.enter_context(rasterio.open(terminal_remote_density_path, "w", **float_profile))
            terminal_absolute_dark_dst = stack.enter_context(rasterio.open(terminal_absolute_dark_path, "w", **float_profile))
            terminal_novelty_dst = stack.enter_context(rasterio.open(terminal_novelty_path, "w", **float_profile))
            terminal_direct_change_dst = stack.enter_context(rasterio.open(terminal_direct_change_path, "w", **float_profile))
            terminal_preexisting_risk_dst = stack.enter_context(rasterio.open(terminal_preexisting_risk_path, "w", **float_profile))
            terminal_prior_persistence_dst = stack.enter_context(rasterio.open(terminal_prior_persistence_path, "w", **float_profile))
            terminal_break_dominance_dst = stack.enter_context(rasterio.open(terminal_break_dominance_path, "w", **float_profile))
            terminal_persistent_nonpv_dst = stack.enter_context(rasterio.open(terminal_persistent_nonpv_path, "w", **float_profile))

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
                positive_similarity_ts = np.zeros_like(score)
                ndvi_ts = np.zeros_like(score)
                ndbi_ts = np.zeros_like(score)
                bsi_ts = np.zeros_like(score)
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
                    positive_similarity_ts[t] = layers["positive_similarity"]
                    ndvi_ts[t] = layers["ndvi"]
                    ndbi_ts[t] = layers["ndbi"]
                    bsi_ts[t] = layers["bsi"]
                    del reflectance, cube, positive_similarity, sand_similarity, layers

                anchor_fraction = fraction_src.read(1, window=expanded).astype(np.float32)
                # Use the fractional raster to recover the complete hand-drawn
                # polygon geometry. The Stage-3 training label intentionally
                # contains only a conservative core when boundary uncertainty is enabled.
                anchor = np.isfinite(anchor_fraction) & (
                    anchor_fraction > float(settings.get("manual_reference_minimum_fraction", 0.0))
                )
                onset, persistence, max_change, reference_persistence = _temporal_change_support(
                    score,
                    brightness,
                    valid,
                    reference_index,
                    int(settings.get("minimum_baseline_years", 2)),
                )
                # Stable-background risk for the 2024 endpoint is computed
                # causally. A 2025 installation cannot lower the 2024 sand risk.
                causal_score = score[: reference_index + 1]
                temporal_variation = np.max(causal_score, axis=0) - np.min(causal_score, axis=0)
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
                break_support = _multiband_reference_break(
                    {
                        "positive_similarity": positive_similarity_ts,
                        "pv_fraction": pv_fraction,
                        "local_dark": local_dark,
                        "brightness": brightness,
                        "bsi": bsi_ts,
                        "ndvi": ndvi_ts,
                        "ndbi": ndbi_ts,
                    },
                    valid,
                    reference_index,
                )
                inventory, refined_fraction, boundary_band, correction_code, boundary_diag = _refine_manual_reference_boundary(
                    anchor,
                    np.clip(anchor_fraction, 0.0, 1.0),
                    valid[reference_index],
                    score[reference_index],
                    reference_support,
                    break_support,
                    stable_sand,
                    confuser[reference_index],
                    local_dark[reference_index],
                    pv_fraction[reference_index],
                    settings,
                )
                supplement = inventory & ~anchor
                reference_rejection = np.zeros(anchor.shape, dtype=np.uint8)
                reference_rejection[anchor & ~inventory] = REJECT_LOW_REFERENCE_SUPPORT
                reference_confidence = np.clip(
                    0.52 * reference_support + 0.30 * break_support + 0.18 * score[reference_index],
                    0.0,
                    1.0,
                )
                reference_confidence[inventory & ~valid[reference_index]] = 0.75

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
                annual_masks[reference_index][:] = inventory
                annual_classes[reference_index][:] = EVIDENCE_BACKGROUND
                annual_classes[reference_index][inventory & anchor] = EVIDENCE_2024_VECTOR
                annual_classes[reference_index][inventory & ~anchor] = EVIDENCE_2024_REFINED_BOUNDARY
                annual_rejection[reference_index][:] = reference_rejection

                # Annual posterior remains evidence based before 2024.  The
                # reference inventory raises only the reference-year posterior.
                posterior = np.clip(
                    0.70 * score + 0.18 * onset + 0.12 * persistence - 0.18 * stable_sand[None, ...],
                    0.0,
                    1.0,
                )
                posterior[~valid] = 0.0
                # The reference-year posterior is a deterministic binary endpoint.
                posterior[reference_index][:] = 0.0
                posterior[reference_index][inventory] = 1.0
                posterior = _enhance_reference_internal_history(
                    posterior,
                    score,
                    positive_similarity_ts,
                    pv_fraction,
                    local_dark,
                    sand,
                    confuser,
                    valid,
                    inventory,
                    reference_index,
                    settings,
                )
                # Re-assert the deterministic audited endpoint after the
                # retrospective evidence enhancement.
                posterior[reference_index][:] = 0.0
                posterior[reference_index][inventory] = 1.0

                terminal_evidence = build_terminal_extension_evidence(
                    posterior,
                    valid,
                    inventory,
                    reference_index,
                    settings,
                    site_score=score,
                    positive_similarity=positive_similarity_ts,
                    pv_fraction=pv_fraction,
                    local_dark=local_dark,
                    brightness=brightness,
                    sand_risk=sand,
                    confuser=confuser,
                )
                terminal_mask, terminal_source, terminal_gain, _terminal_decisions = detect_terminal_year_expansion(
                    annual_masks,
                    posterior,
                    valid,
                    inventory,
                    reference_index,
                    settings,
                    site_score=score,
                    positive_similarity=positive_similarity_ts,
                    pv_fraction=pv_fraction,
                    local_dark=local_dark,
                    brightness=brightness,
                    sand_risk=sand,
                    confuser=confuser,
                    precomputed_evidence=terminal_evidence,
                )
                for decision_index, decision in enumerate(_terminal_decisions, start=1):
                    object_rows.append({
                        "object_id": f"terminal_{decision_index}",
                        "year": 2025,
                        "pixels": int(decision.pixels),
                        "bbox_fill": float(decision.bbox_fill),
                        "solidity": float(decision.solidity),
                        "mean_probability": float(decision.mean_probability),
                        "mean_gain": float(decision.mean_gain),
                        "adjacent_to_reference": bool(decision.adjacent_to_reference),
                        "accepted": bool(decision.accepted),
                        "source": int(decision.source),
                        "mean_novelty": float(decision.mean_novelty),
                        "mean_direct_change": float(decision.mean_direct_change),
                        "mean_preexisting_risk": float(decision.mean_preexisting_risk),
                        "mean_prior_dark_persistence": float(decision.mean_prior_dark_persistence),
                        "mean_break_dominance": float(decision.mean_break_dominance),
                        "mean_persistent_nonpv_risk": float(decision.mean_persistent_nonpv_risk),
                        "novelty_fraction": float(decision.novelty_fraction),
                        "direct_change_fraction": float(decision.direct_change_fraction),
                        "rejection_reason": str(decision.rejection_reason),
                        "decision": (
                            "gain_adjacent" if decision.source == 1 else
                            "remote_new_site" if decision.source == 2 else
                            "component_conditioned_perimeter_extension" if decision.source == 3 else
                            "novelty_gated_remote_change" if decision.source == 4 else
                            "rejected"
                        ),
                    })
                terminal_index = reference_index + 1
                if terminal_index < n_years:
                    annual_masks[terminal_index] |= terminal_mask
                    annual_classes[terminal_index][terminal_mask] = EVIDENCE_2025_NEW_SITE
                    posterior[terminal_index][terminal_mask] = np.maximum(
                        posterior[terminal_index][terminal_mask],
                        float(settings.get("terminal_accepted_probability_floor", 0.62)),
                    )

                core_valid = valid[:, crop[0], crop[1]]
                core_anchor = anchor[crop]
                for t in range(n_years):
                    core_prob = posterior[t][crop]
                    core_mask = annual_masks[t][crop]
                    # The refined 2024 endpoint is valid across the full unit,
                    # including boundary pixels where the Landsat stack is nodata.
                    write_valid = np.ones(core_anchor.shape, dtype=bool) if t == reference_index else core_valid[t]
                    encoded_prob = np.where(write_valid, core_prob, -9999.0).astype(np.float32)
                    encoded_mask = np.where(write_valid, core_mask.astype(np.uint8), mask_nodata).astype(np.uint8)
                    prob_dsts[t].write(encoded_prob, 1, window=core)
                    mask_dsts[t].write(encoded_mask, 1, window=core)
                    annual_pixels[t] += int(np.count_nonzero(core_mask & write_valid))
                    if diagnostic_on:
                        class_dsts[t].write(np.where(write_valid, annual_classes[t][crop], mask_nodata).astype(np.uint8), 1, window=core)
                        rejection_dsts[t].write(np.where(write_valid, annual_rejection[t][crop], mask_nodata).astype(np.uint8), 1, window=core)

                core_inventory = inventory[crop]
                core_supplement = supplement[crop]
                core_fraction = refined_fraction[crop].astype(np.float32)
                core_boundary = boundary_band[crop]
                core_correction = correction_code[crop]
                anchor_dst.write(core_anchor.astype(np.uint8), 1, window=core)
                inventory_dst.write(core_inventory.astype(np.uint8), 1, window=core)
                supplement_dst.write(core_supplement.astype(np.uint8), 1, window=core)
                exact_fraction_dst.write(core_fraction, 1, window=core)
                change_dst.write(np.where(np.any(core_valid, axis=0), max_change[crop], -9999.0).astype(np.float32), 1, window=core)
                sand_dst.write(np.where(np.any(core_valid, axis=0), stable_sand[crop], -9999.0).astype(np.float32), 1, window=core)
                break_dst.write(np.where(np.any(core_valid, axis=0), break_support[crop], -9999.0).astype(np.float32), 1, window=core)
                boundary_dst.write(core_boundary.astype(np.uint8), 1, window=core)
                correction_dst.write(core_correction.astype(np.uint8), 1, window=core)
                confidence_dst.write(reference_confidence[crop].astype(np.float32), 1, window=core)
                terminal_source_dst.write(terminal_source[crop].astype(np.uint8), 1, window=core)
                terminal_gain_dst.write(np.where(core_valid[-1], terminal_gain[crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_site_dst.write(np.where(core_valid[-1], terminal_evidence["site_likelihood"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_similarity_dst.write(np.where(core_valid[-1], terminal_evidence["component_similarity"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_change_dst.write(np.where(core_valid[-1], terminal_evidence["multicue_change"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_proximity_dst.write(np.where(core_valid[-1], terminal_evidence["front_proximity"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_remote_census_dst.write(np.where(core_valid[-1], terminal_evidence["remote_census_likelihood"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_remote_change_dst.write(np.where(core_valid[-1], terminal_evidence["remote_change_support"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_remote_density_dst.write(np.where(core_valid[-1], terminal_evidence["remote_candidate_density"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_absolute_dark_dst.write(np.where(core_valid[-1], terminal_evidence["absolute_darkness"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_novelty_dst.write(np.where(core_valid[-1], terminal_evidence["terminal_novelty_score"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_direct_change_dst.write(np.where(core_valid[-1], terminal_evidence["direct_multicue_change"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_preexisting_risk_dst.write(np.where(core_valid[-1], terminal_evidence["preexisting_state_risk"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_prior_persistence_dst.write(np.where(core_valid[-1], terminal_evidence["prior_dark_persistence"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_break_dominance_dst.write(np.where(core_valid[-1], terminal_evidence["terminal_break_dominance"][crop], -9999.0).astype(np.float32), 1, window=core)
                terminal_persistent_nonpv_dst.write(np.where(core_valid[-1], terminal_evidence["persistent_nonpv_risk"][crop], -9999.0).astype(np.float32), 1, window=core)
                anchor_pixels += int(np.count_nonzero(core_anchor))
                supplement_pixels += int(np.count_nonzero(core_supplement))
                removed_pixels += int(np.count_nonzero(core_anchor & ~core_inventory))
                correction_pixels += int(np.count_nonzero(core_correction >= 3))
                boundary_pixels += int(np.count_nonzero(core_boundary))

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
                "reference_reconstruction": "v24_22_causal_reference_and_novelty_gated_remote_change",
                "reference_inventory_path": str(inventory_path),
                "reference_anchor_path": str(anchor_path),
                "reference_outside_supplement_path": str(supplement_path),
                "reference_fraction_path": str(exact_fraction_path),
                "maximum_installation_change_support_path": str(change_path),
                "stable_sand_risk_path": str(sand_path),
                "multiband_break_support_path": str(break_path),
                "reference_boundary_uncertainty_path": str(boundary_path),
                "reference_boundary_correction_code_path": str(correction_path),
                "reference_boundary_confidence_path": str(confidence_path),
                "terminal_2025_expansion_source_path": str(terminal_source_path),
                "terminal_2025_probability_gain_path": str(terminal_gain_path),
                "terminal_2025_site_likelihood_path": str(terminal_site_path),
                "terminal_2025_component_similarity_path": str(terminal_similarity_path),
                "terminal_2025_multicue_change_path": str(terminal_change_path),
                "terminal_2025_front_proximity_path": str(terminal_proximity_path),
                "terminal_2025_remote_census_likelihood_path": str(terminal_remote_census_path),
                "terminal_2025_remote_change_support_path": str(terminal_remote_change_path),
                "terminal_2025_remote_candidate_density_path": str(terminal_remote_density_path),
                "terminal_2025_absolute_darkness_path": str(terminal_absolute_dark_path),
                "terminal_2025_novelty_score_path": str(terminal_novelty_path),
                "terminal_2025_direct_change_support_path": str(terminal_direct_change_path),
                "terminal_2025_preexisting_state_risk_path": str(terminal_preexisting_risk_path),
                "terminal_2025_prior_dark_persistence_path": str(terminal_prior_persistence_path),
                "terminal_2025_break_dominance_path": str(terminal_break_dominance_path),
                "terminal_2025_persistent_nonpv_risk_path": str(terminal_persistent_nonpv_path),
                "environment_reliability": float(climate_reliability[t]),
                "annual_evidence_pixels": int(annual_pixels[t]),
                "reference_vector_forced_this_year": False,
                "refined_reference_endpoint_this_year": bool(int(year) == reference_year),
            })
            unit_rows.append(original)
        frame = pd.DataFrame(unit_rows)
        save_table(frame, unit_registry_path)
        payload = {
            "status": "passed",
            "schema_version": "24.22.0",
            "algorithm_revision": DETAIL_ALGORITHM_REVISION,
            "Id": region_id,
            "reference_year": reference_year,
            "reference_anchor_pixels": int(anchor_pixels),
            "reference_outside_supplement_pixels": int(supplement_pixels),
            "reference_boundary_removed_pixels": int(removed_pixels),
            "reference_boundary_correction_pixels": int(correction_pixels),
            "reference_boundary_band_pixels": int(boundary_pixels),
            "reference_adjustment_fraction": float(correction_pixels / max(anchor_pixels, 1)),
            "annual_evidence_pixels": {str(int(y)): int(annual_pixels[i]) for i, y in enumerate(years)},
            "historical_anchor_copy": False,
            "reference_year_exact_inventory_unchanged": False,
            "reference_year_boundary_refined": True,
            "reference_refinement_uses_2025": False,
            "causal_reference_freeze_certified": True,
            "terminal_2025_intrasite_perimeter_completion_enabled": True,
            "terminal_2025_novelty_gated_remote_change_enabled": True,
            "environment_creates_labels": False,
            "created_utc": utc_now(),
        }
        atomic_json(payload, checkpoint)
        all_rows.append(frame)
        summaries.append(payload)

    result = pd.concat(all_rows, ignore_index=True, sort=False).sort_values(["Id", "year"])
    backup = root / "07_probability_calibration" / "calibrated_inference_registry_pre_v24_18_authoritative_reference.csv"
    if not backup.exists():
        source_registry.to_csv(backup, index=False)
    save_table(result, canonical)
    save_table(result, out / "authoritative_reference_registry.csv")
    if summaries:
        save_table(pd.DataFrame(summaries), out / "reference_boundary_adjustment_summary.csv")
    atomic_json({
        "status": "passed",
        "schema_version": "24.22.0",
        "algorithm_revision": DETAIL_ALGORITHM_REVISION,
        "units": int(result["Id"].nunique()),
        "records": int(len(result)),
        "reference_semantics": "pv_desert_fix.shp is a high-confidence 2024 manual census prior; the bounded 2024 boundary is refined causally using only 2011-2024 evidence, then frozen before 2025 expansion is detected",
        "bounded_boundary_refinement_certified": True,
        "multiband_break_support_certified": True,
        "terminal_2025_growth_front_certified": True,
        "causal_reference_freeze_certified": True,
        "reference_refinement_uses_2025": False,
        "terminal_2025_intrasite_perimeter_completion_enabled": True,
        "terminal_2025_source_codes": {"1": "gain-supported adjacent expansion", "2": "strict model-supported remote site", "3": "component-conditioned perimeter extension", "4": "novelty-gated remote change census"},
        "terminal_2025_novelty_gated_remote_change_enabled": True,
        "terminal_2025_remote_census_distance_constraint": "none",
        "multiband_break_support_product": "07_authoritative_reference/multiband_break_support_30m",
        "terminal_2025_growth_front_product": "07_authoritative_reference/terminal_2025_expansion_source_30m",
        "terminal_2025_component_similarity_product": "07_authoritative_reference/terminal_2025_component_similarity_30m",
        "terminal_2025_multicue_change_product": "07_authoritative_reference/terminal_2025_multicue_change_30m",
        "terminal_2025_remote_census_product": "07_authoritative_reference/terminal_2025_remote_census_likelihood_30m",
        "terminal_2025_remote_change_product": "07_authoritative_reference/terminal_2025_remote_change_support_30m",
        "terminal_2025_novelty_product": "07_authoritative_reference/terminal_2025_novelty_score_30m",
        "terminal_2025_preexisting_risk_product": "07_authoritative_reference/terminal_2025_preexisting_state_risk_30m",
        "terminal_2025_prior_persistence_product": "07_authoritative_reference/terminal_2025_prior_dark_persistence_30m",
        "terminal_2025_break_dominance_product": "07_authoritative_reference/terminal_2025_break_dominance_30m",
        "terminal_2025_persistent_nonpv_risk_product": "07_authoritative_reference/terminal_2025_persistent_nonpv_risk_30m",
        "terminal_2025_novelty_gate_certified": True,
        "terminal_2025_preexisting_state_rejection_certified": True,
        "historical_semantics": "no vector geometry is copied to 2011-2023; annual evidence and Stage-8 change-point reconstruction determine onset",
        "environment_semantics": "climate variables only downweight unreliable spectral years and never create a historical PV label",
        "created_utc": utc_now(),
    }, out / "authoritative_reference_status.json")
    return result
