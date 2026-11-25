from __future__ import annotations

"""Site-first photovoltaic detection for annual Landsat composites.

The v24.15 detector deliberately does not start from a pixel segmentation mask.
It first proposes whole, spatially coherent dark-array sites from annual Landsat
reflectance and then uses model/prototype evidence only as supporting evidence.
Persistent sand/dune texture is represented as a hard-negative process and is
allowed to veto model detections.  A cross-year site graph reconstructs the
installation onset and prevents late sites from being backfilled into early
sand-only years.

Historical mapping contract
---------------------------
* Environmental suitability variables never create a historical PV label.
* The 2024 reference SHP is forced positive in 2024 only.
* A remote site may be accepted without model or anchor agreement when its
  absolute darkness, local contrast, object interior, geometry and temporal
  persistence are independently strong.
* A persistent sand object may be rejected even when the segmentation model is
  confident.
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
    _oriented_fill,
    _prototype_similarity,
    _read_reflectance,
    _robust01,
    _texture_features,
    _windows,
)
from .spectral_mixture import load_endmember_matrix
from .study_area import expected_counts, test_mode_enabled
from .utils import atomic_json, ensure_dir, save_table, utc_now

EPS = 1e-6

# Accepted-source classes.
CLASS_BACKGROUND = 0
CLASS_SITE_FIRST_REMOTE = 1
CLASS_REFERENCE_ANCHOR = 2
CLASS_MODEL_CONFIRMED_SITE = 3
CLASS_ADJACENCY_COMPLETION = 4
CLASS_TEMPORAL_SITE_RESCUE = 5
CLASS_EXPERT_POSITIVE = 6

# Rejection codes.
REJECT_NONE = 0
REJECT_PERSISTENT_SAND = 31
REJECT_DIRECTIONAL_DUNE_TEXTURE = 32
REJECT_FRAGMENTED_SPECKLE = 33
REJECT_HOLLOW_EDGE = 34
REJECT_WEAK_SITE_INTERIOR = 35
REJECT_WEAK_TEMPORAL_ONSET = 36
REJECT_IMPLAUSIBLE_GEOMETRY = 37
REJECT_WATER_OR_VEGETATION = 38
REJECT_LOW_SITE_EVIDENCE = 39
REJECT_EXPERT_NEGATIVE = 40
REJECT_PREMATURE_BACKFILL = 41


@dataclass(frozen=True)
class SiteMetrics:
    pixels: int
    bbox_fill: float
    solid_fill: float
    oriented_fill: float
    elongation: float
    interior_fraction: float
    perimeter_density: float
    core_fraction: float
    dark_fraction: float
    edge_only_score: float
    child_components: int


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(x, dtype=np.float32), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def _safe_nanmedian(x: np.ndarray, axis: int = 0, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis)
    # np.nanmedian emits RuntimeWarning for all-NaN slices.  Replace all-empty
    # slices temporarily, then restore the explicit fill value.
    work = np.where(valid, arr, np.nan)
    if axis == 0:
        empty = count == 0
        if np.any(empty):
            work = work.copy()
            work[(slice(None),) + np.where(empty)] = fill
    with np.errstate(all="ignore"):
        out = np.nanmedian(work, axis=axis)
    return np.where(count > 0, out, fill).astype(np.float32)


def _safe_range(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.where(valid, x, 0.0).astype(np.float32)
    any_valid = valid.any(axis=0)
    lo = np.where(valid, arr, np.inf).min(axis=0)
    hi = np.where(valid, arr, -np.inf).max(axis=0)
    return np.where(any_valid, hi - lo, 0.0).astype(np.float32)


def _safe_mean_bands(reflectance: np.ndarray, valid: np.ndarray) -> np.ndarray:
    bands = np.asarray(reflectance[:6], dtype=np.float32)
    finite = np.isfinite(bands)
    count = finite.sum(axis=0)
    total = np.where(finite, bands, 0.0).sum(axis=0)
    out = np.zeros(valid.shape, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    out[~valid] = 0.0
    return out


def _spectral_scores(
    reflectance: np.ndarray,
    cube: np.ndarray,
    valid: np.ndarray,
    texture: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = {name: FEATURE_CHANNELS.index(name) for name in FEATURE_CHANNELS}
    blue = cube[idx["BLUE"]]
    green = cube[idx["GREEN"]]
    red = cube[idx["RED"]]
    nir = cube[idx["NIR"]]
    swir1 = cube[idx["SWIR1"]]
    swir2 = cube[idx["SWIR2"]]
    ndvi = cube[idx["NDVI"]]
    mndwi = cube[idx["MNDWI"]]
    tc_brightness = cube[idx["TC_Brightness"]]
    pv_fraction = np.clip(np.nan_to_num(cube[idx["PV_Fraction_Evidence"]]), 0.0, 1.0)
    bare = np.clip(np.nan_to_num(cube[idx["Bare_Evidence"]]), 0.0, 1.0)

    visible = (blue + green + red) / 3.0
    nirswir = (nir + swir1 + swir2) / 3.0
    raw_brightness = _safe_mean_bands(reflectance, valid)

    low_visible = 1.0 - _robust01(visible, valid, 0.02, 0.96)
    low_nirswir = 1.0 - _robust01(nirswir, valid, 0.02, 0.96)
    low_tc = 1.0 - _robust01(tc_brightness, valid, 0.02, 0.96)
    low_raw = 1.0 - _robust01(raw_brightness, valid, 0.02, 0.96)

    fill = float(np.nanmedian(raw_brightness[valid])) if np.any(valid) else 0.0
    local = ndimage.median_filter(np.where(valid, raw_brightness, fill), size=21, mode="nearest")
    broad = ndimage.median_filter(np.where(valid, raw_brightness, fill), size=51, mode="nearest")
    local_contrast = _robust01(np.maximum(local - raw_brightness, broad - raw_brightness), valid, 0.02, 0.995)

    water = np.clip((mndwi - 0.02) / 0.28, 0.0, 1.0)
    vegetation = np.clip((ndvi - 0.14) / 0.34, 0.0, 1.0)
    confuser = np.clip(0.62 * water + 0.38 * vegetation, 0.0, 1.0)

    # This absolute site score is intentionally independent of the segmentation
    # model and 2024 prototype.  It enables discovery of large new black arrays.
    absolute_dark = np.clip(
        0.24 * low_visible
        + 0.25 * low_nirswir
        + 0.18 * low_tc
        + 0.13 * low_raw
        + 0.14 * local_contrast
        + 0.10 * pv_fraction
        + 0.03 * np.clip(texture, 0.0, 1.0)
        - 0.34 * confuser,
        0.0,
        1.0,
    )
    for value in (absolute_dark, local_contrast, confuser, pv_fraction, bare, raw_brightness):
        value[~valid] = 0.0
    return {
        "absolute_dark": absolute_dark.astype(np.float32),
        "local_contrast": local_contrast.astype(np.float32),
        "confuser": confuser.astype(np.float32),
        "pv_fraction": pv_fraction.astype(np.float32),
        "bare": bare.astype(np.float32),
        "brightness": raw_brightness.astype(np.float32),
    }


def _temporal_site_evidence(
    absolute_dark: np.ndarray,
    brightness: np.ndarray,
    valid: np.ndarray,
    years: np.ndarray,
    dark_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return onset, future persistence, stability, static risk and baseline drop."""
    n = len(years)
    onset = np.zeros_like(absolute_dark, dtype=np.float32)
    persistence = np.zeros_like(absolute_dark, dtype=np.float32)
    stability = np.zeros_like(absolute_dark, dtype=np.float32)
    baseline_drop = np.zeros_like(absolute_dark, dtype=np.float32)

    for t in range(n):
        pre0 = max(0, t - 5)
        pre = slice(pre0, t)
        post = slice(t, min(n, t + 4))
        if t > 0:
            pre_dark = _safe_nanmedian(np.where(valid[pre], absolute_dark[pre], np.nan), axis=0)
            pre_brightness = _safe_nanmedian(np.where(valid[pre], brightness[pre], np.nan), axis=0)
        else:
            pre_dark = np.zeros(absolute_dark.shape[1:], dtype=np.float32)
            pre_brightness = brightness[t]
        post_dark = _safe_nanmedian(np.where(valid[post], absolute_dark[post], np.nan), axis=0)
        current = absolute_dark[t]
        current_brightness = brightness[t]

        post_valid = valid[post].sum(axis=0)
        persistent_count = ((absolute_dark[post] >= dark_threshold) & valid[post]).sum(axis=0)
        future = np.zeros_like(current, dtype=np.float32)
        np.divide(persistent_count, np.maximum(post_valid, 1), out=future, where=post_valid > 0)

        dark_gain = _sigmoid((post_dark - pre_dark - 0.035) / 0.09)
        reflectance_drop = _sigmoid((pre_brightness - current_brightness - 0.008) / 0.025)
        one_year_gain = _sigmoid((current - pre_dark - 0.04) / 0.09)
        onset[t] = np.clip(0.42 * dark_gain + 0.33 * reflectance_drop + 0.15 * one_year_gain + 0.10 * future, 0.0, 1.0)
        persistence[t] = future
        stability[t] = _safe_nanmedian(np.where(valid[post], absolute_dark[post], np.nan), axis=0)
        baseline_drop[t] = np.clip(pre_brightness - current_brightness, -1.0, 1.0)

    temporal_range = _safe_range(absolute_dark, valid)
    static_risk = np.clip(1.0 - temporal_range / 0.24, 0.0, 1.0).astype(np.float32)
    return onset, persistence, stability, static_risk, baseline_drop


def _site_metrics(component: np.ndarray, core: np.ndarray, dark: np.ndarray, dark_threshold: float) -> SiteMetrics:
    size = int(component.sum())
    if size == 0:
        return SiteMetrics(0, 0.0, 0.0, 0.0, 999.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0)
    rows, cols = np.where(component)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    bbox_fill = float(size / max((r1 - r0) * (c1 - c0), 1))
    filled = ndimage.binary_fill_holes(component)
    solid_fill = float(size / max(int(filled.sum()), 1))
    oriented_fill, elongation = _oriented_fill(component)
    eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool), iterations=1)
    boundary = component & ~eroded
    interior_fraction = float(eroded.sum() / max(size, 1))
    perimeter_density = float(boundary.sum() / max(size, 1))
    core_fraction = float(np.count_nonzero(core & component) / max(size, 1))
    dark_fraction = float(np.count_nonzero((dark >= dark_threshold) & component) / max(size, 1))
    edge_only = float(np.clip(perimeter_density * (1.0 - interior_fraction), 0.0, 1.0))
    child_components = int(ndimage.label(core & component, structure=np.ones((3, 3), dtype=bool))[1])
    return SiteMetrics(
        pixels=size,
        bbox_fill=bbox_fill,
        solid_fill=solid_fill,
        oriented_fill=float(oriented_fill),
        elongation=float(elongation),
        interior_fraction=interior_fraction,
        perimeter_density=perimeter_density,
        core_fraction=core_fraction,
        dark_fraction=dark_fraction,
        edge_only_score=edge_only,
        child_components=child_components,
    )


def _component_boundary_contrast(component: np.ndarray, brightness: np.ndarray) -> float:
    structure = np.ones((3, 3), dtype=bool)
    ring = ndimage.binary_dilation(component, structure=structure, iterations=2) & ~component
    inside = brightness[component]
    outside = brightness[ring]
    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(np.median(outside) - np.median(inside))


def _group_site_proposals(core: np.ndarray, growable: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=bool)
    opening_iterations = int(settings.get("site_opening_iterations", 1))
    if opening_iterations > 0:
        clean = ndimage.binary_opening(core, structure=structure, iterations=opening_iterations)
        clean |= core & ndimage.binary_dilation(clean, structure=structure, iterations=1)
    else:
        clean = core.copy()
    closing_iterations = int(settings.get("site_closing_iterations", 2))
    grouped = ndimage.binary_closing(clean, structure=structure, iterations=closing_iterations) if closing_iterations > 0 else clean
    radius = int(settings.get("site_group_bridge_iterations", 2))
    expanded = ndimage.binary_dilation(grouped, structure=structure, iterations=max(radius, 0))
    labels, count = ndimage.label(expanded, structure=structure)
    # The labels are envelopes only.  Accepted pixels remain constrained to the
    # physical growable mask; blank inter-row gaps are not painted as panels.
    return labels, count


def _read_optional_mask(
    path: Path | None,
    reference: rasterio.DatasetReader,
    window: Any,
) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((int(window.height), int(window.width)), dtype=bool)
    with rasterio.open(path) as src:
        if src.crs != reference.crs or src.transform != reference.transform or src.width != reference.width or src.height != reference.height:
            raise RuntimeError(f"Expert correction mask is not aligned to canonical grid: {path}")
        return src.read(1, window=window) > 0


def _expert_mask_paths(settings: dict[str, Any], region_id: int, year: int) -> tuple[Path | None, Path | None]:
    root_value = settings.get("expert_corrections_root")
    if not root_value:
        return None, None
    root = Path(str(root_value))
    candidates_pos = [
        root / f"unit_{region_id:03d}_{year}_positive.tif",
        root / f"unit_{region_id:03d}_positive.tif",
    ]
    candidates_neg = [
        root / f"unit_{region_id:03d}_{year}_negative.tif",
        root / f"unit_{region_id:03d}_negative.tif",
    ]
    return next((p for p in candidates_pos if p.exists()), None), next((p for p in candidates_neg if p.exists()), None)


def _evaluate_site_first_year(
    *,
    region_id: int,
    year: int,
    reference_anchor: np.ndarray,
    expert_positive: np.ndarray,
    expert_negative: np.ndarray,
    site_core: np.ndarray,
    site_growable: np.ndarray,
    absolute_dark: np.ndarray,
    local_contrast: np.ndarray,
    onset: np.ndarray,
    persistence: np.ndarray,
    stability: np.ndarray,
    sand_risk: np.ndarray,
    confuser: np.ndarray,
    brightness: np.ndarray,
    model_score: np.ndarray,
    positive_similarity: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    accepted = np.zeros(site_core.shape, dtype=bool)
    classes = np.zeros(site_core.shape, dtype=np.uint8)
    rejection = np.zeros(site_core.shape, dtype=np.uint8)
    site_proposal = np.zeros(site_core.shape, dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    structure = np.ones((3, 3), dtype=bool)

    # Expert-negative masks are optional, auditable corrections.  They are not
    # required for the algorithm but allow georeferenced field interpretation
    # to become a hard negative in later iterations.
    rejection[expert_negative] = REJECT_EXPERT_NEGATIVE

    labels, count = _group_site_proposals(site_core & ~expert_negative, site_growable & ~expert_negative, settings)
    for object_id in range(1, count + 1):
        envelope = labels == object_id
        candidate = envelope & site_growable & ~expert_negative
        candidate |= envelope & site_core & ~expert_negative
        if not np.any(candidate):
            continue
        site_proposal[candidate] = 1
        metrics = _site_metrics(candidate, site_core, absolute_dark, float(settings.get("site_dark_pixel_threshold", 0.50)))
        mean_dark = float(np.mean(absolute_dark[candidate]))
        q75_dark = float(np.quantile(absolute_dark[candidate], 0.75))
        mean_contrast = float(np.mean(local_contrast[candidate]))
        mean_onset = float(np.mean(onset[candidate]))
        mean_persistence = float(np.mean(persistence[candidate]))
        mean_stability = float(np.mean(stability[candidate]))
        mean_sand = float(np.mean(sand_risk[candidate]))
        mean_confuser = float(np.mean(confuser[candidate]))
        mean_model = float(np.mean(model_score[candidate]))
        mean_positive = float(np.mean(positive_similarity[candidate]))
        boundary_contrast = _component_boundary_contrast(candidate, brightness)
        large_site = metrics.pixels >= int(settings.get("large_site_minimum_pixels", 80))
        preexisting = (
            year <= int(settings.get("preexisting_year_cutoff", 2012))
            and mean_stability >= float(settings.get("preexisting_minimum_stability", 0.70))
            and mean_dark >= float(settings.get("preexisting_minimum_dark_score", 0.62))
        )
        temporal_ok = (
            mean_persistence >= float(settings.get("site_minimum_persistence", 0.48))
            and (mean_onset >= float(settings.get("site_minimum_onset", 0.30)) or preexisting)
        )
        geometry_ok = (
            metrics.pixels >= int(settings.get("minimum_site_pixels", 18))
            and metrics.core_fraction >= float(settings.get("minimum_core_fraction", 0.20))
            and metrics.dark_fraction >= float(settings.get("minimum_dark_fraction", 0.48))
            and metrics.solid_fill >= float(settings.get("minimum_solid_fill", 0.52))
            and metrics.oriented_fill >= float(settings.get("minimum_oriented_fill", 0.08))
            and metrics.elongation <= float(settings.get("maximum_elongation", 34.0))
            and metrics.perimeter_density <= float(settings.get("maximum_perimeter_density", 0.88))
            and metrics.edge_only_score <= float(settings.get("maximum_edge_only_score", 0.78))
        )
        physical_ok = (
            mean_dark >= float(settings.get("minimum_mean_dark_score", 0.48))
            and q75_dark >= float(settings.get("minimum_q75_dark_score", 0.55))
            and mean_contrast >= float(settings.get("minimum_mean_local_contrast", 0.20))
            and boundary_contrast >= float(settings.get("minimum_boundary_contrast", 0.002))
            and mean_confuser <= float(settings.get("maximum_confuser", 0.48))
            and mean_sand <= float(settings.get("maximum_site_sand_risk", 0.50))
        )
        accepted_object = bool(geometry_ok and physical_ok and temporal_ok)

        # Large, very dark and durable arrays form an independent recall path.
        # This is the explicit mechanism for the large black Unit-006 sites.
        if not accepted_object and large_site:
            accepted_object = bool(
                mean_dark >= float(settings.get("large_site_minimum_dark_score", 0.58))
                and q75_dark >= float(settings.get("large_site_minimum_q75_dark_score", 0.64))
                and mean_persistence >= float(settings.get("large_site_minimum_persistence", 0.55))
                and mean_confuser <= float(settings.get("large_site_maximum_confuser", 0.42))
                and mean_sand <= float(settings.get("large_site_maximum_sand_risk", 0.48))
                and metrics.dark_fraction >= float(settings.get("large_site_minimum_dark_fraction", 0.46))
                and metrics.solid_fill >= float(settings.get("large_site_minimum_solid_fill", 0.46))
                and metrics.edge_only_score <= float(settings.get("large_site_maximum_edge_only_score", 0.84))
            )

        # A model can support an otherwise borderline whole site, but it cannot
        # create a fragmented sand site by itself.
        if not accepted_object:
            accepted_object = bool(
                geometry_ok
                and mean_model >= float(settings.get("model_supported_site_minimum_score", 0.72))
                and mean_positive >= float(settings.get("model_supported_site_minimum_positive_similarity", 0.48))
                and mean_sand <= float(settings.get("model_supported_site_maximum_sand_risk", 0.42))
                and mean_persistence >= float(settings.get("model_supported_site_minimum_persistence", 0.44))
                and mean_dark >= float(settings.get("model_supported_site_minimum_dark_score", 0.42))
            )

        if accepted_object:
            accepted[candidate] = True
            classes[candidate] = CLASS_SITE_FIRST_REMOTE if mean_model < 0.65 else CLASS_MODEL_CONFIRMED_SITE
            reason = "accepted_site_first"
        else:
            if mean_confuser > float(settings.get("maximum_confuser", 0.48)):
                code, reason = REJECT_WATER_OR_VEGETATION, "water_or_vegetation"
            elif mean_sand > float(settings.get("persistent_sand_rejection_threshold", 0.56)):
                code, reason = REJECT_PERSISTENT_SAND, "persistent_sand"
            elif metrics.pixels < int(settings.get("minimum_site_pixels", 18)):
                code, reason = REJECT_FRAGMENTED_SPECKLE, "fragmented_speckle"
            elif metrics.edge_only_score > float(settings.get("maximum_edge_only_score", 0.78)) or metrics.solid_fill < float(settings.get("minimum_solid_fill", 0.52)):
                code, reason = REJECT_HOLLOW_EDGE, "hollow_or_edge_only"
            elif metrics.dark_fraction < float(settings.get("minimum_dark_fraction", 0.48)):
                code, reason = REJECT_WEAK_SITE_INTERIOR, "weak_site_interior"
            elif not temporal_ok:
                code, reason = REJECT_WEAK_TEMPORAL_ONSET, "weak_temporal_onset"
            elif not geometry_ok:
                code, reason = REJECT_IMPLAUSIBLE_GEOMETRY, "implausible_geometry"
            else:
                code, reason = REJECT_LOW_SITE_EVIDENCE, "low_site_evidence"
            rejection[candidate & (rejection == REJECT_NONE)] = code

        rows.append({
            "Id": region_id,
            "year": year,
            "branch": "site_first",
            "object_id": object_id,
            **metrics.__dict__,
            "mean_absolute_dark": mean_dark,
            "q75_absolute_dark": q75_dark,
            "mean_local_contrast": mean_contrast,
            "mean_onset": mean_onset,
            "mean_persistence": mean_persistence,
            "mean_stability": mean_stability,
            "mean_sand_risk": mean_sand,
            "mean_confuser": mean_confuser,
            "mean_model_score": mean_model,
            "mean_positive_similarity": mean_positive,
            "boundary_contrast": boundary_contrast,
            "large_site": bool(large_site),
            "preexisting": bool(preexisting),
            "accepted": bool(accepted_object),
            "reason": reason,
        })

    # Conservative completion around already accepted whole sites or the 2024
    # reference anchor.  Pixel-model predictions away from a site are not kept.
    reference = accepted | reference_anchor | expert_positive
    if np.any(reference):
        distance = ndimage.distance_transform_edt(~reference)
        completion = (
            (distance <= float(settings.get("maximum_adjacency_completion_pixels", 5)))
            & (absolute_dark >= float(settings.get("adjacency_minimum_dark_score", 0.38)))
            & (sand_risk <= float(settings.get("adjacency_maximum_sand_risk", 0.42)))
            & (confuser <= float(settings.get("maximum_confuser", 0.48)))
            & ~expert_negative
        )
        # Model/prototype support is required only for the lower-darkness edge.
        completion &= (
            (model_score >= float(settings.get("adjacency_minimum_model_score", 0.50)))
            | (positive_similarity >= float(settings.get("adjacency_minimum_positive_similarity", 0.46)))
            | (absolute_dark >= float(settings.get("adjacency_dark_override", 0.62)))
        )
        propagated = ndimage.binary_propagation(reference, structure=structure, mask=reference | completion)
        new = propagated & ~accepted & ~reference_anchor & ~expert_positive
        accepted[new] = True
        classes[new] = CLASS_ADJACENCY_COMPLETION
        rejection[new] = REJECT_NONE

    # Force only explicit positives after all rejection rules.
    accepted[reference_anchor] = True
    classes[reference_anchor] = CLASS_REFERENCE_ANCHOR
    rejection[reference_anchor] = REJECT_NONE
    accepted[expert_positive] = True
    classes[expert_positive] = CLASS_EXPERT_POSITIVE
    rejection[expert_positive] = REJECT_NONE
    accepted[expert_negative] = False
    classes[expert_negative] = CLASS_BACKGROUND
    rejection[expert_negative] = REJECT_EXPERT_NEGATIVE
    return accepted, classes, rejection, site_proposal, rows


def _reconcile_site_graph(
    masks: np.ndarray,
    classes: np.ndarray,
    rejection: np.ndarray,
    absolute_dark: np.ndarray,
    site_growable: np.ndarray,
    onset: np.ndarray,
    persistence: np.ndarray,
    sand_risk: np.ndarray,
    anchor_by_year: np.ndarray,
    expert_positive: np.ndarray,
    expert_negative: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = masks.shape[0]
    structure = np.ones((3, 3), dtype=bool)

    # Rescue only a one-year omission between spatially matching sites and only
    # where current-year physical dark evidence remains.
    for t in range(1, n - 1):
        linked = (
            ndimage.binary_dilation(masks[t - 1], structure=structure, iterations=2)
            & ndimage.binary_dilation(masks[t + 1], structure=structure, iterations=2)
        )
        rescue = (
            linked
            & site_growable[t]
            & (absolute_dark[t] >= float(settings.get("temporal_rescue_minimum_dark_score", 0.42)))
            & (sand_risk[t] <= float(settings.get("temporal_rescue_maximum_sand_risk", 0.46)))
            & ~expert_negative[t]
        )
        new = rescue & ~masks[t]
        masks[t] |= rescue
        classes[t][new] = CLASS_TEMPORAL_SITE_RESCUE
        rejection[t][new] = REJECT_NONE

    # Remove isolated site objects unless they contain a strong installation
    # event and durable physical evidence.
    for t in range(n):
        neighbour = np.zeros_like(masks[t])
        if t > 0:
            neighbour |= ndimage.binary_dilation(masks[t - 1], structure=structure, iterations=2)
        if t + 1 < n:
            neighbour |= ndimage.binary_dilation(masks[t + 1], structure=structure, iterations=2)
        labels, count = ndimage.label(masks[t], structure=structure)
        for object_id in range(1, count + 1):
            obj = labels == object_id
            if np.any(anchor_by_year[t] & obj) or np.any(expert_positive[t] & obj):
                continue
            linked = bool(np.any(neighbour & obj))
            strong = (
                float(np.mean(onset[t][obj])) >= float(settings.get("singleton_minimum_onset", 0.66))
                and float(np.mean(persistence[t][obj])) >= float(settings.get("singleton_minimum_persistence", 0.70))
                and float(np.mean(absolute_dark[t][obj])) >= float(settings.get("singleton_minimum_dark_score", 0.66))
                and float(np.mean(sand_risk[t][obj])) <= float(settings.get("singleton_maximum_sand_risk", 0.38))
            )
            if not linked and not strong:
                masks[t][obj] = False
                classes[t][obj] = CLASS_BACKGROUND
                rejection[t][obj] = REJECT_WEAK_TEMPORAL_ONSET

    # Prevent a site first observed in a later year from being copied into
    # earlier years by model persistence.  Early pre-existing sites are allowed
    # only when they are already physically strong in the first two years.
    strong_support = (
        ((onset >= float(settings.get("first_support_minimum_onset", 0.34)))
         & (persistence >= float(settings.get("first_support_minimum_persistence", 0.48))))
        | ((absolute_dark >= float(settings.get("first_support_dark_override", 0.70)))
           & (persistence >= float(settings.get("first_support_dark_persistence", 0.62)))
           & (sand_risk <= float(settings.get("first_support_maximum_sand_risk", 0.40))))
    )
    has_support = strong_support.any(axis=0)
    first_support = np.argmax(strong_support, axis=0)
    for t in range(n):
        premature = (
            masks[t]
            & has_support
            & (t < first_support)
            & ~anchor_by_year[t]
            & ~expert_positive[t]
        )
        if np.any(premature):
            masks[t][premature] = False
            classes[t][premature] = CLASS_BACKGROUND
            rejection[t][premature] = REJECT_PREMATURE_BACKFILL

    # Persistent dune texture is a hard negative at the track level.  It can
    # veto an accepted track when the whole-period sand risk is high and there
    # is no installation onset.  Large strong-dark sites are excluded from this
    # veto to preserve real pre-existing arrays.
    track_sand = np.median(sand_risk, axis=0)
    track_onset = np.max(onset, axis=0)
    track_dark = np.max(absolute_dark, axis=0)
    hard_sand = (
        (track_sand >= float(settings.get("track_sand_rejection_threshold", 0.62)))
        & (track_onset <= float(settings.get("track_sand_maximum_onset", 0.24)))
        & (track_dark <= float(settings.get("track_sand_dark_override", 0.76)))
    )
    for t in range(n):
        reject = masks[t] & hard_sand & ~anchor_by_year[t] & ~expert_positive[t]
        masks[t][reject] = False
        classes[t][reject] = CLASS_BACKGROUND
        rejection[t][reject] = REJECT_PERSISTENT_SAND

    masks[expert_negative] = False
    classes[expert_negative] = CLASS_BACKGROUND
    rejection[expert_negative] = REJECT_EXPERT_NEGATIVE
    masks[expert_positive] = True
    classes[expert_positive] = CLASS_EXPERT_POSITIVE
    rejection[expert_positive] = REJECT_NONE
    masks[anchor_by_year] = True
    classes[anchor_by_year] = CLASS_REFERENCE_ANCHOR
    rejection[anchor_by_year] = REJECT_NONE
    return masks, classes, rejection


def refine_site_first_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("site_first_refinement", {})
    if not bool(settings.get("enabled", True)):
        return pd.read_csv(Path(cfg["paths"]["output_root"]) / "07_probability_calibration" / "calibrated_inference_registry.csv")

    root = Path(cfg["paths"]["output_root"])
    canonical_registry_path = root / "07_probability_calibration" / "calibrated_inference_registry.csv"
    source_registry = pd.read_csv(canonical_registry_path).sort_values(["Id", "year"])
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")

    out = ensure_dir(root / "07_site_first_refinement")
    probability_root = ensure_dir(out / "site_probability_30m")
    mask_root = ensure_dir(out / "site_masks_30m")
    class_root = ensure_dir(out / "accept_source_30m")
    rejection_root = ensure_dir(out / "rejection_code_30m")
    proposal_root = ensure_dir(out / "site_proposal_30m")
    dark_root = ensure_dir(out / "absolute_dark_score_30m")
    contrast_root = ensure_dir(out / "local_dark_contrast_30m")
    sand_root = ensure_dir(out / "persistent_sand_risk_30m")
    onset_root = ensure_dir(out / "installation_onset_30m")
    persistence_root = ensure_dir(out / "site_persistence_30m")
    prototype_root = ensure_dir(out / "unit_prototypes")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    registry_root = ensure_dir(out / "unit_registries")
    object_table_root = ensure_dir(out / "object_tables")

    years = np.arange(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1)
    reference_year = int(cfg["study"]["reference_year"])
    tile_size = int(settings.get("tile_size", 384))
    halo = int(settings.get("halo_pixels", 96))
    mask_nodata = int(cfg.get("calibration", {}).get("mask_nodata", 255))
    diagnostics_units = {int(x) for x in settings.get("diagnostic_unit_ids", [6, 28])}
    write_diagnostics = bool(settings.get("write_diagnostics", False)) or test_mode_enabled(cfg)
    all_registries: list[pd.DataFrame] = []
    unit_summaries: list[dict[str, Any]] = []

    prototype_cfg = dict(cfg)
    prototype_cfg["object_graph_refinement"] = {
        **cfg.get("object_graph_refinement", {}),
        "prototype_tile_size": int(settings.get("prototype_tile_size", 512)),
        "maximum_prototype_samples": int(settings.get("maximum_prototype_samples", 120000)),
        "positive_prototype_clusters": int(settings.get("positive_prototype_clusters", 3)),
        "sand_prototype_clusters": int(settings.get("sand_prototype_clusters", 5)),
    }

    for region_id, group in tqdm(
        list(source_registry.groupby("Id")),
        desc="Site-first PV detection",
        unit="region",
        disable=not cfg.get("processing", {}).get("show_progress", True),
    ):
        region_id = int(region_id)
        group = group.sort_values("year")
        aligned_group = aligned.loc[aligned["Id"].eq(region_id)].sort_values("year")
        if len(group) != len(years) or len(aligned_group) != len(years):
            raise RuntimeError(f"Id {region_id}: site-first refinement requires {len(years)} annual records")
        if group["inference_mode"].astype(str).eq("area_only_fallback").all():
            passthrough = group.copy()
            passthrough["site_first_refinement"] = "not_applicable_area_only"
            all_registries.append(passthrough)
            continue

        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        unit_registry_path = registry_root / f"unit_{region_id:03d}.csv"
        if checkpoint.exists() and unit_registry_path.exists() and not cfg.get("processing", {}).get("overwrite", False):
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing = pd.read_csv(unit_registry_path)
                if payload.get("status") == "passed" and len(existing) == len(years) and all(Path(p).exists() for p in existing["calibrated_probability_path"].astype(str)):
                    all_registries.append(existing)
                    unit_summaries.append(payload)
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
            prototype_root / f"unit_{region_id:03d}_spectral_prototypes.npz",
        )

        prob_paths = [ensure_dir(probability_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_pv_probability_site_first_30m.tif" for y in years]
        mask_paths = [ensure_dir(mask_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_pv_mask_site_first_30m.tif" for y in years]
        class_paths = [ensure_dir(class_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_accept_source_30m.tif" for y in years]
        rejection_paths = [ensure_dir(rejection_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_rejection_code_30m.tif" for y in years]
        proposal_paths = [ensure_dir(proposal_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_site_proposal_30m.tif" for y in years]
        dark_paths = [ensure_dir(dark_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_absolute_dark_score_30m.tif" for y in years]
        contrast_paths = [ensure_dir(contrast_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_local_dark_contrast_30m.tif" for y in years]
        sand_paths = [ensure_dir(sand_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_persistent_sand_risk_30m.tif" for y in years]
        onset_paths = [ensure_dir(onset_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_installation_onset_30m.tif" for y in years]
        persistence_paths = [ensure_dir(persistence_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_site_persistence_30m.tif" for y in years]

        predicted_pixels = np.zeros(len(years), dtype=np.int64)
        remote_pixels = np.zeros(len(years), dtype=np.int64)
        model_site_pixels = np.zeros(len(years), dtype=np.int64)
        adjacency_pixels = np.zeros(len(years), dtype=np.int64)
        temporal_rescue_pixels = np.zeros(len(years), dtype=np.int64)
        sand_rejected_pixels = np.zeros(len(years), dtype=np.int64)
        backfill_rejected_pixels = np.zeros(len(years), dtype=np.int64)
        object_rows: list[dict[str, Any]] = []

        with ExitStack() as stack:
            calibrated_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].calibrated_probability_path))) for y in years]
            raw_srcs = [stack.enter_context(rasterio.open(str(getattr(year_rows[int(y)], "raw_probability_path", "") or year_rows[int(y)].calibrated_probability_path))) for y in years]
            uncertainty_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].epistemic_std_path))) for y in years]
            landsat_srcs = [stack.enter_context(rasterio.open(landsat_paths[int(y)])) for y in years]
            fraction_src = stack.enter_context(rasterio.open(fraction_path))
            ref = calibrated_srcs[0]
            pixel_area_km2 = abs(float(ref.transform.a * ref.transform.e)) / 1_000_000.0
            float_profile = ref.profile.copy()
            float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            byte_profile = ref.profile.copy()
            byte_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            prob_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in prob_paths]
            mask_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in mask_paths]
            diagnostic_on = write_diagnostics and region_id in diagnostics_units
            class_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in class_paths] if diagnostic_on else []
            rejection_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in rejection_paths] if diagnostic_on else []
            proposal_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in proposal_paths] if diagnostic_on else []
            dark_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in dark_paths] if diagnostic_on else []
            contrast_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in contrast_paths] if diagnostic_on else []
            sand_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in sand_paths] if diagnostic_on else []
            onset_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in onset_paths] if diagnostic_on else []
            persistence_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in persistence_paths] if diagnostic_on else []

            for core in _windows(ref.width, ref.height, tile_size):
                expanded, crop = _expanded(core, ref.width, ref.height, halo)
                shape = (int(expanded.height), int(expanded.width))
                n_years = len(years)
                raw_probability = np.zeros((n_years, *shape), dtype=np.float32)
                calibrated_probability = np.zeros_like(raw_probability)
                uncertainty = np.ones_like(raw_probability)
                valid = np.zeros_like(raw_probability, dtype=bool)
                positive_similarity = np.zeros_like(raw_probability)
                sand_similarity = np.zeros_like(raw_probability)
                coherence = np.zeros_like(raw_probability)
                texture = np.zeros_like(raw_probability)
                absolute_dark = np.zeros_like(raw_probability)
                local_contrast = np.zeros_like(raw_probability)
                confuser = np.zeros_like(raw_probability)
                pv_fraction = np.zeros_like(raw_probability)
                bare = np.zeros_like(raw_probability)
                brightness = np.zeros_like(raw_probability)

                for t, (raw_src, cal_src, unc_src, land_src) in enumerate(zip(raw_srcs, calibrated_srcs, uncertainty_srcs, landsat_srcs)):
                    raw = raw_src.read(1, window=expanded).astype(np.float32)
                    calibrated = cal_src.read(1, window=expanded).astype(np.float32)
                    unc = unc_src.read(1, window=expanded).astype(np.float32)
                    refl = _read_reflectance(land_src, expanded)
                    valid[t] = np.isfinite(raw) & (raw > -9990) & np.isfinite(calibrated) & (calibrated > -9990) & np.isfinite(refl).all(axis=0)
                    raw_probability[t] = np.where(valid[t], np.clip(raw, 0.0, 1.0), 0.0)
                    calibrated_probability[t] = np.where(valid[t], np.clip(calibrated, 0.0, 1.0), 0.0)
                    uncertainty[t] = np.where(valid[t], np.clip(np.nan_to_num(unc, nan=1.0), 0.0, 1.0), 1.0)
                    cube = _feature_cube(refl, endmembers)
                    positive_similarity[t] = _prototype_similarity(cube, prototypes.positive_centers, prototypes.positive_scales)
                    sand_similarity[t] = _prototype_similarity(cube, prototypes.sand_centers, prototypes.sand_scales)
                    _, coherence[t], _, texture[t] = _texture_features(refl, valid[t])
                    spec = _spectral_scores(refl, cube, valid[t], texture[t])
                    absolute_dark[t] = spec["absolute_dark"]
                    local_contrast[t] = spec["local_contrast"]
                    confuser[t] = spec["confuser"]
                    pv_fraction[t] = spec["pv_fraction"]
                    bare[t] = spec["bare"]
                    brightness[t] = spec["brightness"]

                onset, persistence, stability, static_risk, _ = _temporal_site_evidence(
                    absolute_dark, brightness, valid, years, float(settings.get("site_core_dark_threshold", 0.54))
                )
                anchor = fraction_src.read(1, window=expanded).astype(np.float32) >= float(cfg.get("supervision", {}).get("strong_positive_fraction", 0.7))
                masks = np.zeros_like(valid)
                classes = np.zeros_like(raw_probability, dtype=np.uint8)
                rejection = np.zeros_like(raw_probability, dtype=np.uint8)
                proposals = np.zeros_like(raw_probability, dtype=np.uint8)
                sand_risks = np.zeros_like(raw_probability)
                model_scores = np.zeros_like(raw_probability)
                site_growables = np.zeros_like(valid)
                expert_positive_stack = np.zeros_like(valid)
                expert_negative_stack = np.zeros_like(valid)

                for t, year in enumerate(years):
                    uncertainty_penalty = np.clip(uncertainty[t] / 0.45, 0.0, 1.0)
                    no_onset = 1.0 - onset[t]
                    sand_risk = np.clip(
                        0.25 * sand_similarity[t]
                        + 0.20 * bare[t]
                        + 0.20 * coherence[t]
                        + 0.12 * texture[t]
                        + 0.16 * static_risk
                        + 0.10 * no_onset
                        - 0.28 * absolute_dark[t]
                        - 0.14 * local_contrast[t]
                        - 0.10 * persistence[t],
                        0.0,
                        1.0,
                    )
                    sand_risks[t] = sand_risk
                    model_score = np.clip(
                        0.34 * raw_probability[t]
                        + 0.28 * calibrated_probability[t]
                        + 0.18 * positive_similarity[t]
                        + 0.10 * pv_fraction[t]
                        + 0.06 * persistence[t]
                        + 0.04 * onset[t]
                        - 0.28 * sand_risk
                        - 0.14 * confuser[t]
                        - 0.08 * uncertainty_penalty,
                        0.0,
                        1.0,
                    ) * valid[t]
                    model_scores[t] = model_score

                    # Site-first proposals are independent of model/prototype.
                    site_core = valid[t] & (
                        (
                            (absolute_dark[t] >= float(settings.get("site_core_dark_threshold", 0.54)))
                            & (local_contrast[t] >= float(settings.get("site_core_contrast_threshold", 0.18)))
                        )
                        | (absolute_dark[t] >= float(settings.get("site_core_dark_override", 0.70)))
                    )
                    site_core &= confuser[t] <= float(settings.get("maximum_confuser", 0.48))
                    site_core &= sand_risk <= float(settings.get("site_core_maximum_sand_risk", 0.58))
                    site_growable = valid[t] & (
                        (absolute_dark[t] >= float(settings.get("site_growth_dark_threshold", 0.38)))
                        & (confuser[t] <= float(settings.get("maximum_confuser", 0.48)))
                        & (sand_risk <= float(settings.get("site_growth_maximum_sand_risk", 0.62)))
                    )
                    site_growables[t] = site_growable

                    pos_path, neg_path = _expert_mask_paths(settings, region_id, int(year))
                    expert_positive = _read_optional_mask(pos_path, ref, expanded)
                    expert_negative = _read_optional_mask(neg_path, ref, expanded)
                    expert_positive_stack[t] = expert_positive
                    expert_negative_stack[t] = expert_negative
                    year_anchor = anchor & valid[t] if int(year) == reference_year else np.zeros_like(anchor)

                    accepted, object_classes, object_rejection, site_proposal, rows = _evaluate_site_first_year(
                        region_id=region_id,
                        year=int(year),
                        reference_anchor=year_anchor,
                        expert_positive=expert_positive,
                        expert_negative=expert_negative,
                        site_core=site_core,
                        site_growable=site_growable,
                        absolute_dark=absolute_dark[t],
                        local_contrast=local_contrast[t],
                        onset=onset[t],
                        persistence=persistence[t],
                        stability=stability[t],
                        sand_risk=sand_risk,
                        confuser=confuser[t],
                        brightness=brightness[t],
                        model_score=model_score,
                        positive_similarity=positive_similarity[t],
                        settings=settings,
                    )
                    masks[t] = accepted
                    classes[t] = object_classes
                    rejection[t] = object_rejection
                    proposals[t] = site_proposal
                    object_rows.extend(rows)

                reference_index = int(reference_year - int(years[0]))
                anchor_by_year = np.zeros_like(masks)
                anchor_by_year[reference_index] = anchor & valid[reference_index]
                masks, classes, rejection = _reconcile_site_graph(
                    masks,
                    classes,
                    rejection,
                    absolute_dark,
                    site_growables,
                    onset,
                    persistence,
                    sand_risks,
                    anchor_by_year,
                    expert_positive_stack,
                    expert_negative_stack,
                    settings,
                )

                for t, year in enumerate(years):
                    core_valid = valid[t][crop]
                    core_mask = masks[t][crop]
                    core_class = classes[t][crop]
                    core_rejection = rejection[t][crop]
                    accepted_strength = np.clip(
                        0.50 * absolute_dark[t] + 0.22 * persistence[t] + 0.16 * onset[t] + 0.12 * model_scores[t],
                        0.0,
                        1.0,
                    )
                    final_probability = np.where(
                        masks[t],
                        np.clip(0.76 + 0.24 * accepted_strength, 0.0, 1.0),
                        np.clip(0.06 * model_scores[t] + 0.05 * absolute_dark[t], 0.0, 0.16),
                    )
                    encoded_mask = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                    encoded_mask[core_valid] = core_mask[core_valid].astype(np.uint8)
                    prob_dsts[t].write(np.where(core_valid, final_probability[crop], -9999.0).astype(np.float32), 1, window=core)
                    mask_dsts[t].write(encoded_mask, 1, window=core)
                    if diagnostic_on:
                        encoded_class = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        encoded_rejection = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        encoded_proposal = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        encoded_class[core_valid] = core_class[core_valid]
                        encoded_rejection[core_valid] = core_rejection[core_valid]
                        encoded_proposal[core_valid] = proposals[t][crop][core_valid]
                        class_dsts[t].write(encoded_class, 1, window=core)
                        rejection_dsts[t].write(encoded_rejection, 1, window=core)
                        proposal_dsts[t].write(encoded_proposal, 1, window=core)
                        dark_dsts[t].write(np.where(core_valid, absolute_dark[t][crop], -9999.0).astype(np.float32), 1, window=core)
                        contrast_dsts[t].write(np.where(core_valid, local_contrast[t][crop], -9999.0).astype(np.float32), 1, window=core)
                        sand_dsts[t].write(np.where(core_valid, sand_risks[t][crop], -9999.0).astype(np.float32), 1, window=core)
                        onset_dsts[t].write(np.where(core_valid, onset[t][crop], -9999.0).astype(np.float32), 1, window=core)
                        persistence_dsts[t].write(np.where(core_valid, persistence[t][crop], -9999.0).astype(np.float32), 1, window=core)

                    predicted_pixels[t] += int(np.count_nonzero(core_mask & core_valid))
                    remote_pixels[t] += int(np.count_nonzero((core_class == CLASS_SITE_FIRST_REMOTE) & core_valid))
                    model_site_pixels[t] += int(np.count_nonzero((core_class == CLASS_MODEL_CONFIRMED_SITE) & core_valid))
                    adjacency_pixels[t] += int(np.count_nonzero((core_class == CLASS_ADJACENCY_COMPLETION) & core_valid))
                    temporal_rescue_pixels[t] += int(np.count_nonzero((core_class == CLASS_TEMPORAL_SITE_RESCUE) & core_valid))
                    sand_rejected_pixels[t] += int(np.count_nonzero((core_rejection == REJECT_PERSISTENT_SAND) & core_valid))
                    backfill_rejected_pixels[t] += int(np.count_nonzero((core_rejection == REJECT_PREMATURE_BACKFILL) & core_valid))

        object_table = pd.DataFrame(object_rows)
        save_table(object_table, object_table_root / f"unit_{region_id:03d}_site_decisions.csv")
        unit_rows: list[dict[str, Any]] = []
        for t, year in enumerate(years):
            original = year_rows[int(year)]._asdict()
            unit_rows.append({
                **original,
                "pre_site_first_probability_path": original.get("calibrated_probability_path", ""),
                "pre_site_first_mask_path": original.get("calibrated_mask_path", ""),
                "pre_site_first_predicted_pixels": int(original.get("calibrated_predicted_pixels", 0) or 0),
                "pre_site_first_predicted_area_km2": float(original.get("calibrated_predicted_pixels", 0) or 0) * pixel_area_km2,
                "calibrated_probability_path": str(prob_paths[t]),
                "calibrated_mask_path": str(mask_paths[t]),
                "calibrated_predicted_pixels": int(predicted_pixels[t]),
                "calibrated_predicted_area_km2": float(predicted_pixels[t] * pixel_area_km2),
                "site_first_refinement": "v24_15_site_first_dark_array_detection_and_persistent_sand_rejection",
                "accept_source_path": str(class_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "site_first_rejection_code_path": str(rejection_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "site_proposal_path": str(proposal_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "absolute_dark_score_path": str(dark_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "local_dark_contrast_path": str(contrast_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "persistent_sand_risk_path": str(sand_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "installation_onset_path": str(onset_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "site_persistence_path": str(persistence_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "site_first_remote_pixels": int(remote_pixels[t]),
                "model_confirmed_site_pixels": int(model_site_pixels[t]),
                "adjacency_completed_pixels": int(adjacency_pixels[t]),
                "temporal_site_rescued_pixels": int(temporal_rescue_pixels[t]),
                "persistent_sand_rejected_pixels": int(sand_rejected_pixels[t]),
                "premature_backfill_rejected_pixels": int(backfill_rejected_pixels[t]),
            })
        unit_registry = pd.DataFrame(unit_rows).sort_values("year")
        save_table(unit_registry, unit_registry_path)
        payload = {
            "status": "passed",
            "created_utc": utc_now(),
            "Id": region_id,
            "records": int(len(unit_registry)),
            "registry_path": str(unit_registry_path),
            "positive_prototype_samples": prototypes.positive_samples,
            "sand_prototype_samples": prototypes.sand_samples,
            "accepted_object_decisions": int(object_table["accepted"].sum()) if not object_table.empty else 0,
            "rejected_object_decisions": int((~object_table["accepted"]).sum()) if not object_table.empty else 0,
            "site_first_remote_pixel_years": int(remote_pixels.sum()),
            "model_confirmed_site_pixel_years": int(model_site_pixels.sum()),
            "persistent_sand_rejected_pixel_years": int(sand_rejected_pixels.sum()),
            "premature_backfill_rejected_pixel_years": int(backfill_rejected_pixels.sum()),
            "diagnostic_rasters_written": bool(write_diagnostics and region_id in diagnostics_units),
        }
        atomic_json(payload, checkpoint)
        unit_summaries.append(payload)
        all_registries.append(unit_registry)

    refined = pd.concat(all_registries, ignore_index=True, sort=False).sort_values(["Id", "year"])
    backup = root / "07_probability_calibration" / "calibrated_inference_registry_pre_site_first_refinement.csv"
    if not backup.exists():
        save_table(source_registry, backup)
    save_table(refined, out / "site_first_inference_registry.csv")
    save_table(refined, canonical_registry_path)
    save_table(pd.DataFrame(unit_summaries), out / "site_first_refinement_by_unit.csv")

    expected_units, _, expected_records = expected_counts(cfg)
    coverage_ok = len(refined) == expected_records and refined["Id"].nunique() == expected_units
    status = {
        "status": "passed" if coverage_ok else "failed",
        "created_utc": utc_now(),
        "records": int(len(refined)),
        "expected_records": int(expected_records),
        "units": int(refined["Id"].nunique()),
        "expected_units": int(expected_units),
        "method": "site-first absolute-dark array proposals + whole-site geometry/interior validation + persistent-sand hard-negative veto + cross-year onset graph",
        "historical_detection_contract": "historical PV requires Landsat-derived physical site and temporal evidence; environmental suitability cannot create historical PV",
        "false_positive_control": "model-only speckle, directional persistent sand, hollow edges, weak site interiors and premature historical backfill are rejected",
        "false_negative_control": "large spatially separate black arrays are proposed independently of the segmentation model and 2024 prototype",
        "reference_anchor_rule": "2024 SHP is forced positive in 2024 only; no historical backfill without annual evidence",
        "expert_corrections": "optional georeferenced unit/year positive and negative masks are supported and audited",
        "diagnostic_units": sorted(diagnostics_units),
        "canonical_registry_replaced": str(canonical_registry_path),
        "pre_refinement_registry_backup": str(backup),
    }
    atomic_json(status, out / "site_first_refinement_status.json")
    if not coverage_ok:
        raise RuntimeError("Site-first refinement coverage incomplete")
    return refined
