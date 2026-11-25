from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class Thresholds:
    strong: float
    weak: float
    anchor_reference_median: float
    background_strong_quantile: float
    background_weak_quantile: float


def clip01(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def combine_evidence(
    evidence: Mapping[str, np.ndarray | None],
    valid: np.ndarray,
    weights: Mapping[str, float],
    minimum_available_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros(valid.shape, dtype=np.float32)
    denominator = np.zeros(valid.shape, dtype=np.float32)
    for name, weight in weights.items():
        array = clip01(evidence.get(name))
        if array is None:
            continue
        finite = valid & np.isfinite(array)
        if not finite.any():
            continue
        numerator[finite] += float(weight) * array[finite]
        denominator[finite] += float(weight)
    score = np.zeros(valid.shape, dtype=np.float32)
    usable = valid & (denominator >= float(minimum_available_weight))
    score[usable] = numerator[usable] / denominator[usable]
    score[~usable] = np.nan
    return score, denominator


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndi.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def remove_small(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    if minimum_pixels <= 1:
        return mask.astype(bool, copy=True)
    labels, count = label_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum_pixels)
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, maximum_pixels: int) -> np.ndarray:
    if maximum_pixels <= 0:
        return mask.astype(bool, copy=True)
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    labels, count = label_components(holes)
    result = mask.astype(bool, copy=True)
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) <= maximum_pixels:
            result |= component
    return result


def connected_hysteresis(strong: np.ndarray, weak: np.ndarray, domain: np.ndarray) -> np.ndarray:
    weak = weak & domain
    strong = strong & weak
    if not strong.any():
        return np.zeros_like(domain, dtype=bool)
    labels, count = label_components(weak)
    touched = np.unique(labels[strong])
    touched = touched[touched > 0]
    if touched.size == 0:
        return np.zeros_like(domain, dtype=bool)
    return np.isin(labels, touched)


def _safe_quantile(values: np.ndarray, quantile: float, fallback: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.quantile(values, quantile))


def calibrate_thresholds(
    reference_score: np.ndarray,
    anchor: np.ndarray,
    valid: np.ndarray,
    cfg: Mapping[str, Any],
) -> Thresholds:
    anchor_values = reference_score[anchor & valid & np.isfinite(reference_score)]
    outer = ndi.binary_dilation(anchor, iterations=5) & ~ndi.binary_dilation(anchor, iterations=1)
    background_values = reference_score[outer & valid & np.isfinite(reference_score)]
    if background_values.size < 32:
        background_values = reference_score[(~anchor) & valid & np.isfinite(reference_score)]

    min_strong = float(cfg["minimum_strong_threshold"])
    max_strong = float(cfg["maximum_strong_threshold"])
    min_weak = float(cfg["minimum_weak_threshold"])
    anchor_low = _safe_quantile(anchor_values, float(cfg["anchor_low_quantile"]), min_strong)
    anchor_median = _safe_quantile(anchor_values, 0.50, min_strong)
    bg_strong = _safe_quantile(
        background_values,
        float(cfg["background_strong_quantile"]),
        min_strong - float(cfg["strong_background_margin"]),
    )
    bg_weak = _safe_quantile(
        background_values,
        float(cfg["background_weak_quantile"]),
        min_weak - float(cfg["weak_background_margin"]),
    )

    # The anchor distribution limits the global threshold from becoming so high
    # that older sensors can never reach it. Background quantiles still protect
    # against widespread sand/dark-surface false positives.
    strong = max(
        min_strong,
        bg_strong + float(cfg["strong_background_margin"]),
        0.68 * anchor_low,
    )
    strong = min(strong, max_strong, max(min_strong, 0.92 * anchor_median))
    weak = max(
        min_weak,
        bg_weak + float(cfg["weak_background_margin"]),
        strong - float(cfg["hysteresis_width"]),
    )
    weak = min(weak, strong - 0.025)
    return Thresholds(
        strong=float(strong),
        weak=float(weak),
        anchor_reference_median=float(anchor_median),
        background_strong_quantile=float(bg_strong),
        background_weak_quantile=float(bg_weak),
    )


def classify_annual_state(
    score: np.ndarray,
    reference_score: np.ndarray,
    anchor: np.ndarray,
    valid: np.ndarray,
    thresholds: Thresholds,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float | int]]:
    score_clean = np.nan_to_num(score, nan=0.0)
    ref_clean = np.nan_to_num(reference_score, nan=0.0)
    ratio = score_clean / np.maximum(ref_clean, float(cfg["anchor_absolute_floor"]))

    anchor_strong = anchor & valid & (
        (score_clean >= thresholds.strong)
        | (
            (ratio >= float(cfg["anchor_relative_strong_ratio"]))
            & (score_clean >= float(cfg["anchor_absolute_floor"]))
        )
    )
    anchor_weak = anchor & valid & (
        (score_clean >= thresholds.weak)
        | (
            (ratio >= float(cfg["anchor_relative_weak_ratio"]))
            & (score_clean >= float(cfg["anchor_absolute_floor"]))
        )
    )

    inside = np.zeros_like(anchor, dtype=bool)
    labels, count = label_components(anchor)
    for label_id in range(1, count + 1):
        component = labels == label_id
        strong = anchor_strong & component
        weak = anchor_weak & component
        weak_fraction = float(weak.sum() / max(component.sum(), 1))
        # Older sensors may preserve the complete PV-field pattern while the
        # absolute score is uniformly below the modern strong threshold. When
        # broad weak support exists across the component, create local upper-tail
        # seeds instead of rejecting the entire historical facility.
        if not strong.any() and weak_fraction >= float(cfg["whole_component_fill_weak_fraction"]):
            component_values = score_clean[component & valid]
            if component_values.size:
                local_seed_threshold = max(
                    float(cfg["anchor_absolute_floor"]),
                    float(np.quantile(component_values, 0.75)),
                )
                strong = component & weak & (score_clean >= local_seed_threshold - 1e-7)
        if not strong.any():
            continue
        grown = connected_hysteresis(strong, weak, component)
        strong_fraction = float(strong.sum() / max(component.sum(), 1))
        if (
            weak_fraction >= float(cfg["whole_component_fill_weak_fraction"])
            and strong_fraction >= float(cfg["whole_component_fill_strong_fraction"])
        ):
            inside |= component
        else:
            inside |= fill_small_holes(
                ndi.binary_closing(grown, iterations=int(cfg["closing_iterations"])),
                int(cfg["maximum_hole_pixels"]),
            ) & component
    inside = remove_small(inside, int(cfg["inside_anchor_min_component_pixels"]))

    outside_domain = valid & ~anchor
    outside_strong_threshold = min(
        0.97,
        thresholds.strong + float(cfg["outside_anchor_strong_offset"]),
    )
    outside_strong = outside_domain & (score_clean >= outside_strong_threshold)
    outside_weak = outside_domain & (score_clean >= thresholds.weak)
    outside = connected_hysteresis(outside_strong, outside_weak, outside_domain)
    outside = ndi.binary_closing(outside, iterations=int(cfg["closing_iterations"]))
    outside = fill_small_holes(outside, int(cfg["maximum_hole_pixels"]))
    outside = remove_small(outside, int(cfg["outside_anchor_min_component_pixels"]))

    active = inside | outside
    audit = {
        "inside_pixels_raw": int(inside.sum()),
        "outside_pixels_raw": int(outside.sum()),
        "strong_threshold": thresholds.strong,
        "weak_threshold": thresholds.weak,
        "anchor_score_mean": float(np.nanmean(score[anchor & valid])) if (anchor & valid).any() else float("nan"),
        "outside_score_mean": float(np.nanmean(score[(~anchor) & valid])) if ((~anchor) & valid).any() else float("nan"),
    }
    return active, audit


def _bridge_short_gaps(stack: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = stack.astype(bool, copy=True)
    if maximum_gap <= 0 or result.shape[0] < 3:
        return result
    for gap in range(1, maximum_gap + 1):
        for start in range(1, result.shape[0] - gap):
            left = result[start - 1]
            right = result[start + gap]
            between = ~result[start : start + gap].any(axis=0)
            fill = left & right & between
            if fill.any():
                result[start : start + gap] |= fill
    return result


def temporal_regularize(
    raw_stack: np.ndarray,
    score_stack: np.ndarray,
    anchor: np.ndarray,
    years: list[int],
    thresholds: Thresholds,
    cfg: Mapping[str, Any],
    legacy_positive_stack: np.ndarray | None = None,
) -> np.ndarray:
    active = _bridge_short_gaps(raw_stack, int(cfg["maximum_internal_gap_years"]))
    support = active.sum(axis=0)
    inside_min = int(cfg["minimum_persistent_years_inside_anchor"])
    outside_min = int(cfg["minimum_persistent_years_outside_anchor"])
    strong_single = np.nan_to_num(score_stack, nan=0.0) >= (
        thresholds.strong + float(cfg["retain_single_year_if_score_margin"])
    )

    for index in range(active.shape[0]):
        inside_keep = anchor & ((support >= inside_min) | strong_single[index])
        outside_keep = (~anchor) & ((support >= outside_min) | strong_single[index])
        active[index] &= inside_keep | outside_keep

    if legacy_positive_stack is not None and bool(cfg.get("preserve_original_positive_evidence_all_years", True)):
        # Existing annual masks are positive evidence only. They never suppress a
        # newly recovered historical PV object.
        active |= legacy_positive_stack.astype(bool)

    if bool(cfg.get("preserve_original_terminal_year_mask", True)) and legacy_positive_stack is not None:
        terminal_index = len(years) - 1
        active[terminal_index] |= legacy_positive_stack[terminal_index]

    # A two-year pre-series state means the facility predates or begins at the
    # first available observation. Do not require an artificial change point.
    pre_count = min(int(cfg.get("pre_series_year_count", 2)), active.shape[0])
    if pre_count >= 2:
        preexisting = active[:pre_count].sum(axis=0) >= pre_count
        active[0] |= preexisting

    return active


def _sector_coverage(mask: np.ndarray, center_y: float, center_x: float, sectors: int) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    angles = (np.arctan2(ys - center_y, xs - center_x) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(angles / (2 * np.pi) * sectors).astype(int)
    return float(np.unique(np.clip(bins, 0, sectors - 1)).size / sectors)


def refine_reference_year(
    active_2024: np.ndarray,
    anchor: np.ndarray,
    score_2023: np.ndarray,
    score_2024: np.ndarray,
    score_2025: np.ndarray,
    valid: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    result = active_2024.astype(bool, copy=True) | anchor
    original_anchor_pixels = int(anchor.sum())
    audit: dict[str, float | int | bool] = {
        "anchor_pixels": original_anchor_pixels,
        "anchor_trimmed_pixels": 0,
        "anchor_retention_fraction": 1.0,
        "inner_annular_trim_applied": False,
        "outside_annular_suppression_applied": False,
    }

    if bool(cfg.get("allow_conservative_inner_ring_trim", True)) and original_anchor_pixels > 0:
        width = int(cfg.get("inner_ring_width_pixels", 1))
        eroded = ndi.binary_erosion(anchor, iterations=max(width, 1), border_value=0)
        inner_ring = anchor & ~eroded
        support = (
            np.nan_to_num(score_2023, nan=0.0)
            + np.nan_to_num(score_2024, nan=0.0)
            + np.nan_to_num(score_2025, nan=0.0)
        ) / 3.0
        ring_values = support[inner_ring & valid]
        low_threshold = (
            float(np.quantile(ring_values, float(cfg["low_evidence_quantile"])))
            if ring_values.size
            else 0.0
        )
        low = inner_ring & valid & (
            support <= min(low_threshold, float(cfg["three_year_support_threshold"]))
        )
        ys, xs = np.nonzero(anchor)
        if ys.size:
            coverage = _sector_coverage(
                low,
                float(np.mean(ys)),
                float(np.mean(xs)),
                int(cfg["annular_sector_count"]),
            )
        else:
            coverage = 0.0
        max_trim = int(np.floor(original_anchor_pixels * float(cfg["maximum_anchor_trim_fraction"])))
        if coverage >= float(cfg["annular_sector_coverage_threshold"]) and max_trim > 0:
            # Remove only the lowest-evidence ring pixels up to the strict area cap.
            candidate_indices = np.flatnonzero(low.ravel())
            if candidate_indices.size:
                candidate_scores = support.ravel()[candidate_indices]
                order = np.argsort(candidate_scores)
                chosen = candidate_indices[order[:max_trim]]
                trim = np.zeros_like(anchor, dtype=bool)
                trim.ravel()[chosen] = True
                result[trim] = False
                audit["anchor_trimmed_pixels"] = int(trim.sum())
                audit["inner_annular_trim_applied"] = bool(trim.any())
                audit["inner_annular_sector_coverage"] = coverage

    outside = result & ~anchor
    if outside.any():
        distance = ndi.distance_transform_edt(~anchor)
        ring = outside & (distance <= float(cfg["outside_ring_width_pixels"]))
        ys, xs = np.nonzero(anchor)
        coverage = (
            _sector_coverage(ring, float(np.mean(ys)), float(np.mean(xs)), int(cfg["annular_sector_count"]))
            if ys.size
            else 0.0
        )
        mean_score = float(np.nanmean(score_2024[ring])) if ring.any() else float("nan")
        if (
            ring.any()
            and coverage >= float(cfg["annular_sector_coverage_threshold"])
            and np.isfinite(mean_score)
            and mean_score <= float(cfg["outside_annular_max_mean_score"])
        ):
            result[ring & (np.nan_to_num(score_2024, nan=0.0) < float(cfg["outside_annular_max_mean_score"]))] = False
            audit["outside_annular_suppression_applied"] = True
            audit["outside_annular_sector_coverage"] = coverage
            audit["outside_annular_mean_score"] = mean_score

    retained = int((result & anchor).sum())
    audit["anchor_retention_fraction"] = float(retained / max(original_anchor_pixels, 1))
    return result, audit


def build_state_products(active: np.ndarray, years: list[int]) -> dict[str, np.ndarray]:
    cumulative = np.zeros_like(active, dtype=bool)
    newly = np.zeros_like(active, dtype=bool)
    inactive = np.zeros_like(active, dtype=bool)
    running = np.zeros(active.shape[1:], dtype=bool)
    first_year = np.zeros(active.shape[1:], dtype=np.uint16)
    for index, year in enumerate(years):
        previous = running.copy()
        running |= active[index]
        cumulative[index] = running
        newly[index] = running & ~previous
        inactive[index] = running & ~active[index]
        unseen = active[index] & (first_year == 0)
        first_year[unseen] = np.uint16(year)
    state = np.zeros_like(active, dtype=np.uint8)
    state[active] = 1
    state[newly] = 2
    state[inactive] = 3
    return {
        "active": active,
        "cumulative": cumulative,
        "newly": newly,
        "inactive": inactive,
        "state": state,
        "first_year": first_year,
    }
