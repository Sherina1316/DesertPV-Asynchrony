from __future__ import annotations

"""Spatially complete PV expansion-front reconstruction (v24.19).

The routines in this module solve two failure modes that cannot be handled by a
single pixel threshold:

1. a newly installed site may contain only a high-confidence spectral core,
   while panel rows and boundary pixels have weaker mixed-pixel probabilities;
2. the final observation year (2025) has no future image with which to prove
   persistence.

The implementation therefore uses two-threshold hysteresis, connected-component
geometry, local probability gain and explicit adjacency to the audited 2024
inventory.  Most objects require a high-confidence seed.  The sole exception is
a parcel-wide mixed-pixel state whose weak evidence is spatially distributed
across the component and persists in multiple annual images.  No reference
geometry is copied to an unsupported earlier year.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .mixed_parcel import distributed_support_metrics, stabilise_presence_track

EPS = 1e-6


@dataclass(frozen=True)
class ComponentDecision:
    pixels: int
    bbox_fill: float
    solidity: float
    mean_probability: float
    mean_gain: float
    adjacent_to_reference: bool
    accepted: bool
    source: int
    mean_novelty: float = 0.0
    mean_direct_change: float = 0.0
    mean_preexisting_risk: float = 0.0
    mean_prior_dark_persistence: float = 0.0
    mean_break_dominance: float = 0.0
    mean_persistent_nonpv_risk: float = 0.0
    novelty_fraction: float = 0.0
    direct_change_fraction: float = 0.0
    rejection_reason: str = ""


def _safe_temporal_median(
    probability: np.ndarray,
    valid: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Running temporal median without warnings on all-invalid pixels."""
    p = np.asarray(probability, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    out = np.zeros_like(p, dtype=np.float32)
    n = p.shape[0]
    for t in range(n):
        lo = max(0, t - int(radius))
        hi = min(n, t + int(radius) + 1)
        block = np.where(v[lo:hi], p[lo:hi], np.nan)
        count = np.isfinite(block).sum(axis=0)
        work = block.copy()
        if np.any(count == 0):
            rr, cc = np.where(count == 0)
            work[0, rr, cc] = 0.0
        with np.errstate(all="ignore"):
            med = np.nanmedian(work, axis=0)
        out[t] = np.where(count > 0, med, 0.0).astype(np.float32)
    return out


def _component_geometry(mask: np.ndarray) -> tuple[int, float, float]:
    pixels = int(mask.sum())
    if pixels <= 0:
        return 0, 0.0, 0.0
    rr, cc = np.where(mask)
    bbox_area = max(
        (int(rr.max()) - int(rr.min()) + 1)
        * (int(cc.max()) - int(cc.min()) + 1),
        1,
    )
    bbox_fill = float(pixels / bbox_area)
    filled = ndimage.binary_fill_holes(mask)
    solidity = float(pixels / max(int(filled.sum()), 1))
    return pixels, bbox_fill, solidity


def _propagate(seed: np.ndarray, support: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if not np.any(seed):
        return np.zeros(seed.shape, dtype=bool)
    support = support | seed
    return ndimage.binary_propagation(
        seed,
        structure=np.ones((3, 3), dtype=bool),
        mask=support,
    ).astype(bool)


def _component_presence_track(
    probability: np.ndarray,
    smooth: np.ndarray,
    valid: np.ndarray,
    component: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> np.ndarray:
    """Return years with coherent object-level PV state before 2024."""
    presence = np.zeros(reference_index, dtype=bool)
    pixels = int(component.sum())
    if pixels <= 0 or reference_index <= 0:
        return presence

    quantile = float(settings.get("historical_component_presence_quantile", 0.70))
    quantile_floor = float(settings.get("historical_component_presence_probability", 0.36))
    weak_pixel = float(settings.get("historical_component_weak_pixel_probability", 0.30))
    strong_pixel = float(settings.get("historical_component_strong_pixel_probability", 0.56))
    minimum_fraction = float(settings.get("historical_component_minimum_supported_fraction", 0.06))
    minimum_strong_fraction = float(settings.get("historical_component_minimum_strong_fraction", 0.015))
    small_limit = max(1, int(settings.get("historical_small_component_pixels", 16)))
    small_median_floor = float(settings.get("historical_small_component_median_probability", 0.34))
    small_peak_floor = float(settings.get("historical_small_component_peak_probability", 0.48))

    for t in range(reference_index):
        domain = component & valid[t]
        values = smooth[t][domain]
        raw_values = probability[t][domain]
        if values.size == 0:
            continue
        q = float(np.quantile(values, quantile))
        supported_fraction = float(np.count_nonzero(values >= weak_pixel) / values.size)
        strong_fraction = float(np.count_nonzero(raw_values >= strong_pixel) / raw_values.size)
        if pixels <= small_limit:
            presence[t] = bool(
                float(np.median(values)) >= small_median_floor
                or float(np.max(raw_values)) >= small_peak_floor
            )
        else:
            presence[t] = bool(
                (q >= quantile_floor and supported_fraction >= minimum_fraction)
                or strong_fraction >= minimum_strong_fraction
            )

    # A single cloudy/weak year may interrupt a physically persistent facility.
    for t in range(1, reference_index - 1):
        if not presence[t] and presence[t - 1] and presence[t + 1]:
            presence[t] = True

    window = max(1, int(settings.get("historical_component_support_window_years", 3)))
    minimum_years = max(1, int(settings.get("historical_component_minimum_support_years", 2)))
    stable = np.zeros_like(presence)
    for t in range(reference_index):
        hi = min(reference_index, t + window)
        required = min(minimum_years, hi - t)
        stable[t] = presence[t] and int(np.count_nonzero(presence[t:hi])) >= required
    return stable



def _component_mixed_parcel_track(
    probability: np.ndarray,
    smooth: np.ndarray,
    valid: np.ndarray,
    component: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> np.ndarray:
    """Identify years when a complete component is a mixed-pixel PV parcel.

    The gate is deliberately stricter than ordinary object presence.  Evidence
    must be spread across spatial sectors and both the component interior and
    boundary.  A high-probability patch confined to one side therefore remains a
    partial build-out and is processed by the normal growth-front pathway.
    """
    track = np.zeros(reference_index, dtype=bool)
    if reference_index <= 0 or not bool(settings.get("historical_mixed_parcel_enabled", True)):
        return track
    comp = np.asarray(component, dtype=bool)
    p = np.asarray(probability, dtype=np.float32)
    s = np.asarray(smooth, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    if not np.any(comp):
        return track

    weak_floor = float(settings.get("historical_parcel_weak_probability", 0.24))
    moderate_floor = float(settings.get("historical_parcel_moderate_probability", 0.32))
    strong_floor = float(settings.get("historical_parcel_strong_probability", 0.46))
    for t in range(reference_index):
        domain = comp & v[t]
        weak = domain & ((s[t] >= weak_floor) | (p[t] >= weak_floor + 0.04))
        moderate = domain & ((s[t] >= moderate_floor) | (p[t] >= moderate_floor + 0.06))
        strong = domain & ((s[t] >= strong_floor) | (p[t] >= strong_floor + 0.06))
        metrics = distributed_support_metrics(
            comp,
            weak,
            moderate,
            strong,
            v[t],
            np.maximum(p[t], s[t]),
            dilation_pixels=max(1, int(settings.get("historical_parcel_dilation_pixels", 2))),
            grid_size=max(2, int(settings.get("historical_parcel_grid_size", 3))),
        )
        track[t] = bool(
            metrics.valid_fraction >= float(settings.get("historical_parcel_minimum_valid_fraction", 0.50))
            and metrics.supported_fraction >= float(settings.get("historical_parcel_minimum_supported_fraction", 0.22))
            and metrics.moderate_fraction >= float(settings.get("historical_parcel_minimum_moderate_fraction", 0.08))
            and metrics.dilated_coverage >= float(settings.get("historical_parcel_minimum_dilated_coverage", 0.70))
            and metrics.sector_coverage >= float(settings.get("historical_parcel_minimum_sector_coverage", 0.55))
            and metrics.boundary_supported_fraction >= float(settings.get("historical_parcel_minimum_boundary_support", 0.18))
            and metrics.interior_supported_fraction >= float(settings.get("historical_parcel_minimum_interior_support", 0.22))
            and metrics.median_evidence >= float(settings.get("historical_parcel_minimum_median_probability", 0.23))
            and metrics.upper_quantile_evidence >= float(settings.get("historical_parcel_minimum_upper_probability", 0.32))
        )
    return stabilise_presence_track(
        track,
        window=max(1, int(settings.get("historical_parcel_support_window_years", 3))),
        minimum_years=max(1, int(settings.get("historical_parcel_minimum_support_years", 2))),
    )

def _stabilize_thin_component_shell(
    grown: np.ndarray,
    component: np.ndarray,
    annual_probability: np.ndarray,
    annual_smooth: np.ndarray,
    annual_valid: np.ndarray,
    component_present: bool,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fill only a thin, wrap-around mixed-pixel shell around an existing site.

    Interior missing blocks are preserved as real partial build-out.  The helper
    therefore removes contour-only year-to-year jitter without converting a
    partial station into the complete 2024 polygon.
    """
    grown = np.asarray(grown, dtype=bool).copy()
    rescued = np.zeros(grown.shape, dtype=bool)
    if not component_present or not np.any(grown):
        return grown, rescued

    pixels = int(component.sum())
    coverage = float(np.count_nonzero(grown & component) / max(pixels, 1))
    minimum_coverage = float(settings.get("historical_shell_minimum_component_coverage", 0.68))
    if coverage < minimum_coverage:
        return grown, rescued

    shell_width = max(1, int(settings.get("historical_shell_width_pixels", 2)))
    interior = ndimage.binary_erosion(
        component,
        structure=np.ones((3, 3), dtype=bool),
        iterations=shell_width,
        border_value=0,
    )
    missing = component & ~grown
    if not np.any(missing):
        return grown, rescued
    interior_missing_fraction = float(
        np.count_nonzero(missing & interior) / max(int(missing.sum()), 1)
    )
    if interior_missing_fraction > float(
        settings.get("historical_shell_maximum_interior_missing_fraction", 0.08)
    ):
        return grown, rescued

    boundary = component & ~ndimage.binary_erosion(
        component,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary_covered = boundary & ndimage.binary_dilation(
        missing,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    wrap_fraction = float(
        np.count_nonzero(boundary_covered) / max(int(boundary.sum()), 1)
    )
    if wrap_fraction < float(settings.get("historical_shell_minimum_wrap_fraction", 0.50)):
        return grown, rescued

    floor = float(settings.get("historical_shell_probability_floor", 0.20))
    near_grown = ndimage.binary_dilation(
        grown,
        structure=np.ones((3, 3), dtype=bool),
        iterations=shell_width,
    )
    fill = (
        missing
        & annual_valid
        & near_grown
        & ((annual_smooth >= floor) | (annual_probability >= floor + 0.04))
    )
    grown |= fill
    rescued |= fill
    return grown, rescued


def reconstruct_reference_growth_front(
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    onset_index: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct spatially complete pre-2024 installation extents.

    Pixel-level change points are combined with object-level annual spectral
    persistence.  The object evidence can relax a weak edge pixel, but cannot
    activate a component without a Landsat-supported seed.  Small components use
    adaptive seed counts, and contour-only gaps are stabilised separately from
    true interior partial build-out.
    """
    p = np.asarray(probability, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    inventory = np.asarray(inventory, dtype=bool)
    onset_index = np.asarray(onset_index)
    n = p.shape[0]

    # The forced 2024 endpoint must not leak into the 2023 temporal median.
    history_valid = v.copy()
    history_valid[reference_index] = False
    smooth = _safe_temporal_median(
        p,
        history_valid,
        int(settings.get("historical_temporal_median_radius", 1)),
    )

    seed_threshold = float(settings.get("historical_seed_probability", 0.54))
    strong_threshold = float(settings.get("historical_strong_probability", 0.70))
    extent_threshold = float(settings.get("historical_extent_probability", 0.34))
    edge_threshold = float(settings.get("historical_edge_probability", 0.28))
    component_seed_threshold = float(settings.get("historical_component_seed_probability", 0.42))
    onset_relax_floor = float(settings.get("historical_component_onset_relax_probability", 0.30))
    onset_tolerance = int(settings.get("historical_onset_tolerance_years", 1))
    min_seed_pixels = int(settings.get("historical_minimum_seed_pixels", 2))
    min_seed_fraction = float(settings.get("historical_minimum_seed_fraction", 0.004))
    max_gap_pixels = int(settings.get("historical_completion_gap_pixels", 2))
    min_component_support = float(settings.get("historical_minimum_component_support", 0.30))
    small_limit = max(1, int(settings.get("historical_small_component_pixels", 16)))

    structure = np.ones((3, 3), dtype=bool)
    labels, component_count = ndimage.label(inventory, structure=structure)
    components = [labels == label_id for label_id in range(1, component_count + 1)]
    presence_tracks = [
        _component_presence_track(p, smooth, v, component, reference_index, settings)
        for component in components
    ]
    mixed_parcel_tracks = [
        _component_mixed_parcel_track(p, smooth, v, component, reference_index, settings)
        for component in components
    ]

    active = np.zeros((n, *inventory.shape), dtype=bool)
    rescued = np.zeros_like(active, dtype=bool)
    previous = np.zeros(inventory.shape, dtype=bool)

    for t in range(reference_index):
        year_mask = np.zeros(inventory.shape, dtype=bool)
        for component, component_presence, mixed_parcel_presence in zip(
            components,
            presence_tracks,
            mixed_parcel_tracks,
        ):
            component_pixels = int(component.sum())
            if component_pixels == 0:
                continue

            if mixed_parcel_presence[t]:
                # Every cell is admitted only after the current annual image has
                # passed the distributed mixed-pixel gate. This is not blind
                # vector backcasting: a localised partial build-out cannot reach
                # the required sector/boundary coverage.
                year_mask |= component
                # ``rescued`` records the part completed by parcel topology,
                # rather than pixels that merely clear the permissive mixed-pixel
                # support gate.  Use the component seed/strong threshold here so
                # weak but spatially distributed panel fractions remain auditable
                # as reconstructed cells.
                direct_support_threshold = max(
                    float(settings.get("historical_component_seed_probability", 0.42)),
                    float(settings.get("historical_parcel_strong_probability", 0.46)),
                )
                directly_supported_now = component & v[t] & (
                    (smooth[t] >= direct_support_threshold)
                    | (p[t] >= direct_support_threshold)
                )
                rescued[t] |= component & ~directly_supported_now
                continue

            if not component_presence[t]:
                # Already installed pixels remain persistent even if the current
                # image is weak; a new component cannot start without support.
                year_mask |= previous & component
                continue

            eligible_onset = (
                (onset_index <= min(t + onset_tolerance, reference_index))
                | (
                    component
                    & v[t]
                    & (smooth[t] >= onset_relax_floor)
                )
            )
            seed = component & eligible_onset & v[t] & (
                (smooth[t] >= seed_threshold)
                | (p[t] >= strong_threshold)
                | (smooth[t] >= component_seed_threshold)
            )
            minimum_seeds = max(
                1 if component_pixels <= small_limit else min_seed_pixels,
                int(np.ceil(component_pixels * min_seed_fraction)),
            )
            minimum_seeds = min(minimum_seeds, component_pixels)
            if int(seed.sum()) < minimum_seeds:
                domain = component & v[t]
                if not np.any(domain):
                    year_mask |= previous & component
                    continue
                values = np.where(domain, np.maximum(p[t], smooth[t]), -np.inf)
                top = np.flatnonzero(domain)
                order = np.argsort(values.ravel()[top])[::-1]
                selected = top[order[:minimum_seeds]]
                if selected.size == 0 or float(values.ravel()[selected[0]]) < component_seed_threshold:
                    year_mask |= previous & component
                    continue
                seed = seed.copy()
                seed.ravel()[selected] = True

            support = component & v[t] & (
                (smooth[t] >= extent_threshold)
                | (
                    eligible_onset
                    & (smooth[t] >= edge_threshold)
                    & (p[t] >= edge_threshold)
                )
            )
            support |= previous & component
            grown = _propagate(seed | (previous & component), support)

            if max_gap_pixels > 0 and np.any(grown):
                closed = ndimage.binary_closing(
                    grown,
                    structure=structure,
                    iterations=max_gap_pixels,
                )
                distance = ndimage.distance_transform_edt(~grown)
                gap_fill = (
                    closed
                    & component
                    & v[t]
                    & (distance <= float(max_gap_pixels) + 0.01)
                    & (smooth[t] >= edge_threshold)
                )
                grown |= gap_fill

            if np.any(grown):
                mean_support = float(np.mean(smooth[t][grown]))
                adaptive_minimum = min_component_support
                if component_pixels <= small_limit:
                    adaptive_minimum = min(
                        adaptive_minimum,
                        float(settings.get("historical_small_component_minimum_support", 0.24)),
                    )
                if mean_support < adaptive_minimum:
                    year_mask |= previous & component
                    continue
                filled = ndimage.binary_fill_holes(grown) & component
                near_support = ndimage.binary_dilation(
                    support,
                    structure=structure,
                    iterations=max(1, max_gap_pixels),
                )
                grown |= filled & near_support
                grown, shell_rescue = _stabilize_thin_component_shell(
                    grown,
                    component,
                    p[t],
                    smooth[t],
                    v[t],
                    bool(component_presence[t]),
                    settings,
                )
                year_mask |= grown
                rescued[t] |= (grown & ~seed) | shell_rescue

        year_mask |= previous
        active[t] = year_mask
        previous = year_mask

    active[reference_index] = inventory
    for t in range(reference_index + 1, n):
        active[t] = inventory
    return active, rescued

def _baseline_median(
    values: np.ndarray,
    valid: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Robust baseline median without warnings on all-invalid pixels."""
    x = np.asarray(values, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    block = np.where(v[start:stop], x[start:stop], np.nan)
    count = np.isfinite(block).sum(axis=0)
    work = block.copy()
    if work.shape[0] == 0:
        return np.zeros(x.shape[1:], dtype=np.float32)
    if np.any(count == 0):
        rr, cc = np.where(count == 0)
        work[0, rr, cc] = 0.0
    with np.errstate(all="ignore"):
        median = np.nanmedian(work, axis=0)
    return np.where(count > 0, median, 0.0).astype(np.float32)


def _normalised_positive_gain(
    terminal: np.ndarray,
    baseline: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Map a positive terminal-minus-baseline change to [0, 1]."""
    delta = np.asarray(terminal, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    return np.clip(delta / max(float(scale), EPS), 0.0, 1.0).astype(np.float32)


def _temporal_fraction_above(
    values: np.ndarray,
    valid: np.ndarray,
    start: int,
    stop: int,
    threshold: float,
) -> np.ndarray:
    """Fraction of valid pre-terminal years exceeding a state threshold."""
    x = np.asarray(values, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    if stop <= start:
        return np.zeros(x.shape[1:], dtype=np.float32)
    vv = v[start:stop] & np.isfinite(x[start:stop])
    count = vv.sum(axis=0)
    hits = (vv & (x[start:stop] >= float(threshold))).sum(axis=0)
    out = np.zeros(x.shape[1:], dtype=np.float32)
    np.divide(hits, count, out=out, where=count > 0)
    return out


def _maximum_prior_positive_jump(
    values: np.ndarray,
    valid: np.ndarray,
    start: int,
    stop: int,
    scale: float,
    *,
    reverse: bool = False,
) -> np.ndarray:
    """Largest positive annual jump before the terminal transition.

    ``reverse=True`` is used for variables such as brightness for which a
    decrease is the PV-like direction.  The interval is [start, stop), so the
    2024-to-2025 transition is never included in the historical maximum.
    """
    x = np.asarray(values, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    out = np.zeros(x.shape[1:], dtype=np.float32)
    for year in range(max(start + 1, 1), stop):
        pair_valid = v[year - 1] & v[year]
        if reverse:
            jump = _normalised_positive_gain(x[year - 1], x[year], scale)
        else:
            jump = _normalised_positive_gain(x[year], x[year - 1], scale)
        out = np.maximum(out, np.where(pair_valid, jump, 0.0))
    return out.astype(np.float32)


def _spatial_darkness(
    brightness: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Robust absolute darkness within the current processing window.

    Unlike a local-contrast-only score, this remains high inside a large,
    spatially uniform black PV block.  Quantiles are computed only from valid
    pixels, and the result is bounded to [0, 1].
    """
    x = np.asarray(brightness, dtype=np.float32)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if int(v.sum()) < 16:
        return np.zeros(x.shape, dtype=np.float32)
    values = x[v]
    q05 = float(np.quantile(values, 0.05))
    q55 = float(np.quantile(values, 0.55))
    scale = max(q55 - q05, 1e-4)
    out = np.clip((q55 - x) / scale, 0.0, 1.0).astype(np.float32)
    out[~v] = 0.0
    return out


def build_terminal_extension_evidence(
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
    *,
    site_score: np.ndarray | None = None,
    positive_similarity: np.ndarray | None = None,
    pv_fraction: np.ndarray | None = None,
    local_dark: np.ndarray | None = None,
    brightness: np.ndarray | None = None,
    sand_risk: np.ndarray | None = None,
    confuser: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build causal evidence for 2025 expansion over the *entire* unit.

    The 2024 inventory is frozen before this function is evaluated.  Two
    evidence families are deliberately separated:

    * inventory-conditioned evidence for nearby growth fronts;
    * full-domain remote-census evidence, which does not use distance to the
      2024 inventory and can therefore discover a completely new 2025 site.

    The remote census combines absolute 2025 darkness, 2024-to-2025 spectral
    change, weak model/prototype support, neighbourhood density and explicit
    sand/confuser penalties.  A large uniform black site can be recovered even
    when local contrast and segmentation probability are weak, while stable
    dark sand is rejected because it lacks coherent temporal change.
    """
    p = np.asarray(probability, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    inventory = np.asarray(inventory, dtype=bool)
    n = p.shape[0]
    t = reference_index + 1
    shape = inventory.shape
    zeros = np.zeros(shape, dtype=np.float32)
    if t >= n:
        return {
            "probability_gain": zeros.copy(),
            "multicue_change": zeros.copy(),
            "site_likelihood": zeros.copy(),
            "component_similarity": zeros.copy(),
            "front_proximity": zeros.copy(),
            "sand": zeros.copy(),
            "confuser": zeros.copy(),
            "absolute_darkness": zeros.copy(),
            "brightness_drop": zeros.copy(),
            "remote_census_likelihood": zeros.copy(),
            "remote_change_support": zeros.copy(),
            "remote_candidate_density": zeros.copy(),
            "terminal_novelty_score": zeros.copy(),
            "direct_multicue_change": zeros.copy(),
            "preexisting_state_risk": zeros.copy(),
            "prior_dark_persistence": zeros.copy(),
            "terminal_break_dominance": zeros.copy(),
            "persistent_nonpv_risk": zeros.copy(),
            "remote_census_available": zeros.copy(),
        }

    baseline_years = int(settings.get("terminal_baseline_years", 3))
    baseline_start = max(0, reference_index - baseline_years)
    p25 = p[t]
    p_base = _baseline_median(p, v, baseline_start, reference_index)
    probability_gain = np.clip(p25 - p_base, -1.0, 1.0).astype(np.float32)

    def feature_or_probability(values: np.ndarray | None) -> np.ndarray:
        return p if values is None else np.asarray(values, dtype=np.float32)

    score_ts = feature_or_probability(site_score)
    positive_ts = feature_or_probability(positive_similarity)
    fraction_ts = feature_or_probability(pv_fraction)
    dark_ts = feature_or_probability(local_dark)
    brightness_ts = None if brightness is None else np.asarray(brightness, dtype=np.float32)
    sand_ts = np.zeros_like(p, dtype=np.float32) if sand_risk is None else np.asarray(sand_risk, dtype=np.float32)
    confuser_ts = np.zeros_like(p, dtype=np.float32) if confuser is None else np.asarray(confuser, dtype=np.float32)

    score25 = np.clip(score_ts[t], 0.0, 1.0)
    positive25 = np.clip(positive_ts[t], 0.0, 1.0)
    fraction25 = np.clip(fraction_ts[t], 0.0, 1.0)
    dark25 = np.clip(dark_ts[t], 0.0, 1.0)
    sand25 = np.clip(sand_ts[t], 0.0, 1.0)
    confuser25 = np.clip(confuser_ts[t], 0.0, 1.0)

    score_base = _baseline_median(score_ts, v, baseline_start, reference_index)
    positive_base = _baseline_median(positive_ts, v, baseline_start, reference_index)
    fraction_base = _baseline_median(fraction_ts, v, baseline_start, reference_index)
    dark_base = _baseline_median(dark_ts, v, baseline_start, reference_index)

    remote_census_available = brightness_ts is not None
    reference_valid = v[reference_index] & v[t]
    score_ref = np.clip(score_ts[reference_index], 0.0, 1.0)
    positive_ref = np.clip(positive_ts[reference_index], 0.0, 1.0)
    fraction_ref = np.clip(fraction_ts[reference_index], 0.0, 1.0)
    dark_ref = np.clip(dark_ts[reference_index], 0.0, 1.0)
    p_ref = np.clip(p[reference_index], 0.0, 1.0)

    if brightness_ts is not None:
        brightness25 = np.asarray(brightness_ts[t], dtype=np.float32)
        brightness_ref = np.asarray(brightness_ts[reference_index], dtype=np.float32)
        brightness_base = _baseline_median(brightness_ts, v, baseline_start, reference_index)
        absolute_darkness = _spatial_darkness(brightness25, v[t])
        reference_absolute_darkness = _spatial_darkness(brightness_ref, v[reference_index])
        brightness_drop = _normalised_positive_gain(
            brightness_base,
            brightness25,
            float(settings.get("terminal_brightness_drop_scale", 0.055)),
        )
        direct_brightness_drop = _normalised_positive_gain(
            brightness_ref,
            brightness25,
            float(settings.get("terminal_direct_brightness_drop_scale", 0.045)),
        )
        prior_dark_layers = [
            _spatial_darkness(brightness_ts[year], v[year])
            for year in range(baseline_start, reference_index + 1)
        ]
        prior_dark_stack = np.stack(prior_dark_layers, axis=0)
        prior_absolute_darkness = np.median(prior_dark_stack, axis=0).astype(np.float32)
        prior_dark_persistence = np.mean(
            prior_dark_stack
            >= float(settings.get("terminal_prior_darkness_state_threshold", 0.42)),
            axis=0,
        ).astype(np.float32)
        maximum_prior_brightness_jump = _maximum_prior_positive_jump(
            brightness_ts,
            v,
            baseline_start,
            reference_index + 1,
            float(settings.get("terminal_direct_brightness_drop_scale", 0.045)),
            reverse=True,
        )
    else:
        # Backward-compatible fallback used by unit tests and the temporal
        # fallback path.  Production Stage 7 always provides brightness.
        absolute_darkness = dark25.astype(np.float32)
        reference_absolute_darkness = dark_ref.astype(np.float32)
        brightness_drop = _normalised_positive_gain(
            dark25,
            dark_base,
            float(settings.get("terminal_dark_gain_scale", 0.14)),
        )
        direct_brightness_drop = _normalised_positive_gain(
            dark25,
            dark_ref,
            float(settings.get("terminal_direct_dark_gain_scale", 0.11)),
        )
        prior_absolute_darkness = _baseline_median(
            dark_ts, v, baseline_start, reference_index + 1
        )
        prior_dark_persistence = _temporal_fraction_above(
            dark_ts,
            v,
            baseline_start,
            reference_index + 1,
            float(settings.get("terminal_prior_darkness_state_threshold", 0.42)),
        )
        maximum_prior_brightness_jump = _maximum_prior_positive_jump(
            dark_ts,
            v,
            baseline_start,
            reference_index + 1,
            float(settings.get("terminal_direct_dark_gain_scale", 0.11)),
            reverse=False,
        )

    score_gain = _normalised_positive_gain(
        score25, score_base, float(settings.get("terminal_score_gain_scale", 0.16))
    )
    similarity_gain = _normalised_positive_gain(
        positive25, positive_base, float(settings.get("terminal_similarity_gain_scale", 0.16))
    )
    fraction_gain = _normalised_positive_gain(
        fraction25, fraction_base, float(settings.get("terminal_fraction_gain_scale", 0.16))
    )
    dark_gain = _normalised_positive_gain(
        dark25, dark_base, float(settings.get("terminal_dark_gain_scale", 0.14))
    )
    direct_score_gain = _normalised_positive_gain(
        score25, score_ref, float(settings.get("terminal_direct_score_gain_scale", 0.12))
    )
    direct_similarity_gain = _normalised_positive_gain(
        positive25, positive_ref, float(settings.get("terminal_direct_similarity_gain_scale", 0.12))
    )
    direct_fraction_gain = _normalised_positive_gain(
        fraction25, fraction_ref, float(settings.get("terminal_direct_fraction_gain_scale", 0.12))
    )
    direct_dark_gain = _normalised_positive_gain(
        dark25, dark_ref, float(settings.get("terminal_direct_dark_gain_scale", 0.11))
    )
    direct_probability_gain = _normalised_positive_gain(
        p25, p_ref, float(settings.get("terminal_direct_probability_gain_scale", 0.12))
    )

    multicue_change = np.clip(
        0.30 * score_gain
        + 0.22 * similarity_gain
        + 0.18 * fraction_gain
        + 0.12 * dark_gain
        + 0.18 * brightness_drop,
        0.0,
        1.0,
    ).astype(np.float32)
    direct_multicue_change = np.clip(
        0.24 * direct_score_gain
        + 0.18 * direct_similarity_gain
        + 0.17 * direct_fraction_gain
        + 0.11 * direct_dark_gain
        + 0.20 * direct_brightness_drop
        + 0.10 * direct_probability_gain,
        0.0,
        1.0,
    ).astype(np.float32)

    site_likelihood = np.clip(
        0.25 * p25
        + 0.20 * score25
        + 0.16 * positive25
        + 0.14 * fraction25
        + 0.10 * dark25
        + 0.07 * absolute_darkness
        + 0.04 * (1.0 - sand25)
        + 0.04 * (1.0 - confuser25),
        0.0,
        1.0,
    ).astype(np.float32)

    remote_census_likelihood = np.clip(
        0.28 * absolute_darkness
        + 0.18 * dark25
        + 0.14 * score25
        + 0.10 * positive25
        + 0.10 * fraction25
        + 0.08 * p25
        + 0.06 * (1.0 - sand25)
        + 0.06 * (1.0 - confuser25),
        0.0,
        1.0,
    ).astype(np.float32)
    reference_remote_state = np.clip(
        0.28 * reference_absolute_darkness
        + 0.18 * dark_ref
        + 0.14 * score_ref
        + 0.10 * positive_ref
        + 0.10 * fraction_ref
        + 0.08 * p_ref
        + 0.06 * (1.0 - np.clip(sand_ts[reference_index], 0.0, 1.0))
        + 0.06 * (1.0 - np.clip(confuser_ts[reference_index], 0.0, 1.0)),
        0.0,
        1.0,
    ).astype(np.float32)
    terminal_state_gain = _normalised_positive_gain(
        remote_census_likelihood,
        reference_remote_state,
        float(settings.get("terminal_remote_state_gain_scale", 0.18)),
    )

    prior_state_jump = np.maximum.reduce([
        _maximum_prior_positive_jump(
            score_ts, v, baseline_start, reference_index + 1,
            float(settings.get("terminal_direct_score_gain_scale", 0.12))
        ),
        _maximum_prior_positive_jump(
            positive_ts, v, baseline_start, reference_index + 1,
            float(settings.get("terminal_direct_similarity_gain_scale", 0.12))
        ),
        _maximum_prior_positive_jump(
            fraction_ts, v, baseline_start, reference_index + 1,
            float(settings.get("terminal_direct_fraction_gain_scale", 0.12))
        ),
        maximum_prior_brightness_jump,
    ]).astype(np.float32)
    terminal_jump = np.clip(
        0.62 * direct_multicue_change + 0.38 * direct_brightness_drop,
        0.0,
        1.0,
    )
    terminal_break_dominance = np.clip(
        (
            terminal_jump
            - prior_state_jump
            + float(settings.get("terminal_break_dominance_margin", 0.06))
        )
        / max(float(settings.get("terminal_break_dominance_scale", 0.34)), EPS),
        0.0,
        1.0,
    ).astype(np.float32)

    preexisting_state_risk = np.clip(
        0.44 * reference_remote_state
        + 0.26 * prior_absolute_darkness
        + 0.22 * prior_dark_persistence
        + 0.08 * np.clip(p_ref, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    terminal_novelty_score = np.clip(
        0.34 * direct_multicue_change
        + 0.22 * direct_brightness_drop
        + 0.18 * terminal_state_gain
        + 0.14 * terminal_break_dominance
        + 0.12 * (1.0 - preexisting_state_risk),
        0.0,
        1.0,
    ).astype(np.float32)
    persistent_nonpv_risk = np.clip(
        0.46 * preexisting_state_risk
        + 0.34 * prior_dark_persistence
        + 0.20 * prior_absolute_darkness * (1.0 - terminal_break_dominance),
        0.0,
        1.0,
    ).astype(np.float32)

    remote_change_support = np.clip(
        0.52 * terminal_novelty_score
        + 0.24 * direct_multicue_change
        + 0.14 * direct_brightness_drop
        + 0.10 * multicue_change,
        0.0,
        1.0,
    ).astype(np.float32)

    density_window = max(3, int(settings.get("terminal_remote_census_density_window", 7)))
    if density_window % 2 == 0:
        density_window += 1
    density_basis = (
        v[t]
        & (~inventory)
        & (remote_census_likelihood >= float(settings.get("terminal_remote_census_density_likelihood", 0.34)))
        & (absolute_darkness >= float(settings.get("terminal_remote_census_density_darkness", 0.28)))
        & (remote_change_support >= float(settings.get("terminal_remote_census_density_change", 0.025)))
        & (terminal_novelty_score >= float(settings.get("terminal_remote_census_density_novelty", 0.20)))
        & (direct_multicue_change >= float(settings.get("terminal_remote_census_density_direct_change", 0.12)))
        & (preexisting_state_risk <= float(settings.get("terminal_remote_census_density_maximum_preexisting", 0.58)))
        & (persistent_nonpv_risk <= float(settings.get("terminal_remote_census_density_maximum_persistent_risk", 0.62)))
        & (sand25 <= float(settings.get("terminal_remote_census_density_maximum_sand", 0.72)))
        & (confuser25 <= float(settings.get("terminal_remote_census_density_maximum_confuser", 0.70)))
    )
    remote_candidate_density = ndimage.uniform_filter(
        density_basis.astype(np.float32),
        size=density_window,
        mode="nearest",
    ).astype(np.float32)

    radius = max(1, int(settings.get("terminal_perimeter_search_pixels", 22)))
    outside = ~inventory
    distance = ndimage.distance_transform_edt(outside).astype(np.float32)
    front_proximity = np.clip(1.0 - (distance - 1.0) / float(radius), 0.0, 1.0)
    front_proximity[inventory] = 0.0

    # Component-conditioned similarity for nearby extensions.
    component_similarity = np.zeros(shape, dtype=np.float32)
    labels, component_count = ndimage.label(inventory, structure=np.ones((3, 3), dtype=bool))
    terminal_features = [score25, positive25, fraction25, dark25]
    reference_features = [
        np.clip(score_ts[reference_index], 0.0, 1.0),
        np.clip(positive_ts[reference_index], 0.0, 1.0),
        np.clip(fraction_ts[reference_index], 0.0, 1.0),
        np.clip(dark_ts[reference_index], 0.0, 1.0),
    ]
    feature_floors = [0.10, 0.10, 0.10, 0.08]
    for label_id in range(1, component_count + 1):
        component = labels == label_id
        component_valid = component & v[reference_index]
        if int(component_valid.sum()) < 2:
            component_valid = component
        ring = ndimage.binary_dilation(
            component,
            structure=np.ones((3, 3), dtype=bool),
            iterations=radius,
        ) & outside
        if not np.any(ring):
            continue
        similarities: list[np.ndarray] = []
        for ref_feature, terminal_feature, floor in zip(reference_features, terminal_features, feature_floors):
            values = ref_feature[component_valid]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) if values.size > 1 else 0.0
            scale = max(1.4826 * mad, float(floor))
            z = np.abs(terminal_feature - median) / scale
            similarities.append(np.exp(-0.5 * (z / 2.25) ** 2).astype(np.float32))
        local_similarity = (
            np.mean(np.stack(similarities, axis=0), axis=0).astype(np.float32)
            if similarities
            else site_likelihood
        )
        component_similarity[ring] = np.maximum(component_similarity[ring], local_similarity[ring])

    return {
        "probability_gain": probability_gain,
        "multicue_change": multicue_change,
        "site_likelihood": site_likelihood,
        "component_similarity": component_similarity,
        "front_proximity": front_proximity.astype(np.float32),
        "sand": sand25.astype(np.float32),
        "confuser": confuser25.astype(np.float32),
        "absolute_darkness": absolute_darkness.astype(np.float32),
        "brightness_drop": brightness_drop.astype(np.float32),
        "remote_census_likelihood": remote_census_likelihood,
        "remote_change_support": remote_change_support,
        "remote_candidate_density": remote_candidate_density,
        "terminal_novelty_score": terminal_novelty_score,
        "direct_multicue_change": direct_multicue_change,
        "preexisting_state_risk": preexisting_state_risk,
        "prior_dark_persistence": prior_dark_persistence,
        "terminal_break_dominance": terminal_break_dominance,
        "persistent_nonpv_risk": persistent_nonpv_risk,
        "remote_census_available": np.full(shape, 1.0 if remote_census_available else 0.0, dtype=np.float32),
    }


def _is_annular_boundary_displacement(
    obj: np.ndarray,
    inventory_labels: np.ndarray,
    inventory_count: int,
    distance_to_inventory: np.ndarray,
    settings: dict[str, Any],
) -> bool:
    """Identify thin candidate contours wrapping around an existing 2024 site."""
    if not bool(settings.get("terminal_suppress_annular_boundary_change", True)):
        return False
    if inventory_count <= 0 or not np.any(obj):
        return False

    structure = np.ones((3, 3), dtype=bool)
    contact_tolerance = max(1, int(settings.get("terminal_annular_contact_tolerance_pixels", 2)))
    maximum_distance = float(settings.get("terminal_annular_maximum_distance_pixels", 2.5))
    minimum_wrap = float(settings.get("terminal_annular_minimum_wrap_fraction", 0.55))
    minimum_thin_fraction = float(settings.get("terminal_annular_minimum_thin_fraction", 0.78))
    thin_fraction = float(
        np.count_nonzero(obj & (distance_to_inventory <= maximum_distance))
        / max(int(obj.sum()), 1)
    )
    if thin_fraction < minimum_thin_fraction:
        return False

    near_obj = ndimage.binary_dilation(
        obj,
        structure=structure,
        iterations=contact_tolerance,
    )
    for label_id in range(1, inventory_count + 1):
        component = inventory_labels == label_id
        if not np.any(near_obj & component):
            continue
        boundary = component & ~ndimage.binary_erosion(
            component,
            structure=structure,
            border_value=0,
        )
        wrap = float(np.count_nonzero(boundary & near_obj) / max(int(boundary.sum()), 1))
        if wrap >= minimum_wrap:
            return True
    return False

def detect_terminal_year_expansion(
    annual_candidate: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
    *,
    site_score: np.ndarray | None = None,
    positive_similarity: np.ndarray | None = None,
    pv_fraction: np.ndarray | None = None,
    local_dark: np.ndarray | None = None,
    brightness: np.ndarray | None = None,
    sand_risk: np.ndarray | None = None,
    confuser: np.ndarray | None = None,
    precomputed_evidence: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ComponentDecision]]:
    """Detect 2025 PV expansion with four independently audited pathways.

    Source codes:

    1. probability-gain-supported adjacent expansion;
    2. strict legacy remote site (high model probability and gain);
    3. component-conditioned perimeter extension;
    4. novelty-gated remote change census, independent of distance to 2024 sites and of
       the annual candidate mask.

    Path 4 is the key v24.22 change.  It first generates candidates over every
    valid 2025 pixel, bridges panel-row gaps at site scale, and then accepts
    only large, dense, temporally new, dark and low-confuser objects.  Distance
    to the 2024 inventory is never used as a positive requirement.
    """
    n = annual_candidate.shape[0]
    t = reference_index + 1
    empty = np.zeros(inventory.shape, dtype=bool)
    source = np.zeros(inventory.shape, dtype=np.uint8)
    gain_out = np.zeros(inventory.shape, dtype=np.float32)
    if t >= n:
        return empty, source, gain_out, []

    p25 = np.asarray(probability[t], dtype=np.float32)
    v = np.asarray(valid[t], dtype=bool)
    inventory = np.asarray(inventory, dtype=bool)
    outside = ~inventory
    candidate = np.asarray(annual_candidate[t], dtype=bool) & v & outside

    evidence = precomputed_evidence or build_terminal_extension_evidence(
        probability,
        valid,
        inventory,
        reference_index,
        settings,
        site_score=site_score,
        positive_similarity=positive_similarity,
        pv_fraction=pv_fraction,
        local_dark=local_dark,
        brightness=brightness,
        sand_risk=sand_risk,
        confuser=confuser,
    )
    gain = np.asarray(evidence["probability_gain"], dtype=np.float32)
    multicue_change = np.asarray(evidence["multicue_change"], dtype=np.float32)
    site_likelihood = np.asarray(evidence["site_likelihood"], dtype=np.float32)
    component_similarity = np.asarray(evidence["component_similarity"], dtype=np.float32)
    front_proximity = np.asarray(evidence["front_proximity"], dtype=np.float32)
    terminal_sand = np.asarray(evidence["sand"], dtype=np.float32)
    terminal_confuser = np.asarray(evidence["confuser"], dtype=np.float32)
    absolute_darkness = np.asarray(evidence.get("absolute_darkness", site_likelihood), dtype=np.float32)
    remote_likelihood = np.asarray(evidence.get("remote_census_likelihood", site_likelihood), dtype=np.float32)
    remote_change = np.asarray(evidence.get("remote_change_support", multicue_change), dtype=np.float32)
    remote_density = np.asarray(evidence.get("remote_candidate_density", np.zeros_like(p25)), dtype=np.float32)
    terminal_novelty = np.asarray(evidence.get("terminal_novelty_score", remote_change), dtype=np.float32)
    direct_change = np.asarray(evidence.get("direct_multicue_change", multicue_change), dtype=np.float32)
    preexisting_risk = np.asarray(evidence.get("preexisting_state_risk", np.zeros_like(p25)), dtype=np.float32)
    prior_dark_persistence = np.asarray(evidence.get("prior_dark_persistence", np.zeros_like(p25)), dtype=np.float32)
    break_dominance = np.asarray(evidence.get("terminal_break_dominance", remote_change), dtype=np.float32)
    persistent_nonpv_risk = np.asarray(evidence.get("persistent_nonpv_risk", preexisting_risk), dtype=np.float32)
    remote_census_available = bool(np.any(np.asarray(evidence.get("remote_census_available", np.zeros_like(p25))) > 0.5))

    adjacent_radius = max(1, int(settings.get("terminal_adjacent_search_pixels", 12)))
    adjacent_zone = ndimage.binary_dilation(
        inventory,
        structure=np.ones((3, 3), dtype=bool),
        iterations=adjacent_radius,
    ) & outside
    perimeter_radius = max(adjacent_radius, int(settings.get("terminal_perimeter_search_pixels", 22)))
    perimeter_zone = ndimage.binary_dilation(
        inventory,
        structure=np.ones((3, 3), dtype=bool),
        iterations=perimeter_radius,
    ) & outside

    base = v & outside
    adjacent_seed = (
        base
        & adjacent_zone
        & (p25 >= float(settings.get("terminal_adjacent_seed_probability", 0.52)))
        & (gain >= float(settings.get("terminal_adjacent_minimum_gain", 0.055)))
    )
    legacy_remote_seed = (
        base
        & ~perimeter_zone
        & (p25 >= float(settings.get("terminal_remote_seed_probability", 0.68)))
        & (gain >= float(settings.get("terminal_remote_minimum_gain", 0.12)))
    )
    very_strong = base & (
        p25 >= float(settings.get("terminal_absolute_override_probability", 0.82))
    ) & (gain >= float(settings.get("terminal_absolute_override_gain", 0.08)))

    perimeter_seed = (
        base
        & perimeter_zone
        & (site_likelihood >= float(settings.get("terminal_perimeter_seed_likelihood", 0.55)))
        & (component_similarity >= float(settings.get("terminal_perimeter_seed_similarity", 0.54)))
        & (terminal_sand <= float(settings.get("terminal_perimeter_maximum_sand", 0.58)))
        & (terminal_confuser <= float(settings.get("terminal_perimeter_maximum_confuser", 0.58)))
        & (
            (multicue_change >= float(settings.get("terminal_perimeter_minimum_multicue_change", 0.18)))
            | (site_likelihood >= float(settings.get("terminal_perimeter_strong_likelihood", 0.67)))
        )
    )

    # Full-domain candidate generation: no annual-mask requirement and no
    # distance-to-reference requirement.
    remote_census_seed = (
        remote_census_available
        & base
        & (remote_likelihood >= float(settings.get("terminal_remote_census_seed_likelihood", 0.49)))
        & (absolute_darkness >= float(settings.get("terminal_remote_census_seed_darkness", 0.40)))
        & (remote_density >= float(settings.get("terminal_remote_census_seed_density", 0.22)))
        & (terminal_novelty >= float(settings.get("terminal_remote_census_seed_novelty", 0.46)))
        & (direct_change >= float(settings.get("terminal_remote_census_seed_direct_change", 0.25)))
        & (preexisting_risk <= float(settings.get("terminal_remote_census_seed_maximum_preexisting", 0.48)))
        & (prior_dark_persistence <= float(settings.get("terminal_remote_census_seed_maximum_prior_dark_persistence", 0.48)))
        & (persistent_nonpv_risk <= float(settings.get("terminal_remote_census_seed_maximum_persistent_risk", 0.52)))
        & (terminal_sand <= float(settings.get("terminal_remote_census_seed_maximum_sand", 0.64)))
        & (terminal_confuser <= float(settings.get("terminal_remote_census_seed_maximum_confuser", 0.62)))
        & (
            (remote_change >= float(settings.get("terminal_remote_census_seed_change", 0.10)))
            | (
                (absolute_darkness >= float(settings.get("terminal_remote_census_strong_darkness", 0.72)))
                & (terminal_novelty >= float(settings.get("terminal_remote_census_strong_dark_minimum_novelty", 0.52)))
                & (break_dominance >= float(settings.get("terminal_remote_census_strong_dark_minimum_dominance", 0.38)))
            )
        )
    )
    remote_census_support = (
        remote_census_available
        & base
        & (remote_likelihood >= float(settings.get("terminal_remote_census_extent_likelihood", 0.34)))
        & (absolute_darkness >= float(settings.get("terminal_remote_census_extent_darkness", 0.27)))
        & (remote_density >= float(settings.get("terminal_remote_census_extent_density", 0.12)))
        & (remote_change >= float(settings.get("terminal_remote_census_extent_change", 0.025)))
        & (terminal_novelty >= float(settings.get("terminal_remote_census_extent_novelty", 0.16)))
        & (direct_change >= float(settings.get("terminal_remote_census_extent_direct_change", 0.08)))
        & (preexisting_risk <= float(settings.get("terminal_remote_census_extent_maximum_preexisting", 0.64)))
        & (persistent_nonpv_risk <= float(settings.get("terminal_remote_census_extent_maximum_persistent_risk", 0.66)))
        & (terminal_sand <= float(settings.get("terminal_remote_census_extent_maximum_sand", 0.70)))
        & (terminal_confuser <= float(settings.get("terminal_remote_census_extent_maximum_confuser", 0.68)))
    )

    gain_support = outside & v & (
        (
            candidate
            & (p25 >= float(settings.get("terminal_extent_probability", 0.38)))
            & (gain >= float(settings.get("terminal_extent_minimum_gain", 0.025)))
        )
        | (
            (p25 >= float(settings.get("terminal_nonmask_extent_probability", 0.50)))
            & (gain >= float(settings.get("terminal_nonmask_extent_gain", 0.08)))
        )
    )
    perimeter_support = (
        base
        & perimeter_zone
        & (site_likelihood >= float(settings.get("terminal_perimeter_extent_likelihood", 0.42)))
        & (component_similarity >= float(settings.get("terminal_perimeter_extent_similarity", 0.43)))
        & (terminal_sand <= float(settings.get("terminal_perimeter_support_maximum_sand", 0.64)))
        & (terminal_confuser <= float(settings.get("terminal_perimeter_support_maximum_confuser", 0.64)))
        & (
            candidate
            | (multicue_change >= float(settings.get("terminal_perimeter_extent_multicue_change", 0.08)))
            | (site_likelihood >= float(settings.get("terminal_perimeter_absolute_extent_likelihood", 0.58)))
        )
    )
    legacy_remote_support = outside & v & ~perimeter_zone & (
        (p25 >= float(settings.get("terminal_remote_extent_probability", 0.54)))
        & (gain >= float(settings.get("terminal_remote_extent_gain", 0.07)))
    )

    ordinary_seed = adjacent_seed | legacy_remote_seed | very_strong | perimeter_seed
    ordinary_support = gain_support | perimeter_support | legacy_remote_support | ordinary_seed
    ordinary_grown = _propagate(ordinary_seed, ordinary_support)
    ordinary_grown = ndimage.binary_closing(
        ordinary_grown,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(settings.get("terminal_closing_iterations", 1)),
    ) & ordinary_support

    # Bridge panel-row gaps only inside the independently generated remote
    # support field.  The bridge is constrained by a weak absolute-state gate,
    # so it cannot connect across arbitrary background.
    bridge_iterations = max(0, int(settings.get("terminal_remote_census_bridge_iterations", 2)))
    bridged_remote_support = remote_census_support.copy()
    if bridge_iterations > 0:
        bridged_remote_support = ndimage.binary_closing(
            bridged_remote_support,
            structure=np.ones((3, 3), dtype=bool),
            iterations=bridge_iterations,
        )
        weak_bridge_gate = (
            base
            & (remote_likelihood >= float(settings.get("terminal_remote_census_bridge_likelihood", 0.29)))
            & (absolute_darkness >= float(settings.get("terminal_remote_census_bridge_darkness", 0.20)))
            & (terminal_novelty >= float(settings.get("terminal_remote_census_bridge_novelty", 0.10)))
            & (preexisting_risk <= float(settings.get("terminal_remote_census_bridge_maximum_preexisting", 0.70)))
            & (persistent_nonpv_risk <= float(settings.get("terminal_remote_census_bridge_maximum_persistent_risk", 0.72)))
            & (terminal_sand <= float(settings.get("terminal_remote_census_bridge_maximum_sand", 0.74)))
            & (terminal_confuser <= float(settings.get("terminal_remote_census_bridge_maximum_confuser", 0.72)))
        )
        bridged_remote_support &= weak_bridge_gate | remote_census_support | remote_census_seed
    remote_grown = _propagate(remote_census_seed, bridged_remote_support | remote_census_seed)
    remote_grown = ndimage.binary_closing(
        remote_grown,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, int(settings.get("terminal_remote_census_object_closing_iterations", 2))),
    ) & base & (bridged_remote_support | remote_census_seed)

    grown = ordinary_grown | remote_grown
    labels, count = ndimage.label(grown, structure=np.ones((3, 3), dtype=bool))
    accepted = np.zeros(inventory.shape, dtype=bool)
    decisions: list[ComponentDecision] = []
    reference_contact = ndimage.binary_dilation(
        inventory,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, int(settings.get("terminal_contact_tolerance_pixels", 2))),
    ) & outside
    inventory_labels, inventory_count = ndimage.label(
        inventory,
        structure=np.ones((3, 3), dtype=bool),
    )
    distance_to_inventory = ndimage.distance_transform_edt(~inventory)

    for label_id in range(1, count + 1):
        obj = labels == label_id
        pixels, bbox_fill, solidity = _component_geometry(obj)
        if pixels <= 0:
            continue
        adjacent = bool(np.any(obj & adjacent_zone))
        mean_p = float(np.mean(p25[obj]))
        mean_gain = float(np.mean(gain[obj]))
        mean_change = float(np.mean(multicue_change[obj]))
        mean_site = float(np.mean(site_likelihood[obj]))
        mean_similarity = float(np.mean(component_similarity[obj]))
        mean_sand = float(np.mean(terminal_sand[obj]))
        mean_confuser = float(np.mean(terminal_confuser[obj]))
        mean_proximity = float(np.mean(front_proximity[obj]))
        mean_remote = float(np.mean(remote_likelihood[obj]))
        mean_remote_change = float(np.mean(remote_change[obj]))
        mean_darkness = float(np.mean(absolute_darkness[obj]))
        mean_novelty = float(np.mean(terminal_novelty[obj]))
        mean_direct_change = float(np.mean(direct_change[obj]))
        mean_preexisting_risk = float(np.mean(preexisting_risk[obj]))
        mean_prior_dark_persistence = float(np.mean(prior_dark_persistence[obj]))
        mean_break_dominance = float(np.mean(break_dominance[obj]))
        mean_persistent_nonpv_risk = float(np.mean(persistent_nonpv_risk[obj]))
        remote_seed_fraction = float(np.count_nonzero(obj & remote_census_seed) / max(pixels, 1))
        remote_change_fraction = float(
            np.count_nonzero(
                obj
                & (remote_change >= float(settings.get("terminal_remote_census_object_change_pixel", 0.055)))
            )
            / max(pixels, 1)
        )
        remote_dark_fraction = float(
            np.count_nonzero(
                obj
                & (absolute_darkness >= float(settings.get("terminal_remote_census_object_dark_pixel", 0.38)))
            )
            / max(pixels, 1)
        )
        remote_novelty_fraction = float(
            np.count_nonzero(
                obj
                & (terminal_novelty >= float(settings.get("terminal_remote_census_object_novelty_pixel", 0.38)))
            )
            / max(pixels, 1)
        )
        remote_direct_change_fraction = float(
            np.count_nonzero(
                obj
                & (direct_change >= float(settings.get("terminal_remote_census_object_direct_change_pixel", 0.22)))
            )
            / max(pixels, 1)
        )
        boundary = obj & ~ndimage.binary_erosion(obj, structure=np.ones((3, 3), dtype=bool))
        contact_ratio = float(np.count_nonzero(boundary & reference_contact) / max(int(boundary.sum()), 1))
        has_perimeter_seed = bool(np.any(obj & perimeter_seed))
        has_gain_seed = bool(np.any(obj & (adjacent_seed | very_strong)))
        has_legacy_remote_seed = bool(np.any(obj & legacy_remote_seed))
        has_remote_census_seed = bool(np.any(obj & remote_census_seed))
        annular_displacement = adjacent and _is_annular_boundary_displacement(
            obj,
            inventory_labels,
            inventory_count,
            distance_to_inventory,
            settings,
        )

        keep_gain = adjacent and (not annular_displacement) and has_gain_seed and (
            pixels >= int(settings.get("terminal_adjacent_minimum_pixels", 4))
            and mean_p >= float(settings.get("terminal_adjacent_mean_probability", 0.48))
            and mean_gain >= float(settings.get("terminal_adjacent_mean_gain", 0.04))
            and solidity >= float(settings.get("terminal_adjacent_minimum_solidity", 0.30))
            and mean_sand <= float(settings.get("terminal_adjacent_maximum_sand", 0.66))
        )
        keep_perimeter = adjacent and (not annular_displacement) and has_perimeter_seed and (
            pixels >= int(settings.get("terminal_perimeter_minimum_pixels", 6))
            and mean_site >= float(settings.get("terminal_perimeter_mean_likelihood", 0.49))
            and mean_similarity >= float(settings.get("terminal_perimeter_mean_similarity", 0.50))
            and mean_sand <= float(settings.get("terminal_perimeter_object_maximum_sand", 0.60))
            and mean_confuser <= float(settings.get("terminal_perimeter_object_maximum_confuser", 0.60))
            and solidity >= float(settings.get("terminal_perimeter_minimum_solidity", 0.24))
            and mean_proximity >= float(settings.get("terminal_perimeter_minimum_proximity", 0.20))
            and contact_ratio >= float(settings.get("terminal_perimeter_minimum_contact_ratio", 0.015))
            and (
                mean_change >= float(settings.get("terminal_perimeter_mean_multicue_change", 0.10))
                or mean_site >= float(settings.get("terminal_perimeter_object_strong_likelihood", 0.62))
            )
        )
        keep_legacy_remote = (not adjacent) and has_legacy_remote_seed and (
            pixels >= int(settings.get("terminal_remote_minimum_pixels", 14))
            and mean_p >= float(settings.get("terminal_remote_mean_probability", 0.62))
            and mean_gain >= float(settings.get("terminal_remote_mean_gain", 0.095))
            and mean_novelty >= float(settings.get("terminal_remote_minimum_novelty", 0.34))
            and mean_preexisting_risk <= float(settings.get("terminal_remote_maximum_preexisting", 0.48))
            and mean_persistent_nonpv_risk <= float(settings.get("terminal_remote_maximum_persistent_risk", 0.52))
            and bbox_fill >= float(settings.get("terminal_remote_minimum_bbox_fill", 0.07))
            and solidity >= float(settings.get("terminal_remote_minimum_solidity", 0.34))
            and mean_sand <= float(settings.get("terminal_remote_maximum_sand", 0.55))
            and mean_confuser <= float(settings.get("terminal_remote_maximum_confuser", 0.55))
        )
        keep_remote_census = (not adjacent) and has_remote_census_seed and (
            pixels >= int(settings.get("terminal_remote_census_minimum_pixels", 18))
            and mean_remote >= float(settings.get("terminal_remote_census_mean_likelihood", 0.39))
            and mean_remote_change >= float(settings.get("terminal_remote_census_mean_change", 0.065))
            and mean_darkness >= float(settings.get("terminal_remote_census_mean_darkness", 0.34))
            and mean_novelty >= float(settings.get("terminal_remote_census_mean_novelty", 0.34))
            and mean_direct_change >= float(settings.get("terminal_remote_census_mean_direct_change", 0.20))
            and mean_break_dominance >= float(settings.get("terminal_remote_census_mean_break_dominance", 0.28))
            and mean_preexisting_risk <= float(settings.get("terminal_remote_census_maximum_preexisting_risk", 0.42))
            and mean_prior_dark_persistence <= float(settings.get("terminal_remote_census_maximum_prior_dark_persistence", 0.38))
            and mean_persistent_nonpv_risk <= float(settings.get("terminal_remote_census_maximum_persistent_nonpv_risk", 0.44))
            and remote_seed_fraction >= float(settings.get("terminal_remote_census_minimum_seed_fraction", 0.025))
            and remote_change_fraction >= float(settings.get("terminal_remote_census_minimum_change_fraction", 0.16))
            and remote_dark_fraction >= float(settings.get("terminal_remote_census_minimum_dark_fraction", 0.32))
            and remote_novelty_fraction >= float(settings.get("terminal_remote_census_minimum_novelty_fraction", 0.18))
            and remote_direct_change_fraction >= float(settings.get("terminal_remote_census_minimum_direct_change_fraction", 0.16))
            and bbox_fill >= float(settings.get("terminal_remote_census_minimum_bbox_fill", 0.045))
            and solidity >= float(settings.get("terminal_remote_census_minimum_solidity", 0.22))
            and mean_sand <= float(settings.get("terminal_remote_census_object_maximum_sand", 0.64))
            and mean_confuser <= float(settings.get("terminal_remote_census_object_maximum_confuser", 0.62))
        )

        if keep_gain:
            keep, src = True, 1
        elif keep_perimeter:
            keep, src = True, 3
        elif keep_legacy_remote:
            keep, src = True, 2
        elif keep_remote_census:
            keep, src = True, 4
        else:
            keep, src = False, 0
        if keep:
            accepted |= obj
            source[obj] = src
            rejection_reason = ""
        elif annular_displacement:
            rejection_reason = "annular_boundary_displacement"
        elif mean_preexisting_risk > float(settings.get("terminal_remote_census_maximum_preexisting_risk", 0.42)):
            rejection_reason = "preexisting_state_risk"
        elif mean_prior_dark_persistence > float(settings.get("terminal_remote_census_maximum_prior_dark_persistence", 0.38)):
            rejection_reason = "persistent_prior_dark_state"
        elif mean_persistent_nonpv_risk > float(settings.get("terminal_remote_census_maximum_persistent_nonpv_risk", 0.44)):
            rejection_reason = "persistent_nonpv_risk"
        elif mean_novelty < float(settings.get("terminal_remote_census_mean_novelty", 0.34)):
            rejection_reason = "insufficient_terminal_novelty"
        elif mean_direct_change < float(settings.get("terminal_remote_census_mean_direct_change", 0.20)):
            rejection_reason = "insufficient_direct_2024_2025_change"
        elif mean_break_dominance < float(settings.get("terminal_remote_census_mean_break_dominance", 0.28)):
            rejection_reason = "terminal_break_not_dominant"
        else:
            rejection_reason = "object_geometry_or_support_contract"
        decisions.append(
            ComponentDecision(
                pixels=pixels,
                bbox_fill=bbox_fill,
                solidity=solidity,
                mean_probability=mean_p,
                mean_gain=mean_gain,
                adjacent_to_reference=adjacent,
                accepted=bool(keep),
                source=src,
                mean_novelty=mean_novelty,
                mean_direct_change=mean_direct_change,
                mean_preexisting_risk=mean_preexisting_risk,
                mean_prior_dark_persistence=mean_prior_dark_persistence,
                mean_break_dominance=mean_break_dominance,
                mean_persistent_nonpv_risk=mean_persistent_nonpv_risk,
                novelty_fraction=remote_novelty_fraction,
                direct_change_fraction=remote_direct_change_fraction,
                rejection_reason=rejection_reason,
            )
        )

    gain_out[accepted] = gain[accepted]
    return accepted, source, gain_out, decisions


def reconstruct_historical_outside_tracks(
    annual_candidate: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
    *,
    terminal_source_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-threshold historical tracks outside the audited 2024 inventory."""
    p = np.asarray(probability, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    n = p.shape[0]
    outside = ~np.asarray(inventory, dtype=bool)
    smooth = _safe_temporal_median(
        p,
        v,
        int(settings.get("outside_temporal_median_radius", 1)),
    )
    seed_threshold = float(settings.get("outside_seed_probability", 0.56))
    extent_threshold = float(settings.get("outside_extent_probability", 0.40))
    min_pixels = int(settings.get("outside_minimum_component_pixels", 8))

    expanded = np.zeros_like(annual_candidate, dtype=bool)
    for t in range(reference_index):
        seed = annual_candidate[t] & outside & v[t] & (p[t] >= seed_threshold)
        support = outside & v[t] & (
            (annual_candidate[t] & (p[t] >= extent_threshold))
            | (smooth[t] >= max(extent_threshold + 0.06, 0.46))
        )
        grown = _propagate(seed, support)
        labels, count = ndimage.label(grown, structure=np.ones((3, 3), dtype=bool))
        for label_id in range(1, count + 1):
            obj = labels == label_id
            if int(obj.sum()) >= min_pixels:
                expanded[t] |= obj

    persistent = np.zeros_like(expanded, dtype=bool)
    for t in range(reference_index):
        left = expanded[t - 1] if t > 0 else np.zeros(inventory.shape, dtype=bool)
        right = expanded[t + 1] if t + 1 < reference_index else np.zeros(inventory.shape, dtype=bool)
        overlap_left = ndimage.binary_dilation(left, iterations=1)
        overlap_right = ndimage.binary_dilation(right, iterations=1)
        persistent[t] = expanded[t] & (overlap_left | overlap_right)
    for t in range(1, max(reference_index - 1, 1)):
        bridge = (
            ~persistent[t]
            & ndimage.binary_dilation(persistent[t - 1], iterations=1)
            & ndimage.binary_dilation(persistent[t + 1], iterations=1)
            & (smooth[t] >= extent_threshold)
        )
        persistent[t] |= bridge

    if terminal_source_override is not None:
        terminal_source = np.asarray(terminal_source_override, dtype=np.uint8)
        terminal = terminal_source > 0
    else:
        terminal, terminal_source, _terminal_gain, _ = detect_terminal_year_expansion(
            annual_candidate,
            probability,
            valid,
            inventory,
            reference_index,
            settings,
        )
    if reference_index + 1 < n:
        persistent[reference_index + 1] = terminal

    onset = np.full(inventory.shape, n, dtype=np.int16)
    retirement = np.full(inventory.shape, n, dtype=np.int16)
    for t in range(n):
        first = persistent[t] & (onset == n)
        onset[first] = t
    for t in range(reference_index):
        future = np.any(persistent[t + 1 : reference_index], axis=0)
        ended = persistent[t] & ~future
        retirement[ended] = np.minimum(retirement[ended], t + 1)
    return persistent, onset, retirement, terminal_source
