from __future__ import annotations

"""Two-stage 2024-reference temporal reconstruction.

The 2024 manual inventory has a simple semantic contract: every raster cell with
positive overlap belongs to a photovoltaic parcel in 2024, apart from the small
boundary corrections already audited in Stage 7.  Historical geometry is not
copied backwards.  Instead, annual imagery is analysed twice:

1. a full-domain detector provides the ordinary annual PV/change trajectory;
2. an independent, parcel-internal detector compares each 2011--2025 image with
   the spectral state observed inside the same 2024 parcel and reconstructs the
   progressive 2011--2024 build-out.

The two trajectories are fused only after both have been calculated.  This
separation prevents a weak segmentation probability from erasing a real
mixed-pixel parcel, while still requiring annual Landsat evidence before any
pre-2024 part of the 2024 geometry is activated.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .mixed_parcel import distributed_support_metrics

EPS = 1e-6


@dataclass(frozen=True)
class DualStageReferenceResult:
    internal_active: np.ndarray
    internal_evidence: np.ndarray
    spectral_similarity: np.ndarray
    parcel_presence: np.ndarray
    parcel_wide_presence: np.ndarray
    component_onset_index: np.ndarray


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def _spectral_features(reflectance: np.ndarray) -> np.ndarray:
    """Return radiometrically stable Landsat features [T, C, H, W]."""
    x = np.asarray(reflectance, dtype=np.float32)
    if x.ndim != 4 or x.shape[1] < 6:
        raise ValueError("reflectance must have shape [T, 6, H, W]")
    blue, green, red, nir, swir1, swir2 = [x[:, i] for i in range(6)]
    total = np.maximum(blue + green + red + nir + swir1 + swir2, EPS)
    normalised = [band / total for band in (blue, green, red, nir, swir1, swir2)]
    ndvi = _safe_ratio(nir, red)
    ndmi = _safe_ratio(nir, swir1)
    ndbi = _safe_ratio(swir1, nir)
    mndwi = _safe_ratio(green, swir1)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)
    brightness = (
        0.3029 * blue
        + 0.2786 * green
        + 0.4733 * red
        + 0.5599 * nir
        + 0.5080 * swir1
        + 0.1872 * swir2
    )
    return np.stack([*normalised, ndvi, ndmi, ndbi, mndwi, bsi, brightness], axis=1).astype(np.float32)


def _component_reference_similarity(
    reflectance: np.ndarray,
    reflectance_valid: np.ndarray,
    component: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> np.ndarray:
    """Compare every annual pixel with its own parcel's 2024 spectral state.

    Local background centring reduces year-to-year atmospheric and illumination
    shifts.  The metric therefore responds to the panel-versus-surrounding-land
    contrast, rather than merely to absolute darkness.
    """
    features = _spectral_features(reflectance)
    n, channels, height, width = features.shape
    similarity = np.zeros((n, height, width), dtype=np.float32)
    comp = np.asarray(component, dtype=bool)
    valid = np.asarray(reflectance_valid, dtype=bool)
    if not np.any(comp):
        return similarity

    ring_pixels = max(2, int(settings.get("internal_reference_context_ring_pixels", 6)))
    inner_exclusion = max(1, int(settings.get("internal_reference_context_inner_pixels", 1)))
    outer = ndimage.binary_dilation(
        comp,
        structure=np.ones((3, 3), dtype=bool),
        iterations=ring_pixels,
    )
    inner = ndimage.binary_dilation(
        comp,
        structure=np.ones((3, 3), dtype=bool),
        iterations=inner_exclusion,
    )
    ring = outer & ~inner

    centred = np.full_like(features, np.nan, dtype=np.float32)
    minimum_context = max(8, int(settings.get("internal_reference_minimum_context_pixels", 20)))
    for t in range(n):
        context = ring & valid[t]
        if int(np.count_nonzero(context)) >= minimum_context:
            context_median = np.nanmedian(features[t][:, context], axis=1)
        else:
            domain = valid[t] & ~comp
            if int(np.count_nonzero(domain)) >= minimum_context:
                context_median = np.nanmedian(features[t][:, domain], axis=1)
            else:
                context_median = np.zeros(channels, dtype=np.float32)
        centred[t] = features[t] - context_median[:, None, None]

    ref_domain = comp & valid[reference_index]
    if int(np.count_nonzero(ref_domain)) == 0:
        return similarity
    ref_values = centred[reference_index][:, ref_domain]
    ref_median = np.nanmedian(ref_values, axis=1).astype(np.float32)
    q25 = np.nanquantile(ref_values, 0.25, axis=1)
    q75 = np.nanquantile(ref_values, 0.75, axis=1)
    scale = np.maximum((q75 - q25).astype(np.float32), 1e-4)
    # Fixed floors stop a spectrally uniform reference parcel from producing an
    # unrealistically brittle distance metric.
    scale_floor = np.asarray(
        settings.get(
            "internal_reference_feature_scale_floors",
            [0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.07, 0.07, 0.07, 0.07, 0.08, 0.045],
        ),
        dtype=np.float32,
    )
    if scale_floor.size != channels:
        scale_floor = np.full(channels, float(np.nanmedian(scale_floor)) if scale_floor.size else 0.05, dtype=np.float32)
    scale = np.maximum(scale, scale_floor)

    distance_scale = max(float(settings.get("internal_reference_similarity_distance_scale", 1.15)), 0.1)
    for t in range(n):
        domain = comp & valid[t]
        if not np.any(domain):
            continue
        z = np.abs((centred[t] - ref_median[:, None, None]) / scale[:, None, None])
        distance = np.nanmedian(z, axis=0)
        annual_similarity = np.exp(-distance / distance_scale).astype(np.float32)
        annual_similarity[~domain] = 0.0
        similarity[t] = annual_similarity
    return np.clip(similarity, 0.0, 1.0)


def _weighted_evidence(
    raw_probability: np.ndarray,
    global_probability: np.ndarray,
    annual_candidate: np.ndarray,
    spectral_similarity: np.ndarray,
    valid: np.ndarray,
    *,
    anchor_similarity: np.ndarray | None,
    candidate_score: np.ndarray | None,
    settings: dict[str, Any],
) -> np.ndarray:
    cues: list[tuple[np.ndarray, float]] = [
        (np.asarray(raw_probability, dtype=np.float32), float(settings.get("internal_evidence_raw_probability_weight", 0.20))),
        (np.asarray(global_probability, dtype=np.float32), float(settings.get("internal_evidence_global_probability_weight", 0.10))),
        (np.asarray(spectral_similarity, dtype=np.float32), float(settings.get("internal_evidence_spectral_similarity_weight", 0.46))),
        (np.asarray(annual_candidate, dtype=np.float32), float(settings.get("internal_evidence_candidate_mask_weight", 0.04))),
    ]
    if anchor_similarity is not None:
        cues.append((np.asarray(anchor_similarity, dtype=np.float32), float(settings.get("internal_evidence_anchor_similarity_weight", 0.12))))
    if candidate_score is not None:
        cues.append((np.asarray(candidate_score, dtype=np.float32), float(settings.get("internal_evidence_candidate_score_weight", 0.08))))
    numerator = np.zeros_like(raw_probability, dtype=np.float32)
    denominator = np.zeros_like(raw_probability, dtype=np.float32)
    for cue, weight in cues:
        if weight <= 0:
            continue
        cue_valid = np.isfinite(cue)
        numerator += np.where(cue_valid, np.clip(cue, 0.0, 1.0) * weight, 0.0).astype(np.float32)
        denominator += cue_valid.astype(np.float32) * weight
    evidence = np.divide(numerator, np.maximum(denominator, EPS), where=denominator > 0)
    evidence[~np.asarray(valid, dtype=bool)] = 0.0
    radius = max(0, int(settings.get("internal_evidence_spatial_median_radius", 1)))
    if radius > 0:
        size = 2 * radius + 1
        for t in range(evidence.shape[0]):
            filtered = ndimage.median_filter(evidence[t], size=size, mode="nearest")
            evidence[t] = np.where(valid[t], 0.65 * evidence[t] + 0.35 * filtered, 0.0)
    return np.clip(evidence, 0.0, 1.0).astype(np.float32)


def _persistent_state(raw: np.ndarray, reference_index: int, settings: dict[str, Any]) -> np.ndarray:
    """Stabilise a historical state while allowing 2024 as endpoint support only."""
    raw = np.asarray(raw, dtype=bool).copy()
    if raw.size != reference_index:
        raise ValueError("raw state length must equal reference_index")
    for t in range(1, max(reference_index - 1, 1)):
        if not raw[t] and raw[t - 1] and raw[t + 1]:
            raw[t] = True
    window = max(2, int(settings.get("internal_presence_support_window_years", 3)))
    minimum = max(1, int(settings.get("internal_presence_minimum_support_years", 2)))
    stable = np.zeros_like(raw)
    for t in range(reference_index):
        hi = min(reference_index, t + window)
        count = int(np.count_nonzero(raw[t:hi]))
        required = min(minimum, max(1, hi - t))
        # The audited 2024 endpoint may supply the second persistence observation
        # for 2023, but it never supplies the annual spectral evidence itself.
        endpoint_help = t == reference_index - 1 and raw[t]
        stable[t] = bool(raw[t] and (count >= required or endpoint_help))
    return stable


def _earliest_supported_onset(
    stable: np.ndarray,
    strength: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> int:
    if not np.any(stable):
        return reference_index
    lookback = max(1, int(settings.get("internal_onset_baseline_years", 2)))
    minimum_gain = float(settings.get("internal_onset_minimum_strength_gain", 0.045))
    strong_override = float(settings.get("internal_onset_strong_state_override", 0.64))
    for t in np.flatnonzero(stable):
        t = int(t)
        if t == 0:
            return 0
        lo = max(0, t - lookback)
        baseline = float(np.median(strength[lo:t])) if t > lo else 0.0
        gain = float(strength[t] - baseline)
        if gain >= minimum_gain or float(strength[t]) >= strong_override:
            return t
    return int(np.flatnonzero(stable)[0])


def reconstruct_internal_reference_buildout(
    raw_probability: np.ndarray,
    global_probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    annual_candidate: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
    *,
    reflectance: np.ndarray,
    reflectance_valid: np.ndarray,
    anchor_similarity: np.ndarray | None = None,
    candidate_score: np.ndarray | None = None,
) -> DualStageReferenceResult:
    """Reconstruct 2011--2024 build-out independently inside the 2024 SHP."""
    raw_p = np.asarray(raw_probability, dtype=np.float32)
    global_p = np.asarray(global_probability, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    inventory = np.asarray(inventory, dtype=bool)
    annual_candidate = np.asarray(annual_candidate, dtype=bool)
    n = raw_p.shape[0]
    if not (raw_p.shape == global_p.shape == valid.shape == annual_candidate.shape):
        raise ValueError("annual evidence arrays must have identical shapes")
    if reflectance.shape[0] != n or reflectance.shape[2:] != inventory.shape:
        raise ValueError("reflectance cube does not match annual evidence")

    spectral_similarity = np.zeros_like(raw_p, dtype=np.float32)
    internal_active = np.zeros_like(valid, dtype=bool)
    parcel_presence = np.zeros((n, *inventory.shape), dtype=bool)
    parcel_wide_presence = np.zeros_like(parcel_presence)
    component_onset = np.full(inventory.shape, reference_index, dtype=np.int16)

    labels, count = ndimage.label(inventory, structure=np.ones((3, 3), dtype=bool))
    for label_id in range(1, count + 1):
        component = labels == label_id
        pixels = int(np.count_nonzero(component))
        if pixels == 0:
            continue
        component_similarity = _component_reference_similarity(
            reflectance,
            reflectance_valid,
            component,
            reference_index,
            settings,
        )
        spectral_similarity[:, component] = component_similarity[:, component]

    evidence = _weighted_evidence(
        raw_p,
        global_p,
        annual_candidate,
        spectral_similarity,
        valid,
        anchor_similarity=anchor_similarity,
        candidate_score=candidate_score,
        settings=settings,
    )

    for label_id in range(1, count + 1):
        component = labels == label_id
        pixels = int(np.count_nonzero(component))
        if pixels == 0:
            continue
        ref_domain = component & valid[reference_index]
        ref_values = evidence[reference_index][ref_domain]
        if ref_values.size:
            q25, q50, q75 = np.quantile(ref_values, [0.25, 0.50, 0.75])
        else:
            q25, q50, q75 = 0.24, 0.34, 0.46
        weak = float(np.clip(
            max(float(settings.get("internal_dynamic_weak_minimum", 0.14)), q25 * float(settings.get("internal_dynamic_weak_reference_ratio", 0.62))),
            0.0,
            float(settings.get("internal_dynamic_weak_maximum", 0.38)),
        ))
        moderate = float(np.clip(
            max(float(settings.get("internal_dynamic_moderate_minimum", 0.20)), q50 * float(settings.get("internal_dynamic_moderate_reference_ratio", 0.70))),
            weak + 0.02,
            float(settings.get("internal_dynamic_moderate_maximum", 0.50)),
        ))
        strong = float(np.clip(
            max(float(settings.get("internal_dynamic_strong_minimum", 0.30)), q75 * float(settings.get("internal_dynamic_strong_reference_ratio", 0.76))),
            moderate + 0.03,
            float(settings.get("internal_dynamic_strong_maximum", 0.68)),
        ))

        object_raw = np.zeros(reference_index, dtype=bool)
        wide_raw = np.zeros(reference_index, dtype=bool)
        strength = np.zeros(reference_index, dtype=np.float32)
        yearly_metrics = []
        for t in range(reference_index):
            domain = component & valid[t]
            weak_support = domain & ((evidence[t] >= weak) | annual_candidate[t])
            moderate_support = domain & (evidence[t] >= moderate)
            strong_support = domain & (evidence[t] >= strong)
            metrics = distributed_support_metrics(
                component,
                weak_support,
                moderate_support,
                strong_support,
                valid[t],
                evidence[t],
                dilation_pixels=max(1, int(settings.get("internal_support_dilation_pixels", 2))),
                grid_size=max(2, int(settings.get("internal_support_grid_size", 3))),
                sector_minimum_fraction=float(settings.get("internal_sector_minimum_fraction", 0.035)),
            )
            yearly_metrics.append(metrics)
            strength[t] = float(np.clip(
                0.20 * metrics.supported_fraction
                + 0.10 * metrics.moderate_fraction
                + 0.17 * metrics.dilated_coverage
                + 0.13 * metrics.sector_coverage
                + 0.09 * metrics.boundary_supported_fraction
                + 0.09 * metrics.interior_supported_fraction
                + 0.10 * metrics.median_evidence
                + 0.12 * metrics.upper_quantile_evidence,
                0.0,
                1.0,
            ))
            minimum_seed_pixels = max(1, min(
                pixels,
                max(
                    int(settings.get("internal_minimum_strong_pixels", 2)),
                    int(np.ceil(pixels * float(settings.get("internal_minimum_strong_fraction", 0.004)))),
                ),
            ))
            has_seed = int(np.count_nonzero(strong_support)) >= minimum_seed_pixels
            spectral_values = component_similarity[t][domain]
            spectral_median = float(np.median(spectral_values)) if spectral_values.size else 0.0
            spectral_upper = float(np.quantile(spectral_values, 0.75)) if spectral_values.size else 0.0
            spectral_supported = bool(
                spectral_median >= float(settings.get("internal_object_minimum_spectral_similarity", 0.62))
                or spectral_upper >= float(settings.get("internal_object_minimum_upper_spectral_similarity", 0.70))
            )
            object_raw[t] = bool(
                metrics.valid_fraction >= float(settings.get("internal_minimum_valid_fraction", 0.24))
                and (
                    (
                        spectral_supported
                        and metrics.supported_fraction >= float(settings.get("internal_object_minimum_supported_fraction", 0.055))
                        and metrics.dilated_coverage >= float(settings.get("internal_object_minimum_dilated_coverage", 0.30))
                        and metrics.upper_quantile_evidence >= max(weak, float(settings.get("internal_object_minimum_upper_evidence", 0.24)))
                    )
                    or (
                        has_seed
                        and spectral_upper >= float(settings.get("internal_seed_minimum_spectral_similarity", 0.54))
                    )
                )
            )
            wide_raw[t] = bool(
                object_raw[t]
                and metrics.supported_fraction >= float(settings.get("internal_wide_minimum_supported_fraction", 0.28))
                and metrics.dilated_coverage >= float(settings.get("internal_wide_minimum_dilated_coverage", 0.75))
                and metrics.sector_coverage >= float(settings.get("internal_wide_minimum_sector_coverage", 0.65))
                and metrics.interior_supported_fraction >= float(settings.get("internal_wide_minimum_interior_support", 0.24))
                and (
                    metrics.boundary_supported_fraction >= float(settings.get("internal_wide_minimum_boundary_support", 0.18))
                    or pixels <= int(settings.get("internal_small_component_pixels", 20))
                )
                and spectral_median >= float(settings.get("internal_wide_minimum_spectral_similarity", 0.66))
                and metrics.median_evidence >= max(
                    float(settings.get("internal_wide_minimum_median_evidence", 0.15)),
                    weak * float(settings.get("internal_wide_median_to_weak_ratio", 0.78)),
                )
            )

        stable_object = _persistent_state(object_raw, reference_index, settings)
        stable_wide = _persistent_state(wide_raw, reference_index, settings)
        onset = _earliest_supported_onset(stable_object, strength, reference_index, settings)
        wide_onset = _earliest_supported_onset(stable_wide, strength, reference_index, settings)
        if wide_onset < onset:
            onset = wide_onset
        component_onset[component] = np.int16(onset)

        previous = np.zeros(component.shape, dtype=bool)
        for t in range(reference_index):
            if t < onset:
                continue
            parcel_presence[t] |= component if stable_object[t] else False
            if wide_onset < reference_index and t >= wide_onset:
                current = component.copy()
                parcel_wide_presence[t] |= component
            else:
                domain = component & valid[t]
                support = domain & ((evidence[t] >= weak) | annual_candidate[t])
                seed = domain & ((evidence[t] >= strong) | annual_candidate[t])
                minimum_seed_pixels = max(1, min(
                    pixels,
                    max(
                        int(settings.get("internal_minimum_strong_pixels", 2)),
                        int(np.ceil(pixels * float(settings.get("internal_minimum_strong_fraction", 0.004)))),
                    ),
                ))
                if int(np.count_nonzero(seed)) < minimum_seed_pixels and stable_object[t] and np.any(domain):
                    flat = np.flatnonzero(domain)
                    order = np.argsort(evidence[t].ravel()[flat])[::-1]
                    chosen = flat[order[:minimum_seed_pixels]]
                    if chosen.size and float(evidence[t].ravel()[chosen[0]]) >= weak:
                        seed = seed.copy()
                        seed.ravel()[chosen] = True
                current = previous.copy()
                if np.any(seed):
                    current |= ndimage.binary_propagation(
                        seed | previous,
                        structure=np.ones((3, 3), dtype=bool),
                        mask=(support | previous | seed) & component,
                    )
                if stable_object[t] and np.any(current):
                    close_iterations = max(0, int(settings.get("internal_partial_closing_iterations", 1)))
                    if close_iterations:
                        closed = ndimage.binary_closing(
                            current,
                            structure=np.ones((3, 3), dtype=bool),
                            iterations=close_iterations,
                        )
                        current |= closed & support & component
            current |= previous
            internal_active[t] |= current & component
            previous = current & component

        internal_active[reference_index] |= component
        parcel_presence[reference_index] |= component
        parcel_wide_presence[reference_index] |= component
        for t in range(reference_index + 1, n):
            internal_active[t] |= component
            parcel_presence[t] |= component
            parcel_wide_presence[t] |= component

    return DualStageReferenceResult(
        internal_active=internal_active,
        internal_evidence=evidence,
        spectral_similarity=spectral_similarity,
        parcel_presence=parcel_presence,
        parcel_wide_presence=parcel_wide_presence,
        component_onset_index=component_onset,
    )


def fuse_reference_tracks(
    global_reference_active: np.ndarray,
    internal_result: DualStageReferenceResult,
    inventory: np.ndarray,
    annual_candidate: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the independent full-domain and SHP-internal trajectories.

    Source codes: 0 none; 1 global detector only; 2 internal detector only;
    3 agreement; 4 authoritative 2024 endpoint / retained post-reference parcel.
    """
    global_active = np.asarray(global_reference_active, dtype=bool)
    internal_active = np.asarray(internal_result.internal_active, dtype=bool)
    inventory = np.asarray(inventory, dtype=bool)
    candidate = np.asarray(annual_candidate, dtype=bool)
    evidence = np.asarray(internal_result.internal_evidence, dtype=np.float32)
    fused = np.zeros_like(global_active, dtype=bool)
    source = np.zeros_like(global_active, dtype=np.uint8)
    agreement_floor = float(settings.get("dual_stage_global_agreement_evidence_floor", 0.16))
    previous = np.zeros(inventory.shape, dtype=bool)
    for t in range(reference_index):
        global_supported = global_active[t] & inventory & (
            (evidence[t] >= agreement_floor) | candidate[t] | internal_active[t]
        )
        inside = internal_active[t] | global_supported | previous
        both = global_supported & internal_active[t]
        source[t][global_supported & ~internal_active[t]] = 1
        source[t][internal_active[t] & ~global_supported] = 2
        source[t][both] = 3
        fused[t] = inside
        previous = inside
    fused[reference_index] = inventory
    source[reference_index][inventory] = 4
    for t in range(reference_index + 1, fused.shape[0]):
        fused[t] = inventory
        source[t][inventory] = 4
    return fused, source
