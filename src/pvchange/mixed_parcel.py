from __future__ import annotations

"""Object-level support diagnostics for mixed-pixel photovoltaic parcels.

A 30 m Landsat pixel can contain panel rows, aisle gaps, bare soil and shadows at
once.  Consequently, a real PV parcel may show only moderate evidence in each
individual pixel even though the evidence is spatially distributed across the
whole parcel.  These helpers distinguish that pattern from a local partial
build-out, a single dark patch and a thin boundary ring.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DistributedSupportMetrics:
    component_pixels: int
    valid_pixels: int
    valid_fraction: float
    supported_fraction: float
    moderate_fraction: float
    strong_fraction: float
    dilated_coverage: float
    sector_coverage: float
    boundary_supported_fraction: float
    interior_supported_fraction: float
    median_evidence: float
    upper_quantile_evidence: float


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / max(int(denominator), 1))


def distributed_support_metrics(
    component: np.ndarray,
    weak_support: np.ndarray,
    moderate_support: np.ndarray | None = None,
    strong_support: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    evidence: np.ndarray | None = None,
    *,
    dilation_pixels: int = 2,
    grid_size: int = 3,
    sector_minimum_fraction: float = 0.05,
    upper_quantile: float = 0.75,
) -> DistributedSupportMetrics:
    """Summarise how broadly annual evidence occupies a reference component.

    The sector and dilation terms are intentionally spatial.  A concentrated
    high-confidence core cannot masquerade as parcel-wide mixed-pixel support,
    while repeated weak evidence spread across panel rows can.
    """
    comp = np.asarray(component, dtype=bool)
    valid_mask = np.ones(comp.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    domain = comp & valid_mask
    weak = np.asarray(weak_support, dtype=bool) & domain
    moderate = weak if moderate_support is None else np.asarray(moderate_support, dtype=bool) & domain
    strong = moderate if strong_support is None else np.asarray(strong_support, dtype=bool) & domain

    component_pixels = int(np.count_nonzero(comp))
    valid_pixels = int(np.count_nonzero(domain))
    if component_pixels == 0 or valid_pixels == 0:
        return DistributedSupportMetrics(
            component_pixels=component_pixels,
            valid_pixels=valid_pixels,
            valid_fraction=0.0,
            supported_fraction=0.0,
            moderate_fraction=0.0,
            strong_fraction=0.0,
            dilated_coverage=0.0,
            sector_coverage=0.0,
            boundary_supported_fraction=0.0,
            interior_supported_fraction=0.0,
            median_evidence=0.0,
            upper_quantile_evidence=0.0,
        )

    if np.any(weak):
        expanded = ndimage.binary_dilation(
            weak,
            structure=np.ones((3, 3), dtype=bool),
            iterations=max(0, int(dilation_pixels)),
        )
        dilated_coverage = _safe_fraction(np.count_nonzero(expanded & domain), valid_pixels)
    else:
        dilated_coverage = 0.0

    rr, cc = np.where(comp)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    row_edges = np.linspace(r0, r1, max(1, int(grid_size)) + 1, dtype=int)
    col_edges = np.linspace(c0, c1, max(1, int(grid_size)) + 1, dtype=int)
    occupied_sectors = 0
    supported_sectors = 0
    for ri in range(len(row_edges) - 1):
        for ci in range(len(col_edges) - 1):
            rs = slice(row_edges[ri], max(row_edges[ri + 1], row_edges[ri] + 1))
            cs = slice(col_edges[ci], max(col_edges[ci + 1], col_edges[ci] + 1))
            sector_domain = domain[rs, cs]
            sector_pixels = int(np.count_nonzero(sector_domain))
            if sector_pixels == 0:
                continue
            occupied_sectors += 1
            sector_supported = int(np.count_nonzero(weak[rs, cs] & sector_domain))
            required = max(1, int(np.ceil(sector_pixels * float(sector_minimum_fraction))))
            if sector_supported >= required:
                supported_sectors += 1
    sector_coverage = _safe_fraction(supported_sectors, occupied_sectors)

    boundary = comp & ~ndimage.binary_erosion(
        comp,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary_domain = boundary & valid_mask
    interior_domain = (comp & ~boundary) & valid_mask
    boundary_supported_fraction = _safe_fraction(
        np.count_nonzero(weak & boundary_domain),
        np.count_nonzero(boundary_domain),
    )
    interior_supported_fraction = _safe_fraction(
        np.count_nonzero(weak & interior_domain),
        np.count_nonzero(interior_domain),
    )
    if not np.any(interior_domain):
        interior_supported_fraction = _safe_fraction(np.count_nonzero(weak), valid_pixels)

    if evidence is None:
        evidence_values = weak[domain].astype(np.float32)
    else:
        ev = np.asarray(evidence, dtype=np.float32)
        evidence_values = ev[domain]
        evidence_values = evidence_values[np.isfinite(evidence_values)]
    if evidence_values.size:
        median_evidence = float(np.median(evidence_values))
        upper_quantile_evidence = float(np.quantile(evidence_values, float(upper_quantile)))
    else:
        median_evidence = 0.0
        upper_quantile_evidence = 0.0

    return DistributedSupportMetrics(
        component_pixels=component_pixels,
        valid_pixels=valid_pixels,
        valid_fraction=_safe_fraction(valid_pixels, component_pixels),
        supported_fraction=_safe_fraction(np.count_nonzero(weak), valid_pixels),
        moderate_fraction=_safe_fraction(np.count_nonzero(moderate), valid_pixels),
        strong_fraction=_safe_fraction(np.count_nonzero(strong), valid_pixels),
        dilated_coverage=dilated_coverage,
        sector_coverage=sector_coverage,
        boundary_supported_fraction=boundary_supported_fraction,
        interior_supported_fraction=interior_supported_fraction,
        median_evidence=median_evidence,
        upper_quantile_evidence=upper_quantile_evidence,
    )


def stabilise_presence_track(
    presence: np.ndarray,
    *,
    window: int = 3,
    minimum_years: int = 2,
    bridge_single_gap: bool = True,
) -> np.ndarray:
    """Require short-term persistence while allowing one cloudy intermediate year."""
    raw = np.asarray(presence, dtype=bool).copy()
    if bridge_single_gap and raw.size >= 3:
        for t in range(1, raw.size - 1):
            if not raw[t] and raw[t - 1] and raw[t + 1]:
                raw[t] = True
    stable = np.zeros_like(raw)
    width = max(1, int(window))
    minimum = max(1, int(minimum_years))
    for t in range(raw.size):
        hi = min(raw.size, t + width)
        required = min(minimum, hi - t)
        stable[t] = bool(raw[t] and np.count_nonzero(raw[t:hi]) >= required)
    return stable
