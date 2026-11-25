from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

EPS = 1e-6
ENDMEMBER_NAMES = ("HA", "LA", "VG", "SH")
BAND_NAMES = ("BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2")


@dataclass(frozen=True)
class EndmemberBundle:
    matrix: np.ndarray  # [6, 4], columns HA/LA/VG/SH
    source: str
    condition_number: float
    minimum_spectral_angle_deg: float
    anchor_pixels: int
    background_pixels: int

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "condition_number": float(self.condition_number),
            "minimum_spectral_angle_deg": float(self.minimum_spectral_angle_deg),
            "anchor_pixels": int(self.anchor_pixels),
            "background_pixels": int(self.background_pixels),
        }
        for j, name in enumerate(ENDMEMBER_NAMES):
            for i, band in enumerate(BAND_NAMES):
                payload[f"{name}_{band}"] = float(self.matrix[i, j])
        return payload


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def _brightness(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    count = finite.sum(axis=0)
    total = np.where(finite, x, 0.0).sum(axis=0)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _sample_rows(values: np.ndarray, mask: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if values.ndim != 3 or values.shape[0] != 6:
        raise ValueError("values must be [6, H, W]")
    flat = values.reshape(6, -1).T
    ids = np.flatnonzero(mask.reshape(-1) & np.all(np.isfinite(flat), axis=1))
    if ids.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if ids.size > maximum:
        ids = rng.choice(ids, maximum, replace=False)
    return flat[ids].astype(np.float32, copy=False)


def _robust_center(rows: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return fallback.astype(np.float32)
    lo = np.nanquantile(rows, 0.05, axis=0)
    hi = np.nanquantile(rows, 0.95, axis=0)
    clipped = np.clip(rows, lo, hi)
    return np.nanmedian(clipped, axis=0).astype(np.float32)


def _spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _bundle_quality(matrix: np.ndarray) -> tuple[float, float]:
    condition = float(np.linalg.cond(matrix)) if np.all(np.isfinite(matrix)) else float("inf")
    angles = [
        _spectral_angle_deg(matrix[:, i], matrix[:, j])
        for i in range(matrix.shape[1])
        for j in range(i + 1, matrix.shape[1])
    ]
    return condition, float(min(angles)) if angles else 0.0


def _default_endmembers() -> np.ndarray:
    # Conservative Landsat reflectance prototypes. They are used only when a
    # unit lacks enough valid anchor/background pixels. Columns are HA, LA, VG, SH.
    return np.asarray(
        [
            [0.34, 0.055, 0.045, 0.010],
            [0.38, 0.060, 0.075, 0.010],
            [0.42, 0.055, 0.045, 0.010],
            [0.44, 0.060, 0.420, 0.010],
            [0.39, 0.070, 0.180, 0.010],
            [0.32, 0.065, 0.100, 0.010],
        ],
        dtype=np.float32,
    )


def derive_endmembers(
    reflectance: np.ndarray,
    anchor_fraction: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> EndmemberBundle:
    """Derive unit-adaptive HA/LA/VG/SH endmembers from the reference-year image.

    The validated SHP is treated as complete-reference evidence. LA is estimated from
    confirmed anchor pixels. HA, VG and SH are derived from spectrally extreme
    background candidates that are sufficiently far from the anchor. No pixel
    outside the SHP is assumed to be PV-negative merely because it is outside.
    """
    if reflectance.shape[0] != 6:
        raise ValueError("Expected six Landsat bands")
    valid = np.all(np.isfinite(reflectance), axis=0)
    strong = anchor_fraction >= float(cfg["supervision"]["strong_positive_fraction"])
    background = (anchor_fraction <= 0) & valid
    rng = np.random.default_rng(seed)
    maximum = int(cfg.get("spectral_mixture", {}).get("maximum_endmember_samples", 120000))

    blue, green, red, nir, swir1, swir2 = reflectance
    ndvi = _safe_ratio(nir, red)
    bright = _brightness(reflectance)
    shadow_score = bright + 0.25 * np.nan_to_num(ndvi, nan=0.0)

    default = _default_endmembers()
    anchor_rows = _sample_rows(reflectance, strong & valid, maximum, rng)
    background_rows = _sample_rows(reflectance, background, maximum, rng)

    if anchor_rows.size:
        anchor_brightness = np.nanmean(anchor_rows, axis=1)
        low = np.nanquantile(anchor_brightness, 0.15)
        high = np.nanquantile(anchor_brightness, 0.85)
        la_rows = anchor_rows[(anchor_brightness >= low) & (anchor_brightness <= high)]
        if len(la_rows) < 20:
            la_rows = anchor_rows
    else:
        la_rows = np.empty((0, 6), dtype=np.float32)

    if np.any(background):
        bg_ndvi = ndvi[background]
        bg_brightness = bright[background]
        vg_cut = float(np.nanquantile(bg_ndvi, 0.985)) if bg_ndvi.size else 0.3
        ha_cut = float(np.nanquantile(bg_brightness, 0.985)) if bg_brightness.size else 0.4
        sh_cut = float(np.nanquantile(shadow_score[background], 0.02)) if bg_ndvi.size else 0.03
        vg_mask = background & (ndvi >= vg_cut)
        ha_mask = background & (bright >= ha_cut) & (ndvi < max(vg_cut, 0.25))
        sh_mask = background & (shadow_score <= sh_cut)
    else:
        vg_mask = ha_mask = sh_mask = np.zeros_like(valid)

    vg_rows = _sample_rows(reflectance, vg_mask, maximum // 4, rng)
    ha_rows = _sample_rows(reflectance, ha_mask, maximum // 4, rng)
    sh_rows = _sample_rows(reflectance, sh_mask, maximum // 4, rng)

    ha = _robust_center(ha_rows, default[:, 0])
    la = _robust_center(la_rows, default[:, 1])
    vg = _robust_center(vg_rows, default[:, 2])
    sh = _robust_center(sh_rows, default[:, 3])

    matrix = np.column_stack([ha, la, vg, sh]).astype(np.float32)
    matrix = np.clip(matrix, 0.0, 1.2)
    condition, angle = _bundle_quality(matrix)
    source = "unit_adaptive"
    min_anchor = int(cfg.get("spectral_mixture", {}).get("minimum_anchor_pixels", 20))
    min_background = int(cfg.get("spectral_mixture", {}).get("minimum_background_pixels", 100))
    if len(anchor_rows) < min_anchor or len(background_rows) < min_background or not np.isfinite(condition):
        # Preserve any reliable unit-derived endmember and replace unstable columns.
        fallback = default.copy()
        if len(anchor_rows) >= min_anchor:
            fallback[:, 1] = la
        if len(vg_rows) >= 20:
            fallback[:, 2] = vg
        if len(ha_rows) >= 20:
            fallback[:, 0] = ha
        if len(sh_rows) >= 20:
            fallback[:, 3] = sh
        matrix = fallback
        condition, angle = _bundle_quality(matrix)
        source = "hybrid_with_default"

    return EndmemberBundle(
        matrix=matrix,
        source=source,
        condition_number=condition,
        minimum_spectral_angle_deg=angle,
        anchor_pixels=int(len(anchor_rows)),
        background_pixels=int(len(background_rows)),
    )


def unmix_lsma(reflectance: np.ndarray, endmembers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast non-negative sum-to-one approximate LSMA.

    A least-squares solution is projected to the simplex. This is substantially
    faster than solving NNLS for every Landsat pixel and is stable for tiled CPU
    processing. Fractions are returned in HA/LA/VG/SH order.
    """
    if reflectance.ndim != 3 or reflectance.shape[0] != 6:
        raise ValueError("reflectance must have shape [6, H, W]")
    if endmembers.shape != (6, 4):
        raise ValueError("endmembers must have shape [6, 4]")
    h, w = reflectance.shape[1:]
    pixels = reflectance.reshape(6, -1)
    valid = np.all(np.isfinite(pixels), axis=0)
    fractions = np.full((4, pixels.shape[1]), np.nan, dtype=np.float32)
    rmse = np.full(pixels.shape[1], np.nan, dtype=np.float32)
    if np.any(valid):
        e_aug = np.vstack([endmembers.astype(np.float64), np.ones((1, 4), dtype=np.float64) * 0.25])
        x_aug = np.vstack([pixels[:, valid].astype(np.float64), np.ones((1, int(valid.sum())), dtype=np.float64) * 0.25])
        coef = np.linalg.pinv(e_aug) @ x_aug
        coef = np.maximum(coef, 0.0)
        denom = np.sum(coef, axis=0)
        zero = denom <= EPS
        coef[:, ~zero] /= denom[~zero]
        if np.any(zero):
            coef[:, zero] = 0.25
        reconstructed = endmembers.astype(np.float64) @ coef
        residual = np.sqrt(np.mean((pixels[:, valid] - reconstructed) ** 2, axis=0))
        fractions[:, valid] = coef.astype(np.float32)
        rmse[valid] = residual.astype(np.float32)
    return fractions.reshape(4, h, w), rmse.reshape(h, w)


def pavg_fraction(
    vg: np.ndarray,
    la: np.ndarray,
    sh: np.ndarray,
    baseline_la: np.ndarray | float,
    baseline_sh: np.ndarray | float,
    maximum_denominator_correction: float = 0.85,
) -> np.ndarray:
    """Photovoltaics-adjusted vegetation fraction following Xia et al. (2022).

    PAVG = VG / (1 - ΔSH - ΔLA), with physically safe clipping. Positive changes
    in LA/SH relative to the pre-deployment baseline are interpreted as panel and
    panel-shadow obstruction. Negative changes do not inflate the denominator.
    """
    delta_la = np.maximum(la - baseline_la, 0.0)
    delta_sh = np.maximum(sh - baseline_sh, 0.0)
    correction = np.clip(delta_la + delta_sh, 0.0, maximum_denominator_correction)
    denominator = np.maximum(1.0 - correction, 0.15)
    out = vg / denominator
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_endmember_matrix(path: str | Path) -> np.ndarray:
    data = np.load(path)
    matrix = np.asarray(data["endmembers"], dtype=np.float32)
    if matrix.shape != (6, 4):
        raise RuntimeError(f"Invalid endmember matrix at {path}: {matrix.shape}")
    return matrix


def save_endmember_bundle(bundle: EndmemberBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, endmembers=bundle.matrix.astype(np.float32))
    metadata = path.with_suffix(".json")
    metadata.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")


def stream_reference_samples(
    raster_path: str | Path,
    fraction_path: str | Path,
    cfg: dict[str, Any],
    seed: int,
    maximum_side: int = 1800,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a bounded reference-year representation for endmember fitting."""
    with rasterio.open(raster_path) as src, rasterio.open(fraction_path) as frac_src:
        scale = max(1.0, max(src.height, src.width) / float(maximum_side))
        out_h = max(1, int(round(src.height / scale)))
        out_w = max(1, int(round(src.width / scale)))
        refl = src.read([1, 2, 3, 4, 5, 6], out_shape=(6, out_h, out_w), masked=True).filled(np.nan).astype(np.float32)
        fraction = frac_src.read(1, out_shape=(out_h, out_w), masked=True).filled(0).astype(np.float32)
    refl[refl <= -9990] = np.nan
    return refl, fraction


def summarize_fraction_cube(fractions: np.ndarray, rmse: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    names = ENDMEMBER_NAMES
    for i, name in enumerate(names):
        values = fractions[i][mask & np.isfinite(fractions[i])]
        result[f"{name}_median"] = float(np.nanmedian(values)) if values.size else float("nan")
        result[f"{name}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
    values = rmse[mask & np.isfinite(rmse)]
    result["LSMA_RMSE_median"] = float(np.nanmedian(values)) if values.size else float("nan")
    result["LSMA_RMSE_p90"] = float(np.nanquantile(values, 0.9)) if values.size else float("nan")
    return result
