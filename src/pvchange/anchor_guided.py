from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import box, mapping

from .features import compute_features
from .spectral_mixture import load_endmember_matrix, stream_reference_samples


@dataclass(frozen=True)
class AnchorPrototype:
    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    anchor_similarity_floor: float
    background_similarity_ceiling: float
    anchor_pixels: int
    background_pixels: int


def _robust_scale(
    values: np.ndarray,
    fallback_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - center), axis=0)
    scale = 1.4826 * mad
    if fallback_values is not None and len(fallback_values):
        fallback_center = np.nanmedian(fallback_values, axis=0)
        fallback_mad = np.nanmedian(np.abs(fallback_values - fallback_center), axis=0)
        # A small mixed-pixel parcel can have almost no within-anchor variance.
        # Borrow a conservative fraction of the local background scale so its
        # prototype is not unrealistically narrow.
        scale = np.maximum(scale, 0.20 * 1.4826 * fallback_mad)
    scale = np.maximum(scale, 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def _similarity_from_matrix(matrix: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.abs((matrix - center[None, :]) / scale[None, :])
    z = np.minimum(z, 6.0)
    distance = np.sqrt(np.nanmean(z * z, axis=1))
    return np.exp(-0.5 * distance * distance).astype(np.float32)


def build_anchor_prototype(
    reference_raster: str | Path,
    fraction_raster: str | Path,
    endmember_path: str | Path,
    cfg: dict[str, Any],
) -> AnchorPrototype | None:
    """Build a local robust feature prototype from certain-positive 2024 anchors.

    The SHP is used only to define certain positive pixels. Background samples are
    explicit distant/low-evidence pixels from the supervision raster, never the
    complete outside-SHP domain.
    """
    reflectance, fraction = stream_reference_samples(
        reference_raster,
        fraction_raster,
        cfg,
        int(cfg["project"]["random_seed"]),
        maximum_side=int(cfg.get("anchor_guided", {}).get("prototype_maximum_side", 1800)),
    )
    endmembers = load_endmember_matrix(endmember_path)
    channels = list(cfg["features"]["channels"])
    static = compute_features(reflectance, channels, False, endmembers=endmembers)
    selected = tuple(
        name for name in cfg.get("anchor_guided", {}).get(
            "feature_names",
            ["PV_Spectral_Evidence", "LSMA_LA", "LSMA_SH", "LSMA_HA", "LSMA_VG", "NDVI", "NDBI", "BSI"],
        )
        if name in channels
    )
    if not selected:
        return None
    indices = [channels.index(name) for name in selected]
    matrix = np.moveaxis(static[indices], 0, -1).reshape(-1, len(indices))
    fraction_flat = fraction.reshape(-1)
    valid = np.isfinite(matrix).all(axis=1)
    settings = cfg.get("anchor_guided", {})
    strong = float(cfg["supervision"]["strong_positive_fraction"])
    anchor = valid & (fraction_flat >= strong)
    background = valid & (fraction_flat <= 0)
    minimum_anchor = int(settings.get("minimum_anchor_prototype_pixels", 20))
    minimum_background = int(settings.get("minimum_background_prototype_pixels", 100))

    # Mixed-pixel rescue: a small but certain 2024 parcel may contain fewer than
    # the formal strong-pixel minimum.  Use the highest-fraction positive cells
    # rather than abandoning spatial inference.  This fallback is still wholly
    # inside the supplied 2024 SHP and requires several valid positive pixels.
    if int(anchor.sum()) < minimum_anchor:
        soft_floor = float(settings.get("mixed_pixel_prototype_fraction_floor", 0.02))
        soft = valid & (fraction_flat >= soft_floor)
        minimum_soft = max(3, int(settings.get("minimum_mixed_pixel_prototype_pixels", 3)))
        if int(soft.sum()) < minimum_soft:
            return None
        soft_indices = np.flatnonzero(soft)
        order = np.argsort(fraction_flat[soft_indices])[::-1]
        maximum_soft = int(settings.get("maximum_mixed_pixel_prototype_pixels", 5000))
        selected_soft = soft_indices[order[:maximum_soft]]
        anchor = np.zeros_like(valid, dtype=bool)
        anchor[selected_soft] = True

    anchor_values = matrix[anchor]
    fallback_values = matrix[background] if np.any(background) else matrix[valid]
    center, scale = _robust_scale(anchor_values, fallback_values)
    anchor_similarity = _similarity_from_matrix(anchor_values, center, scale)
    if int(background.sum()) >= minimum_background:
        background_similarity = _similarity_from_matrix(matrix[background], center, scale)
        background_ceiling = float(np.nanquantile(background_similarity, 0.995))
    else:
        background_ceiling = 0.35
    anchor_floor = float(np.nanquantile(anchor_similarity, 0.05))
    return AnchorPrototype(
        feature_names=selected,
        center=center,
        scale=scale,
        anchor_similarity_floor=float(np.clip(anchor_floor, 0.05, 0.95)),
        background_similarity_ceiling=float(np.clip(background_ceiling, 0.0, 0.95)),
        anchor_pixels=int(anchor.sum()),
        background_pixels=int(background.sum()),
    )


def anchor_similarity(static_cube: np.ndarray, channels: list[str], prototype: AnchorPrototype) -> np.ndarray:
    indices = [channels.index(name) for name in prototype.feature_names]
    matrix = np.moveaxis(static_cube[indices], 0, -1).reshape(-1, len(indices))
    valid = np.isfinite(matrix).all(axis=1)
    output = np.zeros(matrix.shape[0], dtype=np.float32)
    if valid.any():
        output[valid] = _similarity_from_matrix(matrix[valid], prototype.center, prototype.scale)
    return output.reshape(static_cube.shape[1:])


def anchor_similarity_matrix(static_matrix: np.ndarray, channels: list[str], prototype: AnchorPrototype) -> np.ndarray:
    """Compute local anchor similarity for a pixel-by-feature static matrix."""
    indices = [channels.index(name) for name in prototype.feature_names]
    matrix = np.asarray(static_matrix, dtype=np.float32)[:, indices]
    valid = np.isfinite(matrix).all(axis=1)
    output = np.zeros(matrix.shape[0], dtype=np.float32)
    if valid.any():
        output[valid] = _similarity_from_matrix(matrix[valid], prototype.center, prototype.scale)
    return output


def anchor_guided_score(
    probability: np.ndarray,
    similarity: np.ndarray,
    uncertainty: np.ndarray | None,
    cfg: dict[str, Any],
) -> np.ndarray:
    settings = cfg.get("anchor_guided", {})
    wp = float(settings.get("probability_weight", 0.60))
    ws = float(settings.get("similarity_weight", 0.35))
    wu = float(settings.get("uncertainty_penalty_weight", 0.05))
    total = max(wp + ws, 1e-6)
    score = (wp * np.asarray(probability, dtype=np.float32) + ws * np.asarray(similarity, dtype=np.float32)) / total
    if uncertainty is not None:
        score -= wu * np.clip(np.asarray(uncertainty, dtype=np.float32), 0, 1)
    return np.clip(score, 0, 1).astype(np.float32)


def _component_filter(
    mask: np.ndarray,
    near_anchor: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    settings = cfg.get("anchor_guided", {})
    minimum = int(settings.get("minimum_component_pixels", 6))
    remote_minimum = int(settings.get("remote_minimum_component_pixels", 20))
    minimum_fill = float(settings.get("minimum_component_bbox_fill", 0.08))
    close_iterations = int(settings.get("binary_closing_iterations", 1))
    working = np.asarray(mask, dtype=bool)
    if close_iterations > 0:
        working = ndimage.binary_closing(working, iterations=close_iterations)
    labels, count = ndimage.label(working)
    if count == 0:
        return working
    keep = np.zeros(count + 1, dtype=bool)
    objects = ndimage.find_objects(labels)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        size = int(sizes[label_id])
        component = labels[slc] == label_id
        fill = float(size / max(component.size, 1))
        touches_anchor_zone = bool(np.any(near_anchor[slc] & component))
        required = minimum if touches_anchor_zone else remote_minimum
        if size >= required and (touches_anchor_zone or fill >= minimum_fill):
            keep[label_id] = True
    return keep[labels]


def classify_anchor_guided(
    probability: np.ndarray,
    similarity: np.ndarray,
    uncertainty: np.ndarray | None,
    near_anchor: np.ndarray,
    valid: np.ndarray,
    anchor_mask: np.ndarray | None,
    is_reference_year: bool,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return high-confidence mask, medium-confidence mask and score.

    Candidate pixels are accepted either near a confirmed anchor with moderate
    joint evidence, or remotely only with much stronger model and local-prototype
    evidence. The reference-year SHP anchor is forced positive only in 2024.
    """
    settings = cfg.get("anchor_guided", {})
    score = anchor_guided_score(probability, similarity, uncertainty, cfg)
    near_threshold = float(settings.get("near_anchor_score_threshold", 0.62))
    medium_threshold = float(settings.get("medium_score_threshold", 0.52))
    remote_threshold = float(settings.get("remote_score_threshold", 0.82))
    remote_probability = float(settings.get("remote_probability_threshold", 0.90))
    remote_similarity = float(settings.get("remote_similarity_threshold", 0.70))
    minimum_similarity = float(settings.get("minimum_similarity", 0.20))

    high = valid & (
        (near_anchor & (score >= near_threshold) & (similarity >= minimum_similarity))
        | ((~near_anchor) & (score >= remote_threshold) & (probability >= remote_probability) & (similarity >= remote_similarity))
    )
    medium = valid & near_anchor & (score >= medium_threshold) & ~high
    if is_reference_year and anchor_mask is not None:
        high |= valid & np.asarray(anchor_mask, dtype=bool)
        medium &= ~np.asarray(anchor_mask, dtype=bool)
    high = _component_filter(high, near_anchor, cfg) & valid
    medium = _component_filter(medium, near_anchor, cfg) & valid & ~high
    return high, medium, score


def select_anchor_geometry(
    shape: gpd.GeoDataFrame,
    region_id: int,
    crs: Any,
    bounds: rasterio.coords.BoundingBox,
    cfg: dict[str, Any],
):
    if shape.empty:
        return None
    projected = shape.to_crs(crs)
    id_field = str(cfg["supervision"].get("id_field", "Id"))
    subset = projected.iloc[0:0]
    if id_field in projected.columns:
        numeric = np.asarray(projected[id_field])
        try:
            subset = projected.loc[np.asarray(numeric, dtype=float) == float(region_id)]
        except Exception:
            subset = projected.loc[projected[id_field].astype(str) == str(region_id)]
    footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    if subset.empty:
        subset = projected.loc[projected.geometry.intersects(footprint)]
    if subset.empty:
        return None
    geometry = subset.geometry.intersection(footprint).union_all()
    return None if geometry.is_empty else geometry


def buffered_support_mask(
    geometry,
    transform,
    shape: tuple[int, int],
    cfg: dict[str, Any],
) -> np.ndarray:
    if geometry is None:
        return np.zeros(shape, dtype=bool)
    distance = float(cfg.get("anchor_guided", {}).get("search_radius_m", 5000.0))
    buffered = geometry.buffer(distance)
    return geometry_mask([mapping(buffered)], out_shape=shape, transform=transform, invert=True, all_touched=True)


def anchor_mask_for_window(
    fraction_src: rasterio.DatasetReader,
    window: Window,
    cfg: dict[str, Any],
) -> np.ndarray:
    fraction = fraction_src.read(1, window=window).astype(np.float32)
    return fraction >= float(cfg["supervision"]["strong_positive_fraction"])
