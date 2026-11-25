from __future__ import annotations

import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .features import compute_features
from .spectral_mixture import load_endmember_matrix
from .study_area import expected_counts, test_mode_enabled
from .utils import atomic_json, ensure_dir, save_table, utc_now

EPS = 1e-6

# Diagnostic decision classes written to object_class_30m.
CLASS_BACKGROUND = 0
CLASS_HIGH_CONFIDENCE_SEED = 1
CLASS_ADJACENCY_RESCUE = 2
CLASS_REMOTE_NEW_SITE = 3
CLASS_REFERENCE_ANCHOR = 4

# Rejection codes written to rejection_code_30m.
REJECT_NONE = 0
REJECT_LOW_EVIDENCE = 11
REJECT_SAND_TEXTURE = 12
REJECT_HOLLOW_BOUNDARY = 13
REJECT_IMPLAUSIBLE_GEOMETRY = 14
REJECT_WEAK_TEMPORAL_SUPPORT = 15
REJECT_PROTOTYPE_MISMATCH = 16
REJECT_TEMPORAL_SINGLETON = 17

FEATURE_CHANNELS = [
    "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
    "NDVI", "NDBI", "BSI", "MNDWI", "TC_Brightness",
    "PV_Fraction_Evidence", "Bare_Evidence",
]


@dataclass
class PrototypeBundle:
    positive_centers: np.ndarray
    positive_scales: np.ndarray
    sand_centers: np.ndarray
    sand_scales: np.ndarray
    positive_samples: int
    sand_samples: int
    feature_channels: list[str]


def _windows(width: int, height: int, size: int) -> Iterator[Window]:
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


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    arr = np.clip(np.asarray(x, dtype=np.float32), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _robust01(x: np.ndarray, valid: np.ndarray | None = None, qlo: float = 0.05, qhi: float = 0.95) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(arr)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    values = arr[mask]
    if values.size < 20:
        out = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.0)
        out[~mask] = 0.0
        return out.astype(np.float32)
    lo, hi = np.nanquantile(values, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + EPS:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + EPS:
        out = np.zeros(arr.shape, dtype=np.float32)
        out[mask] = 0.5
        return out
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _read_reflectance(src: rasterio.DatasetReader, window: Window) -> np.ndarray:
    refl = src.read([1, 2, 3, 4, 5, 6], window=window).astype(np.float32)
    refl[refl <= -9990] = np.nan
    return refl


def _feature_cube(reflectance: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    return compute_features(reflectance, FEATURE_CHANNELS, False, endmembers=endmembers).astype(np.float32)


def _sample_rows(cube: np.ndarray, mask: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if maximum <= 0:
        return np.empty((0, cube.shape[0]), dtype=np.float32)
    flat = cube.reshape(cube.shape[0], -1).T
    ids = np.flatnonzero(mask.reshape(-1) & np.all(np.isfinite(flat), axis=1))
    if ids.size == 0:
        return np.empty((0, cube.shape[0]), dtype=np.float32)
    if ids.size > maximum:
        ids = rng.choice(ids, maximum, replace=False)
    return flat[ids].astype(np.float32, copy=False)


def _fit_prototypes(samples: np.ndarray, maximum_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 2 or samples.shape[0] == 0:
        return np.zeros((1, len(FEATURE_CHANNELS)), dtype=np.float32), np.ones((1, len(FEATURE_CHANNELS)), dtype=np.float32)
    global_center = np.nanmedian(samples, axis=0)
    global_scale = np.maximum(1.4826 * np.nanmedian(np.abs(samples - global_center), axis=0), 0.015)
    whitened = np.clip((samples - global_center) / global_scale, -8.0, 8.0)
    n_clusters = min(maximum_clusters, max(1, samples.shape[0] // 500))
    if n_clusters <= 1 or samples.shape[0] < 80:
        centers = global_center[None, :]
        scales = global_scale[None, :]
        return centers.astype(np.float32), scales.astype(np.float32)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, max(256, samples.shape[0])),
        n_init=5,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(whitened)
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for k in range(n_clusters):
        part = samples[labels == k]
        if len(part) < 20:
            continue
        center = np.nanmedian(part, axis=0)
        scale = np.maximum(1.4826 * np.nanmedian(np.abs(part - center), axis=0), 0.015)
        centers.append(center)
        scales.append(scale)
    if not centers:
        centers = [global_center]
        scales = [global_scale]
    return np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def _prototype_similarity(cube: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    flat = cube.reshape(cube.shape[0], -1).T
    valid = np.all(np.isfinite(flat), axis=1)
    best = np.zeros(flat.shape[0], dtype=np.float32)
    if np.any(valid):
        values = flat[valid]
        for center, scale in zip(centers, scales):
            z2 = np.square((values - center) / np.maximum(scale, 0.015))
            distance = np.mean(np.clip(z2, 0.0, 16.0), axis=1)
            similarity = np.exp(-0.5 * distance).astype(np.float32)
            best[valid] = np.maximum(best[valid], similarity)
    return best.reshape(cube.shape[1:])


def _build_unit_prototypes(
    cfg: dict[str, Any],
    region_id: int,
    reference_row: Any,
    reference_landsat: Path,
    fraction_path: Path,
    label_path: Path,
    support_path: Path,
    endmembers: np.ndarray,
    output_path: Path,
) -> PrototypeBundle:
    settings = cfg.get("object_graph_refinement", {})
    maximum_samples = int(settings.get("maximum_prototype_samples", 120000))
    tile_size = int(settings.get("prototype_tile_size", 512))
    rng = np.random.default_rng(int(cfg.get("project", {}).get("random_seed", 2027)) + region_id * 31)
    positive_parts: list[np.ndarray] = []
    sand_parts: list[np.ndarray] = []
    positive_remaining = maximum_samples
    sand_remaining = maximum_samples

    raw_path = Path(str(getattr(reference_row, "raw_probability_path", "") or reference_row.calibrated_probability_path))
    refined_path = Path(str(reference_row.calibrated_probability_path))
    uncertainty_path = Path(str(reference_row.epistemic_std_path))
    strong_fraction = float(cfg.get("supervision", {}).get("strong_positive_fraction", 0.7))

    with rasterio.open(raw_path) as raw_src, rasterio.open(refined_path) as refined_src, rasterio.open(uncertainty_path) as uncertainty_src, rasterio.open(reference_landsat) as land_src, rasterio.open(fraction_path) as fraction_src, rasterio.open(label_path) as label_src, rasterio.open(support_path) as support_src:
        for window in _windows(raw_src.width, raw_src.height, tile_size):
            if positive_remaining <= 0 and sand_remaining <= 0:
                break
            raw = raw_src.read(1, window=window).astype(np.float32)
            refined = refined_src.read(1, window=window).astype(np.float32)
            uncertainty = uncertainty_src.read(1, window=window).astype(np.float32)
            refl = _read_reflectance(land_src, window)
            valid = np.isfinite(raw) & (raw > -9990) & np.isfinite(refl).all(axis=0)
            if not np.any(valid):
                continue
            cube = _feature_cube(refl, endmembers)
            fraction = fraction_src.read(1, window=window).astype(np.float32)
            label = label_src.read(1, window=window)
            support = support_src.read(1, window=window) >= 0.5
            frac_ev = cube[FEATURE_CHANNELS.index("PV_Fraction_Evidence")]
            bare = cube[FEATURE_CHANNELS.index("Bare_Evidence")]

            anchor = valid & (fraction >= strong_fraction)
            pseudo_positive = valid & ~anchor & (raw >= 0.88) & (refined >= 0.72) & (frac_ev >= 0.55) & (uncertainty <= 0.28) & (bare <= 0.72)
            positive = anchor | pseudo_positive

            hard_negative = valid & (label == 0)
            pseudo_sand = valid & ~support & (bare >= 0.55) & (frac_ev <= 0.36) & (raw <= 0.35) & (refined <= 0.42)
            sand = hard_negative | pseudo_sand

            if positive_remaining > 0:
                part = _sample_rows(cube, positive, min(positive_remaining, 8000), rng)
                if len(part):
                    positive_parts.append(part)
                    positive_remaining -= len(part)
            if sand_remaining > 0:
                part = _sample_rows(cube, sand, min(sand_remaining, 8000), rng)
                if len(part):
                    sand_parts.append(part)
                    sand_remaining -= len(part)

    positive_samples = np.concatenate(positive_parts, axis=0) if positive_parts else np.empty((0, len(FEATURE_CHANNELS)), dtype=np.float32)
    sand_samples = np.concatenate(sand_parts, axis=0) if sand_parts else np.empty((0, len(FEATURE_CHANNELS)), dtype=np.float32)
    if len(positive_samples) < 20:
        raise RuntimeError(f"Id {region_id}: insufficient confirmed/pseudo-positive pixels for anchor prototype ({len(positive_samples)})")
    positive_centers, positive_scales = _fit_prototypes(positive_samples, int(settings.get("positive_prototype_clusters", 3)), region_id + 17)
    sand_centers, sand_scales = _fit_prototypes(sand_samples, int(settings.get("sand_prototype_clusters", 3)), region_id + 71)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        positive_centers=positive_centers,
        positive_scales=positive_scales,
        sand_centers=sand_centers,
        sand_scales=sand_scales,
        feature_channels=np.asarray(FEATURE_CHANNELS),
    )
    output_path.with_suffix(".json").write_text(json.dumps({
        "Id": region_id,
        "positive_samples": int(len(positive_samples)),
        "sand_samples": int(len(sand_samples)),
        "positive_clusters": int(len(positive_centers)),
        "sand_clusters": int(len(sand_centers)),
        "feature_channels": FEATURE_CHANNELS,
        "semantics": "2024 positive anchor plus ultra-high-confidence pseudo-positive spectra; hard negatives plus high-confidence sand/background spectra",
    }, indent=2), encoding="utf-8")
    return PrototypeBundle(
        positive_centers=positive_centers,
        positive_scales=positive_scales,
        sand_centers=sand_centers,
        sand_scales=sand_scales,
        positive_samples=int(len(positive_samples)),
        sand_samples=int(len(sand_samples)),
        feature_channels=list(FEATURE_CHANNELS),
    )


def _texture_features(reflectance: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    brightness = np.nanmean(reflectance[:6], axis=0).astype(np.float32)
    filled = np.where(valid, brightness, np.nanmedian(brightness[valid]) if np.any(valid) else 0.0)
    smooth = ndimage.gaussian_filter(filled, sigma=1.0, mode="nearest")
    gy = ndimage.sobel(smooth, axis=0, mode="nearest") / 8.0
    gx = ndimage.sobel(smooth, axis=1, mode="nearest") / 8.0
    edge = np.sqrt(gx * gx + gy * gy)
    jxx = ndimage.gaussian_filter(gx * gx, sigma=2.0, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, sigma=2.0, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, sigma=2.0, mode="nearest")
    root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    l1 = 0.5 * (jxx + jyy + root)
    l2 = 0.5 * (jxx + jyy - root)
    coherence = np.clip((l1 - l2) / (l1 + l2 + EPS), 0.0, 1.0).astype(np.float32)
    mean5 = ndimage.uniform_filter(filled, size=5, mode="nearest")
    mean_sq5 = ndimage.uniform_filter(filled * filled, size=5, mode="nearest")
    local_std = np.sqrt(np.maximum(mean_sq5 - mean5 * mean5, 0.0))
    outer_mean = ndimage.uniform_filter(filled, size=13, mode="nearest")
    relative_darkness = _robust01(outer_mean - filled, valid, 0.10, 0.98)
    edge01 = _robust01(edge, valid, 0.10, 0.98)
    texture01 = _robust01(local_std, valid, 0.10, 0.98)
    for arr in (brightness, coherence, relative_darkness, edge01, texture01):
        arr[~valid] = 0.0
    return brightness, coherence, relative_darkness, np.clip(0.55 * edge01 + 0.45 * texture01, 0.0, 1.0).astype(np.float32)


def _temporal_evidence(
    raw_probability: np.ndarray,
    refined_probability: np.ndarray,
    fraction: np.ndarray,
    brightness: np.ndarray,
    positive_similarity: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = raw_probability.shape[0]
    onset = np.zeros_like(raw_probability, dtype=np.float32)
    future_persistence = np.zeros_like(raw_probability, dtype=np.float32)
    stable_presence = np.zeros_like(raw_probability, dtype=np.float32)
    joint = np.clip(0.58 * raw_probability + 0.42 * refined_probability, 0.0, 1.0)
    for t in range(n):
        pre0 = max(0, t - 4)
        post1 = min(n, t + 3)
        pre_joint = np.nanmedian(np.where(valid[pre0:t], joint[pre0:t], np.nan), axis=0) if t > 0 else np.zeros(joint.shape[1:], dtype=np.float32)
        post_joint = np.nanmedian(np.where(valid[t:post1], joint[t:post1], np.nan), axis=0)
        pre_fraction = np.nanmedian(np.where(valid[pre0:t], fraction[pre0:t], np.nan), axis=0) if t > 0 else np.zeros(joint.shape[1:], dtype=np.float32)
        post_fraction = np.nanmedian(np.where(valid[t:post1], fraction[t:post1], np.nan), axis=0)
        pre_brightness = np.nanmedian(np.where(valid[pre0:t], brightness[pre0:t], np.nan), axis=0) if t > 0 else brightness[t]
        post_brightness = np.nanmedian(np.where(valid[t:post1], brightness[t:post1], np.nan), axis=0)
        gain = _sigmoid((np.nan_to_num(post_joint) - np.nan_to_num(pre_joint)) / 0.09)
        fraction_gain = _sigmoid((np.nan_to_num(post_fraction) - np.nan_to_num(pre_fraction)) / 0.10)
        darkening = _sigmoid((np.nan_to_num(pre_brightness) - np.nan_to_num(post_brightness)) / 0.035)
        valid_count = np.maximum(valid[t:post1].sum(axis=0), 1)
        persistence = ((joint[t:post1] >= 0.56) & valid[t:post1]).sum(axis=0) / valid_count
        future_persistence[t] = persistence.astype(np.float32)
        onset[t] = np.clip(0.42 * gain + 0.22 * fraction_gain + 0.16 * darkening + 0.20 * persistence, 0.0, 1.0)
        lo = max(0, t - 1)
        hi = min(n, t + 2)
        stable_presence[t] = np.clip(
            0.45 * np.nanmedian(np.where(valid[lo:hi], joint[lo:hi], np.nan), axis=0)
            + 0.35 * np.nanmedian(np.where(valid[lo:hi], positive_similarity[lo:hi], np.nan), axis=0)
            + 0.20 * persistence,
            0.0,
            1.0,
        )
    return np.nan_to_num(onset).astype(np.float32), np.nan_to_num(future_persistence).astype(np.float32), np.nan_to_num(stable_presence).astype(np.float32)


def _oriented_fill(component: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(component)
    if len(coords) < 3:
        return 1.0, 1.0
    centered = coords - coords.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    try:
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        projected = centered @ vectors[:, order]
        spans = np.ptp(projected, axis=0) + 1.0
        area = max(float(spans[0] * spans[1]), 1.0)
        elongation = float(max(spans) / max(min(spans), 1.0))
        return float(len(coords) / area), elongation
    except np.linalg.LinAlgError:
        return 0.0, 999.0


def _component_boundary_contrast(component: np.ndarray, brightness: np.ndarray) -> float:
    inner = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool), iterations=1)
    if not np.any(inner):
        inner = component
    outer = ndimage.binary_dilation(component, structure=np.ones((3, 3), dtype=bool), iterations=2) & ~component
    if not np.any(outer):
        return 0.0
    return float(np.nanmedian(brightness[outer]) - np.nanmedian(brightness[inner]))


def _filter_objects(
    candidate: np.ndarray,
    seed: np.ndarray,
    adjacency: np.ndarray,
    remote: np.ndarray,
    anchor: np.ndarray,
    score: np.ndarray,
    positive_similarity: np.ndarray,
    sand_score: np.ndarray,
    onset: np.ndarray,
    persistence: np.ndarray,
    stable_presence: np.ndarray,
    brightness: np.ndarray,
    settings: dict[str, Any],
    year: int,
    region_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    closed = ndimage.binary_closing(candidate, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, count = ndimage.label(closed)
    accepted = np.zeros(candidate.shape, dtype=bool)
    classes = np.zeros(candidate.shape, dtype=np.uint8)
    rejection = np.zeros(candidate.shape, dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    min_adjacent = int(settings.get("minimum_adjacent_component_pixels", 4))
    min_remote = int(settings.get("minimum_remote_component_pixels", 12))

    for object_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        obj = labels[slc] == object_id
        size = int(obj.sum())
        rows_n = int(slc[0].stop - slc[0].start)
        cols_n = int(slc[1].stop - slc[1].start)
        bbox_fill = float(size / max(rows_n * cols_n, 1))
        filled = ndimage.binary_fill_holes(obj)
        solid_fill = float(size / max(int(filled.sum()), 1))
        oriented_fill, elongation = _oriented_fill(obj)
        anchor_hit = bool(np.any(anchor[slc] & obj))
        seed_hit = bool(np.any(seed[slc] & obj))
        adjacency_hit = bool(np.any(adjacency[slc] & obj))
        remote_hit = bool(np.any(remote[slc] & obj))
        mean_score = float(np.mean(score[slc][obj]))
        q75_score = float(np.quantile(score[slc][obj], 0.75))
        mean_positive = float(np.mean(positive_similarity[slc][obj]))
        mean_sand = float(np.mean(sand_score[slc][obj]))
        mean_onset = float(np.mean(onset[slc][obj]))
        mean_persistence = float(np.mean(persistence[slc][obj]))
        mean_stable = float(np.mean(stable_presence[slc][obj]))
        full_obj = np.zeros(candidate.shape, dtype=bool)
        full_obj[slc] = obj
        boundary_contrast = _component_boundary_contrast(full_obj, brightness)

        decision_class = CLASS_BACKGROUND
        reason = "accepted"
        accepted_object = False
        if anchor_hit:
            accepted_object = True
            decision_class = CLASS_REFERENCE_ANCHOR
        elif seed_hit or adjacency_hit:
            accepted_object = (
                size >= min_adjacent
                and mean_score >= float(settings.get("adjacent_minimum_mean_score", 0.50))
                and mean_positive >= float(settings.get("adjacent_minimum_positive_similarity", 0.34))
                and mean_sand <= float(settings.get("adjacent_maximum_sand_score", 0.64))
                and solid_fill >= float(settings.get("adjacent_minimum_solid_fill", 0.48))
                and elongation <= float(settings.get("adjacent_maximum_elongation", 28.0))
            )
            decision_class = CLASS_HIGH_CONFIDENCE_SEED if seed_hit else CLASS_ADJACENCY_RESCUE
        elif remote_hit:
            accepted_object = (
                size >= min_remote
                and mean_score >= float(settings.get("remote_minimum_mean_score", 0.64))
                and q75_score >= float(settings.get("remote_minimum_q75_score", 0.68))
                and mean_positive >= float(settings.get("remote_minimum_positive_similarity", 0.48))
                and mean_sand <= float(settings.get("remote_maximum_sand_score", 0.46))
                and mean_persistence >= float(settings.get("remote_minimum_persistence", 0.55))
                and (mean_onset >= float(settings.get("remote_minimum_onset", 0.48)) or mean_stable >= float(settings.get("remote_preexisting_minimum_stability", 0.76)))
                and solid_fill >= float(settings.get("remote_minimum_solid_fill", 0.70))
                and oriented_fill >= float(settings.get("remote_minimum_oriented_fill", 0.24))
                and elongation <= float(settings.get("remote_maximum_elongation", 18.0))
                and boundary_contrast >= float(settings.get("remote_minimum_boundary_contrast", -0.015))
            )
            decision_class = CLASS_REMOTE_NEW_SITE

        if not accepted_object:
            if mean_sand > float(settings.get("sand_rejection_threshold", 0.54)) and mean_onset < 0.65:
                reject_code = REJECT_SAND_TEXTURE
                reason = "sand_texture"
            elif solid_fill < float(settings.get("minimum_solid_fill", 0.55)):
                reject_code = REJECT_HOLLOW_BOUNDARY
                reason = "hollow_boundary"
            elif oriented_fill < float(settings.get("minimum_oriented_fill", 0.16)) or elongation > float(settings.get("maximum_object_elongation", 30.0)):
                reject_code = REJECT_IMPLAUSIBLE_GEOMETRY
                reason = "implausible_geometry"
            elif mean_persistence < float(settings.get("minimum_object_persistence", 0.38)) and mean_onset < 0.62:
                reject_code = REJECT_WEAK_TEMPORAL_SUPPORT
                reason = "weak_temporal_support"
            elif mean_positive < float(settings.get("minimum_positive_similarity", 0.30)):
                reject_code = REJECT_PROTOTYPE_MISMATCH
                reason = "prototype_mismatch"
            else:
                reject_code = REJECT_LOW_EVIDENCE
                reason = "low_evidence"
            view = rejection[slc]
            view[obj] = reject_code
            rejection[slc] = view
        else:
            accepted[slc] |= obj
            class_view = classes[slc]
            class_view[obj] = decision_class
            classes[slc] = class_view

        rows.append({
            "Id": region_id,
            "year": year,
            "tile_object_id": object_id,
            "pixels": size,
            "bbox_fill": bbox_fill,
            "solid_fill": solid_fill,
            "oriented_fill": oriented_fill,
            "elongation": elongation,
            "boundary_contrast": boundary_contrast,
            "mean_score": mean_score,
            "q75_score": q75_score,
            "mean_positive_similarity": mean_positive,
            "mean_sand_score": mean_sand,
            "mean_onset": mean_onset,
            "mean_persistence": mean_persistence,
            "mean_stable_presence": mean_stable,
            "anchor_hit": anchor_hit,
            "seed_hit": seed_hit,
            "adjacency_hit": adjacency_hit,
            "remote_hit": remote_hit,
            "accepted": accepted_object,
            "decision_class": decision_class,
            "reason": reason,
        })
    return accepted, classes, rejection, rows


def _temporal_object_graph(
    masks: np.ndarray,
    classes: np.ndarray,
    rejection: np.ndarray,
    growable: np.ndarray,
    scores: np.ndarray,
    positive_similarity: np.ndarray,
    sand_score: np.ndarray,
    onset: np.ndarray,
    stable_presence: np.ndarray,
    anchor_by_year: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = masks.shape[0]
    structure = np.ones((3, 3), dtype=bool)
    # Rescue omitted edge/adjacent pixels when the same object is present in a neighbouring year.
    for t in range(n):
        neighbour = np.zeros_like(masks[t])
        if t > 0:
            neighbour |= masks[t - 1]
        if t + 1 < n:
            neighbour |= masks[t + 1]
        if not np.any(neighbour):
            continue
        neighbour = ndimage.binary_dilation(neighbour, structure=structure, iterations=int(settings.get("temporal_graph_link_radius_pixels", 2)))
        rescue = (
            growable[t]
            & neighbour
            & (scores[t] >= float(settings.get("temporal_rescue_minimum_score", 0.50)))
            & (positive_similarity[t] >= float(settings.get("temporal_rescue_minimum_positive_similarity", 0.35)))
            & (sand_score[t] <= float(settings.get("temporal_rescue_maximum_sand_score", 0.56)))
        )
        new_pixels = rescue & ~masks[t]
        masks[t] |= rescue
        classes[t][new_pixels] = CLASS_ADJACENCY_RESCUE
        rejection[t][new_pixels] = REJECT_NONE

    # Remove one-year isolated objects unless they have a very strong physical onset or are anchored.
    for t in range(n):
        neighbour = np.zeros_like(masks[t])
        if t > 0:
            neighbour |= ndimage.binary_dilation(masks[t - 1], structure=structure, iterations=2)
        if t + 1 < n:
            neighbour |= ndimage.binary_dilation(masks[t + 1], structure=structure, iterations=2)
        labels, count = ndimage.label(masks[t])
        for object_id in range(1, count + 1):
            obj = labels == object_id
            if not np.any(obj):
                continue
            if np.any(anchor_by_year[t] & obj):
                continue
            linked = bool(np.any(neighbour & obj))
            strong_onset = float(np.mean(onset[t][obj])) >= float(settings.get("singleton_keep_minimum_onset", 0.72))
            stable = float(np.mean(stable_presence[t][obj])) >= float(settings.get("singleton_keep_minimum_stability", 0.78))
            if not linked and not strong_onset and not stable:
                masks[t][obj] = False
                classes[t][obj] = CLASS_BACKGROUND
                rejection[t][obj] = REJECT_TEMPORAL_SINGLETON
    return masks, classes, rejection


def refine_object_graph_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("object_graph_refinement", {})
    if not bool(settings.get("enabled", True)):
        return pd.read_csv(Path(cfg["paths"]["output_root"]) / "07_probability_calibration" / "calibrated_inference_registry.csv")

    root = Path(cfg["paths"]["output_root"])
    canonical_registry_path = root / "07_probability_calibration" / "calibrated_inference_registry.csv"
    source_registry = pd.read_csv(canonical_registry_path).sort_values(["Id", "year"])
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")

    out = ensure_dir(root / "07_object_graph_refinement")
    probability_root = ensure_dir(out / "object_graph_probability_30m")
    mask_root = ensure_dir(out / "object_graph_masks_30m")
    class_root = ensure_dir(out / "object_class_30m")
    rejection_root = ensure_dir(out / "rejection_code_30m")
    positive_similarity_root = ensure_dir(out / "positive_similarity_30m")
    sand_score_root = ensure_dir(out / "sand_texture_score_30m")
    prototype_root = ensure_dir(out / "unit_prototypes")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    registry_root = ensure_dir(out / "unit_registries")
    object_table_root = ensure_dir(out / "object_tables")

    years = np.arange(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1)
    reference_year = int(cfg["study"]["reference_year"])
    tile_size = int(settings.get("tile_size", 256))
    halo = int(settings.get("halo_pixels", 64))
    mask_nodata = int(cfg.get("calibration", {}).get("mask_nodata", 255))
    diagnostics_units = {int(x) for x in settings.get("diagnostic_unit_ids", [1, 6, 28])}
    write_diagnostics = bool(settings.get("write_diagnostics", False)) or test_mode_enabled(cfg)
    all_unit_registries: list[pd.DataFrame] = []
    unit_summaries: list[dict[str, Any]] = []

    for region_id, group in tqdm(
        list(source_registry.groupby("Id")),
        desc="Anchor-prototype object graph refinement",
        unit="region",
        disable=not cfg.get("processing", {}).get("show_progress", True),
    ):
        region_id = int(region_id)
        group = group.sort_values("year")
        aligned_group = aligned.loc[aligned["Id"].eq(region_id)].sort_values("year")
        if len(group) != len(years) or len(aligned_group) != len(years):
            raise RuntimeError(f"Id {region_id}: object graph refinement requires {len(years)} annual records")
        if group["inference_mode"].astype(str).eq("area_only_fallback").all():
            passthrough = group.copy()
            passthrough["object_graph_refinement"] = "not_applicable_area_only"
            all_unit_registries.append(passthrough)
            continue

        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        unit_registry_path = registry_root / f"unit_{region_id:03d}.csv"
        if checkpoint.exists() and unit_registry_path.exists() and not cfg.get("processing", {}).get("overwrite", False):
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing = pd.read_csv(unit_registry_path)
                if payload.get("status") == "passed" and len(existing) == len(years) and all(Path(p).exists() for p in existing["calibrated_probability_path"].astype(str)):
                    all_unit_registries.append(existing)
                    unit_summaries.append(payload)
                    continue
            except Exception:
                pass

        endmembers = load_endmember_matrix(quality.loc[region_id, "endmember_path"])
        year_rows = {int(row.year): row for row in group.itertuples(index=False)}
        landsat_paths = {int(row.year): Path(str(row.path)) for row in aligned_group.itertuples(index=False)}
        reference_row = year_rows[reference_year]
        fraction_path = Path(str(supervision.loc[region_id, "pv_fraction_path"]))
        label_path = Path(str(supervision.loc[region_id, "label_path"]))
        support_path = Path(str(supervision.loc[region_id, "anchor_support_path"]))
        prototypes = _build_unit_prototypes(
            cfg, region_id, reference_row, landsat_paths[reference_year], fraction_path, label_path, support_path,
            endmembers, prototype_root / f"unit_{region_id:03d}_spectral_prototypes.npz",
        )

        prob_paths = [ensure_dir(probability_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_pv_probability_object_graph_30m.tif" for y in years]
        mask_paths = [ensure_dir(mask_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_pv_mask_object_graph_30m.tif" for y in years]
        class_paths = [ensure_dir(class_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_object_class_30m.tif" for y in years]
        rejection_paths = [ensure_dir(rejection_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_rejection_code_30m.tif" for y in years]
        positive_similarity_paths = [ensure_dir(positive_similarity_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_positive_similarity_30m.tif" for y in years]
        sand_score_paths = [ensure_dir(sand_score_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{y}_sand_texture_score_30m.tif" for y in years]

        predicted_pixels = np.zeros(len(years), dtype=np.int64)
        valid_pixels = np.zeros(len(years), dtype=np.int64)
        rescued_pixels = np.zeros(len(years), dtype=np.int64)
        remote_pixels = np.zeros(len(years), dtype=np.int64)
        rejected_sand_pixels = np.zeros(len(years), dtype=np.int64)
        object_rows: list[dict[str, Any]] = []

        with ExitStack() as stack:
            refined_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].calibrated_probability_path))) for y in years]
            raw_srcs = [stack.enter_context(rasterio.open(str(getattr(year_rows[int(y)], "raw_probability_path", "") or year_rows[int(y)].calibrated_probability_path))) for y in years]
            uncertainty_srcs = [stack.enter_context(rasterio.open(str(year_rows[int(y)].epistemic_std_path))) for y in years]
            landsat_srcs = [stack.enter_context(rasterio.open(landsat_paths[int(y)])) for y in years]
            fraction_src = stack.enter_context(rasterio.open(fraction_path))
            support_src = stack.enter_context(rasterio.open(support_path))
            ref = refined_srcs[0]
            float_profile = ref.profile.copy()
            float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            byte_profile = ref.profile.copy()
            byte_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            prob_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in prob_paths]
            mask_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in mask_paths]
            diagnostic_on = write_diagnostics and region_id in diagnostics_units
            class_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in class_paths] if diagnostic_on else []
            rejection_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in rejection_paths] if diagnostic_on else []
            positive_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in positive_similarity_paths] if diagnostic_on else []
            sand_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in sand_score_paths] if diagnostic_on else []
            pixel_area_km2 = abs(float(ref.transform.a * ref.transform.e)) / 1_000_000.0

            for core in _windows(ref.width, ref.height, tile_size):
                expanded, crop = _expanded(core, ref.width, ref.height, halo)
                shape = (int(expanded.height), int(expanded.width))
                n_years = len(years)
                raw_probability = np.zeros((n_years, *shape), dtype=np.float32)
                refined_probability = np.zeros_like(raw_probability)
                uncertainty = np.ones_like(raw_probability)
                valid = np.zeros_like(raw_probability, dtype=bool)
                fraction_evidence = np.zeros_like(raw_probability)
                bare_evidence = np.zeros_like(raw_probability)
                confuser = np.ones_like(raw_probability)
                positive_similarity = np.zeros_like(raw_probability)
                sand_similarity = np.zeros_like(raw_probability)
                brightness = np.zeros_like(raw_probability)
                coherence = np.zeros_like(raw_probability)
                relative_darkness = np.zeros_like(raw_probability)
                texture = np.zeros_like(raw_probability)

                for t, (raw_src, refined_src, uncertainty_src, land_src) in enumerate(zip(raw_srcs, refined_srcs, uncertainty_srcs, landsat_srcs)):
                    raw = raw_src.read(1, window=expanded).astype(np.float32)
                    refined = refined_src.read(1, window=expanded).astype(np.float32)
                    unc = uncertainty_src.read(1, window=expanded).astype(np.float32)
                    refl = _read_reflectance(land_src, expanded)
                    valid[t] = np.isfinite(raw) & (raw > -9990) & np.isfinite(refined) & (refined > -9990) & np.isfinite(refl).all(axis=0)
                    raw_probability[t] = np.where(valid[t], np.clip(raw, 0.0, 1.0), 0.0)
                    refined_probability[t] = np.where(valid[t], np.clip(refined, 0.0, 1.0), 0.0)
                    uncertainty[t] = np.where(valid[t], np.clip(np.nan_to_num(unc, nan=1.0), 0.0, 1.0), 1.0)
                    cube = _feature_cube(refl, endmembers)
                    fraction_evidence[t] = np.clip(np.nan_to_num(cube[FEATURE_CHANNELS.index("PV_Fraction_Evidence")]), 0.0, 1.0)
                    bare_evidence[t] = np.clip(np.nan_to_num(cube[FEATURE_CHANNELS.index("Bare_Evidence")]), 0.0, 1.0)
                    confuser_feature = compute_features(refl, ["PV_Confuser_Penalty"], False, endmembers=endmembers)[0]
                    confuser[t] = np.clip(np.nan_to_num(confuser_feature, nan=1.0), 0.0, 1.0)
                    positive_similarity[t] = _prototype_similarity(cube, prototypes.positive_centers, prototypes.positive_scales)
                    sand_similarity[t] = _prototype_similarity(cube, prototypes.sand_centers, prototypes.sand_scales)
                    brightness[t], coherence[t], relative_darkness[t], texture[t] = _texture_features(refl, valid[t])

                anchor = fraction_src.read(1, window=expanded).astype(np.float32) >= float(cfg.get("supervision", {}).get("strong_positive_fraction", 0.7))
                near_anchor = support_src.read(1, window=expanded) >= 0.5
                onset, persistence, stable_presence = _temporal_evidence(
                    raw_probability, refined_probability, fraction_evidence, brightness, positive_similarity, valid,
                )
                masks = np.zeros_like(valid)
                classes = np.zeros_like(raw_probability, dtype=np.uint8)
                rejection = np.zeros_like(raw_probability, dtype=np.uint8)
                scores = np.zeros_like(raw_probability)
                growables = np.zeros_like(valid)
                sand_scores = np.zeros_like(raw_probability)

                for t, year in enumerate(years):
                    uncertainty_penalty = np.clip(uncertainty[t] / 0.40, 0.0, 1.0)
                    # Dune/sand false positives are characterised by a bare-ground spectrum,
                    # directional texture continuity, and weak anchor-prototype similarity.
                    sand_score = np.clip(
                        0.38 * sand_similarity[t]
                        + 0.24 * bare_evidence[t]
                        + 0.20 * coherence[t]
                        + 0.18 * texture[t]
                        - 0.28 * relative_darkness[t]
                        - 0.24 * positive_similarity[t],
                        0.0,
                        1.0,
                    )
                    sand_scores[t] = sand_score
                    score = np.clip(
                        0.25 * raw_probability[t]
                        + 0.19 * refined_probability[t]
                        + 0.19 * positive_similarity[t]
                        + 0.12 * fraction_evidence[t]
                        + 0.11 * onset[t]
                        + 0.07 * persistence[t]
                        + 0.07 * relative_darkness[t]
                        - 0.22 * sand_score
                        - 0.13 * confuser[t]
                        - 0.08 * uncertainty_penalty,
                        0.0,
                        1.0,
                    ) * valid[t]
                    scores[t] = score

                    seed = valid[t] & (
                        (
                            (raw_probability[t] >= float(settings.get("seed_minimum_raw_probability", 0.72)))
                            & (refined_probability[t] >= float(settings.get("seed_minimum_refined_probability", 0.58)))
                            & (positive_similarity[t] >= float(settings.get("seed_minimum_positive_similarity", 0.42)))
                            & (fraction_evidence[t] >= float(settings.get("seed_minimum_fraction_evidence", 0.38)))
                            & (sand_score <= float(settings.get("seed_maximum_sand_score", 0.54)))
                        )
                        | (
                            (score >= float(settings.get("seed_minimum_fused_score", 0.68)))
                            & (positive_similarity[t] >= 0.52)
                            & (sand_score <= 0.44)
                        )
                    )
                    year_anchor = anchor & valid[t] if int(year) == reference_year else np.zeros_like(anchor)
                    seed |= year_anchor

                    growable = valid[t] & (
                        (score >= float(settings.get("growth_minimum_score", 0.45)))
                        & (positive_similarity[t] >= float(settings.get("growth_minimum_positive_similarity", 0.28)))
                        & (fraction_evidence[t] >= float(settings.get("growth_minimum_fraction_evidence", 0.24)))
                        & (sand_score <= float(settings.get("growth_maximum_sand_score", 0.62)))
                        & (confuser[t] <= float(settings.get("growth_maximum_confuser", 0.72)))
                    )
                    growables[t] = growable
                    if np.any(seed):
                        distance = ndimage.distance_transform_edt(~seed)
                        graph_mask = growable & (distance <= float(settings.get("maximum_adjacency_growth_pixels", 18)))
                        adjacency = ndimage.binary_propagation(seed, structure=np.ones((3, 3), dtype=bool), mask=graph_mask)
                    else:
                        adjacency = np.zeros_like(seed)

                    remote = valid[t] & (
                        (score >= float(settings.get("remote_seed_minimum_score", 0.64)))
                        & (raw_probability[t] >= float(settings.get("remote_seed_minimum_raw_probability", 0.58)))
                        & (positive_similarity[t] >= float(settings.get("remote_seed_minimum_positive_similarity", 0.45)))
                        & (sand_score <= float(settings.get("remote_seed_maximum_sand_score", 0.48)))
                        & (
                            (onset[t] >= float(settings.get("remote_seed_minimum_onset", 0.46)))
                            | (stable_presence[t] >= float(settings.get("remote_seed_preexisting_stability", 0.76)))
                        )
                    )
                    candidate = adjacency | remote | year_anchor
                    accepted, object_classes, object_rejection, rows = _filter_objects(
                        candidate, seed, adjacency, remote, year_anchor, score, positive_similarity[t], sand_score,
                        onset[t], persistence[t], stable_presence[t], brightness[t], settings, int(year), region_id,
                    )
                    masks[t] = accepted
                    classes[t] = object_classes
                    rejection[t] = object_rejection
                    object_rows.extend(rows)

                reference_index = int(reference_year - int(years[0]))
                anchor_by_year = np.zeros_like(masks)
                anchor_by_year[reference_index] = anchor & valid[reference_index]
                masks, classes, rejection = _temporal_object_graph(
                    masks, classes, rejection, growables, scores, positive_similarity, sand_scores,
                    onset, stable_presence, anchor_by_year, settings,
                )
                masks[reference_index] |= anchor & valid[reference_index]
                classes[reference_index][anchor & valid[reference_index]] = CLASS_REFERENCE_ANCHOR
                rejection[reference_index][anchor & valid[reference_index]] = REJECT_NONE

                for t, year in enumerate(years):
                    core_valid = valid[t][crop]
                    core_mask = masks[t][crop]
                    core_class = classes[t][crop]
                    core_rejection = rejection[t][crop]
                    # Rejected candidates are intentionally capped below temporal decoder thresholds.
                    final_probability = np.where(
                        masks[t],
                        np.clip(0.62 + 0.38 * scores[t], 0.0, 1.0),
                        np.clip(0.18 * scores[t], 0.0, 0.24),
                    )
                    if int(year) == reference_year:
                        final_probability = np.maximum(final_probability, np.where(anchor, 0.995, 0.0))
                    encoded_mask = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                    encoded_mask[core_valid] = core_mask[core_valid].astype(np.uint8)
                    prob_dsts[t].write(np.where(core_valid, final_probability[crop], -9999.0).astype(np.float32), 1, window=core)
                    mask_dsts[t].write(encoded_mask, 1, window=core)
                    if diagnostic_on:
                        encoded_class = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        encoded_rejection = np.full(core_mask.shape, mask_nodata, dtype=np.uint8)
                        encoded_class[core_valid] = core_class[core_valid]
                        encoded_rejection[core_valid] = core_rejection[core_valid]
                        class_dsts[t].write(encoded_class, 1, window=core)
                        rejection_dsts[t].write(encoded_rejection, 1, window=core)
                        positive_dsts[t].write(np.where(core_valid, positive_similarity[t][crop], -9999.0).astype(np.float32), 1, window=core)
                        sand_dsts[t].write(np.where(core_valid, sand_scores[t][crop], -9999.0).astype(np.float32), 1, window=core)

                    predicted_pixels[t] += int(np.count_nonzero(core_mask & core_valid))
                    valid_pixels[t] += int(np.count_nonzero(core_valid))
                    rescued_pixels[t] += int(np.count_nonzero((core_class == CLASS_ADJACENCY_RESCUE) & core_valid))
                    remote_pixels[t] += int(np.count_nonzero((core_class == CLASS_REMOTE_NEW_SITE) & core_valid))
                    rejected_sand_pixels[t] += int(np.count_nonzero((core_rejection == REJECT_SAND_TEXTURE) & core_valid))

        object_table = pd.DataFrame(object_rows)
        save_table(object_table, object_table_root / f"unit_{region_id:03d}_object_decisions.csv")
        unit_rows: list[dict[str, Any]] = []
        for t, year in enumerate(years):
            original = year_rows[int(year)]._asdict()
            unit_rows.append({
                **original,
                "pre_object_graph_probability_path": original.get("calibrated_probability_path", ""),
                "pre_object_graph_mask_path": original.get("calibrated_mask_path", ""),
                "pre_object_graph_predicted_pixels": int(original.get("calibrated_predicted_pixels", 0) or 0),
                "pre_object_graph_predicted_area_km2": float(original.get("calibrated_predicted_pixels", 0) or 0) * pixel_area_km2,
                "calibrated_probability_path": str(prob_paths[t]),
                "calibrated_mask_path": str(mask_paths[t]),
                "calibrated_predicted_pixels": int(predicted_pixels[t]),
                "calibrated_predicted_area_km2": float(predicted_pixels[t] * pixel_area_km2),
                "object_graph_refinement": "anchor_prototype_topology_temporal_object_graph_v24_13",
                "object_class_path": str(class_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "object_graph_rejection_code_path": str(rejection_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "positive_prototype_similarity_path": str(positive_similarity_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "sand_texture_score_path": str(sand_score_paths[t]) if write_diagnostics and region_id in diagnostics_units else "",
                "adjacency_rescued_pixels": int(rescued_pixels[t]),
                "remote_new_site_pixels": int(remote_pixels[t]),
                "sand_rejected_pixels": int(rejected_sand_pixels[t]),
            })
        unit_registry = pd.DataFrame(unit_rows).sort_values("year")
        save_table(unit_registry, unit_registry_path)
        accepted_objects = int(object_table["accepted"].sum()) if not object_table.empty else 0
        rejected_objects = int((~object_table["accepted"]).sum()) if not object_table.empty else 0
        payload = {
            "status": "passed",
            "created_utc": utc_now(),
            "Id": region_id,
            "records": int(len(unit_registry)),
            "registry_path": str(unit_registry_path),
            "positive_prototype_samples": prototypes.positive_samples,
            "sand_prototype_samples": prototypes.sand_samples,
            "accepted_objects": accepted_objects,
            "rejected_objects": rejected_objects,
            "adjacency_rescued_pixel_years": int(rescued_pixels.sum()),
            "remote_new_site_pixel_years": int(remote_pixels.sum()),
            "sand_rejected_pixel_years": int(rejected_sand_pixels.sum()),
            "diagnostic_rasters_written": bool(write_diagnostics and region_id in diagnostics_units),
        }
        atomic_json(payload, checkpoint)
        unit_summaries.append(payload)
        all_unit_registries.append(unit_registry)

    refined = pd.concat(all_unit_registries, ignore_index=True, sort=False).sort_values(["Id", "year"])
    backup = root / "07_probability_calibration" / "calibrated_inference_registry_pre_object_graph_refinement.csv"
    if not backup.exists():
        save_table(source_registry, backup)
    save_table(refined, out / "object_graph_inference_registry.csv")
    save_table(refined, canonical_registry_path)
    summary = pd.DataFrame(unit_summaries)
    save_table(summary, out / "object_graph_refinement_by_unit.csv")

    expected_units, _, expected_records = expected_counts(cfg)
    coverage_ok = len(refined) == expected_records and refined["Id"].nunique() == expected_units
    status = {
        "status": "passed" if coverage_ok else "failed",
        "created_utc": utc_now(),
        "records": int(len(refined)),
        "expected_records": int(expected_records),
        "units": int(refined["Id"].nunique()),
        "expected_units": int(expected_units),
        "method": "2024 positive-anchor spectral prototypes + sand/background prototypes + bounded topology growth + remote onset proposals + object interior/geometry tests + cross-year object graph",
        "false_positive_control": "directional sand texture, hollow boundary objects, prototype mismatch, weak temporal support and implausible geometry are independently rejected",
        "false_negative_control": "high-confidence seed objects expand into adjacent spectrally similar pixels; remote rectangular sites can enter only with durable onset or stable pre-existing evidence",
        "reference_anchor_rule": "2024 SHP is forced positive only in 2024; other years require image-derived object and temporal evidence",
        "diagnostic_units": sorted(diagnostics_units),
        "canonical_registry_replaced": str(canonical_registry_path),
        "pre_object_graph_registry_backup": str(backup),
    }
    atomic_json(status, out / "object_graph_refinement_status.json")
    if not coverage_ok:
        raise RuntimeError("Object graph refinement coverage incomplete")
    return refined
