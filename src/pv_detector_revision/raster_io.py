from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import math
import os
import tempfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds


@dataclass(frozen=True)
class GridIdentity:
    crs: str
    transform: tuple[float, ...]
    width: int
    height: int

    @classmethod
    def from_profile(cls, profile: Mapping[str, object]) -> "GridIdentity":
        transform = profile["transform"]
        return cls(
            crs=str(profile.get("crs")),
            transform=tuple(float(v) for v in transform)[:6],
            width=int(profile["width"]),
            height=int(profile["height"]),
        )


PATH_COLUMNS = {
    "calibrated_probability": (
        "calibrated_probability_path",
        "probability_path",
    ),
    "raw_probability": (
        "raw_probability_path",
    ),
    "anchor_similarity": (
        "anchor_similarity_path",
        "component_similarity_path",
    ),
    "lsma_fraction": (
        "reference_fraction_path",
        "pv_fraction_path",
        "lsma_pv_fraction_path",
    ),
    "candidate_score": (
        "candidate_score_path",
        "terminal_2025_site_likelihood_path",
        "terminal_2025_remote_census_likelihood_path",
    ),
    "multiband_break_support": (
        "multiband_break_support_path",
        "maximum_installation_change_support_path",
        "terminal_2025_direct_change_support_path",
    ),
    "reference_anchor": (
        "reference_anchor_path",
        "reference_inventory_path",
    ),
    "legacy_active": (
        "active_mask_path",
        "observed_active_mask_path",
    ),
    "legacy_cumulative": (
        "cumulative_mask_path",
        "cumulative_installed_mask_path",
    ),
    "terminal_expansion": (
        "reference_outside_supplement_path",
        "terminal_2025_expansion_source_path",
        "terminal_2025_remote_change_support_path",
    ),
}


def existing_path(row: pd.Series, candidates: Sequence[str]) -> Path | None:
    for column in candidates:
        if column not in row.index:
            continue
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        path = Path(text)
        if path.is_file():
            return path
    return None


def read_float(path: Path) -> tuple[np.ndarray, dict, np.ndarray]:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        valid = np.isfinite(array)
        if nodata is not None and np.isfinite(nodata):
            valid &= array != nodata
    array = np.where(valid, array, np.nan).astype(np.float32, copy=False)
    return array, profile, valid


def read_bool(path: Path) -> tuple[np.ndarray, dict, np.ndarray]:
    # Binary production masks commonly declare nodata=0 even though zero is the
    # valid background class. Treat finite zero as valid; only non-zero nodata
    # sentinels are excluded.
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
    valid = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata) and float(nodata) != 0.0:
        valid &= array != nodata
    return (np.where(valid, array, 0) > 0), profile, valid


def same_grid(a: Mapping[str, object], b: Mapping[str, object], atol: float = 1e-9) -> bool:
    if int(a["width"]) != int(b["width"]) or int(a["height"]) != int(b["height"]):
        return False
    if str(a.get("crs")) != str(b.get("crs")):
        return False
    ta = tuple(float(v) for v in a["transform"])[:6]
    tb = tuple(float(v) for v in b["transform"])[:6]
    return bool(np.allclose(ta, tb, atol=atol, rtol=0.0))


def require_same_grid(reference: Mapping[str, object], other: Mapping[str, object], label: str) -> None:
    if not same_grid(reference, other):
        raise ValueError(
            f"Raster grid mismatch for {label}: "
            f"reference={GridIdentity.from_profile(reference)}, "
            f"other={GridIdentity.from_profile(other)}"
        )


def write_raster_atomic(
    path: Path,
    array: np.ndarray,
    profile: Mapping[str, object],
    *,
    dtype: str | None = None,
    nodata: int | float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_dtype = dtype or str(array.dtype)
    out_profile = dict(profile)
    width = int(out_profile["width"])
    height = int(out_profile["height"])
    out_profile.pop("blockxsize", None)
    out_profile.pop("blockysize", None)
    tiled = width >= 16 and height >= 16
    out_profile.update(
        count=1,
        dtype=target_dtype,
        compress="deflate",
        predictor=2 if np.issubdtype(np.dtype(target_dtype), np.floating) else 1,
        tiled=tiled,
        BIGTIFF="IF_SAFER",
    )
    if tiled:
        out_profile["blockxsize"] = max(16, min(512, (width // 16) * 16))
        out_profile["blockysize"] = max(16, min(512, (height // 16) * 16))
    if nodata is not None:
        out_profile["nodata"] = nodata
    elif "nodata" in out_profile:
        out_profile.pop("nodata", None)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".tmp.tif",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with rasterio.open(tmp, "w", **out_profile) as dst:
            dst.write(array.astype(target_dtype, copy=False), 1)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def pixel_area_km2(profile: Mapping[str, object]) -> float:
    transform = profile["transform"]
    crs = profile.get("crs")
    # The production grids are projected 30 m grids. Refuse silent degree-area use.
    if crs is None or getattr(crs, "is_geographic", False):
        raise ValueError("Pixel-area calculation requires a projected canonical grid")
    return abs(float(transform.a) * float(transform.e)) / 1_000_000.0


def rasterize_reference_on_template(
    reference_shp: Path,
    template_profile: Mapping[str, object],
) -> np.ndarray:
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except Exception as exc:  # pragma: no cover - environment contract
        raise RuntimeError("geopandas and shapely are required to rasterize the 2024 reference") from exc

    gdf = gpd.read_file(reference_shp)
    if gdf.empty:
        raise ValueError(f"Reference vector is empty: {reference_shp}")
    target_crs = template_profile.get("crs")
    if target_crs is None:
        raise ValueError("Template raster has no CRS")
    if gdf.crs is None:
        raise ValueError("Reference vector has no CRS")
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)

    height = int(template_profile["height"])
    width = int(template_profile["width"])
    transform = template_profile["transform"]
    west, south, east, north = array_bounds(height, width, transform)
    extent = box(west, south, east, north)
    selected = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    selected = selected.loc[selected.intersects(extent)]
    if selected.empty:
        return np.zeros((height, width), dtype=bool)
    shapes = [(geometry, 1) for geometry in selected.geometry]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return mask.astype(bool)


def load_reference_anchor(
    row_2024: pd.Series,
    template_profile: Mapping[str, object],
    reference_shp: Path,
) -> tuple[np.ndarray, str]:
    path = existing_path(row_2024, PATH_COLUMNS["reference_anchor"])
    if path is not None:
        mask, profile, _ = read_bool(path)
        require_same_grid(template_profile, profile, "2024 reference anchor")
        if mask.any():
            return mask, str(path)
    mask = rasterize_reference_on_template(reference_shp, template_profile)
    if not mask.any():
        raise ValueError(
            "The 2024 reference has zero pixels after rasterization on the canonical unit grid"
        )
    return mask, f"rasterized:{reference_shp}"


def find_registry(output_root: Path, relative: str) -> Path:
    path = output_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required registry is missing: {path}")
    return path


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp.csv", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def collect_existing_raster_paths(row: pd.Series) -> list[Path]:
    paths: list[Path] = []
    for column in row.index:
        if "path" not in str(column).lower():
            continue
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        p = Path(str(value))
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            paths.append(p)
    return paths


def choose_template_path(row: pd.Series) -> Path:
    preferred = (
        *PATH_COLUMNS["calibrated_probability"],
        *PATH_COLUMNS["raw_probability"],
        *PATH_COLUMNS["anchor_similarity"],
        *PATH_COLUMNS["lsma_fraction"],
        *PATH_COLUMNS["candidate_score"],
        *PATH_COLUMNS["multiband_break_support"],
    )
    path = existing_path(row, preferred)
    if path is not None:
        return path
    all_paths = collect_existing_raster_paths(row)
    if all_paths:
        return all_paths[0]
    raise FileNotFoundError(
        "No canonical pixel raster exists for this annual Stage-7 row. "
        "The row is probably area_only_fallback and cannot be spatially revised."
    )
