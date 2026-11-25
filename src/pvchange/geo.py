from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds


@dataclass(frozen=True)
class Grid:
    region_id: int
    crs: CRS
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    resolution_m: float


def is_metre_projected(crs: Any) -> bool:
    try:
        parsed = CRS.from_user_input(crs)
        if not parsed.is_projected:
            return False
        axis = parsed.axis_info
        factor = float(axis[0].unit_conversion_factor) if axis and axis[0].unit_conversion_factor else 1.0
        return abs(factor - 1.0) < 1e-6
    except Exception:
        return False


def source_pixel_size_m(src: rasterio.DatasetReader) -> tuple[float, float]:
    if is_metre_projected(src.crs):
        return abs(float(src.transform.a)), abs(float(src.transform.e))
    cx = (src.bounds.left + src.bounds.right) / 2.0
    cy = (src.bounds.bottom + src.bounds.top) / 2.0
    to_wgs = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(cx, cy)
    utm = local_utm_crs(lon, lat)
    tr = Transformer.from_crs(src.crs, utm, always_xy=True)
    x0, y0 = tr.transform(cx, cy)
    x1, y1 = tr.transform(cx + src.transform.a, cy)
    x2, y2 = tr.transform(cx, cy + src.transform.e)
    return abs(x1 - x0), abs(y2 - y0)


def local_utm_crs(lon: float, lat: float) -> CRS:
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def target_crs_for_source(src: rasterio.DatasetReader) -> CRS:
    if is_metre_projected(src.crs):
        return CRS.from_user_input(src.crs)
    cx = (src.bounds.left + src.bounds.right) / 2.0
    cy = (src.bounds.bottom + src.bounds.top) / 2.0
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
    return local_utm_crs(lon, lat)


def aligned_bounds(bounds: tuple[float, float, float, float], resolution: float, anchor: float) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    left = np.floor(left / anchor) * anchor
    bottom = np.floor(bottom / anchor) * anchor
    right = np.ceil(right / anchor) * anchor
    top = np.ceil(top / anchor) * anchor
    width = np.ceil((right - left) / resolution)
    height = np.ceil((top - bottom) / resolution)
    return float(left), float(top - height * resolution), float(left + width * resolution), float(top)


def build_grid(region_id: int, reference_path: str | Path, resolution: float = 30.0, anchor: float = 30.0) -> Grid:
    with rasterio.open(reference_path) as src:
        target_crs = target_crs_for_source(src)
        if CRS.from_user_input(src.crs) == target_crs:
            bounds = tuple(map(float, src.bounds))
        else:
            bounds = transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=41)
    left, bottom, right, top = aligned_bounds(bounds, resolution, anchor)
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    actual = array_bounds(height, width, transform)
    return Grid(region_id, target_crs, transform, width, height, tuple(map(float, actual)), resolution)


def aligned_vrt(src: rasterio.DatasetReader, grid: Grid, resampling: Resampling = Resampling.bilinear) -> WarpedVRT:
    return WarpedVRT(
        src,
        crs=grid.crs,
        transform=grid.transform,
        width=grid.width,
        height=grid.height,
        resampling=resampling,
        src_nodata=src.nodata,
        nodata=np.nan,
        dtype="float32",
    )


def grid_profile(grid: Grid, count: int, dtype: str, nodata: Any, compress: str = "DEFLATE") -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": grid.width >= 256 and grid.height >= 256,
        "BIGTIFF": "IF_SAFER",
    }
