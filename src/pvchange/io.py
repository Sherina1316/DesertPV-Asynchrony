from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling


def region_dir(root: str | Path, region_id: int) -> Path:
    root = Path(root)
    for name in (str(region_id), f"{region_id:03d}"):
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Region directory missing: Id={region_id}")


def discover_images(cfg: dict[str, Any], region_id: int) -> dict[int, Path]:
    directory = region_dir(cfg["paths"]["landsat_root"], region_id)
    all_tifs = sorted([*directory.rglob("*.tif"), *directory.rglob("*.tiff")])
    output: dict[int, Path] = {}
    for year in range(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1):
        selected = None
        for pattern in cfg["landsat"]["filename_patterns"]:
            matches = sorted(directory.glob(str(pattern).format(year=year)))
            if matches:
                selected = matches[0]
                break
        if selected is None:
            regex = re.compile(rf"(?<!\d){year}(?!\d)")
            candidates = [p for p in all_tifs if regex.search(p.name)]
            if candidates:
                selected = candidates[0]
        if selected is not None:
            output[year] = selected
    return output


def detect_reflectance_scale(path: str | Path, max_size: int = 256) -> str:
    with rasterio.open(path) as src:
        count = min(6, src.count)
        arr = src.read(
            indexes=list(range(1, count + 1)),
            out_shape=(count, min(max_size, src.height), min(max_size, src.width)),
            masked=True,
            resampling=Resampling.nearest,
        ).filled(np.nan).astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "unit"
    q99 = float(np.nanquantile(np.abs(finite), 0.99))
    if q99 > 1000:
        return "scaled_10000"
    if q99 > 2:
        return "scaled_100"
    return "unit"


def scale_reflectance(x: np.ndarray, mode: str) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if mode == "scaled_10000":
        x = x / 10000.0
    elif mode == "scaled_100":
        x = x / 100.0
    x[(x < -0.2) | (x > 1.6)] = np.nan
    return x
