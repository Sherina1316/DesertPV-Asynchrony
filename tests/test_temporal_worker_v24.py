from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pvchange.temporal_worker import process_unit


def _write_raster(path: Path, array: np.ndarray, *, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[-2],
        "width": array.shape[-1],
        "count": count,
        "dtype": "float32" if array.dtype != np.uint8 else "uint8",
        "crs": "EPSG:32648",
        "transform": from_origin(500000, 4000000, 30, 30),
        "nodata": -9999.0 if array.dtype != np.uint8 else 255,
        "compress": "DEFLATE",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array if count > 1 else array[None, ...])


def test_isolated_temporal_worker_detects_persistent_transition(tmp_path: Path) -> None:
    root = tmp_path / "output"
    years = np.arange(2011, 2019)
    height = width = 32
    event = np.zeros((height, width), dtype=bool)
    event[8:24, 8:24] = True

    probability_rows = []
    aligned_rows = []
    default_endmembers = np.asarray(
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

    for index, year in enumerate(years):
        probability = np.full((height, width), 0.05, dtype=np.float32)
        reflectance = np.repeat(default_endmembers[:, 0, None, None], height, axis=1)
        reflectance = np.repeat(reflectance, width, axis=2).astype(np.float32)
        if index >= 4:
            probability[event] = 0.95
            for band in range(6):
                reflectance[band, event] = default_endmembers[band, 1]
        probability_path = root / "07_probability_calibration" / f"p_{year}.tif"
        landsat_path = root / "02_aligned_30m" / f"l_{year}.tif"
        _write_raster(probability_path, probability, count=1)
        _write_raster(landsat_path, reflectance, count=6)
        probability_rows.append({
            "Id": 1,
            "year": int(year),
            "calibrated_probability_path": str(probability_path),
            "calibrated_threshold": 0.7,
            "inference_mode": "direct_pixel",
            "estimation_route": "direct_pixel",
        })
        aligned_rows.append({"Id": 1, "year": int(year), "path": str(landsat_path)})

    (root / "07_probability_calibration").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probability_rows).to_csv(
        root / "07_probability_calibration" / "calibrated_inference_registry.csv", index=False
    )
    (root / "02_aligned_30m").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aligned_rows).to_csv(
        root / "02_aligned_30m" / "aligned_raster_registry.csv", index=False
    )

    fraction = np.zeros((height, width), dtype=np.float32)
    fraction[event] = 1.0
    support = np.zeros((height, width), dtype=np.uint8)
    support[5:27, 5:27] = 1
    fraction_path = root / "03_supervision" / "masks" / "unit_001_pv_fraction_30m.tif"
    support_path = root / "03_supervision" / "masks" / "unit_001_anchor_support_30m.tif"
    _write_raster(fraction_path, fraction, count=1)
    _write_raster(support_path, support, count=1)
    pd.DataFrame([{
        "Id": 1,
        "pv_fraction_path": str(fraction_path),
        "anchor_support_path": str(support_path),
        "confirmed_anchor_area_km2": float(event.sum() * 0.0009),
    }]).to_csv(root / "03_supervision" / "supervision_registry.csv", index=False)

    endmember_path = root / "04_unit_quality" / "unit_001_endmembers.npz"
    endmember_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(endmember_path, endmembers=default_endmembers)
    pd.DataFrame([{"Id": 1, "endmember_path": str(endmember_path)}]).to_csv(
        root / "04_unit_quality" / "unit_quality_registry.csv", index=False
    )

    cfg = {
        "paths": {"output_root": str(root)},
        "project": {"random_seed": 2027},
        "study": {"start_year": 2011, "end_year": 2018},
        "processing": {"compress": "DEFLATE", "show_progress": False},
        "resource_control": {"gdal_cache_mb": 8},
        "temporal_decoding": {"mask_nodata": 255, "change_year_probability_threshold": 0.7},
        "temporal_signature": {
            "processing_tile_size": 16,
            "processing_halo_pixels": 8,
            "harmonization_tile_size": 16,
            "harmonization_maximum_samples_per_year": 5000,
            "harmonization_maximum_samples_per_tile": 500,
            "harmonization_minimum_valid_years": 6,
            "harmonization_minimum_pixels": 50,
            "stable_background_probability_ceiling": 0.22,
            "stable_background_maximum_range": 0.15,
            "reference_anchor_search_radius_m": 3000.0,
            "minimum_pre_years": 3,
            "minimum_post_years": 2,
            "minimum_probability_level_shift": 0.18,
            "minimum_fraction_level_shift": 0.08,
            "minimum_break_z": 2.0,
            "minimum_post_presence_fraction": 0.75,
            "maximum_pre_presence_fraction": 0.30,
            "maximum_post_probability_mad": 0.18,
            "maximum_post_annual_drop": 0.30,
            "minimum_transition_score": 0.50,
            "near_anchor_score_relaxation": 0.08,
            "remote_score_penalty": 0.08,
            "confuser_penalty_weight": 0.15,
            "near_anchor_minimum_component_pixels": 4,
            "remote_minimum_component_pixels": 24,
            "minimum_bbox_fill": 0.10,
            "maximum_component_area_fraction": 0.50,
            "baseline_preexisting_enabled": False,
        },
    }

    result = process_unit(cfg, 1, overwrite=True)
    assert len(result) == len(years)
    assert result.loc[result.year.eq(2014), "pv_area_km2"].iloc[0] == 0
    assert result.loc[result.year.eq(2015), "pv_area_km2"].iloc[0] > 0
    assert result["pv_area_km2"].iloc[-1] >= result["pv_area_km2"].iloc[-2]
    onset_path = Path(result["change_year_path"].iloc[0])
    with rasterio.open(onset_path) as src:
        onset = src.read(1)
    assert int(np.median(onset[event])) == 2015
