from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pvchange.object_graph_refinement import refine_object_graph_probabilities


def _write_raster(path: Path, data: np.ndarray, *, dtype: str = "float32", nodata: float | int = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None, ...]
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": dtype,
        "crs": "EPSG:32648",
        "transform": from_origin(500000, 4200000, 30, 30),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def test_full_object_graph_refinement_on_synthetic_unit(tmp_path: Path) -> None:
    root = tmp_path / "output"
    for name in ["02_aligned_30m", "03_supervision", "04_unit_quality", "07_probability_calibration"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    years = [2022, 2023, 2024]
    h = w = 32
    transform_profile = np.zeros((6, h, w), dtype=np.float32)
    transform_profile[:] = np.array([0.25, 0.28, 0.30, 0.32, 0.31, 0.29])[:, None, None]
    anchor = np.zeros((h, w), dtype=np.float32)
    anchor[10:18, 10:18] = 1.0
    support = np.zeros((h, w), dtype=np.uint8)
    support[6:22, 6:22] = 1
    label = np.full((h, w), 255, dtype=np.uint8)
    label[:5, :] = 0
    fraction_path = root / "03_supervision" / "fraction.tif"
    support_path = root / "03_supervision" / "support.tif"
    label_path = root / "03_supervision" / "label.tif"
    _write_raster(fraction_path, anchor, dtype="float32", nodata=-9999.0)
    _write_raster(support_path, support, dtype="uint8", nodata=255)
    _write_raster(label_path, label, dtype="uint8", nodata=255)

    endmembers = np.asarray([
        [0.34, 0.055, 0.045, 0.010],
        [0.38, 0.060, 0.075, 0.010],
        [0.42, 0.055, 0.045, 0.010],
        [0.44, 0.060, 0.420, 0.010],
        [0.39, 0.070, 0.180, 0.010],
        [0.32, 0.065, 0.100, 0.010],
    ], dtype=np.float32)
    endmember_path = root / "04_unit_quality" / "endmembers.npz"
    np.savez_compressed(endmember_path, endmembers=endmembers)

    aligned_rows = []
    registry_rows = []
    for i, year in enumerate(years):
        refl = transform_profile.copy()
        if year >= 2023:
            refl[:, 10:18, 10:18] = np.array([0.06, 0.06, 0.055, 0.06, 0.07, 0.065])[:, None, None]
            refl[:, 12:18, 18:23] = np.array([0.061, 0.061, 0.056, 0.061, 0.071, 0.066])[:, None, None]
        landsat_path = root / "02_aligned_30m" / f"land_{year}.tif"
        _write_raster(landsat_path, refl, dtype="float32", nodata=-9999.0)
        aligned_rows.append({"Id": 1, "year": year, "path": str(landsat_path)})

        raw = np.full((h, w), 0.08, dtype=np.float32)
        refined = np.full((h, w), 0.05, dtype=np.float32)
        if year >= 2023:
            raw[10:18, 10:18] = 0.93
            refined[10:18, 10:18] = 0.88
            raw[12:18, 18:23] = 0.78
            refined[12:18, 18:23] = 0.63
        uncertainty = np.full((h, w), 0.10, dtype=np.float32)
        raw_path = root / "07_probability_calibration" / f"raw_{year}.tif"
        refined_path = root / "07_probability_calibration" / f"refined_{year}.tif"
        mask_path = root / "07_probability_calibration" / f"mask_{year}.tif"
        uncertainty_path = root / "07_probability_calibration" / f"unc_{year}.tif"
        _write_raster(raw_path, raw)
        _write_raster(refined_path, refined)
        _write_raster(mask_path, (refined >= 0.5).astype(np.uint8), dtype="uint8", nodata=255)
        _write_raster(uncertainty_path, uncertainty)
        registry_rows.append({
            "Id": 1,
            "year": year,
            "inference_mode": "direct_pixel",
            "estimation_route": "direct_pixel",
            "calibrated_probability_path": str(refined_path),
            "calibrated_mask_path": str(mask_path),
            "raw_probability_path": str(raw_path),
            "epistemic_std_path": str(uncertainty_path),
            "calibrated_predicted_pixels": int((refined >= 0.5).sum()),
        })
    pd.DataFrame(aligned_rows).to_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv", index=False)
    pd.DataFrame(registry_rows).to_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv", index=False)
    pd.DataFrame([{"Id": 1, "endmember_path": str(endmember_path)}]).to_csv(root / "04_unit_quality" / "unit_quality_registry.csv", index=False)
    pd.DataFrame([{
        "Id": 1,
        "pv_fraction_path": str(fraction_path),
        "label_path": str(label_path),
        "anchor_support_path": str(support_path),
    }]).to_csv(root / "03_supervision" / "supervision_registry.csv", index=False)

    cfg = {
        "project": {"random_seed": 2027},
        "paths": {"output_root": str(root)},
        "study": {
            "id_start": 1, "id_end": 1, "selected_unit_ids": [1],
            "start_year": 2022, "end_year": 2024, "reference_year": 2024,
        },
        "test_mode": {"enabled_when_selected_unit_ids": True},
        "processing": {"show_progress": False, "overwrite": True, "compress": "DEFLATE"},
        "calibration": {"mask_nodata": 255},
        "supervision": {"strong_positive_fraction": 0.7},
        "object_graph_refinement": {
            "enabled": True,
            "tile_size": 32,
            "halo_pixels": 8,
            "prototype_tile_size": 32,
            "maximum_prototype_samples": 10000,
            "positive_prototype_clusters": 1,
            "sand_prototype_clusters": 1,
            "diagnostic_unit_ids": [1],
            "write_diagnostics": True,
            "minimum_remote_component_pixels": 6,
            "remote_minimum_mean_score": 0.35,
            "remote_minimum_q75_score": 0.35,
            "remote_minimum_positive_similarity": 0.10,
            "remote_maximum_sand_score": 0.95,
            "remote_minimum_persistence": 0.20,
            "remote_minimum_onset": 0.20,
            "remote_preexisting_minimum_stability": 0.20,
            "remote_minimum_solid_fill": 0.40,
            "remote_minimum_oriented_fill": 0.10,
            "remote_maximum_elongation": 30.0,
            "remote_minimum_boundary_contrast": -1.0,
        },
    }
    result = refine_object_graph_probabilities(cfg)
    assert len(result) == 3
    assert (root / "07_object_graph_refinement" / "object_graph_refinement_status.json").exists()
    row_2024 = result.loc[result["year"].eq(2024)].iloc[0]
    with rasterio.open(row_2024["calibrated_mask_path"]) as src:
        mask = src.read(1)
    assert np.all(mask[10:18, 10:18] == 1)
    assert int(row_2024["calibrated_predicted_pixels"]) >= 64
