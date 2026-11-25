from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pvchange.site_first_refinement import CLASS_SITE_FIRST_REMOTE, refine_site_first_probabilities


def _write_raster(path: Path, data: np.ndarray, *, dtype: str = "float32", nodata: float | int = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None, ...]
    profile = {
        "driver": "GTiff", "height": data.shape[1], "width": data.shape[2],
        "count": data.shape[0], "dtype": dtype, "crs": "EPSG:32648",
        "transform": from_origin(500000, 4200000, 30, 30), "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def test_site_first_integration_recovers_late_remote_site_and_rejects_early_sand(tmp_path: Path) -> None:
    root = tmp_path / "output"
    for name in ["02_aligned_30m", "03_supervision", "04_unit_quality", "07_probability_calibration"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    h = w = 64

    fraction = np.zeros((h, w), dtype=np.float32)
    fraction[4:14, 4:14] = 1.0
    support = np.zeros((h, w), dtype=np.uint8)
    support[2:18, 2:18] = 1
    label = np.full((h, w), 255, dtype=np.uint8)
    label[:4, :] = 0
    fraction_path = root / "03_supervision/fraction.tif"
    support_path = root / "03_supervision/support.tif"
    label_path = root / "03_supervision/label.tif"
    _write_raster(fraction_path, fraction)
    _write_raster(support_path, support, dtype="uint8", nodata=255)
    _write_raster(label_path, label, dtype="uint8", nodata=255)

    endmembers = np.asarray([
        [0.34, 0.055, 0.045, 0.010], [0.38, 0.060, 0.075, 0.010],
        [0.42, 0.055, 0.045, 0.010], [0.44, 0.060, 0.420, 0.010],
        [0.39, 0.070, 0.180, 0.010], [0.32, 0.065, 0.100, 0.010],
    ], dtype=np.float32)
    endmember_path = root / "04_unit_quality/endmembers.npz"
    np.savez_compressed(endmember_path, endmembers=endmembers)

    aligned_rows = []
    registry_rows = []
    background = np.array([0.28, 0.31, 0.34, 0.36, 0.34, 0.30], dtype=np.float32)
    dark_panel = np.array([0.025, 0.030, 0.032, 0.038, 0.042, 0.040], dtype=np.float32)
    sand = np.array([0.20, 0.22, 0.24, 0.27, 0.26, 0.23], dtype=np.float32)
    for year in years:
        refl = np.broadcast_to(background[:, None, None], (6, h, w)).copy()
        # Stable sand-like patch exists in all years and has high model score.
        refl[:, 18:30, 5:22] = sand[:, None, None]
        if year == 2024:
            refl[:, 4:14, 4:14] = dark_panel[:, None, None]
        # Remote black array appears in 2021 and persists.
        if year >= 2021:
            refl[:, 32:56, 34:58] = dark_panel[:, None, None]
        land_path = root / f"02_aligned_30m/land_{year}.tif"
        _write_raster(land_path, refl)
        aligned_rows.append({"Id": 6, "year": year, "path": str(land_path)})

        raw = np.full((h, w), 0.04, dtype=np.float32)
        calibrated = np.full((h, w), 0.03, dtype=np.float32)
        raw[18:30, 5:22] = 0.92
        calibrated[18:30, 5:22] = 0.88
        if year == 2024:
            raw[4:14, 4:14] = 0.95
            calibrated[4:14, 4:14] = 0.92
        # Intentionally low model score over the real remote site.
        unc = np.full((h, w), 0.10, dtype=np.float32)
        raw_path = root / f"07_probability_calibration/raw_{year}.tif"
        cal_path = root / f"07_probability_calibration/cal_{year}.tif"
        mask_path = root / f"07_probability_calibration/mask_{year}.tif"
        unc_path = root / f"07_probability_calibration/unc_{year}.tif"
        _write_raster(raw_path, raw)
        _write_raster(cal_path, calibrated)
        _write_raster(mask_path, (calibrated >= 0.5).astype(np.uint8), dtype="uint8", nodata=255)
        _write_raster(unc_path, unc)
        registry_rows.append({
            "Id": 6, "year": year, "inference_mode": "direct_pixel", "estimation_route": "direct_pixel",
            "calibrated_probability_path": str(cal_path), "calibrated_mask_path": str(mask_path),
            "raw_probability_path": str(raw_path), "epistemic_std_path": str(unc_path),
            "calibrated_predicted_pixels": int((calibrated >= 0.5).sum()),
        })

    pd.DataFrame(aligned_rows).to_csv(root / "02_aligned_30m/aligned_raster_registry.csv", index=False)
    pd.DataFrame(registry_rows).to_csv(root / "07_probability_calibration/calibrated_inference_registry.csv", index=False)
    pd.DataFrame([{"Id": 6, "endmember_path": str(endmember_path)}]).to_csv(root / "04_unit_quality/unit_quality_registry.csv", index=False)
    pd.DataFrame([{
        "Id": 6, "pv_fraction_path": str(fraction_path), "label_path": str(label_path),
        "anchor_support_path": str(support_path),
    }]).to_csv(root / "03_supervision/supervision_registry.csv", index=False)

    cfg = {
        "project": {"random_seed": 2027},
        "paths": {"output_root": str(root)},
        "study": {"id_start": 6, "id_end": 6, "selected_unit_ids": [6], "start_year": 2019, "end_year": 2024, "reference_year": 2024},
        "test_mode": {"enabled_when_selected_unit_ids": True},
        "processing": {"show_progress": False, "overwrite": True, "compress": "DEFLATE"},
        "calibration": {"mask_nodata": 255},
        "supervision": {"strong_positive_fraction": 0.7},
        "site_first_refinement": {
            "enabled": True, "tile_size": 64, "halo_pixels": 8, "prototype_tile_size": 64,
            "maximum_prototype_samples": 10000, "positive_prototype_clusters": 1, "sand_prototype_clusters": 1,
            "diagnostic_unit_ids": [6], "write_diagnostics": True,
            "site_core_dark_threshold": 0.36, "site_core_contrast_threshold": 0.04, "site_core_dark_override": 0.48,
            "site_core_maximum_sand_risk": 0.85, "site_growth_dark_threshold": 0.25, "site_growth_maximum_sand_risk": 0.88,
            "site_opening_iterations": 0, "site_closing_iterations": 1, "site_group_bridge_iterations": 1,
            "site_dark_pixel_threshold": 0.32, "minimum_site_pixels": 10, "minimum_core_fraction": 0.10,
            "minimum_dark_fraction": 0.30, "minimum_solid_fill": 0.30, "minimum_oriented_fill": 0.02,
            "maximum_elongation": 60.0, "maximum_perimeter_density": 0.96, "maximum_edge_only_score": 0.96,
            "minimum_mean_dark_score": 0.30, "minimum_q75_dark_score": 0.34, "minimum_mean_local_contrast": 0.02,
            "minimum_boundary_contrast": -0.05, "maximum_confuser": 0.70, "maximum_site_sand_risk": 0.70,
            "site_minimum_persistence": 0.32, "site_minimum_onset": 0.15,
            "large_site_minimum_pixels": 30, "large_site_minimum_dark_score": 0.35,
            "large_site_minimum_q75_dark_score": 0.38, "large_site_minimum_persistence": 0.35,
            "large_site_maximum_confuser": 0.70, "large_site_maximum_sand_risk": 0.70,
            "large_site_minimum_dark_fraction": 0.25, "large_site_minimum_solid_fill": 0.25,
            "large_site_maximum_edge_only_score": 0.98, "persistent_sand_rejection_threshold": 0.55,
            "track_sand_rejection_threshold": 0.55, "track_sand_maximum_onset": 0.30,
            "track_sand_dark_override": 0.80, "first_support_minimum_onset": 0.18,
            "first_support_minimum_persistence": 0.30, "first_support_dark_override": 0.55,
            "first_support_dark_persistence": 0.40, "first_support_maximum_sand_risk": 0.70,
            "maximum_adjacency_completion_pixels": 2, "adjacency_minimum_dark_score": 0.25,
            "adjacency_maximum_sand_risk": 0.70, "adjacency_minimum_model_score": 0.60,
            "adjacency_minimum_positive_similarity": 0.50, "adjacency_dark_override": 0.55,
            "temporal_rescue_minimum_dark_score": 0.25, "temporal_rescue_maximum_sand_risk": 0.75,
            "singleton_minimum_onset": 0.50, "singleton_minimum_persistence": 0.50,
            "singleton_minimum_dark_score": 0.50, "singleton_maximum_sand_risk": 0.60,
        },
    }
    result = refine_site_first_probabilities(cfg)
    assert len(result) == len(years)
    assert (root / "07_site_first_refinement/site_first_refinement_status.json").exists()
    row_2022 = result.loc[result["year"].eq(2022)].iloc[0]
    with rasterio.open(row_2022["calibrated_mask_path"]) as src:
        mask = src.read(1)
    with rasterio.open(row_2022["accept_source_path"]) as src:
        cls = src.read(1)
    assert np.mean(mask[32:56, 34:58] == 1) > 0.70
    assert np.any(cls[32:56, 34:58] == CLASS_SITE_FIRST_REMOTE)
    # The model-only sand patch must not survive as final PV.
    assert np.mean(mask[18:30, 5:22] == 1) < 0.10
    row_2019 = result.loc[result["year"].eq(2019)].iloc[0]
    with rasterio.open(row_2019["calibrated_mask_path"]) as src:
        early = src.read(1)
    assert np.mean(early[32:56, 34:58] == 1) < 0.05
