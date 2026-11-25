from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import yaml

from pvchange.preparation import build_patch_dataset


def _write_raster(path: Path, data: np.ndarray, nodata, dtype: str) -> None:
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": "EPSG:32648",
        "transform": from_origin(500000, 4000000, 30, 30),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def test_low_valid_positive_preserved_and_all_ignore_unit_audited(tmp_path: Path) -> None:
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    aligned_dir = output / "02_aligned_30m"
    supervision_dir = output / "03_supervision"
    aligned_dir.mkdir(parents=True)
    supervision_dir.mkdir(parents=True)

    aligned_rows = []
    supervision_rows = []
    rng = np.random.default_rng(7)
    for unit in range(1, 5):
        for year in range(2011, 2026):
            raster_path = tmp_path / f"unit_{unit:03d}_{year}.tif"
            reflectance = rng.uniform(0.05, 0.45, size=(6, 16, 16)).astype(np.float32)
            _write_raster(raster_path, reflectance, -9999.0, "float32")
            aligned_rows.append({"Id": unit, "year": year, "path": str(raster_path)})

        label = np.zeros((1, 16, 16), dtype=np.uint8)
        if unit in (1, 2):
            label[:] = 255
            label[0, 0:2, 0:2] = 1
            label[0, 2:4, 0:4] = 0
        elif unit == 4:
            label[:] = 255
        label_path = tmp_path / f"unit_{unit:03d}_label.tif"
        _write_raster(label_path, label, 255, "uint8")
        supervision_rows.append({
            "Id": unit,
            "label_path": str(label_path),
            "strong_positive_pixels": int((label == 1).sum()),
            "has_mapped_pv": bool((label == 1).any()),
        })

    pd.DataFrame(aligned_rows).to_csv(aligned_dir / "aligned_raster_registry.csv", index=False)
    supervision_frame = pd.DataFrame(supervision_rows)
    supervision_frame["split_role"] = "train"
    supervision_frame["confirmed_anchor_area_km2"] = supervision_frame["strong_positive_pixels"] * 0.0009
    supervision_frame.to_csv(supervision_dir / "supervision_registry.csv", index=False)

    quality_dir = output / "04_unit_quality"
    quality_dir.mkdir(parents=True)
    quality_rows = []
    default_endmembers = np.array([
        [0.42, 0.08, 0.05, 0.0],
        [0.47, 0.07, 0.09, 0.0],
        [0.52, 0.06, 0.05, 0.0],
        [0.55, 0.05, 0.48, 0.0],
        [0.48, 0.04, 0.22, 0.0],
        [0.42, 0.03, 0.12, 0.0],
    ], dtype=np.float32)
    for unit in range(1, 5):
        endmember_path = tmp_path / f"unit_{unit:03d}_endmembers.npz"
        np.savez_compressed(endmember_path, endmembers=default_endmembers)
        quality_rows.append({
            "Id": unit, "direct_detection_eligible": True, "estimation_route": "direct_pixel",
            "quality_score": 1.0, "quality_reasons": "", "endmember_path": str(endmember_path),
        })
    pd.DataFrame(quality_rows).to_csv(quality_dir / "unit_quality_registry.csv", index=False)

    cfg = {
        "paths": {"output_root": str(output), "cache_root": str(cache)},
        "study": {"id_start": 1, "id_end": 4, "start_year": 2011, "end_year": 2025, "reference_year": 2025},
        "patches": {
            "patch_size": 16,
            "stride": 16,
            "minimum_valid_fraction": 0.70,
            "minimum_valid_pixels_per_patch": 4,
            "allow_training_unit_exclusion": True,
            "minimum_training_regions": 3,
            "maximum_negative_to_positive_ratio": 2.0,
            "maximum_patches_per_unit": 8,
            "maximum_cached_pixels_per_unit": 256,
            "spatial_folds": 2,
        },
        "supervision": {"ignore_value": 255, "minimum_positive_pixels_per_patch": 6},
        "project": {"random_seed": 42},
        "features": {
            "channels": [
                "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "NDVI", "EVI",
                "NDMI", "NDBI", "MNDWI", "BSI", "TC_Brightness", "TC_Greenness",
                "TC_Wetness", "PV_Spectral_Evidence",
            ],
            "add_valid_mask_channel": True,
        },
        "ccdc": {
            "indices": ["NDVI", "NDMI", "NDBI", "BSI", "TC_Brightness", "PV_Spectral_Evidence"],
            "minimum_observations": 6,
            "break_minimum_segment_years": 3,
        },
        "training": {
            "pixels_per_patch": 64,
            "positive_negative_ratio": 1.5,
            "minimum_valid_spatial_folds": 2,
        },
        "processing": {"show_progress": False},
    }

    build_patch_dataset(cfg)

    audit = pd.read_csv(output / "04_patch_dataset" / "training_unit_audit.csv")
    patches = pd.read_csv(output / "04_patch_dataset" / "patch_index.csv")
    status = yaml.safe_load((output / "04_patch_dataset" / "patch_dataset_status.json").read_text())

    assert status["status"] == "passed"
    assert status["audited_regions"] == 4
    assert status["training_regions"] == 3
    assert status["excluded_training_regions"] == 1
    assert audit.loc[audit.Id.eq(4), "training_status"].iloc[0] == "excluded"
    assert set(patches.loc[patches.Id.isin([1, 2]), "selection_policy"]) == {"positive_preservation"}
