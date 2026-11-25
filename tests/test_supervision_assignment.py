from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from pvchange.preparation import build_supervision


def _write_reference(path: Path, left: float) -> None:
    data = np.full((6, 10, 10), 0.2, dtype=np.float32)
    data[3] = 0.3
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=6,
        dtype="float32", crs="EPSG:32650",
        transform=from_origin(left, 300, 30, 30), nodata=-9999.0,
    ) as dst:
        dst.write(data)


def test_positive_only_spatial_fallback_and_subpixel_anchor(tmp_path: Path) -> None:
    root = tmp_path
    aligned_dir = root / "02_aligned_30m"
    aligned_dir.mkdir(parents=True)
    grid_rows, aligned_rows = [], []
    for region_id in range(1, 5):
        left = (region_id - 1) * 1000
        raster_path = aligned_dir / f"unit_{region_id:03d}_2024.tif"
        _write_reference(raster_path, left)
        grid_rows.append({
            "Id": region_id, "status": "ok", "reference_raster": str(raster_path),
            "target_crs": "EPSG:32650", "target_resolution_m": 30,
            "width": 10, "height": 10, "left": left, "bottom": 0,
            "right": left + 300, "top": 300,
            "transform_a": 30, "transform_b": 0, "transform_c": left,
            "transform_d": 0, "transform_e": -30, "transform_f": 300,
        })
        aligned_rows.append({"Id": region_id, "year": 2024, "path": str(raster_path), "status": "written"})
    pd.DataFrame(grid_rows).to_csv(aligned_dir / "grid_registry.csv", index=False)
    pd.DataFrame(aligned_rows).to_csv(aligned_dir / "aligned_raster_registry.csv", index=False)

    # Invalid IDs force authoritative spatial assignment. The first polygon is
    # subpixel-sized and must still become a positive anchor.
    gdf = gpd.GeoDataFrame(
        {"Id": [999, 999, 999]},
        geometry=[box(5, 5, 10, 10), box(1030, 30, 1150, 150), box(2050, 50, 2200, 200)],
        crs="EPSG:32650",
    )
    shp_path = root / "pv.shp"
    gdf.to_file(shp_path)

    cfg = {
        "project": {"random_seed": 2027},
        "paths": {"output_root": str(root), "reference_shp": str(shp_path)},
        "study": {"id_start": 1, "id_end": 4, "start_year": 2011, "end_year": 2025, "reference_year": 2024},
        "processing": {"show_progress": False, "compress": "DEFLATE"},
        "supervision": {
            "id_field": "Id", "fractional_rasterization_oversample": 8,
            "strong_positive_fraction": 0.70, "weak_positive_fraction": 0.20,
            "ambiguous_outer_buffer_m": 120, "hard_negative_min_distance_m": 500,
            "hard_negative_evidence_quantile": 0.10, "ignore_value": 255,
            "minimum_positive_regions": 3, "adaptive_positive_fraction_floor": 1 / 64,
            "split": {
                "train_fraction": 1/3, "calibration_fraction": 1/3, "test_fraction": 1/3,
                "minimum_positive_train_units": 1, "minimum_positive_calibration_units": 1,
                "minimum_positive_test_units": 1,
            },
        },
        "training": {"minimum_valid_spatial_folds": 3},
    }
    registry = build_supervision(cfg)
    status = json.loads((root / "03_supervision" / "supervision_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "passed"
    assert status["units_with_positive_labels"] == 3
    assert "outside is unlabelled" in status["shp_semantics"]
    assert registry.loc[registry.Id.eq(1), "label_mode"].iloc[0] == "adaptive_subpixel_fraction"
    assert registry.loc[registry.Id.eq(1), "strong_positive_pixels"].iloc[0] >= 1
    assert registry.loc[registry.Id.eq(4), "split_role"].iloc[0] == "inference_only"
    assert registry.loc[registry.Id.eq(4), "outside_shp_semantics"].iloc[0] == "unlabelled_not_negative"
