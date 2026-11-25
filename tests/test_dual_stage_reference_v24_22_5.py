from __future__ import annotations

import numpy as np

from pvchange.dual_stage_reference import (
    fuse_reference_tracks,
    reconstruct_internal_reference_buildout,
)
from pvchange.quality import _reference_internal_analysis_eligible


def _scene(onset: int, *, partial_until: int | None = None):
    years, height, width, reference_index = 15, 32, 32, 13
    inventory = np.zeros((height, width), dtype=bool)
    inventory[8:24, 8:24] = True
    left = np.zeros_like(inventory)
    left[8:24, 8:16] = True
    background = np.array([0.22, 0.24, 0.26, 0.32, 0.36, 0.30], dtype=np.float32)
    photovoltaic = np.array([0.08, 0.09, 0.10, 0.09, 0.12, 0.11], dtype=np.float32)
    reflectance = np.broadcast_to(background, (years, height, width, 6)).transpose(0, 3, 1, 2).copy()
    reflectance += np.random.default_rng(2027).normal(0.0, 0.003, reflectance.shape).astype(np.float32)
    raw_probability = np.full((years, height, width), 0.12, dtype=np.float32)
    for year in range(onset, years):
        footprint = left if partial_until is not None and year < partial_until else inventory
        for band in range(6):
            reflectance[year, band][footprint] = photovoltaic[band]
        raw_probability[year][footprint] = 0.22
    valid = np.ones((years, height, width), dtype=bool)
    candidate = np.zeros_like(valid)
    return raw_probability, valid, inventory, candidate, reference_index, reflectance, left


def test_full_mixed_pixel_parcel_is_recovered_from_true_image_onset():
    probability, valid, inventory, candidate, reference_index, reflectance, _ = _scene(5)
    result = reconstruct_internal_reference_buildout(
        probability,
        probability,
        valid,
        inventory,
        candidate,
        reference_index,
        {},
        reflectance=reflectance,
        reflectance_valid=valid,
    )
    assert not result.internal_active[:5].any()
    assert all(np.array_equal(result.internal_active[t], inventory) for t in range(5, 15))
    assert np.unique(result.component_onset_index[inventory]).tolist() == [5]


def test_partial_buildout_is_not_prematurely_filled_as_complete_parcel():
    probability, valid, inventory, candidate, reference_index, reflectance, left = _scene(5, partial_until=8)
    result = reconstruct_internal_reference_buildout(
        probability,
        probability,
        valid,
        inventory,
        candidate,
        reference_index,
        {},
        reflectance=reflectance,
        reflectance_valid=valid,
    )
    assert np.array_equal(result.internal_active[5], left)
    assert np.array_equal(result.internal_active[7], left)
    assert np.array_equal(result.internal_active[8], inventory)


def test_reference_endpoint_is_not_blindly_copied_backwards():
    probability, valid, inventory, candidate, reference_index, reflectance, _ = _scene(13)
    result = reconstruct_internal_reference_buildout(
        probability,
        probability,
        valid,
        inventory,
        candidate,
        reference_index,
        {},
        reflectance=reflectance,
        reflectance_valid=valid,
    )
    assert not result.internal_active[:reference_index].any()
    assert np.array_equal(result.internal_active[reference_index], inventory)


def test_start_of_series_facility_remains_supported():
    probability, valid, inventory, candidate, reference_index, reflectance, _ = _scene(0)
    result = reconstruct_internal_reference_buildout(
        probability,
        probability,
        valid,
        inventory,
        candidate,
        reference_index,
        {},
        reflectance=reflectance,
        reflectance_valid=valid,
    )
    assert all(np.array_equal(result.internal_active[t], inventory) for t in range(15))
    assert np.unique(result.component_onset_index[inventory]).tolist() == [0]


def test_fusion_uses_internal_branch_when_global_detector_is_empty():
    probability, valid, inventory, candidate, reference_index, reflectance, _ = _scene(5)
    result = reconstruct_internal_reference_buildout(
        probability,
        probability,
        valid,
        inventory,
        candidate,
        reference_index,
        {},
        reflectance=reflectance,
        reflectance_valid=valid,
    )
    global_active = np.zeros_like(valid)
    fused, source = fuse_reference_tracks(
        global_active,
        result,
        inventory,
        candidate,
        reference_index,
        {},
    )
    assert not fused[:5].any()
    assert np.array_equal(fused[5], inventory)
    assert np.all(source[5][inventory] == 2)
    assert np.all(source[reference_index][inventory] == 4)


def test_authoritative_parcel_keeps_spatial_route_without_pure_anchor_pixels():
    row = {
        "has_confirmed_anchor": True,
        "valid_year_count_50pct": 8,
        "median_valid_fraction": 0.52,
        "reference_lsma_rmse_median": 0.31,
        "maximum_annual_median_spectral_jump": 0.40,
    }
    assert _reference_internal_analysis_eligible(row, {"quality_routing": {}})


def test_stage8_dual_stage_integration_writes_fused_historical_masks(tmp_path, monkeypatch):
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin
    import pvchange.authoritative_temporal as temporal

    root = tmp_path / "output"
    calibration = root / "07_probability_calibration"
    aligned_dir = root / "02_aligned_30m"
    calibration.mkdir(parents=True)
    aligned_dir.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    years = list(range(2011, 2026))
    reference_index = years.index(2024)
    transform = from_origin(0, 960, 30, 30)
    float_profile = {
        "driver": "GTiff", "height": 32, "width": 32, "count": 1,
        "dtype": "float32", "crs": "EPSG:32647", "transform": transform,
        "nodata": -9999.0,
    }
    mask_profile = dict(float_profile, dtype="uint8", nodata=255)
    six_profile = dict(float_profile, count=6)
    inventory = np.zeros((32, 32), dtype=np.uint8)
    inventory[8:24, 8:24] = 1
    fraction = inventory.astype(np.float32)
    sand = np.zeros((32, 32), dtype=np.float32)

    def write(path, values, profile):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(values if values.ndim == 3 else values[None, ...])

    inventory_path = data / "inventory.tif"
    anchor_path = data / "anchor.tif"
    fraction_path = data / "fraction.tif"
    sand_path = data / "sand.tif"
    write(inventory_path, inventory, mask_profile)
    write(anchor_path, inventory, mask_profile)
    write(fraction_path, fraction, float_profile)
    write(sand_path, sand, float_profile)

    background = np.array([0.22, 0.24, 0.26, 0.32, 0.36, 0.30], dtype=np.float32)
    photovoltaic = np.array([0.08, 0.09, 0.10, 0.09, 0.12, 0.11], dtype=np.float32)
    registry_rows = []
    aligned_rows = []
    for index, year in enumerate(years):
        probability = np.full((32, 32), 0.12, dtype=np.float32)
        candidate = np.zeros((32, 32), dtype=np.uint8)
        reflectance = np.broadcast_to(background[:, None, None], (6, 32, 32)).copy()
        if year >= 2016:
            probability[inventory == 1] = 0.22
            for band in range(6):
                reflectance[band][inventory == 1] = photovoltaic[band]
        refined_probability_path = data / f"refined_p_{year}.tif"
        raw_probability_path = data / f"raw_p_{year}.tif"
        mask_path = data / f"mask_{year}.tif"
        raw_mask_path = data / f"raw_mask_{year}.tif"
        aligned_path = data / f"landsat_{year}.tif"
        write(refined_probability_path, probability, float_profile)
        write(raw_probability_path, probability, float_profile)
        write(mask_path, candidate, mask_profile)
        write(raw_mask_path, candidate, mask_profile)
        write(aligned_path, reflectance, six_profile)
        registry_rows.append({
            "Id": 35,
            "year": year,
            "calibrated_probability_path": str(refined_probability_path),
            "calibrated_mask_path": str(mask_path),
            "pre_reference_reconstruction_probability_path": str(raw_probability_path),
            "pre_reference_reconstruction_mask_path": str(raw_mask_path),
            "reference_inventory_path": str(inventory_path),
            "reference_anchor_path": str(anchor_path),
            "reference_fraction_path": str(fraction_path),
            "stable_sand_risk_path": str(sand_path),
            "inference_mode": "anchor_guided_candidate",
            "estimation_route": "anchor_guided_candidate",
            "calibrated_threshold": 0.5,
            "confirmed_anchor_area_km2": float(fraction.sum() * 0.0009),
        })
        aligned_rows.append({"Id": 35, "year": year, "path": str(aligned_path), "status": "written"})
    pd.DataFrame(registry_rows).to_csv(calibration / "calibrated_inference_registry.csv", index=False)
    pd.DataFrame(aligned_rows).to_csv(aligned_dir / "aligned_raster_registry.csv", index=False)
    monkeypatch.setattr(temporal, "infer_area_only_units", lambda cfg, direct: (pd.DataFrame(), pd.DataFrame()))
    cfg = {
        "paths": {"output_root": str(root)},
        "study": {
            "id_start": 1, "id_end": 142, "selected_unit_ids": [35],
            "start_year": 2011, "end_year": 2025, "reference_year": 2024,
        },
        "processing": {"overwrite": True, "show_progress": False, "compress": "DEFLATE"},
        "temporal_decoding": {"mask_nodata": 255, "maximum_synchronous_peak_fraction": 1.0},
        "authoritative_temporal_reconstruction": {
            "dual_stage_reference_enabled": True,
            "tile_size": 32,
            "halo_pixels": 0,
            "minimum_outside_component_pixels": 2,
        },
    }
    frame = temporal.decode_authoritative_reference_change(cfg)
    row_2015 = frame.loc[frame["year"].eq(2015)].iloc[0]
    row_2016 = frame.loc[frame["year"].eq(2016)].iloc[0]
    with rasterio.open(row_2015.cumulative_mask_path) as src:
        assert not np.any(src.read(1) == 1)
    with rasterio.open(row_2016.cumulative_mask_path) as src:
        assert np.array_equal(src.read(1) == 1, inventory.astype(bool))
    assert row_2016.temporal_decoder == "v24_22_5_r4_dual_stage_reference_fusion"
    assert (root / "08_temporal_pv_change" / "dual_stage_internal_shp_buildout_30m" / "unit_035" / "unit_035_2016_internal_shp_buildout_30m.tif").is_file()


def test_authoritative_parcel_presence_is_separate_from_training_core():
    import pandas as pd
    from pvchange.preparation import _assign_supervision_roles

    registry = pd.DataFrame([
        {"Id": 35, "has_mapped_pv": True, "has_authoritative_2024_pv": True,
         "has_strong_training_core": False, "confirmed_anchor_area_km2": 2.3},
        {"Id": 107, "has_mapped_pv": True, "has_authoritative_2024_pv": True,
         "has_strong_training_core": True, "confirmed_anchor_area_km2": 1.4},
        {"Id": 6, "has_mapped_pv": True, "has_authoritative_2024_pv": True,
         "has_strong_training_core": True, "confirmed_anchor_area_km2": 0.9},
    ])
    cfg = {
        "project": {"random_seed": 2027},
        "supervision": {"split": {
            "train_fraction": 0.5,
            "calibration_fraction": 0.5,
            "test_fraction": 0.0,
            "minimum_positive_train_units": 1,
            "minimum_positive_calibration_units": 1,
            "minimum_positive_test_units": 0,
        }},
    }
    assigned = _assign_supervision_roles(registry, cfg).set_index("Id")
    # Unit 35 remains a certain 2024 PV parcel for inference, but cannot leak a
    # one-pixel/mixed-pixel fallback seed into formal supervised roles.
    assert bool(assigned.loc[35, "has_mapped_pv"])
    assert assigned.loc[35, "split_role"] == "inference_only"
    assert set(assigned.loc[[6, 107], "split_role"]) == {"train", "calibration"}
