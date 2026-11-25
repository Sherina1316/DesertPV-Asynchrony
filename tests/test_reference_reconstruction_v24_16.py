import numpy as np

from pvchange.reference_reconstruction import _filter_reference_inventory
from pvchange.reference_temporal import _ordered_change_point, _persistent_outside_tracks


def temporal_settings():
    return {
        'change_point_transition_penalty': 0.18,
        'post_installation_high_probability': 0.46,
        'minimum_post_installation_mean_probability': 0.44,
        'minimum_pre_post_probability_gain': 0.11,
        'minimum_post_installation_support_years': 2,
        'strong_post_installation_override': 0.72,
        'preexisting_probability_threshold': 0.58,
        'end_year_singleton_probability': 0.90,
    }


def test_2024_anchor_is_not_backfilled_without_historical_evidence():
    p = np.full((15, 1, 1), 0.08, dtype=np.float32)
    p[13, 0, 0] = 0.99  # 2024 hard endpoint
    p[14, 0, 0] = 0.91
    valid = np.ones_like(p, dtype=bool)
    inventory = np.ones((1, 1), dtype=bool)
    onset, confidence, _, _ = _ordered_change_point(p, valid, inventory, 13, temporal_settings())
    assert int(onset[0, 0]) == 13
    assert confidence[0, 0] >= 0


def test_real_2020_installation_is_reconstructed_from_change_evidence():
    p = np.full((15, 1, 1), 0.07, dtype=np.float32)
    p[9:, 0, 0] = 0.86  # 2020--2025
    valid = np.ones_like(p, dtype=bool)
    inventory = np.ones((1, 1), dtype=bool)
    onset, _, post, pre = _ordered_change_point(p, valid, inventory, 13, temporal_settings())
    assert int(onset[0, 0]) == 9
    assert post[0, 0] > pre[0, 0]


def test_preexisting_2011_site_requires_first_two_year_support():
    p = np.full((15, 1, 1), 0.80, dtype=np.float32)
    valid = np.ones_like(p, dtype=bool)
    inventory = np.ones((1, 1), dtype=bool)
    onset, _, _, _ = _ordered_change_point(p, valid, inventory, 13, temporal_settings())
    assert int(onset[0, 0]) == 0


def test_outside_inventory_requires_temporal_persistence():
    candidate = np.zeros((15, 2, 2), dtype=bool)
    candidate[9:13, 0, 0] = True
    candidate[5, 1, 1] = True  # isolated sand-like false alarm
    p = candidate.astype(np.float32) * 0.8 + 0.05
    valid = np.ones_like(candidate, dtype=bool)
    inventory = np.zeros((2, 2), dtype=bool)
    active, onset = _persistent_outside_tracks(candidate, p, valid, inventory, temporal_settings())
    assert active[9:13, 0, 0].all()
    assert int(onset[0, 0]) == 9
    assert not active[:, 1, 1].any()


def test_reference_inventory_accepts_changed_remote_site_and_rejects_stable_sand():
    shape = (30, 40)
    anchor = np.zeros(shape, dtype=bool)
    anchor[3:8, 3:8] = True
    raw = np.zeros(shape, dtype=bool)
    raw[3:8, 3:8] = True
    raw[10:20, 23:35] = True  # large remote site
    raw[21:23, 3:18] = True   # thin sand stripe
    score = np.zeros(shape, dtype=np.float32)
    change = np.zeros(shape, dtype=np.float32)
    sand = np.zeros(shape, dtype=np.float32)
    ref = np.zeros(shape, dtype=np.float32)
    score[raw] = 0.75
    change[10:20, 23:35] = 0.78
    ref[10:20, 23:35] = 0.76
    sand[10:20, 23:35] = 0.18
    change[21:23, 3:18] = 0.05
    ref[21:23, 3:18] = 0.58
    sand[21:23, 3:18] = 0.90
    cfg = {'reference_reconstruction': {
        'reference_closing_iterations': 1,
        'reference_bridge_iterations': 0,
        'minimum_reference_object_pixels': 8,
        'large_reference_object_pixels': 40,
        'minimum_reference_object_score': 0.50,
        'minimum_reference_change_support': 0.32,
        'maximum_reference_sand_risk': 0.50,
        'minimum_reference_solidity': 0.35,
        'maximum_reference_perimeter_density': 0.95,
        'large_object_minimum_reference_score': 0.44,
        'large_object_minimum_change_support': 0.40,
        'large_object_maximum_sand_risk': 0.58,
    }}
    inventory, rejection, _ = _filter_reference_inventory(anchor, raw, score, change, sand, ref, cfg)
    assert inventory[4, 4]
    assert inventory[14, 28]
    assert not inventory[21, 10]
    assert rejection[21, 10] != 0


def test_temporal_raster_integration_keeps_2024_only_anchor_out_of_2011(tmp_path, monkeypatch):
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin
    import pvchange.reference_temporal as rt

    root = tmp_path / 'out'
    cal = root / '07_probability_calibration'
    cal.mkdir(parents=True)
    data = tmp_path / 'data'
    data.mkdir()
    transform = from_origin(0, 360, 30, 30)
    profile = {
        'driver': 'GTiff', 'height': 12, 'width': 12, 'count': 1,
        'dtype': 'float32', 'crs': 'EPSG:32647', 'transform': transform,
        'nodata': -9999.0,
    }
    mask_profile = dict(profile, dtype='uint8', nodata=255)
    years = list(range(2011, 2026))
    rows = []
    inventory = np.zeros((12, 12), dtype=np.uint8)
    inventory[3:7, 3:7] = 1
    anchor = inventory.copy()
    sand = np.zeros((12, 12), dtype=np.float32)
    inv_path = data / 'inventory.tif'
    anc_path = data / 'anchor.tif'
    sand_path = data / 'sand.tif'
    with rasterio.open(inv_path, 'w', **mask_profile) as dst:
        dst.write(inventory, 1)
    with rasterio.open(anc_path, 'w', **mask_profile) as dst:
        dst.write(anchor, 1)
    with rasterio.open(sand_path, 'w', **profile) as dst:
        dst.write(sand, 1)
    for i, year in enumerate(years):
        p = np.full((12, 12), 0.05, dtype=np.float32)
        m = np.zeros((12, 12), dtype=np.uint8)
        if year >= 2024:
            p[3:7, 3:7] = 0.95
            m[3:7, 3:7] = 1
        pp = data / f'p_{year}.tif'
        mp = data / f'm_{year}.tif'
        with rasterio.open(pp, 'w', **profile) as dst:
            dst.write(p, 1)
        with rasterio.open(mp, 'w', **mask_profile) as dst:
            dst.write(m, 1)
        rows.append({
            'Id': 6, 'year': year, 'calibrated_probability_path': str(pp),
            'calibrated_mask_path': str(mp), 'reference_inventory_path': str(inv_path),
            'reference_anchor_path': str(anc_path), 'stable_sand_risk_path': str(sand_path),
            'inference_mode': 'direct_pixel', 'estimation_route': 'direct_pixel',
            'calibrated_threshold': 0.5, 'confirmed_anchor_area_km2': 0.0144,
        })
    pd.DataFrame(rows).to_csv(cal / 'calibrated_inference_registry.csv', index=False)
    monkeypatch.setattr(rt, 'infer_area_only_units', lambda cfg, direct: (pd.DataFrame(), pd.DataFrame()))
    cfg = {
        'paths': {'output_root': str(root)},
        'study': {'id_start': 1, 'id_end': 142, 'selected_unit_ids': [6], 'start_year': 2011, 'end_year': 2025, 'reference_year': 2024},
        'processing': {'overwrite': True, 'show_progress': False, 'compress': 'DEFLATE'},
        'temporal_decoding': {'mask_nodata': 255, 'synchrony_minimum_peak_gain_km2': 0.01, 'maximum_synchronous_peak_fraction': 1.0},
        'reference_temporal_reconstruction': temporal_settings() | {
            'tile_size': 8, 'halo_pixels': 2, 'minimum_outside_component_pixels': 2,
            'model_only_reference_probability': 0.5, 'stable_sand_exclusion_threshold': 0.68,
        },
    }
    registry = rt.decode_reference_constrained_change(cfg)
    row_2011 = registry.loc[registry['year'] == 2011].iloc[0]
    row_2024 = registry.loc[registry['year'] == 2024].iloc[0]
    with rasterio.open(row_2011['active_mask_path']) as src:
        active_2011 = src.read(1)
    with rasterio.open(row_2024['active_mask_path']) as src:
        active_2024 = src.read(1)
    assert not np.any(active_2011[3:7, 3:7] == 1)
    assert np.all(active_2024[3:7, 3:7] == 1)
