import numpy as np

from pvchange.authoritative_temporal import _ordered_change_point, _persistent_outside_tracks


def test_inside_reference_without_historical_evidence_defaults_to_2024():
    p=np.full((14,2,2),0.08,np.float32); v=np.ones_like(p,bool); inv=np.zeros((2,2),bool); inv[0,0]=True
    p[13,0,0]=1.0
    onset,_,_,_=_ordered_change_point(p,v,inv,13,{"minimum_post_installation_mean_probability":0.46,"minimum_post_installation_support_years":2})
    assert onset[0,0]==13


def test_outside_is_forced_non_pv_in_2024_but_2025_new_site_allowed():
    n=15; ref=13
    c=np.zeros((n,3,3),bool); p=np.zeros((n,3,3),np.float32); v=np.ones_like(c,bool); inv=np.zeros((3,3),bool)
    c[ref,1,1]=True; p[ref,1,1]=0.99
    c[ref+1,1,1]=True; p[ref+1,1,1]=0.96
    active,onset,ret=_persistent_outside_tracks(c,p,v,inv,ref,{"historical_outside_probability":0.62,"end_year_singleton_probability":0.92})
    assert not active[ref,1,1]
    assert active[ref+1,1,1]


def test_historical_outside_requires_multiyear_track_and_retires_before_2024():
    n=15; ref=13
    c=np.zeros((n,2,2),bool); p=np.zeros((n,2,2),np.float32); v=np.ones_like(c,bool); inv=np.zeros((2,2),bool)
    c[4:7,0,0]=True; p[4:7,0,0]=0.8
    active,onset,ret=_persistent_outside_tracks(c,p,v,inv,ref,{"historical_outside_probability":0.62})
    assert active[4:7,0,0].all()
    assert not active[ref,0,0]
    assert onset[0,0]==4
    assert ret[0,0] < ref


def test_authoritative_temporal_raster_endpoint_and_onset(tmp_path, monkeypatch):
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin
    import pvchange.authoritative_temporal as at

    root = tmp_path / "out"
    cal = root / "07_probability_calibration"
    cal.mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()
    transform = from_origin(0, 360, 30, 30)
    profile = {
        "driver": "GTiff", "height": 12, "width": 12, "count": 1,
        "dtype": "float32", "crs": "EPSG:32647", "transform": transform,
        "nodata": -9999.0,
    }
    mask_profile = dict(profile, dtype="uint8", nodata=255)
    years = list(range(2011, 2026))
    reference_index = years.index(2024)

    anchor = np.zeros((12, 12), dtype=np.uint8)
    anchor[2:6, 2:6] = 1                 # installation supported only from 2020
    anchor[7:10, 7:10] = 1              # no historical support; must default to 2024
    fraction = anchor.astype(np.float32)
    fraction[2, 2] = 0.25                # boundary area must be fraction weighted
    sand = np.zeros((12, 12), dtype=np.float32)

    def write(path, array, prof):
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(array, 1)

    inventory_path = data / "inventory.tif"
    anchor_path = data / "anchor.tif"
    fraction_path = data / "fraction.tif"
    sand_path = data / "sand.tif"
    write(inventory_path, anchor, mask_profile)
    write(anchor_path, anchor, mask_profile)
    write(fraction_path, fraction, profile)
    write(sand_path, sand, profile)

    rows = []
    for year in years:
        p = np.full((12, 12), 0.04, dtype=np.float32)
        candidate = np.zeros((12, 12), dtype=np.uint8)
        if year >= 2020:
            p[2:6, 2:6] = 0.88
            candidate[2:6, 2:6] = 1
        if year == 2024:
            # The second object is forced only at the exact reference endpoint.
            p[7:10, 7:10] = 1.0
            candidate[7:10, 7:10] = 1
        if year == 2025:
            p[7:10, 7:10] = 0.90
            candidate[7:10, 7:10] = 1
        pp = data / f"p_{year}.tif"
        mp = data / f"m_{year}.tif"
        write(pp, p, profile)
        write(mp, candidate, mask_profile)
        rows.append({
            "Id": 6,
            "year": year,
            "calibrated_probability_path": str(pp),
            "calibrated_mask_path": str(mp),
            "reference_inventory_path": str(inventory_path),
            "reference_anchor_path": str(anchor_path),
            "reference_fraction_path": str(fraction_path),
            "stable_sand_risk_path": str(sand_path),
            "inference_mode": "direct_pixel",
            "estimation_route": "direct_pixel",
            "calibrated_threshold": 0.5,
            "confirmed_anchor_area_km2": float(fraction.sum() * 0.0009),
        })
    pd.DataFrame(rows).to_csv(cal / "calibrated_inference_registry.csv", index=False)

    monkeypatch.setattr(at, "infer_area_only_units", lambda cfg, direct: (pd.DataFrame(), pd.DataFrame()))
    cfg = {
        "paths": {"output_root": str(root)},
        "study": {
            "id_start": 1, "id_end": 142, "selected_unit_ids": [6],
            "start_year": 2011, "end_year": 2025, "reference_year": 2024,
        },
        "processing": {"overwrite": True, "show_progress": False, "compress": "DEFLATE"},
        "temporal_decoding": {
            "mask_nodata": 255,
            "synchrony_minimum_peak_gain_km2": 0.01,
            "maximum_synchronous_peak_fraction": 1.0,
        },
        "authoritative_temporal_reconstruction": {
            "tile_size": 8, "halo_pixels": 2,
            "change_point_transition_penalty": 0.22,
            "post_installation_high_probability": 0.48,
            "minimum_post_installation_mean_probability": 0.46,
            "minimum_pre_post_probability_gain": 0.13,
            "minimum_post_installation_support_years": 2,
            "strong_post_installation_override": 0.76,
            "preexisting_probability_threshold": 0.62,
            "model_only_reference_probability": 0.50,
            "historical_outside_probability": 0.62,
            "end_year_singleton_probability": 0.92,
            "minimum_outside_component_pixels": 2,
            "stable_sand_exclusion_threshold": 0.64,
        },
    }
    registry = at.decode_authoritative_reference_change(cfg)
    by_year = registry.set_index("year")

    with rasterio.open(by_year.loc[2011, "active_mask_path"]) as src:
        active_2011 = src.read(1)
    with rasterio.open(by_year.loc[2020, "active_mask_path"]) as src:
        active_2020 = src.read(1)
    with rasterio.open(by_year.loc[2024, "active_mask_path"]) as src:
        active_2024 = src.read(1)
    with rasterio.open(by_year.loc[2024, "change_year_path"]) as src:
        onset_year = src.read(1)

    assert not np.any(active_2011 == 1)
    assert np.all(active_2020[2:6, 2:6] == 1)
    assert not np.any(active_2020[7:10, 7:10] == 1)
    assert np.array_equal(active_2024 == 1, anchor == 1)
    assert np.all(onset_year[2:6, 2:6] == 2020)
    assert np.all(onset_year[7:10, 7:10] == 2024)

    expected_2024_area = float(fraction.sum() * 0.0009)
    assert np.isclose(float(by_year.loc[2024, "observed_active_area_km2"]), expected_2024_area)
