from __future__ import annotations

import numpy as np
import pandas as pd

from pvexp2.forecast import recursive_projection, rolling_growth_hindcast


def _panel(n_regions: int = 4, start: int = 2011, end: int = 2025) -> pd.DataFrame:
    rows = []
    for rid in range(1, n_regions + 1):
        area = 0.0
        valid = 10.0 + rid
        for year in range(start, end + 1):
            gain = 0.0
            if year >= 2015 + (rid % 2) and (year - rid) % 3 == 0:
                gain = 0.08 + 0.01 * rid
            area += gain
            rows.append({
                "region": rid,
                "year": year,
                "PV_Cover_Area_km2": area,
                "PV_Cover_Ratio": area / valid,
                "Valid_Area_km2": valid,
                "Annual_Installed_Gain_km2": gain,
                "PVIntegrated_PAVG_Optimized": 0.35 + 0.01 * rid + 0.002 * (year - start),
                "PVIntegrated_EIPAVG": 0.40 + 0.008 * rid + 0.001 * (year - start),
                "ENV_precipitation": 100 + 2 * rid + 0.5 * (year - start),
                "ENV_temperature": 15 + 0.2 * rid + 0.02 * (year - start),
                "ENV_potential_evapotranspiration": 400 + rid,
                "ENV_soil_moisture": 0.15 + 0.005 * rid,
                "ENV_radiation": 200 + 3 * rid,
                "ENV_wind_speed": 4 + 0.1 * rid,
            })
    return pd.DataFrame(rows)


def test_two_part_projection_is_cumulative_and_bounded():
    annual = _panel()
    forecast, audit = recursive_projection(
        annual,
        targets=["PV_Cover_Area_km2", "PV_Cover_Ratio", "PVIntegrated_PAVG_Optimized"],
        forecast_year=2028,
        model_names=["ridge", "extra_trees", "hgb"],
        random_state=7,
    )
    assert not forecast.empty
    assert audit["audit_passed"].fillna(True).all()
    med = forecast[forecast["scenario"].eq("median")]
    area = med[med["target"].eq("PV_Cover_Area_km2")].sort_values(["region", "year"])
    assert len(area) == 4 * 3
    diffs = area.groupby("region")["projected"].diff().dropna()
    assert (diffs >= -1e-10).all()
    ratio = med[med["target"].eq("PV_Cover_Ratio")]
    assert ratio["projected"].between(0, 1).all()
    assert set(med[med["target"].eq("PV_Cover_Area_km2")]["projection_method"]) == {"two_part_multisource_growth"}


def test_two_part_growth_hindcast_is_strict_and_returns_metrics():
    annual = _panel(n_regions=5)
    records, metrics, audit = rolling_growth_hindcast(
        annual,
        model_names=["ridge", "extra_trees", "hgb"],
        min_history=4,
        random_state=11,
    )
    assert not records.empty
    assert not metrics.empty
    assert audit["audit_passed"].all()
    assert {"Annual_Installed_Gain_km2", "PV_Cover_Area_km2", "PV_Cover_Ratio"}.issubset(records["target"].unique())
    assert np.isfinite(records["predicted"]).all()


def test_rolling_features_never_cross_region_boundaries():
    from pvexp2.features import add_strict_lag_features
    frame = pd.DataFrame({
        "region": [1, 1, 1, 2, 2, 2],
        "year": [2011, 2012, 2013, 2011, 2012, 2013],
        "PV_Cover_Area_km2": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        "PV_Cover_Ratio": [0.01, 0.02, 0.03, 0.10, 0.20, 0.30],
    })
    feat = add_strict_lag_features(frame, rolling_window=3)
    first_region2 = feat[(feat["region"] == 2) & (feat["year"] == 2011)].iloc[0]
    assert np.isnan(first_region2["feat__PV_Cover_Area_km2__lag1"])
    assert np.isnan(first_region2["feat__PV_Cover_Area_km2__rolling3_mean"])
    region2_2012 = feat[(feat["region"] == 2) & (feat["year"] == 2012)].iloc[0]
    assert region2_2012["feat__PV_Cover_Area_km2__rolling3_mean"] == 100.0
