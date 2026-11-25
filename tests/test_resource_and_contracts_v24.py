from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from pvchange.future import _dynamic_forecast
from pvchange.resource_control import MemorySnapshot, resource_headroom_ok

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pvchange_run_all_v24", ROOT / "run_all.py")
assert SPEC is not None and SPEC.loader is not None
RUN_ALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_ALL)


def _cfg() -> dict:
    return {
        "study": {"id_start": 1, "id_end": 1, "start_year": 2011, "end_year": 2012, "future_year": 2050},
        "future_2050": {
            "scenarios": ["conservative", "transition", "accelerated"],
            "backtest_holdout_years": 1,
            "scenario_growth_multipliers": {"conservative": 0.7, "transition": 1.0, "accelerated": 1.3},
            "maximum_total_pv_fraction_per_unit": 0.35,
        },
        "resource_control": {
            "minimum_available_memory_gb": 16.0,
            "minimum_swap_free_gb": 0.25,
        },
        "annual_visualization": {"write_individual_year_maps": True, "write_annual_atlas": True},
    }


def test_resource_gate_ignores_summed_competing_rss() -> None:
    snapshot = MemorySnapshot(
        total_gb=128.0,
        available_gb=40.0,
        swap_total_gb=8.0,
        swap_free_gb=1.0,
        competing_python_rss_gb=408.0,
        competing_processes=[{"pid": 1, "rss_mb": 9000.0, "command": "python train.py"}],
    )
    ok, reasons = resource_headroom_ok(snapshot, _cfg())
    assert ok
    assert reasons == []


def test_future_forecast_uses_cumulative_installed_series() -> None:
    group = pd.DataFrame({
        "year": [2011, 2012, 2013, 2014],
        "pv_area_km2": [1.0, 0.8, 0.5, 0.2],
        "cumulative_installed_area_km2": [1.0, 2.0, 3.0, 4.0],
    })
    forecasts, diagnostics = _dynamic_forecast(group, _cfg(), unit_area_km2=100.0)
    assert diagnostics["current_area_km2"] == 4.0
    assert all(item["target_area_km2"] >= 4.0 for item in forecasts.values())


def test_stage8_contract_requires_dual_state_and_area_order(tmp_path: Path) -> None:
    out = tmp_path
    stage = out / "08_temporal_pv_change"
    stage.mkdir(parents=True)
    status = {
        "status": "passed",
        "no_data_is_never_decoded_as_pv": True,
        "test_units_anchored": False,
        "area_only_pixel_maps_fabricated": False,
        "direct_pixel_units": 1,
        "anchor_guided_candidate_units": 0,
        "area_only_units": 0,
        "cumulative_installed_state_is_persistent": True,
        "observed_active_state_allows_consecutive_negative_retirement": True,
        "image_quality_gaps_do_not_create_retirement": True,
        "preexisting_start_of_series_facilities_supported": True,
        "reference_anchor_forced_in_reference_year": True,
        "reference_anchor_copied_to_historical_years_without_change_evidence": False,
        "model_only_diagnostic_product_retained": True,
    }
    (stage / "temporal_decoding_status.json").write_text(json.dumps(status))
    active = tmp_path / "active.tif"; active.write_bytes(b"x")
    cumulative = tmp_path / "cumulative.tif"; cumulative.write_bytes(b"x")
    model_only = tmp_path / "model_only.tif"; model_only.write_bytes(b"x")
    frame = pd.DataFrame([
        {"Id": 1, "year": 2011, "estimation_mode": "direct_pixel", "active_mask_path": active,
         "cumulative_mask_path": cumulative, "observed_active_area_km2": 1.0,
         "cumulative_installed_area_km2": 2.0, "model_only_cumulative_installed_area_km2": 1.8,
         "anchor_constrained_added_area_km2": 0.2, "model_only_cumulative_mask_path": model_only,
         "temporal_candidate_path": "a", "morphology_removed_path": "b",
         "valid_year_count_path": "c", "temporal_exclusion_code_path": "d",
         "anchor_backcast_path": "e", "anchor_forced_reference_path": "f",
         "anchor_backcast_object_fill_path": "g", "anchor_reference_path": "h"},
        {"Id": 1, "year": 2012, "estimation_mode": "direct_pixel", "active_mask_path": active,
         "cumulative_mask_path": cumulative, "observed_active_area_km2": 1.5,
         "cumulative_installed_area_km2": 2.5, "model_only_cumulative_installed_area_km2": 2.3,
         "anchor_constrained_added_area_km2": 0.2, "model_only_cumulative_mask_path": model_only,
         "temporal_candidate_path": "a", "morphology_removed_path": "b",
         "valid_year_count_path": "c", "temporal_exclusion_code_path": "d",
         "anchor_backcast_path": "e", "anchor_forced_reference_path": "f",
         "anchor_backcast_object_fill_path": "g", "anchor_reference_path": "h"},
    ])
    frame.to_csv(stage / "temporal_mask_registry.csv", index=False)
    ok, problems = RUN_ALL.coverage_valid(8, out, _cfg())
    assert ok, problems
    frame.loc[0, "observed_active_area_km2"] = 3.0
    frame.to_csv(stage / "temporal_mask_registry.csv", index=False)
    ok, problems = RUN_ALL.coverage_valid(8, out, _cfg())
    assert not ok
    assert any("exceeds cumulative" in problem for problem in problems)


def test_stage17_contract_requires_colour_safe_dual_state_audit(tmp_path: Path) -> None:
    stage = tmp_path / "17_maps"
    stage.mkdir(parents=True)
    pd.DataFrame([
        {"Id": 1, "year": 2011, "map_type": "annual_pv_change"},
        {"Id": 1, "year": 2012, "map_type": "annual_pv_change"},
        {"Id": 1, "year": None, "map_type": "annual_pv_change_atlas"},
    ]).to_csv(stage / "map_manifest.csv", index=False)
    pd.DataFrame([{
        "Id": 1, "contract_pass": True, "reference_shp_drawn": False,
        "dual_state_visualization": True, "colourblind_safe_high_contrast": True,
        "single_exclusive_class_layer": True, "no_stacked_pv_layers": True,
        "area_annotations_outside_map_axes": True,
    }]).to_csv(stage / "map_rendering_audit.csv", index=False)
    ok, problems = RUN_ALL.coverage_valid(17, tmp_path, _cfg())
    assert ok, problems


def test_dynamics_dual_state_confidence_intervals_are_coherent(tmp_path: Path) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from pvchange.dynamics import summarize_pv_dynamics

    root = tmp_path / "out"
    temporal_dir = root / "08_temporal_pv_change"
    uncertainty_dir = root / "10_uncertainty"
    temporal_dir.mkdir(parents=True)
    uncertainty_dir.mkdir(parents=True)
    rows = []
    for year, active_count, cumulative_count in [(2011, 2, 3), (2012, 1, 4)]:
        active = np.zeros((2, 2), dtype=np.uint8)
        cumulative = np.zeros((2, 2), dtype=np.uint8)
        active.flat[:active_count] = 1
        cumulative.flat[:cumulative_count] = 1
        ap = temporal_dir / f"active_{year}.tif"
        cp = temporal_dir / f"cumulative_{year}.tif"
        profile = {
            "driver": "GTiff", "height": 2, "width": 2, "count": 1,
            "dtype": "uint8", "crs": "EPSG:32648",
            "transform": from_origin(500000, 4000000, 30, 30),
            "nodata": 255,
        }
        for path, array in [(ap, active), (cp, cumulative)]:
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(array, 1)
        rows.append({
            "Id": 1, "year": year, "estimation_mode": "direct_pixel", "estimation_route": "direct_pixel",
            "active_mask_path": str(ap), "cumulative_mask_path": str(cp), "mask_path": str(ap),
            "pv_area_ci_low_km2": np.nan, "pv_area_ci_high_km2": np.nan,
        })
    pd.DataFrame(rows).to_csv(temporal_dir / "temporal_mask_registry.csv", index=False)
    pd.DataFrame([
        {"Id": 1, "year": 2011, "expected_pv_area_km2": 0.0018, "area_ci_low_km2": 0.0010, "area_ci_high_km2": 0.0026},
        {"Id": 1, "year": 2012, "expected_pv_area_km2": 0.0009, "area_ci_low_km2": 0.0002, "area_ci_high_km2": 0.0016},
    ]).to_csv(uncertainty_dir / "annual_area_uncertainty.csv", index=False)
    cfg = {
        "paths": {"output_root": str(root)},
        "processing": {"compress": "DEFLATE", "show_progress": False},
        "temporal_decoding": {"mask_nodata": 255},
        "study": {"id_start": 1, "id_end": 1, "start_year": 2011, "end_year": 2012, "reference_year": 2012},
    }
    annual = summarize_pv_dynamics(cfg)
    assert (annual["observed_active_area_km2"] <= annual["cumulative_installed_area_km2"]).all()
    assert (annual["pv_area_ci_low_km2"] <= annual["pv_area_km2"]).all()
    assert (annual["pv_area_km2"] <= annual["pv_area_ci_high_km2"]).all()
    national = pd.read_csv(root / "11_pv_dynamics" / "national_pv_dynamics.csv")
    assert (national["observed_active_area_km2"] <= national["cumulative_installed_area_km2"]).all()
