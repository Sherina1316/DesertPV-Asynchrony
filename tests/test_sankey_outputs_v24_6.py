from __future__ import annotations

from pathlib import Path

import pandas as pd

from pvchange.sankey import generate_ecology_sankey_figures
from pvchange.tech_route import generate_chinese_technical_route


def test_sankey_and_technical_route_outputs(tmp_path: Path):
    root = tmp_path / "output"
    (root / "18_figures").mkdir(parents=True)
    (root / "16_future_2050").mkdir(parents=True)
    (root / "12_dynamic_ecology").mkdir(parents=True)
    pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "pv_expansion_speed_normalized": [0.1, 0.2, 0.7, 1.3],
        "reference_cumulative_installed_area_km2": [1.0, 2.0, 3.0, 4.0],
        "ecological_adjustment_speed": [0.01, 0.02, 0.05, 0.10],
    }).to_csv(root / "18_figures" / "expansion_ecological_speed_by_unit.csv", index=False)
    pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "response_direction": ["increase", "decline", "insufficient", "increase"],
        "response_lag_years": [1, 3, None, 6],
        "response_amplitude": [0.2, -0.1, 0.0, 0.5],
        "stabilization_observed": [True, False, False, True],
    }).to_csv(root / "18_figures" / "model_consistent_response_timing.csv", index=False)
    pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "scenario": ["transition", "transition", "transition", "transition"],
        "additional_area_2050_km2": [0.2, 0.6, 1.0, 2.0],
    }).to_csv(root / "16_future_2050" / "conditional_2050_candidate_summary.csv", index=False)
    pd.DataFrame({
        "Id": [1, 1, 2, 2, 3, 3, 4, 4],
        "metric": ["NDVI", "NDMI", "NDVI", "NDMI", "NDVI", "NDMI", "NDVI", "NDMI"],
        "external_anomaly": [0.2, 0.1, -0.2, -0.1, 0.01, 0.0, 0.3, -0.2],
    }).to_csv(root / "12_dynamic_ecology" / "external_ecology_anomaly_panel.csv", index=False)
    cfg = {
        "paths": {"output_root": str(root)},
        "mapping": {"map_dpi": 80, "raster_formats": ["png"], "vector_formats": ["svg"]},
    }
    manifest = generate_ecology_sankey_figures(cfg)
    assert len(manifest) == 2
    assert (root / "18_figures" / "figure_10_ecological_response_sankey.png").exists()
    assert (root / "18_figures" / "figure_11_multimetric_ecology_sankey.svg").exists()
    assert (root / "18_figures" / "ecology_response_sankey_nodes.csv").exists()
    assert (root / "18_figures" / "multimetric_ecology_sankey_links.csv").exists()
    route = generate_chinese_technical_route(cfg)
    assert Path(route["png"]).exists()
