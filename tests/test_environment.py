from __future__ import annotations

from pathlib import Path

import pandas as pd

from pvchange.environment import normalize_environmental_panel


def test_v20_environment_aliases(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    tables.mkdir()
    rows = []
    for region_id in [1, 2]:
        for year in [2011, 2012]:
            rows.append({
                "Id": region_id, "year": year,
                "gs_P_mm": 100 + year, "gs_Tmean_C": 15.0,
                "gs_PET_mm": 800.0, "gs_SM_m3m3": 0.15,
                "gs_SSR_MJm2": 2500.0, "gs_Wind_ms": 1.2,
            })
    pd.DataFrame(rows).to_csv(tables / "model_ready_clean_annual_2011_2025.csv", index=False)
    cfg = {
        "paths": {"input_table_dir": str(tables), "input_table_zip": str(tmp_path / "missing.zip")},
        "study": {"id_start": 1, "id_end": 2, "start_year": 2011, "end_year": 2012},
        "ecology": {"climate_covariates": ["precipitation", "temperature", "potential_evapotranspiration", "soil_moisture", "radiation", "wind_speed"]},
        "environment": {
            "preferred_table_names": ["model_ready_clean_annual_2011_2025.csv"],
            "source_column_map": {
                "gs_P_mm": "precipitation", "gs_Tmean_C": "temperature",
                "gs_PET_mm": "potential_evapotranspiration", "gs_SM_m3m3": "soil_moisture",
                "gs_SSR_MJm2": "radiation", "gs_Wind_ms": "wind_speed",
            },
        },
    }
    panel, inventory = normalize_environmental_panel(cfg)
    assert len(panel) == 4
    assert inventory.status.eq("used").any()
    assert panel[["precipitation", "temperature", "potential_evapotranspiration", "soil_moisture", "radiation", "wind_speed"]].notna().all().all()
