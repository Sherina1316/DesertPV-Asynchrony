from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_all


def test_boundary_aware_step3_contract_uses_structured_semantics(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path
    stage = out / "03_supervision"
    stage.mkdir(parents=True)
    pd.DataFrame({
        "Id": [6, 28],
        "split_role": ["train", "calibration"],
        "has_mapped_pv": [True, True],
        "outside_shp_semantics": [
            "boundary_uncertain_far_background_negative",
            "boundary_uncertain_far_background_negative",
        ],
    }).to_csv(stage / "supervision_registry.csv", index=False)
    pd.DataFrame({"Id": [6, 28]}).to_csv(stage / "supervision_split.csv", index=False)
    (stage / "supervision_status.json").write_text(
        '{"boundary_band_is_unlabelled": true, "far_background_is_negative": true, "outside_is_fully_unlabelled": false}',
        encoding="utf-8",
    )
    cfg = {
        "study": {"id_start": 1, "id_end": 142, "start_year": 2011, "end_year": 2025, "selected_unit_ids": [6, 28]},
        "test_mode": {"enabled_when_selected_unit_ids": True},
        "supervision": {
            "boundary_uncertainty_enabled": True,
            "reference_year_complete_inventory_assumed": True,
            "split": {
                "minimum_positive_train_units": 1,
                "minimum_positive_calibration_units": 1,
                "minimum_positive_test_units": 0,
            },
        },
        "future_2050": {"scenarios": ["baseline"]},
    }
    monkeypatch.setattr(run_all, "_table_contract", lambda *args, **kwargs: [])
    ok, problems = run_all.coverage_valid(3, out, cfg)
    assert ok, problems
    assert problems == []
