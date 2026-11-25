from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pvchange_run_all", ROOT / "run_all.py")
assert SPEC is not None and SPEC.loader is not None
RUN_ALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_ALL)


def minimal_config() -> dict:
    return {
        "study": {"id_start": 1, "id_end": 5, "start_year": 2011, "end_year": 2025},
        "future_2050": {"scenarios": ["conservative", "transition", "accelerated"]},
        "processing": {"allow_scientific_gate_failure": False},
        "supervision": {"split": {
            "minimum_positive_train_units": 1,
            "minimum_positive_calibration_units": 1,
            "minimum_positive_test_units": 1,
        }},
    }


def test_empty_stage_contracts_fail_cleanly(tmp_path: Path) -> None:
    cfg = minimal_config()
    stage_map = {step: required for step, _, _, required in RUN_ALL.STAGES}
    for step in (3, 7, 8, 9, 12, 14, 17, 18):
        ok, problems = RUN_ALL.contract_valid(tmp_path, stage_map[step], step, cfg)
        assert ok is False
        assert problems
