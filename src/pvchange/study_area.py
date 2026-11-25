from __future__ import annotations

from typing import Any


def study_ids(cfg: dict[str, Any]) -> list[int]:
    study = cfg.get("study", {})
    selected = study.get("selected_unit_ids") or []
    if selected:
        return sorted({int(x) for x in selected})
    return list(range(int(study["id_start"]), int(study["id_end"]) + 1))


def expected_counts(cfg: dict[str, Any]) -> tuple[int, int, int]:
    ids = study_ids(cfg)
    years = int(cfg["study"]["end_year"]) - int(cfg["study"]["start_year"]) + 1
    return len(ids), years, len(ids) * years


def expected_id_set(cfg: dict[str, Any]) -> set[int]:
    return set(study_ids(cfg))


def selected_mode(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("study", {}).get("selected_unit_ids") or [])


def selected_unit_count(cfg: dict[str, Any]) -> int:
    return len(study_ids(cfg))


def test_mode_enabled(cfg: dict[str, Any]) -> bool:
    return bool(selected_mode(cfg) and cfg.get("test_mode", {}).get("enabled_when_selected_unit_ids", True))
