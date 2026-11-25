from __future__ import annotations

from pathlib import Path


def test_explicit_units_are_routed_without_prototype_build():
    text = (Path(__file__).parents[1] / "src/pvchange/authoritative_reference.py").read_text()
    assert "explicit_area_only_ids = {83, 104, 141}" in text
    assert "explicit_area_only_skip_insufficient_reference_anchor" in text
    assert "_build_unit_prototypes" in text
    assert text.index("explicit_area_only_ids = {83, 104, 141}") < text.index("prototypes: PrototypeBundle = _build_unit_prototypes")


def test_stage7_contract_is_pixel_route_aware():
    text = (Path(__file__).parents[1] / "run_all.py").read_text()
    assert "pixel/area-only contract separation" in text
    assert "Area-only units unexpectedly contain fabricated Stage-7 rasters" in text


def test_stage8_includes_explicit_area_only_routes():
    text = (Path(__file__).parents[1] / "src/pvchange/fallback.py").read_text()
    assert "explicit_fallback_ids" in text
    assert "calibrated_inference_registry.csv" in text
    assert "explicit_stage7_area_only" in text
    for column in [
        "active_mask_path",
        "cumulative_mask_path",
        "model_only_cumulative_mask_path",
        "observed_active_area_km2",
        "cumulative_installed_area_km2",
        "model_only_cumulative_installed_area_km2",
        "terminal_2025_expansion_source_path",
    ]:
        assert f'"{column}"' in text


def test_temporal_registry_rejects_duplicate_id_year_rows():
    text = (Path(__file__).parents[1] / "src/pvchange/authoritative_temporal.py").read_text()
    assert 'registry.duplicated(["Id", "year"], keep=False)' in text
    assert '"explicit_stage7_area_only_unit_ids"' in text
