from pvchange.study_area import test_mode_enabled as _test_mode_enabled, selected_unit_count


def _effective_minimum(cfg):
    minimum = int(cfg["supervision"].get("minimum_positive_regions", 3))
    if _test_mode_enabled(cfg):
        selected_count = max(1, int(selected_unit_count(cfg)))
        diagnostic_minimum = int(cfg.get("test_mode", {}).get("minimum_positive_regions", selected_count))
        minimum = min(minimum, max(1, min(selected_count, diagnostic_minimum)))
    return minimum


def test_two_unit_diagnostic_contract_caps_formal_minimum():
    cfg = {
        "study": {"id_start": 1, "id_end": 142, "selected_unit_ids": [6, 28]},
        "supervision": {"minimum_positive_regions": 3},
        "test_mode": {"enabled_when_selected_unit_ids": True, "minimum_positive_regions": 2},
    }
    assert _effective_minimum(cfg) == 2


def test_full_production_contract_is_unchanged():
    cfg = {
        "study": {"id_start": 1, "id_end": 142, "selected_unit_ids": []},
        "supervision": {"minimum_positive_regions": 3},
        "test_mode": {"enabled_when_selected_unit_ids": True, "minimum_positive_regions": 2},
    }
    assert _effective_minimum(cfg) == 3
