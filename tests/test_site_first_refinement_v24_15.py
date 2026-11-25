from __future__ import annotations

import numpy as np

from pvchange.site_first_refinement import (
    CLASS_SITE_FIRST_REMOTE,
    REJECT_PERSISTENT_SAND,
    _evaluate_site_first_year,
    _reconcile_site_graph,
)


def _settings() -> dict:
    return {
        "site_opening_iterations": 0,
        "site_closing_iterations": 1,
        "site_group_bridge_iterations": 1,
        "site_dark_pixel_threshold": 0.45,
        "minimum_site_pixels": 12,
        "minimum_core_fraction": 0.15,
        "minimum_dark_fraction": 0.35,
        "minimum_solid_fill": 0.40,
        "minimum_oriented_fill": 0.04,
        "maximum_elongation": 50.0,
        "maximum_perimeter_density": 0.94,
        "maximum_edge_only_score": 0.92,
        "minimum_mean_dark_score": 0.40,
        "minimum_q75_dark_score": 0.45,
        "minimum_mean_local_contrast": 0.08,
        "minimum_boundary_contrast": -0.02,
        "maximum_confuser": 0.55,
        "maximum_site_sand_risk": 0.55,
        "site_minimum_persistence": 0.40,
        "site_minimum_onset": 0.22,
        "preexisting_year_cutoff": 2012,
        "preexisting_minimum_stability": 0.60,
        "preexisting_minimum_dark_score": 0.55,
        "large_site_minimum_pixels": 40,
        "large_site_minimum_dark_score": 0.48,
        "large_site_minimum_q75_dark_score": 0.52,
        "large_site_minimum_persistence": 0.45,
        "large_site_maximum_confuser": 0.48,
        "large_site_maximum_sand_risk": 0.52,
        "large_site_minimum_dark_fraction": 0.32,
        "large_site_minimum_solid_fill": 0.32,
        "large_site_maximum_edge_only_score": 0.94,
        "model_supported_site_minimum_score": 0.80,
        "model_supported_site_minimum_positive_similarity": 0.55,
        "model_supported_site_maximum_sand_risk": 0.38,
        "model_supported_site_minimum_persistence": 0.45,
        "model_supported_site_minimum_dark_score": 0.38,
        "persistent_sand_rejection_threshold": 0.58,
        "maximum_adjacency_completion_pixels": 3,
        "adjacency_minimum_dark_score": 0.34,
        "adjacency_maximum_sand_risk": 0.42,
        "adjacency_minimum_model_score": 0.55,
        "adjacency_minimum_positive_similarity": 0.48,
        "adjacency_dark_override": 0.62,
        "temporal_rescue_minimum_dark_score": 0.36,
        "temporal_rescue_maximum_sand_risk": 0.48,
        "singleton_minimum_onset": 0.62,
        "singleton_minimum_persistence": 0.65,
        "singleton_minimum_dark_score": 0.60,
        "singleton_maximum_sand_risk": 0.40,
        "first_support_minimum_onset": 0.26,
        "first_support_minimum_persistence": 0.42,
        "first_support_dark_override": 0.66,
        "first_support_dark_persistence": 0.56,
        "first_support_maximum_sand_risk": 0.42,
        "track_sand_rejection_threshold": 0.60,
        "track_sand_maximum_onset": 0.26,
        "track_sand_dark_override": 0.74,
    }


def _base(shape=(64, 64)) -> dict:
    z = np.zeros(shape, dtype=np.float32)
    return {
        "reference_anchor": np.zeros(shape, dtype=bool),
        "expert_positive": np.zeros(shape, dtype=bool),
        "expert_negative": np.zeros(shape, dtype=bool),
        "site_core": np.zeros(shape, dtype=bool),
        "site_growable": np.zeros(shape, dtype=bool),
        "absolute_dark": z.copy(),
        "local_contrast": z.copy(),
        "onset": z.copy(),
        "persistence": z.copy(),
        "stability": z.copy(),
        "sand_risk": z.copy(),
        "confuser": z.copy(),
        "brightness": np.full(shape, 0.35, dtype=np.float32),
        "model_score": z.copy(),
        "positive_similarity": z.copy(),
    }


def test_large_remote_black_site_is_accepted_without_model_support() -> None:
    d = _base()
    sl = np.s_[20:44, 28:54]
    d["site_core"][sl] = True
    d["site_growable"][sl] = True
    d["absolute_dark"][sl] = 0.82
    d["local_contrast"][sl] = 0.75
    d["onset"][sl] = 0.85
    d["persistence"][sl] = 0.92
    d["stability"][sl] = 0.86
    d["sand_risk"][sl] = 0.10
    d["confuser"][sl] = 0.02
    d["brightness"][sl] = 0.05
    accepted, classes, _, _, rows = _evaluate_site_first_year(
        region_id=6, year=2021, settings=_settings(), **d
    )
    assert accepted[30, 40]
    assert classes[30, 40] == CLASS_SITE_FIRST_REMOTE
    assert any(r["accepted"] and r["branch"] == "site_first" for r in rows)


def test_persistent_sand_is_rejected_even_when_model_score_is_high() -> None:
    d = _base()
    # A filled but sand-like region with weak physical darkness and no onset.
    sl = np.s_[12:30, 8:30]
    d["site_core"][sl] = True
    d["site_growable"][sl] = True
    d["absolute_dark"][sl] = 0.43
    d["local_contrast"][sl] = 0.12
    d["onset"][sl] = 0.04
    d["persistence"][sl] = 0.85
    d["stability"][sl] = 0.50
    d["sand_risk"][sl] = 0.88
    d["model_score"][sl] = 0.94
    d["positive_similarity"][sl] = 0.62
    accepted, _, rejection, _, rows = _evaluate_site_first_year(
        region_id=6, year=2015, settings=_settings(), **d
    )
    assert not accepted[sl].any()
    assert np.any(rejection[sl] == REJECT_PERSISTENT_SAND)
    assert any(not r["accepted"] for r in rows)


def test_late_site_is_not_backfilled_into_early_years() -> None:
    n = 6
    shape = (40, 40)
    masks = np.zeros((n, *shape), dtype=bool)
    masks[:, 10:25, 18:34] = True  # simulate an over-persistent model result
    classes = np.zeros_like(masks, dtype=np.uint8)
    rejection = np.zeros_like(masks, dtype=np.uint8)
    dark = np.zeros((n, *shape), dtype=np.float32)
    dark[3:, 10:25, 18:34] = 0.82
    grow = dark >= 0.30
    onset = np.zeros_like(dark)
    onset[3, 10:25, 18:34] = 0.86
    persistence = np.zeros_like(dark)
    persistence[3:, 10:25, 18:34] = 0.88
    sand = np.full_like(dark, 0.15)
    anchor = np.zeros_like(masks)
    pos = np.zeros_like(masks)
    neg = np.zeros_like(masks)
    out, _, _ = _reconcile_site_graph(
        masks, classes, rejection, dark, grow, onset, persistence, sand,
        anchor, pos, neg, _settings(),
    )
    assert not out[:3, 15, 25].any()
    assert out[3:, 15, 25].all()


def test_expert_negative_overrides_all_sources() -> None:
    d = _base()
    sl = np.s_[20:40, 20:40]
    for key in ["site_core", "site_growable"]:
        d[key][sl] = True
    d["absolute_dark"][sl] = 0.9
    d["local_contrast"][sl] = 0.8
    d["onset"][sl] = 0.9
    d["persistence"][sl] = 0.9
    d["stability"][sl] = 0.9
    d["expert_negative"][sl] = True
    accepted, _, rejection, _, _ = _evaluate_site_first_year(
        region_id=6, year=2024, settings=_settings(), **d
    )
    assert not accepted[sl].any()
    assert np.all(rejection[sl] == 40)
