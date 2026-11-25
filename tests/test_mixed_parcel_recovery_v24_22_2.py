from __future__ import annotations

import numpy as np

from pvchange.anchor_guided import build_anchor_prototype
from pvchange.authoritative_reference import (
    _enhance_reference_internal_history,
    _refine_manual_reference_boundary,
)
from pvchange.growth_front import reconstruct_reference_growth_front
from pvchange.quality import _mixed_pixel_anchor_rescue_eligible


def _growth_settings() -> dict:
    return {
        "historical_temporal_median_radius": 1,
        "historical_seed_probability": 0.50,
        "historical_strong_probability": 0.68,
        "historical_extent_probability": 0.30,
        "historical_edge_probability": 0.24,
        "historical_component_seed_probability": 0.42,
        "historical_component_onset_relax_probability": 0.28,
        "historical_onset_tolerance_years": 1,
        "historical_minimum_seed_pixels": 2,
        "historical_minimum_seed_fraction": 0.003,
        "historical_completion_gap_pixels": 1,
        "historical_minimum_component_support": 0.27,
        "historical_component_presence_probability": 0.36,
        "historical_component_weak_pixel_probability": 0.30,
        "historical_component_strong_pixel_probability": 0.56,
        "historical_component_minimum_supported_fraction": 0.06,
        "historical_component_minimum_strong_fraction": 0.015,
        "historical_small_component_pixels": 16,
        "historical_small_component_median_probability": 0.34,
        "historical_small_component_peak_probability": 0.48,
        "historical_small_component_minimum_support": 0.24,
        "historical_component_support_window_years": 3,
        "historical_component_minimum_support_years": 2,
        "historical_mixed_parcel_enabled": True,
        "historical_parcel_weak_probability": 0.24,
        "historical_parcel_moderate_probability": 0.32,
        "historical_parcel_strong_probability": 0.46,
        "historical_parcel_minimum_valid_fraction": 0.50,
        "historical_parcel_minimum_supported_fraction": 0.22,
        "historical_parcel_minimum_moderate_fraction": 0.08,
        "historical_parcel_minimum_dilated_coverage": 0.70,
        "historical_parcel_minimum_sector_coverage": 0.55,
        "historical_parcel_minimum_boundary_support": 0.18,
        "historical_parcel_minimum_interior_support": 0.22,
        "historical_parcel_minimum_median_probability": 0.23,
        "historical_parcel_minimum_upper_probability": 0.32,
        "historical_parcel_dilation_pixels": 2,
        "historical_parcel_grid_size": 3,
        "historical_parcel_support_window_years": 3,
        "historical_parcel_minimum_support_years": 2,
    }


def _boundary_settings() -> dict:
    return {
        "reference_core_erosion_pixels": 1,
        "reference_boundary_search_pixels": 2,
        "small_reference_component_pixels": 9,
        "boundary_addition_minimum_likelihood": 0.50,
        "boundary_addition_minimum_break": 0.30,
        "boundary_addition_minimum_support": 0.38,
        "boundary_addition_maximum_sand": 0.65,
        "boundary_addition_maximum_confuser": 0.60,
        "boundary_removal_maximum_break": 0.35,
        "boundary_removal_minimum_sand": 0.55,
        "boundary_removal_minimum_confuser": 0.55,
        "boundary_removal_maximum_pv_fraction": 0.15,
        "boundary_removal_maximum_state_likelihood": 0.34,
        "boundary_removal_maximum_reference_score": 0.30,
        "boundary_removal_minimum_negative_cues": 2,
        "minimum_boundary_correction_pixels": 2,
        "maximum_reference_area_adjustment_fraction": 0.50,
        "reference_mixed_parcel_protection": True,
    }


def test_distributed_mixed_pixel_parcel_is_filled_from_true_onset() -> None:
    n, ref, h, w = 15, 13, 36, 36
    inventory = np.zeros((h, w), dtype=bool)
    inventory[8:28, 8:28] = True
    probability = np.full((n, h, w), 0.05, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    install = 5
    probability[install:ref, inventory] = 0.25
    rows, cols = np.indices((h, w))
    distributed_rows = inventory & ((rows % 4 == 0) | (cols % 4 == 0))
    probability[install:ref, distributed_rows] = 0.42
    probability[ref, inventory] = 1.0
    onset = np.full((h, w), ref, dtype=np.int16)

    active, rescued = reconstruct_reference_growth_front(
        probability,
        valid,
        inventory,
        onset,
        ref,
        _growth_settings(),
    )

    assert not active[install - 1].any()
    assert np.array_equal(active[install], inventory)
    assert np.count_nonzero(rescued[install]) > 0


def test_localised_partial_buildout_does_not_activate_complete_parcel() -> None:
    n, ref, h, w = 15, 13, 36, 36
    inventory = np.zeros((h, w), dtype=bool)
    inventory[8:28, 8:28] = True
    early = np.zeros_like(inventory)
    early[8:28, 8:13] = True
    probability = np.full((n, h, w), 0.05, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    probability[4:ref, early] = 0.70
    probability[ref, inventory] = 1.0
    onset = np.full((h, w), ref, dtype=np.int16)
    onset[early] = 4

    active, _ = reconstruct_reference_growth_front(
        probability,
        valid,
        inventory,
        onset,
        ref,
        _growth_settings(),
    )

    assert active[4, 10:26, 8:13].mean() > 0.9
    assert active[4, 10:26, 18:26].mean() == 0.0


def test_stage7_enhancement_raises_whole_distributed_mixed_parcel_only_after_onset() -> None:
    n, ref, h, w = 15, 13, 30, 30
    inventory = np.zeros((h, w), dtype=bool)
    inventory[7:23, 7:23] = True
    valid = np.ones((n, h, w), dtype=bool)
    posterior = np.full((n, h, w), 0.04, dtype=np.float32)
    score = np.full_like(posterior, 0.05)
    similarity = np.full_like(posterior, 0.05)
    fraction = np.full_like(posterior, 0.01)
    dark = np.full_like(posterior, 0.05)
    sand = np.full_like(posterior, 0.10)
    confuser = np.full_like(posterior, 0.10)
    install = 5
    score[install:ref, inventory] = 0.24
    similarity[install:ref, inventory] = 0.28
    fraction[install:ref, inventory] = 0.08
    dark[install:ref, inventory] = 0.25

    enhanced = _enhance_reference_internal_history(
        posterior,
        score,
        similarity,
        fraction,
        dark,
        sand,
        confuser,
        valid,
        inventory,
        ref,
        {
            "historical_mixed_parcel_enabled": True,
            "historical_component_support_window_years": 3,
            "historical_component_minimum_support_years": 2,
        },
    )

    assert float(enhanced[install - 1][inventory].max()) < 0.10
    assert float(enhanced[install][inventory].min()) >= 0.36


def test_reference_boundary_preserves_a_parcel_with_distributed_mixed_pixel_support() -> None:
    anchor = np.zeros((24, 24), dtype=bool)
    anchor[6:18, 6:18] = True
    fraction = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.30, dtype=np.float32)
    support = np.full(anchor.shape, 0.36, dtype=np.float32)
    change = np.full(anchor.shape, 0.05, dtype=np.float32)
    sand = np.full(anchor.shape, 0.12, dtype=np.float32)
    confuser = np.full(anchor.shape, 0.12, dtype=np.float32)
    dark = np.full(anchor.shape, 0.38, dtype=np.float32)
    pv_fraction = np.full(anchor.shape, 0.14, dtype=np.float32)

    refined, _, _, code, diagnostics = _refine_manual_reference_boundary(
        anchor,
        fraction,
        valid,
        score,
        support,
        change,
        sand,
        confuser,
        dark,
        pv_fraction,
        _boundary_settings(),
    )

    assert np.array_equal(refined, anchor)
    assert not np.any(code == 3)
    assert diagnostics["protected_mixed_parcel_components"] == 1


def test_low_quality_but_usable_anchor_remains_on_spatial_route() -> None:
    row = {
        "has_confirmed_anchor": True,
        "valid_year_count_50pct": 10,
        "median_valid_fraction": 0.62,
        "anchor_sample_pixels": 8,
        "reference_lsma_rmse_median": 0.40,
        "maximum_annual_median_spectral_jump": 0.52,
        "endmember_condition_number": 50000.0,
    }
    assert _mixed_pixel_anchor_rescue_eligible(row, {"quality_routing": {}})



def test_soft_mixed_anchor_count_can_rescue_a_unit_without_strong_anchor_pixels() -> None:
    row = {
        "has_confirmed_anchor": True,
        "valid_year_count_50pct": 9,
        "median_valid_fraction": 0.58,
        "anchor_sample_pixels": 0,
        "mixed_anchor_sample_pixels": 12,
        "reference_lsma_rmse_median": 0.41,
        "maximum_annual_median_spectral_jump": 0.50,
        "endmember_condition_number": 70000.0,
    }
    assert _mixed_pixel_anchor_rescue_eligible(row, {"quality_routing": {}})

def test_anchor_prototype_uses_soft_positive_pixels_for_small_mixed_parcel(monkeypatch) -> None:
    h, w = 8, 8
    reflectance = np.zeros((6, h, w), dtype=np.float32)
    fraction = np.zeros((h, w), dtype=np.float32)
    fraction[3:5, 3:5] = 0.25
    feature_cube = np.zeros((2, h, w), dtype=np.float32)
    feature_cube[0] = 0.10
    feature_cube[1] = 0.20
    feature_cube[:, 3:5, 3:5] += 0.35

    monkeypatch.setattr(
        "pvchange.anchor_guided.stream_reference_samples",
        lambda *args, **kwargs: (reflectance, fraction),
    )
    monkeypatch.setattr(
        "pvchange.anchor_guided.load_endmember_matrix",
        lambda *args, **kwargs: np.eye(4, 6, dtype=np.float32),
    )
    monkeypatch.setattr(
        "pvchange.anchor_guided.compute_features",
        lambda *args, **kwargs: feature_cube,
    )
    cfg = {
        "project": {"random_seed": 7},
        "supervision": {"strong_positive_fraction": 0.70},
        "features": {"channels": ["NDVI", "NDBI"]},
        "anchor_guided": {
            "feature_names": ["NDVI", "NDBI"],
            "minimum_anchor_prototype_pixels": 20,
            "minimum_mixed_pixel_prototype_pixels": 3,
            "mixed_pixel_prototype_fraction_floor": 0.02,
            "minimum_background_prototype_pixels": 10,
        },
    }

    prototype = build_anchor_prototype("reference.tif", "fraction.tif", "endmembers.npz", cfg)
    assert prototype is not None
    assert prototype.anchor_pixels == 4
    assert np.all(np.isfinite(prototype.center))
    assert np.all(prototype.scale > 0)
