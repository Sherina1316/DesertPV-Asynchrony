from __future__ import annotations

import numpy as np
from scipy import ndimage

from pvchange.authoritative_reference import _refine_manual_reference_boundary
from pvchange.authoritative_temporal import _ordered_change_point
from pvchange.growth_front import (
    detect_terminal_year_expansion,
    reconstruct_reference_growth_front,
)


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
        "boundary_addition_strong_persistent_support": 0.70,
        "boundary_addition_strong_reference_score": 0.70,
        "boundary_addition_strong_pv_fraction": 0.40,
        "boundary_removal_maximum_break": 0.35,
        "boundary_removal_minimum_sand": 0.55,
        "boundary_removal_minimum_confuser": 0.55,
        "boundary_removal_maximum_pv_fraction": 0.15,
        "boundary_removal_maximum_state_likelihood": 0.34,
        "boundary_removal_maximum_reference_score": 0.30,
        "boundary_removal_minimum_negative_cues": 2,
        "minimum_boundary_correction_pixels": 2,
        "maximum_reference_area_adjustment_fraction": 0.50,
        "suppress_annular_boundary_additions": True,
        "annular_minimum_reference_pixels": 12,
        "annular_minimum_boundary_wrap_fraction": 0.50,
        "annular_minimum_thin_fraction": 0.75,
        "annular_maximum_distance_pixels": 2.25,
        "annular_strong_patch_priority": 0.95,
    }


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
        "historical_component_presence_probability": 0.34,
        "historical_component_weak_pixel_probability": 0.28,
        "historical_component_strong_pixel_probability": 0.56,
        "historical_component_minimum_supported_fraction": 0.05,
        "historical_component_minimum_strong_fraction": 0.01,
        "historical_small_component_pixels": 16,
        "historical_small_component_median_probability": 0.34,
        "historical_small_component_peak_probability": 0.48,
        "historical_small_component_minimum_support": 0.22,
        "historical_component_support_window_years": 3,
        "historical_component_minimum_support_years": 2,
        "historical_shell_minimum_component_coverage": 0.60,
        "historical_shell_width_pixels": 2,
        "historical_shell_maximum_interior_missing_fraction": 0.08,
        "historical_shell_minimum_wrap_fraction": 0.45,
        "historical_shell_probability_floor": 0.18,
    }


def test_persistent_moderate_early_probability_is_not_shifted_to_2023_or_2024() -> None:
    n, ref = 14, 13
    p = np.full((n, 1, 1), 0.43, dtype=np.float32)
    p[ref, 0, 0] = 1.0
    valid = np.ones_like(p, dtype=bool)
    inventory = np.ones((1, 1), dtype=bool)
    onset, _, _, _ = _ordered_change_point(
        p,
        valid,
        inventory,
        ref,
        {
            "reference_endpoint_likelihood_weight": 0.25,
            "preexisting_window_years": 3,
            "preexisting_weak_probability_threshold": 0.38,
            "preexisting_minimum_early_mean_probability": 0.40,
            "preexisting_minimum_early_support_fraction": 0.60,
            "preexisting_persistent_probability_threshold": 0.34,
            "preexisting_minimum_historical_support_fraction": 0.55,
            "preexisting_minimum_early_valid_years": 2,
            "preexisting_maximum_later_jump": 0.24,
            "minimum_post_installation_support_years": 2,
        },
    )
    assert int(onset[0, 0]) == 0


def test_true_late_installation_keeps_its_late_onset() -> None:
    n, ref = 14, 13
    p = np.full((n, 1, 1), 0.05, dtype=np.float32)
    p[8:ref, 0, 0] = 0.82
    p[ref, 0, 0] = 1.0
    valid = np.ones_like(p, dtype=bool)
    inventory = np.ones((1, 1), dtype=bool)
    onset, _, _, _ = _ordered_change_point(
        p,
        valid,
        inventory,
        ref,
        {
            "reference_endpoint_likelihood_weight": 0.25,
            "minimum_post_installation_mean_probability": 0.46,
            "minimum_pre_post_probability_gain": 0.13,
            "minimum_post_installation_support_years": 2,
            "post_installation_high_probability": 0.48,
        },
    )
    assert 7 <= int(onset[0, 0]) <= 8


def test_object_level_history_recovers_weak_edges_from_2011() -> None:
    n, h, w, ref = 15, 32, 32, 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[8:24, 8:24] = True
    core = np.zeros_like(inventory)
    core[12:20, 12:20] = True
    p = np.full((n, h, w), 0.04, dtype=np.float32)
    p[: ref + 1, inventory] = 0.32
    p[: ref + 1, core] = 0.66
    valid = np.ones_like(p, dtype=bool)
    onset = np.full((h, w), ref, dtype=np.int16)
    onset[core] = 0

    active, rescued = reconstruct_reference_growth_front(
        p,
        valid,
        inventory,
        onset,
        ref,
        _growth_settings(),
    )
    assert np.array_equal(active[0], inventory)
    assert np.count_nonzero(rescued[0] & (inventory & ~core)) > 0


def test_object_level_history_does_not_fill_a_missing_interior_buildout() -> None:
    n, h, w, ref = 15, 32, 32, 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[8:24, 8:24] = True
    early = np.zeros_like(inventory)
    early[8:24, 8:16] = True
    late = inventory & ~early
    p = np.full((n, h, w), 0.04, dtype=np.float32)
    p[: ref + 1, early] = 0.66
    p[7 : ref + 1, late] = 0.62
    valid = np.ones_like(p, dtype=bool)
    onset = np.full((h, w), ref, dtype=np.int16)
    onset[early] = 0
    onset[late] = 7

    active, _ = reconstruct_reference_growth_front(
        p,
        valid,
        inventory,
        onset,
        ref,
        _growth_settings(),
    )
    assert active[0, 12:20, 18:22].mean() == 0.0
    assert active[7, 12:20, 18:22].mean() == 1.0


def test_small_manual_pv_block_is_never_discarded() -> None:
    anchor = np.zeros((16, 16), dtype=bool)
    anchor[7:9, 7:9] = True
    frac = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.15, dtype=np.float32)
    support = np.full(anchor.shape, 0.15, dtype=np.float32)
    change = np.zeros(anchor.shape, dtype=np.float32)
    sand = np.full(anchor.shape, 0.90, dtype=np.float32)
    conf = np.full(anchor.shape, 0.90, dtype=np.float32)
    dark = np.full(anchor.shape, 0.05, dtype=np.float32)
    pvfrac = np.full(anchor.shape, 0.02, dtype=np.float32)

    refined, _, _, _, diag = _refine_manual_reference_boundary(
        anchor,
        frac,
        valid,
        score,
        support,
        change,
        sand,
        conf,
        dark,
        pvfrac,
        _boundary_settings(),
    )
    assert np.array_equal(refined, anchor)
    assert diag["protected_small_component_pixels"] == 4


def test_oversized_manual_outer_shell_is_removed_but_inner_site_is_kept() -> None:
    anchor = np.zeros((24, 24), dtype=bool)
    anchor[6:18, 6:18] = True
    inner = np.zeros_like(anchor)
    inner[7:17, 7:17] = True
    shell = anchor & ~inner
    frac = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.80, dtype=np.float32)
    support = np.full(anchor.shape, 0.75, dtype=np.float32)
    change = np.full(anchor.shape, 0.05, dtype=np.float32)  # old site: little recent break
    sand = np.full(anchor.shape, 0.08, dtype=np.float32)
    conf = np.full(anchor.shape, 0.08, dtype=np.float32)
    dark = np.full(anchor.shape, 0.72, dtype=np.float32)
    pvfrac = np.full(anchor.shape, 0.68, dtype=np.float32)
    score[shell] = 0.05
    support[shell] = 0.05
    sand[shell] = 0.92
    conf[shell] = 0.80
    dark[shell] = 0.05
    pvfrac[shell] = 0.01

    refined, _, _, _, _ = _refine_manual_reference_boundary(
        anchor,
        frac,
        valid,
        score,
        support,
        change,
        sand,
        conf,
        dark,
        pvfrac,
        _boundary_settings(),
    )
    assert np.array_equal(refined, inner)


def test_full_external_annulus_is_not_added_to_2024_inventory() -> None:
    anchor = np.zeros((28, 28), dtype=bool)
    anchor[8:20, 8:20] = True
    ring = ndimage.binary_dilation(anchor, iterations=1) & ~anchor
    frac = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.20, dtype=np.float32)
    support = np.full(anchor.shape, 0.20, dtype=np.float32)
    change = np.full(anchor.shape, 0.10, dtype=np.float32)
    sand = np.full(anchor.shape, 0.10, dtype=np.float32)
    conf = np.full(anchor.shape, 0.05, dtype=np.float32)
    dark = np.full(anchor.shape, 0.20, dtype=np.float32)
    pvfrac = np.full(anchor.shape, 0.10, dtype=np.float32)
    for layer in (score, support, change, dark, pvfrac):
        layer[ring] = 0.95

    refined, _, _, _, diag = _refine_manual_reference_boundary(
        anchor,
        frac,
        valid,
        score,
        support,
        change,
        sand,
        conf,
        dark,
        pvfrac,
        _boundary_settings(),
    )
    assert not np.any(refined & ring)
    assert diag["annular_addition_pixels_suppressed"] > 0


def test_terminal_full_ring_is_rejected_as_boundary_displacement() -> None:
    n, h, w, ref = 15, 48, 48, 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[16:32, 16:32] = True
    ring = ndimage.binary_dilation(inventory, iterations=1) & ~inventory
    p = np.full((n, h, w), 0.05, dtype=np.float32)
    valid = np.ones_like(p, dtype=bool)
    candidate = np.zeros_like(valid)
    p[ref, inventory] = 1.0
    p[ref + 1, ring] = 0.86
    candidate[ref + 1, ring] = True
    settings = {
        "terminal_adjacent_search_pixels": 4,
        "terminal_perimeter_search_pixels": 6,
        "terminal_adjacent_seed_probability": 0.45,
        "terminal_adjacent_minimum_gain": 0.02,
        "terminal_extent_probability": 0.30,
        "terminal_extent_minimum_gain": 0.01,
        "terminal_nonmask_extent_probability": 0.45,
        "terminal_nonmask_extent_gain": 0.02,
        "terminal_adjacent_minimum_pixels": 2,
        "terminal_adjacent_mean_probability": 0.40,
        "terminal_adjacent_mean_gain": 0.02,
        "terminal_adjacent_minimum_solidity": 0.10,
        "terminal_adjacent_maximum_sand": 1.0,
        "terminal_closing_iterations": 1,
        "terminal_suppress_annular_boundary_change": True,
        "terminal_annular_contact_tolerance_pixels": 2,
        "terminal_annular_maximum_distance_pixels": 2.5,
        "terminal_annular_minimum_wrap_fraction": 0.50,
        "terminal_annular_minimum_thin_fraction": 0.75,
    }
    mask, _, _, decisions = detect_terminal_year_expansion(
        candidate,
        p,
        valid,
        inventory,
        ref,
        settings,
    )
    assert not mask.any()
    assert any(d.rejection_reason == "annular_boundary_displacement" for d in decisions)
