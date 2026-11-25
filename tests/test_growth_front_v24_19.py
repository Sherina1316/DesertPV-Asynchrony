from __future__ import annotations

import numpy as np

from pvchange.growth_front import (
    detect_terminal_year_expansion,
    reconstruct_reference_growth_front,
)


def _settings() -> dict:
    return {
        "historical_temporal_median_radius": 0,
        "historical_seed_probability": 0.50,
        "historical_strong_probability": 0.68,
        "historical_extent_probability": 0.30,
        "historical_edge_probability": 0.24,
        "historical_onset_tolerance_years": 1,
        "historical_minimum_seed_pixels": 2,
        "historical_minimum_seed_fraction": 0.003,
        "historical_completion_gap_pixels": 2,
        "historical_minimum_component_support": 0.27,
        "terminal_adjacent_search_pixels": 8,
        "terminal_adjacent_seed_probability": 0.48,
        "terminal_adjacent_minimum_gain": 0.04,
        "terminal_remote_seed_probability": 0.62,
        "terminal_remote_minimum_gain": 0.10,
        "terminal_absolute_override_probability": 0.78,
        "terminal_absolute_override_gain": 0.06,
        "terminal_extent_probability": 0.34,
        "terminal_extent_minimum_gain": 0.02,
        "terminal_nonmask_extent_probability": 0.46,
        "terminal_nonmask_extent_gain": 0.06,
        "terminal_adjacent_minimum_pixels": 4,
        "terminal_adjacent_mean_probability": 0.44,
        "terminal_adjacent_mean_gain": 0.03,
        "terminal_adjacent_minimum_solidity": 0.26,
        "terminal_remote_minimum_pixels": 12,
        "terminal_remote_mean_probability": 0.56,
        "terminal_remote_mean_gain": 0.075,
        "terminal_remote_minimum_bbox_fill": 0.06,
        "terminal_remote_minimum_solidity": 0.30,
        "terminal_closing_iterations": 1,
    }


def test_historical_growth_front_recovers_weak_panel_edges_without_backfill():
    n, h, w = 15, 48, 48
    reference_index = 13
    probability = np.full((n, h, w), 0.04, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    inventory = np.zeros((h, w), dtype=bool)
    inventory[10:34, 10:34] = True

    install = 7
    probability[install:reference_index + 1, 10:34, 10:34] = 0.38
    probability[install:reference_index + 1, 15:29, 15:29] = 0.76
    onset = np.full((h, w), reference_index, dtype=np.int16)
    onset[15:29, 15:29] = install
    onset[10:34, 10:34] = install + 1
    onset[15:29, 15:29] = install

    active, rescued = reconstruct_reference_growth_front(
        probability, valid, inventory, onset, reference_index, _settings()
    )

    assert not active[install - 1].any()
    strong_core_pixels = 14 * 14
    assert active[install].sum() > strong_core_pixels
    assert rescued[install].sum() > 0
    assert np.array_equal(active[reference_index], inventory)


def test_terminal_adjacent_expansion_is_detected_from_gain_and_object_extent():
    n, h, w = 15, 64, 64
    reference_index = 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[20:38, 10:26] = True
    probability = np.full((n, h, w), 0.05, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    candidate = np.zeros_like(valid)

    probability[reference_index, inventory] = 1.0
    probability[14, 22:38, 26:42] = 0.58
    candidate[14, 23:37, 27:41] = True

    mask, source, gain, decisions = detect_terminal_year_expansion(
        candidate, probability, valid, inventory, reference_index, _settings()
    )

    assert mask[25:35, 28:38].mean() > 0.8
    assert np.any(source == 1)
    assert gain[mask].mean() > 0.4
    assert any(d.accepted and d.adjacent_to_reference for d in decisions)


def test_terminal_remote_large_site_is_detected_but_stable_speckle_is_not():
    n, h, w = 15, 80, 80
    reference_index = 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[8:20, 8:20] = True
    probability = np.full((n, h, w), 0.06, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    candidate = np.zeros_like(valid)
    probability[reference_index, inventory] = 1.0

    probability[14, 45:58, 48:66] = 0.68
    candidate[14, 45:58, 48:66] = True
    probability[reference_index, 70:72, 70:72] = 0.60
    probability[14, 70:72, 70:72] = 0.65  # small gain and tiny object
    candidate[14, 70:72, 70:72] = True

    mask, source, _, decisions = detect_terminal_year_expansion(
        candidate, probability, valid, inventory, reference_index, _settings()
    )

    assert mask[47:56, 50:64].mean() > 0.9
    assert np.any(source == 2)
    assert not mask[70:72, 70:72].any()
    assert any(d.accepted and not d.adjacent_to_reference for d in decisions)


def test_terminal_stable_dark_background_is_rejected_without_probability_gain():
    n, h, w = 15, 48, 48
    reference_index = 13
    inventory = np.zeros((h, w), dtype=bool)
    probability = np.full((n, h, w), 0.05, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    candidate = np.zeros_like(valid)
    probability[10:reference_index, 20:34, 20:34] = 0.67
    probability[reference_index, 20:34, 20:34] = 0.67
    probability[14, 20:34, 20:34] = 0.69
    candidate[14, 20:34, 20:34] = True

    mask, source, _, _ = detect_terminal_year_expansion(
        candidate, probability, valid, inventory, reference_index, _settings()
    )

    assert not mask.any()
    assert not source.any()
