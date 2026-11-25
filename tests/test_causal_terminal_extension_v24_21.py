from __future__ import annotations

import numpy as np

from pvchange.authoritative_reference import _multiband_reference_break
from pvchange.growth_front import (
    detect_terminal_year_expansion,
    reconstruct_historical_outside_tracks,
)
from pvchange.mapping import _exclusive_classes


def _settings() -> dict[str, float | int]:
    return {
        "terminal_baseline_years": 3,
        "terminal_adjacent_search_pixels": 8,
        "terminal_perimeter_search_pixels": 14,
        "terminal_adjacent_seed_probability": 0.58,
        "terminal_adjacent_minimum_gain": 0.08,
        "terminal_remote_seed_probability": 0.75,
        "terminal_remote_minimum_gain": 0.15,
        "terminal_absolute_override_probability": 0.90,
        "terminal_absolute_override_gain": 0.10,
        "terminal_extent_probability": 0.40,
        "terminal_extent_minimum_gain": 0.05,
        "terminal_nonmask_extent_probability": 0.60,
        "terminal_nonmask_extent_gain": 0.10,
        "terminal_adjacent_minimum_pixels": 4,
        "terminal_adjacent_mean_probability": 0.55,
        "terminal_adjacent_mean_gain": 0.07,
        "terminal_adjacent_minimum_solidity": 0.20,
        "terminal_perimeter_seed_likelihood": 0.50,
        "terminal_perimeter_seed_similarity": 0.48,
        "terminal_perimeter_minimum_multicue_change": 0.08,
        "terminal_perimeter_strong_likelihood": 0.60,
        "terminal_perimeter_maximum_sand": 0.55,
        "terminal_perimeter_maximum_confuser": 0.55,
        "terminal_perimeter_extent_likelihood": 0.38,
        "terminal_perimeter_extent_similarity": 0.38,
        "terminal_perimeter_extent_multicue_change": 0.04,
        "terminal_perimeter_absolute_extent_likelihood": 0.52,
        "terminal_perimeter_support_maximum_sand": 0.60,
        "terminal_perimeter_support_maximum_confuser": 0.60,
        "terminal_perimeter_minimum_pixels": 6,
        "terminal_perimeter_mean_likelihood": 0.45,
        "terminal_perimeter_mean_similarity": 0.44,
        "terminal_perimeter_mean_multicue_change": 0.04,
        "terminal_perimeter_object_strong_likelihood": 0.56,
        "terminal_perimeter_object_maximum_sand": 0.58,
        "terminal_perimeter_object_maximum_confuser": 0.58,
        "terminal_perimeter_minimum_solidity": 0.18,
        "terminal_perimeter_minimum_proximity": 0.10,
        "terminal_perimeter_minimum_contact_ratio": 0.005,
        "terminal_contact_tolerance_pixels": 2,
        "terminal_remote_minimum_pixels": 20,
        "terminal_remote_mean_probability": 0.70,
        "terminal_remote_mean_gain": 0.12,
        "terminal_remote_minimum_bbox_fill": 0.10,
        "terminal_remote_minimum_solidity": 0.30,
        "terminal_remote_maximum_sand": 0.50,
        "terminal_remote_maximum_confuser": 0.50,
        "terminal_closing_iterations": 1,
        "outside_temporal_median_radius": 1,
        "outside_seed_probability": 0.60,
        "outside_extent_probability": 0.45,
        "outside_minimum_component_pixels": 4,
    }


def test_2025_signal_cannot_refine_2024_break_support() -> None:
    n, h, w = 15, 8, 8
    reference_index = 13
    valid = np.ones((n, h, w), dtype=bool)
    names = [
        "positive_similarity",
        "pv_fraction",
        "local_dark",
        "brightness",
        "bsi",
        "ndvi",
        "ndbi",
    ]
    only_2025 = {name: np.zeros((n, h, w), dtype=np.float32) for name in names}
    only_2024 = {name: np.zeros((n, h, w), dtype=np.float32) for name in names}

    # Directionally PV-like changes.  In the first case they occur only in 2025.
    for name in ["positive_similarity", "pv_fraction", "local_dark", "ndbi"]:
        only_2025[name][14] = 1.0
        only_2024[name][13] = 1.0
    for name in ["brightness", "bsi", "ndvi"]:
        only_2025[name][14] = -1.0
        only_2024[name][13] = -1.0

    future_only_score = _multiband_reference_break(only_2025, valid, reference_index)
    reference_score = _multiband_reference_break(only_2024, valid, reference_index)

    assert float(reference_score.mean()) > float(future_only_score.mean()) + 0.35
    assert float(future_only_score.max()) < 0.35


def _terminal_case(high_sand: bool = False):
    n, h, w = 15, 36, 42
    ref = 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[10:24, 8:19] = True
    extension = np.zeros((h, w), dtype=bool)
    extension[11:23, 19:28] = True

    valid = np.ones((n, h, w), dtype=bool)
    probability = np.full((n, h, w), 0.08, dtype=np.float32)
    site_score = np.full_like(probability, 0.10)
    positive = np.full_like(probability, 0.10)
    fraction = np.full_like(probability, 0.08)
    dark = np.full_like(probability, 0.10)
    sand = np.full_like(probability, 0.18)
    confuser = np.full_like(probability, 0.12)

    # Audited 2024 site prototype.
    for arr, value in [
        (site_score, 0.78),
        (positive, 0.80),
        (fraction, 0.72),
        (dark, 0.76),
    ]:
        arr[ref][inventory] = value
        arr[14][inventory] = value
    probability[ref][inventory] = 1.0
    probability[14][inventory] = 0.82

    # The new wing has strong absolute panel state but only modest probability
    # gain, which should activate the component-conditioned perimeter path.
    probability[10:13, extension] = 0.47
    probability[14][extension] = 0.50
    site_score[10:13, extension] = 0.48
    site_score[14][extension] = 0.74
    positive[10:13, extension] = 0.45
    positive[14][extension] = 0.77
    fraction[10:13, extension] = 0.40
    fraction[14][extension] = 0.68
    dark[10:13, extension] = 0.45
    dark[14][extension] = 0.73
    sand[14][extension] = 0.90 if high_sand else 0.16
    confuser[14][extension] = 0.12

    candidate = np.zeros((n, h, w), dtype=bool)
    candidate[14][extension] = True
    return candidate, probability, valid, inventory, ref, site_score, positive, fraction, dark, sand, confuser, extension


def test_low_gain_intrasite_2025_wing_is_recovered_as_source_3() -> None:
    args = _terminal_case(high_sand=False)
    candidate, probability, valid, inventory, ref, site_score, positive, fraction, dark, sand, confuser, extension = args
    accepted, source, _, decisions = detect_terminal_year_expansion(
        candidate,
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        site_score=site_score,
        positive_similarity=positive,
        pv_fraction=fraction,
        local_dark=dark,
        sand_risk=sand,
        confuser=confuser,
    )
    recall = np.count_nonzero(accepted & extension) / extension.sum()
    assert recall > 0.85
    assert np.any(source[extension] == 3)
    assert any(d.accepted and d.source == 3 for d in decisions)


def test_adjacent_persistent_sand_is_not_recovered() -> None:
    args = _terminal_case(high_sand=True)
    candidate, probability, valid, inventory, ref, site_score, positive, fraction, dark, sand, confuser, extension = args
    accepted, source, _, _ = detect_terminal_year_expansion(
        candidate,
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        site_score=site_score,
        positive_similarity=positive,
        pv_fraction=fraction,
        local_dark=dark,
        sand_risk=sand,
        confuser=confuser,
    )
    assert not np.any(accepted & extension)
    assert not np.any(source[extension] > 0)


def test_stage7_terminal_source_is_preserved_by_temporal_decoder() -> None:
    n, h, w = 15, 20, 24
    ref = 13
    inventory = np.zeros((h, w), dtype=bool)
    inventory[5:11, 5:11] = True
    terminal = np.zeros((h, w), dtype=np.uint8)
    terminal[6:12, 11:16] = 3
    probability = np.full((n, h, w), 0.05, dtype=np.float32)
    valid = np.ones_like(probability, dtype=bool)
    candidate = np.zeros_like(probability, dtype=bool)

    persistent, onset, _, source = reconstruct_historical_outside_tracks(
        candidate,
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        terminal_source_override=terminal,
    )
    assert np.all(persistent[14][terminal == 3])
    assert np.all(onset[terminal == 3] == 14)
    assert np.array_equal(source, terminal)


def test_terminal_extension_is_rendered_as_new_installation() -> None:
    h, w = 12, 14
    inventory = np.zeros((h, w), dtype=bool)
    inventory[2:8, 2:7] = True
    extension = np.zeros((h, w), dtype=bool)
    extension[3:8, 7:11] = True
    previous_active = inventory.copy()
    previous_cumulative = inventory.copy()
    active = inventory | extension
    cumulative = inventory | extension
    classes = _exclusive_classes(active, previous_active, cumulative, previous_cumulative)
    assert np.all(classes[extension] == 2)
    assert np.all(classes[inventory] == 1)
