from __future__ import annotations

import numpy as np

from pvchange.growth_front import detect_terminal_year_expansion


def _settings() -> dict[str, float | int | bool]:
    return {
        "terminal_baseline_years": 3,
        "terminal_adjacent_search_pixels": 8,
        "terminal_perimeter_search_pixels": 14,
        "terminal_adjacent_seed_probability": 0.60,
        "terminal_adjacent_minimum_gain": 0.10,
        "terminal_remote_seed_probability": 0.80,
        "terminal_remote_minimum_gain": 0.18,
        "terminal_absolute_override_probability": 0.92,
        "terminal_absolute_override_gain": 0.14,
        "terminal_extent_probability": 0.50,
        "terminal_extent_minimum_gain": 0.10,
        "terminal_nonmask_extent_probability": 0.70,
        "terminal_nonmask_extent_gain": 0.16,
        "terminal_closing_iterations": 1,
        "terminal_remote_census_density_window": 7,
        "terminal_remote_census_density_likelihood": 0.30,
        "terminal_remote_census_density_darkness": 0.22,
        "terminal_remote_census_density_change": 0.015,
        "terminal_remote_census_density_maximum_sand": 0.75,
        "terminal_remote_census_density_maximum_confuser": 0.72,
        "terminal_remote_census_seed_likelihood": 0.44,
        "terminal_remote_census_seed_darkness": 0.34,
        "terminal_remote_census_seed_density": 0.16,
        "terminal_remote_census_seed_change": 0.07,
        "terminal_remote_census_seed_maximum_sand": 0.66,
        "terminal_remote_census_seed_maximum_confuser": 0.64,
        "terminal_remote_census_strong_darkness": 0.68,
        "terminal_remote_census_strong_dark_minimum_change": 0.03,
        "terminal_remote_census_extent_likelihood": 0.29,
        "terminal_remote_census_extent_darkness": 0.20,
        "terminal_remote_census_extent_density": 0.08,
        "terminal_remote_census_extent_change": 0.012,
        "terminal_remote_census_extent_maximum_sand": 0.72,
        "terminal_remote_census_extent_maximum_confuser": 0.70,
        "terminal_remote_census_bridge_iterations": 3,
        "terminal_remote_census_bridge_likelihood": 0.24,
        "terminal_remote_census_bridge_darkness": 0.15,
        "terminal_remote_census_bridge_maximum_sand": 0.76,
        "terminal_remote_census_bridge_maximum_confuser": 0.74,
        "terminal_remote_census_object_closing_iterations": 2,
        "terminal_remote_census_minimum_pixels": 18,
        "terminal_remote_census_mean_likelihood": 0.34,
        "terminal_remote_census_mean_change": 0.045,
        "terminal_remote_census_mean_darkness": 0.27,
        "terminal_remote_census_minimum_seed_fraction": 0.015,
        "terminal_remote_census_object_change_pixel": 0.035,
        "terminal_remote_census_object_dark_pixel": 0.30,
        "terminal_remote_census_minimum_change_fraction": 0.10,
        "terminal_remote_census_minimum_dark_fraction": 0.24,
        "terminal_remote_census_minimum_bbox_fill": 0.03,
        "terminal_remote_census_minimum_solidity": 0.16,
        "terminal_remote_census_object_maximum_sand": 0.66,
        "terminal_remote_census_object_maximum_confuser": 0.64,
        "terminal_brightness_drop_scale": 0.055,
        "terminal_remote_probability_gain_scale": 0.14,
    }


def _case(stable_dark: bool = False):
    n, h, w = 15, 150, 160
    ref = 13
    valid = np.ones((n, h, w), dtype=bool)
    inventory = np.zeros((h, w), dtype=bool)
    inventory[8:22, 8:24] = True

    # A new site more than 100 pixels from the nearest 2024 component.
    remote = np.zeros((h, w), dtype=bool)
    remote[96:122, 112:145] = True
    # Simulate panel-row gaps that must be bridged at object scale.
    remote[102:105, 112:145] = False
    remote[113:116, 112:145] = False

    probability = np.full((n, h, w), 0.08, dtype=np.float32)
    score = np.full_like(probability, 0.10)
    positive = np.full_like(probability, 0.08)
    fraction = np.full_like(probability, 0.06)
    local_dark = np.full_like(probability, 0.10)
    brightness = np.full_like(probability, 0.48)
    sand = np.full_like(probability, 0.16)
    confuser = np.full_like(probability, 0.10)
    candidate = np.zeros((n, h, w), dtype=bool)

    probability[ref][inventory] = 1.0
    score[ref][inventory] = 0.80
    positive[ref][inventory] = 0.82
    fraction[ref][inventory] = 0.74
    local_dark[ref][inventory] = 0.70
    brightness[ref][inventory] = 0.14

    if stable_dark:
        brightness[10:15, remote] = 0.12
        local_dark[10:15, remote] = 0.38
        score[10:15, remote] = 0.20
        positive[10:15, remote] = 0.16
        fraction[10:15, remote] = 0.18
        probability[10:15, remote] = 0.16
    else:
        # Intentionally weak model/prototype evidence and no annual candidate.
        # The direct 2024-to-2025 brightness transition is the decisive signal.
        brightness[14, remote] = 0.10
        local_dark[14, remote] = 0.40
        score[14, remote] = 0.23
        positive[14, remote] = 0.18
        fraction[14, remote] = 0.22
        probability[14, remote] = 0.18

    return (
        candidate,
        probability,
        valid,
        inventory,
        ref,
        score,
        positive,
        fraction,
        local_dark,
        brightness,
        sand,
        confuser,
        remote,
    )


def test_far_2025_site_is_found_without_candidate_or_distance_support() -> None:
    args = _case(stable_dark=False)
    (
        candidate, probability, valid, inventory, ref, score, positive,
        fraction, local_dark, brightness, sand, confuser, remote,
    ) = args
    accepted, source, _, decisions = detect_terminal_year_expansion(
        candidate,
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        site_score=score,
        positive_similarity=positive,
        pv_fraction=fraction,
        local_dark=local_dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
    )
    recall = np.count_nonzero(accepted & remote) / max(int(remote.sum()), 1)
    assert recall > 0.78
    assert np.count_nonzero(source == 4) > 500
    assert any(d.accepted and d.source == 4 and not d.adjacent_to_reference for d in decisions)


def test_stable_far_dark_sand_is_rejected_by_change_contract() -> None:
    args = _case(stable_dark=True)
    (
        candidate, probability, valid, inventory, ref, score, positive,
        fraction, local_dark, brightness, sand, confuser, remote,
    ) = args
    accepted, source, _, _ = detect_terminal_year_expansion(
        candidate,
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        site_score=score,
        positive_similarity=positive,
        pv_fraction=fraction,
        local_dark=local_dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
    )
    assert not np.any(accepted & remote)
    assert not np.any(source[remote] == 4)
