from __future__ import annotations

import numpy as np

from pvchange.growth_front import (
    build_terminal_extension_evidence,
    detect_terminal_year_expansion,
)


def _settings() -> dict[str, float | int | bool]:
    return {
        "terminal_baseline_years": 3,
        "terminal_adjacent_search_pixels": 8,
        "terminal_perimeter_search_pixels": 14,
        "terminal_remote_seed_probability": 0.68,
        "terminal_remote_minimum_gain": 0.12,
        "terminal_remote_extent_probability": 0.54,
        "terminal_remote_extent_gain": 0.07,
        "terminal_remote_minimum_pixels": 14,
        "terminal_remote_mean_probability": 0.62,
        "terminal_remote_mean_gain": 0.095,
        "terminal_remote_minimum_bbox_fill": 0.07,
        "terminal_remote_minimum_solidity": 0.34,
        "terminal_remote_maximum_sand": 0.55,
        "terminal_remote_maximum_confuser": 0.55,
        "terminal_remote_minimum_novelty": 0.34,
        "terminal_remote_maximum_preexisting": 0.48,
        "terminal_remote_maximum_persistent_risk": 0.52,
        "terminal_remote_census_density_window": 7,
        "terminal_remote_census_density_likelihood": 0.30,
        "terminal_remote_census_density_darkness": 0.22,
        "terminal_remote_census_density_change": 0.015,
        "terminal_remote_census_density_novelty": 0.18,
        "terminal_remote_census_density_direct_change": 0.10,
        "terminal_remote_census_density_maximum_preexisting": 0.58,
        "terminal_remote_census_density_maximum_persistent_risk": 0.62,
        "terminal_remote_census_density_maximum_sand": 0.75,
        "terminal_remote_census_density_maximum_confuser": 0.72,
        "terminal_remote_census_seed_likelihood": 0.44,
        "terminal_remote_census_seed_darkness": 0.34,
        "terminal_remote_census_seed_density": 0.16,
        "terminal_remote_census_seed_change": 0.07,
        "terminal_remote_census_seed_novelty": 0.42,
        "terminal_remote_census_seed_direct_change": 0.20,
        "terminal_remote_census_seed_maximum_preexisting": 0.48,
        "terminal_remote_census_seed_maximum_prior_dark_persistence": 0.48,
        "terminal_remote_census_seed_maximum_persistent_risk": 0.52,
        "terminal_remote_census_seed_maximum_sand": 0.66,
        "terminal_remote_census_seed_maximum_confuser": 0.64,
        "terminal_remote_census_strong_darkness": 0.68,
        "terminal_remote_census_strong_dark_minimum_novelty": 0.48,
        "terminal_remote_census_strong_dark_minimum_dominance": 0.30,
        "terminal_remote_census_extent_likelihood": 0.29,
        "terminal_remote_census_extent_darkness": 0.20,
        "terminal_remote_census_extent_density": 0.08,
        "terminal_remote_census_extent_change": 0.012,
        "terminal_remote_census_extent_novelty": 0.13,
        "terminal_remote_census_extent_direct_change": 0.06,
        "terminal_remote_census_extent_maximum_preexisting": 0.64,
        "terminal_remote_census_extent_maximum_persistent_risk": 0.66,
        "terminal_remote_census_extent_maximum_sand": 0.72,
        "terminal_remote_census_extent_maximum_confuser": 0.70,
        "terminal_remote_census_bridge_iterations": 2,
        "terminal_remote_census_bridge_likelihood": 0.24,
        "terminal_remote_census_bridge_darkness": 0.15,
        "terminal_remote_census_bridge_novelty": 0.08,
        "terminal_remote_census_bridge_maximum_preexisting": 0.70,
        "terminal_remote_census_bridge_maximum_persistent_risk": 0.72,
        "terminal_remote_census_bridge_maximum_sand": 0.76,
        "terminal_remote_census_bridge_maximum_confuser": 0.74,
        "terminal_remote_census_object_closing_iterations": 2,
        "terminal_remote_census_minimum_pixels": 18,
        "terminal_remote_census_mean_likelihood": 0.34,
        "terminal_remote_census_mean_change": 0.045,
        "terminal_remote_census_mean_darkness": 0.27,
        "terminal_remote_census_mean_novelty": 0.30,
        "terminal_remote_census_mean_direct_change": 0.17,
        "terminal_remote_census_mean_break_dominance": 0.20,
        "terminal_remote_census_maximum_preexisting_risk": 0.46,
        "terminal_remote_census_maximum_prior_dark_persistence": 0.42,
        "terminal_remote_census_maximum_persistent_nonpv_risk": 0.48,
        "terminal_remote_census_minimum_seed_fraction": 0.015,
        "terminal_remote_census_object_change_pixel": 0.035,
        "terminal_remote_census_object_dark_pixel": 0.30,
        "terminal_remote_census_object_novelty_pixel": 0.32,
        "terminal_remote_census_object_direct_change_pixel": 0.17,
        "terminal_remote_census_minimum_change_fraction": 0.10,
        "terminal_remote_census_minimum_dark_fraction": 0.24,
        "terminal_remote_census_minimum_novelty_fraction": 0.14,
        "terminal_remote_census_minimum_direct_change_fraction": 0.13,
        "terminal_remote_census_minimum_bbox_fill": 0.03,
        "terminal_remote_census_minimum_solidity": 0.16,
        "terminal_remote_census_object_maximum_sand": 0.66,
        "terminal_remote_census_object_maximum_confuser": 0.64,
        "terminal_brightness_drop_scale": 0.055,
        "terminal_direct_brightness_drop_scale": 0.045,
        "terminal_break_dominance_margin": 0.06,
        "terminal_break_dominance_scale": 0.34,
    }


def _arrays() -> tuple[np.ndarray, ...]:
    n, h, w = 15, 130, 150
    ref = 13
    valid = np.ones((n, h, w), dtype=bool)
    inventory = np.zeros((h, w), dtype=bool)
    inventory[5:18, 5:20] = True
    candidate = np.zeros((n, h, w), dtype=bool)
    probability = np.full((n, h, w), 0.08, dtype=np.float32)
    score = np.full_like(probability, 0.10)
    positive = np.full_like(probability, 0.08)
    fraction = np.full_like(probability, 0.06)
    dark = np.full_like(probability, 0.10)
    brightness = np.full_like(probability, 0.48)
    sand = np.full_like(probability, 0.14)
    confuser = np.full_like(probability, 0.10)
    probability[ref][inventory] = 1.0
    return (
        candidate,
        probability,
        valid,
        inventory,
        ref,
        score,
        positive,
        fraction,
        dark,
        brightness,
        sand,
        confuser,
    )


def test_true_2025_remote_site_has_high_novelty_and_is_accepted() -> None:
    args = list(_arrays())
    candidate, probability, valid, inventory, ref, score, positive, fraction, dark, brightness, sand, confuser = args
    remote = np.zeros(inventory.shape, dtype=bool)
    remote[82:108, 104:137] = True
    brightness[14, remote] = 0.10
    dark[14, remote] = 0.42
    score[14, remote] = 0.24
    positive[14, remote] = 0.19
    fraction[14, remote] = 0.23
    probability[14, remote] = 0.19

    evidence = build_terminal_extension_evidence(
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        site_score=score,
        positive_similarity=positive,
        pv_fraction=fraction,
        local_dark=dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
    )
    assert float(np.mean(evidence["terminal_novelty_score"][remote])) > 0.45
    assert float(np.mean(evidence["preexisting_state_risk"][remote])) < 0.35

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
        local_dark=dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
        precomputed_evidence=evidence,
    )
    assert np.count_nonzero(accepted & remote) / remote.sum() > 0.75
    assert np.any(source[remote] == 4)


def test_longstanding_dark_nonpv_with_small_2025_noise_is_rejected() -> None:
    args = list(_arrays())
    candidate, probability, valid, inventory, ref, score, positive, fraction, dark, brightness, sand, confuser = args
    old_dark = np.zeros(inventory.shape, dtype=bool)
    old_dark[80:110, 100:138] = True

    # This parcel is dark and model-confusing for years.  A small 2025
    # fluctuation must not be interpreted as installation.
    brightness[9:14, old_dark] = 0.135
    brightness[14, old_dark] = 0.115
    dark[9:14, old_dark] = 0.38
    dark[14, old_dark] = 0.42
    score[9:14, old_dark] = 0.22
    score[14, old_dark] = 0.25
    positive[9:14, old_dark] = 0.18
    positive[14, old_dark] = 0.20
    fraction[9:14, old_dark] = 0.20
    fraction[14, old_dark] = 0.23
    probability[9:14, old_dark] = 0.18
    probability[14, old_dark] = 0.21

    evidence = build_terminal_extension_evidence(
        probability,
        valid,
        inventory,
        ref,
        _settings(),
        site_score=score,
        positive_similarity=positive,
        pv_fraction=fraction,
        local_dark=dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
    )
    assert float(np.mean(evidence["prior_dark_persistence"][old_dark])) > 0.70
    assert float(np.mean(evidence["persistent_nonpv_risk"][old_dark])) > 0.50

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
        local_dark=dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
        precomputed_evidence=evidence,
    )
    assert not np.any(accepted & old_dark)
    assert not np.any(source[old_dark] > 0)


def test_persistent_high_model_false_positive_cannot_use_legacy_remote_path() -> None:
    args = list(_arrays())
    candidate, probability, valid, inventory, ref, score, positive, fraction, dark, brightness, sand, confuser = args
    false_site = np.zeros(inventory.shape, dtype=bool)
    false_site[78:108, 102:137] = True
    probability[10:14, false_site] = 0.72
    probability[14, false_site] = 0.83
    score[10:14, false_site] = 0.60
    score[14, false_site] = 0.65
    positive[10:14, false_site] = 0.55
    positive[14, false_site] = 0.58
    fraction[10:14, false_site] = 0.50
    fraction[14, false_site] = 0.55
    dark[10:14, false_site] = 0.55
    dark[14, false_site] = 0.58
    brightness[10:14, false_site] = 0.13
    brightness[14, false_site] = 0.12

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
        local_dark=dark,
        brightness=brightness,
        sand_risk=sand,
        confuser=confuser,
    )
    assert not np.any(accepted & false_site)
    assert not np.any(source[false_site] == 2)
    assert not np.any(source[false_site] == 4)
