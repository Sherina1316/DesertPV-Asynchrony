from __future__ import annotations

import numpy as np

from pvchange.temporal_signature import (
    annual_latent_masks_from_onset,
    detect_temporal_signature,
    robust_probability_harmonization,
)


def _cfg() -> dict:
    return {
        "calibration": {"minimum_anchor_recall": 0.9},
        "temporal_decoding": {"change_year_probability_threshold": 0.7},
        "temporal_signature": {
            "minimum_pre_years": 3,
            "minimum_post_years": 2,
            "minimum_probability_level_shift": 0.18,
            "minimum_fraction_level_shift": 0.08,
            "minimum_break_z": 2.0,
            "minimum_post_presence_fraction": 0.75,
            "maximum_pre_presence_fraction": 0.30,
            "maximum_post_probability_mad": 0.18,
            "maximum_post_annual_drop": 0.30,
            "minimum_transition_score": 0.58,
            "near_anchor_score_relaxation": 0.06,
            "remote_score_penalty": 0.08,
            "confuser_penalty_weight": 0.25,
            "stable_background_probability_ceiling": 0.22,
            "stable_background_maximum_range": 0.15,
            "harmonization_minimum_pixels": 4,
            "baseline_preexisting_enabled": True,
            "baseline_early_year_count": 3,
            "baseline_minimum_near_anchor_probability": 0.82,
            "baseline_minimum_remote_probability": 0.94,
            "baseline_minimum_near_anchor_fraction_evidence": 0.48,
            "baseline_minimum_remote_fraction_evidence": 0.68,
            "baseline_minimum_presence_fraction": 0.85,
            "baseline_maximum_confuser_penalty": 0.28,
            "baseline_maximum_probability_mad": 0.14,
            "baseline_maximum_annual_drop": 0.22,
        },
    }


def test_persistent_abrupt_installation_is_detected() -> None:
    p = np.array([0.08, 0.10, 0.11, 0.87, 0.91, 0.90], dtype=np.float32)[:, None, None]
    f = np.array([0.10, 0.11, 0.10, 0.62, 0.66, 0.65], dtype=np.float32)[:, None, None]
    c = np.full_like(p, 0.05)
    valid = np.ones_like(p, dtype=bool)
    result = detect_temporal_signature(p, f, c, valid, np.ones((1, 1), bool), _cfg())
    assert bool(result.qualified[0, 0])
    assert int(result.onset_index[0, 0]) == 3
    assert not bool(result.baseline_preexisting[0, 0])


def test_one_year_quality_spike_is_rejected() -> None:
    p = np.array([0.08, 0.09, 0.10, 0.93, 0.15, 0.11], dtype=np.float32)[:, None, None]
    f = np.array([0.10, 0.10, 0.11, 0.68, 0.14, 0.12], dtype=np.float32)[:, None, None]
    c = np.full_like(p, 0.05)
    result = detect_temporal_signature(p, f, c, np.ones_like(p, bool), np.ones((1, 1), bool), _cfg())
    assert not bool(result.qualified[0, 0])


def test_stationary_confuser_is_rejected() -> None:
    p = np.full((6, 1, 1), 0.88, dtype=np.float32)
    f = np.full_like(p, 0.42)
    c = np.full_like(p, 0.92)
    result = detect_temporal_signature(p, f, c, np.ones_like(p, bool), np.zeros((1, 1), bool), _cfg())
    assert not bool(result.qualified[0, 0])


def test_preexisting_near_anchor_facility_is_supported() -> None:
    p = np.array([0.91, 0.90, 0.92, 0.91, 0.90, 0.91], dtype=np.float32)[:, None, None]
    f = np.array([0.64, 0.63, 0.65, 0.66, 0.64, 0.65], dtype=np.float32)[:, None, None]
    c = np.full_like(p, 0.05)
    result = detect_temporal_signature(p, f, c, np.ones_like(p, bool), np.ones((1, 1), bool), _cfg())
    assert bool(result.qualified[0, 0])
    assert bool(result.baseline_preexisting[0, 0])
    assert int(result.onset_index[0, 0]) == 0


def test_latent_state_does_not_retire_when_observation_is_missing() -> None:
    onset = np.array([[2]], dtype=np.int16)
    qualified = np.array([[True]])
    latent = annual_latent_masks_from_onset(onset, qualified, 6)
    assert latent[:, 0, 0].tolist() == [False, False, True, True, True, True]


def test_harmonization_removes_global_year_bias() -> None:
    base = np.array([0.08, 0.10, 0.09, 0.11], dtype=np.float32)
    p = np.stack([base, base + 0.08, base - 0.01], axis=0).reshape(3, 2, 2)
    valid = np.ones_like(p, dtype=bool)
    adjusted, bias = robust_probability_harmonization(p, valid, _cfg())
    medians = np.nanmedian(adjusted.reshape(3, -1), axis=1)
    assert float(np.ptp(medians)) < float(np.ptp(np.nanmedian(p.reshape(3, -1), axis=1)))
    assert bias.shape == (3,)
