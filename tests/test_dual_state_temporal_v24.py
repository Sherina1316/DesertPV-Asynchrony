from __future__ import annotations

import numpy as np

from pvchange.temporal_signature import (
    annual_active_masks_from_evidence,
    annual_latent_masks_from_onset,
    detect_temporal_signature,
    filter_transition_components,
)


def _cfg() -> dict:
    return {
        "study": {"start_year": 2011, "end_year": 2025, "reference_year": 2024},
        "temporal_decoding": {"change_year_probability_threshold": 0.7},
        "temporal_signature": {
            "minimum_pre_years": 3,
            "minimum_post_years": 2,
            "recent_installation_window_years": 3,
            "minimum_probability_level_shift": 0.18,
            "minimum_fraction_level_shift": 0.08,
            "minimum_break_z": 2.0,
            "minimum_post_presence_fraction": 0.75,
            "maximum_pre_presence_fraction": 0.30,
            "maximum_post_probability_mad": 0.18,
            "maximum_post_annual_drop": 0.30,
            "minimum_transition_score": 0.58,
            "minimum_anchor_transition_score": 0.46,
            "minimum_near_transition_score": 0.52,
            "anchor_core_score_relaxation": 0.12,
            "near_anchor_score_relaxation": 0.06,
            "remote_score_penalty": 0.08,
            "confuser_penalty_weight": 0.25,
            "baseline_preexisting_enabled": True,
            "baseline_early_year_count": 4,
            "baseline_minimum_anchor_early_valid_years": 1,
            "baseline_minimum_near_early_valid_years": 2,
            "baseline_minimum_remote_early_valid_years": 3,
            "baseline_minimum_anchor_probability": 0.62,
            "baseline_minimum_near_anchor_probability": 0.74,
            "baseline_minimum_remote_probability": 0.92,
            "baseline_minimum_anchor_fraction_evidence": 0.18,
            "baseline_minimum_near_anchor_fraction_evidence": 0.35,
            "baseline_minimum_remote_fraction_evidence": 0.62,
            "baseline_minimum_anchor_presence_fraction": 0.58,
            "baseline_minimum_near_presence_fraction": 0.68,
            "baseline_minimum_remote_presence_fraction": 0.84,
            "baseline_maximum_anchor_confuser_penalty": 0.48,
            "baseline_maximum_near_confuser_penalty": 0.38,
            "baseline_maximum_remote_confuser_penalty": 0.26,
            "anchor_minimum_pre_observations": 1,
            "near_anchor_minimum_pre_observations": 2,
            "recent_anchor_minimum_post_observations": 1,
            "recent_near_minimum_post_observations": 1,
            "anchor_core_minimum_component_pixels": 1,
            "near_anchor_minimum_component_pixels": 2,
            "remote_minimum_component_pixels": 16,
            "minimum_bbox_fill": 0.06,
            "maximum_component_area_fraction": 0.12,
            "edge_recovery_iterations": 2,
            "edge_recovery_minimum_score": 0.42,
            "edge_recovery_minimum_reference_probability": 0.55,
            "edge_recovery_maximum_confuser_penalty": 0.55,
        },
        "active_state": {
            "retirement_probability_maximum": 0.22,
            "retirement_fraction_maximum": 0.18,
            "retirement_consecutive_valid_years": 3,
            "reactivation_probability_minimum": 0.72,
            "reactivation_fraction_minimum": 0.30,
            "reactivation_consecutive_valid_years": 2,
        },
    }


def test_preexisting_anchor_survives_missing_early_observations() -> None:
    p = np.full((15, 1, 1), 0.91, np.float32)
    f = np.full_like(p, 0.62)
    c = np.full_like(p, 0.05)
    valid = np.ones_like(p, bool)
    valid[0:3] = False
    valid[3:] = True
    result = detect_temporal_signature(
        p, f, c, valid, np.ones((1, 1), bool), _cfg(), anchor_core=np.ones((1, 1), bool)
    )
    assert bool(result.qualified[0, 0])
    assert bool(result.baseline_preexisting[0, 0])


def test_recent_anchor_installation_is_not_rejected_by_right_censoring() -> None:
    p = np.full((15, 1, 1), 0.08, np.float32)
    f = np.full_like(p, 0.08)
    p[13:] = 0.96
    f[13:] = 0.66
    c = np.full_like(p, 0.03)
    valid = np.ones_like(p, bool)
    valid[14] = False  # only one valid post-installation observation
    result = detect_temporal_signature(
        p, f, c, valid, np.ones((1, 1), bool), _cfg(), anchor_core=np.ones((1, 1), bool)
    )
    assert bool(result.qualified[0, 0])
    assert int(result.onset_index[0, 0]) == 13
    assert bool(result.recent_right_censored[0, 0])


def test_cumulative_and_active_states_are_separate() -> None:
    onset = np.array([[2]], dtype=np.int16)
    qualified = np.array([[True]])
    p = np.array([0.05, 0.05, 0.90, 0.88, 0.86, 0.10, 0.08, 0.07], np.float32)[:, None, None]
    f = np.array([0.05, 0.05, 0.60, 0.58, 0.55, 0.10, 0.09, 0.08], np.float32)[:, None, None]
    valid = np.ones_like(p, bool)
    active, observed, retirement = annual_active_masks_from_evidence(
        onset, qualified, p, f, valid, _cfg()
    )
    cumulative = annual_latent_masks_from_onset(onset, qualified, len(p))
    assert cumulative[:, 0, 0].tolist() == [False, False, True, True, True, True, True, True]
    assert active[:, 0, 0].tolist() == [False, False, True, True, True, True, True, False]
    assert observed[:, 0, 0].tolist() == active[:, 0, 0].tolist()
    assert int(retirement[0, 0]) == 7


def test_anchor_connected_edge_pixels_are_recovered() -> None:
    raw = np.zeros((12, 12), bool)
    raw[5, 2:10] = True
    anchor = np.zeros_like(raw)
    anchor[5, 5] = True
    near = np.zeros_like(raw)
    near[3:8, 1:11] = True
    score = np.full(raw.shape, 0.8, np.float32)
    probability = np.full(raw.shape, 0.9, np.float32)
    confuser = np.full(raw.shape, 0.05, np.float32)
    filtered = filter_transition_components(
        raw,
        near,
        _cfg(),
        transition_score=score,
        reference_probability=probability,
        confuser_penalty=confuser,
        anchor_core=anchor,
    )
    assert np.array_equal(filtered, raw)
