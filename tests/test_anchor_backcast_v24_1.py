from __future__ import annotations

import numpy as np

from pvchange.mapping import _exclusive_classes
from pvchange.temporal_signature import (
    detect_temporal_signature,
    propagate_anchor_backcast_within_objects,
)


def _cfg() -> dict:
    return {
        "study": {"start_year": 2011, "end_year": 2025, "reference_year": 2024},
        "temporal_decoding": {"change_year_probability_threshold": 0.70},
        "temporal_signature": {
            "baseline_preexisting_enabled": False,
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
            "minimum_anchor_transition_score": 0.40,
            "minimum_near_transition_score": 0.52,
            "anchor_minimum_pre_observations": 1,
            "near_anchor_minimum_pre_observations": 2,
            "recent_anchor_minimum_post_observations": 1,
            "recent_near_minimum_post_observations": 1,
            "recent_installation_window_years": 3,
            "confuser_penalty_weight": 0.25,
            "near_anchor_score_relaxation": 0.06,
            "anchor_core_score_relaxation": 0.12,
            "remote_score_penalty": 0.08,
        },
        "anchor_backcasting": {
            "enabled": True,
            "target_start_year": 2011,
            "target_end_year": 2015,
            "local_pre_window_years": 2,
            "local_post_window_years": 2,
            "minimum_pre_observations": 1,
            "minimum_post_observations": 1,
            "minimum_probability_shift": 0.08,
            "minimum_fraction_shift": 0.025,
            "minimum_break_z": 0.75,
            "minimum_evidence_votes": 2,
            "minimum_later_presence_fraction": 0.25,
            "later_presence_probability_threshold": 0.55,
            "maximum_confuser_penalty": 0.62,
            "minimum_score": 0.42,
            "reference_probability_support_threshold": 0.45,
            "reference_fraction_support_threshold": 0.08,
            "earliest_credible_onset_preference": 0.03,
            "object_consensus_enabled": True,
            "object_consensus_minimum_supported_fraction": 0.20,
            "object_consensus_maximum_year_spread": 3,
            "object_consensus_minimum_supported_pixels": 1,
        },
    }


def test_early_anchor_change_is_backcast_with_short_post_support() -> None:
    # Single anchor pixel changes in 2013. Only one immediately post-event
    # observation is needed by the retrospective route; later/reference support
    # confirms that it is not a transient false positive.
    probability = np.full((15, 1, 1), 0.10, dtype=np.float32)
    fraction = np.full((15, 1, 1), 0.03, dtype=np.float32)
    confuser = np.full((15, 1, 1), 0.05, dtype=np.float32)
    valid = np.ones((15, 1, 1), dtype=bool)
    probability[2:, 0, 0] = 0.90
    fraction[2:, 0, 0] = 0.45
    # Only 2011, 2012, 2013 and the 2024 endpoint are observable. This is
    # intentionally insufficient for the standard multi-year post-event rule.
    valid[3:, 0, 0] = False
    valid[13, 0, 0] = True
    anchor = np.ones((1, 1), dtype=bool)
    cfg = _cfg()
    cfg["temporal_signature"]["minimum_post_years"] = 4

    result = detect_temporal_signature(
        probability, fraction, confuser, valid, anchor, cfg, anchor_core=anchor
    )

    assert result.qualified[0, 0]
    assert result.anchor_backcast[0, 0]
    assert int(result.onset_index[0, 0]) == 2  # 2013


def test_anchor_without_historical_change_is_not_backcast() -> None:
    probability = np.full((15, 1, 1), 0.12, dtype=np.float32)
    fraction = np.full((15, 1, 1), 0.03, dtype=np.float32)
    confuser = np.full((15, 1, 1), 0.05, dtype=np.float32)
    valid = np.ones((15, 1, 1), dtype=bool)
    # Strong endpoint only, no 2011-2015 transition evidence.
    probability[13, 0, 0] = 0.98
    fraction[13, 0, 0] = 0.60
    anchor = np.ones((1, 1), dtype=bool)

    result = detect_temporal_signature(
        probability, fraction, confuser, valid, anchor, _cfg(), anchor_core=anchor
    )

    assert not result.anchor_backcast[0, 0]


def test_object_consensus_never_expands_outside_reference_anchor() -> None:
    anchor = np.zeros((4, 5), dtype=bool)
    anchor[1:3, 1:4] = True
    onset = np.full(anchor.shape, -1, dtype=np.int16)
    direct = np.zeros(anchor.shape, dtype=bool)
    score = np.zeros(anchor.shape, dtype=np.float32)
    direct[1, 1:3] = True
    onset[1, 1:3] = 2
    score[1, 1:3] = 0.8

    propagated_onset, propagated, filled = propagate_anchor_backcast_within_objects(
        onset, direct, anchor, score, _cfg()
    )

    assert np.all(~propagated[~anchor])
    assert np.all(~filled[~anchor])
    assert np.all(propagated[anchor])
    assert np.all(propagated_onset[anchor] == 2)


def test_exclusive_map_classes_are_not_stacked() -> None:
    active = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    previous_active = np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
    cumulative = np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)
    previous_cumulative = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)

    classes = _exclusive_classes(active, previous_active, cumulative, previous_cumulative)

    assert set(np.unique(classes)).issubset({0, 1, 2, 3})
    # Every pixel has a single categorical code rather than multiple overlays.
    one_hot = np.stack([classes == code for code in (1, 2, 3)], axis=0)
    assert np.all(one_hot.sum(axis=0) <= 1)
    assert classes[0, 0] == 1  # existing active
    assert classes[0, 1] == 2  # newly installed and active
    assert classes[0, 2] == 3  # installed but inactive
