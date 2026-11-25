from __future__ import annotations

import numpy as np

from pvchange.multisource_refinement import _object_filter, _temporal_evidence


def _cfg() -> dict:
    return {
        "multisource_refinement": {
            "minimum_component_pixels": 4,
            "remote_minimum_component_pixels": 8,
            "minimum_bbox_fill": 0.08,
            "maximum_elongation": 12.0,
            "minimum_object_mean_score": 0.55,
            "minimum_object_mean_votes": 2.0,
            "minimum_object_mean_persistence": 0.35,
        }
    }


def test_temporal_evidence_rewards_persistent_installation() -> None:
    prob = np.zeros((6, 8, 8), dtype=np.float32)
    spec = np.zeros_like(prob)
    valid = np.ones_like(prob, dtype=bool)
    prob[3:, 2:6, 2:6] = 0.90
    spec[3:, 2:6, 2:6] = 0.85
    change, persistence, spectral_change = _temporal_evidence(prob, spec, valid, 0.48)
    assert float(change[3, 3, 3]) > 0.9
    assert float(spectral_change[3, 3, 3]) > 0.9
    assert float(persistence[4, 3, 3]) >= 0.66
    assert float(persistence[1, 3, 3]) < float(persistence[4, 3, 3])


def test_object_filter_suppresses_small_remote_false_positive_and_keeps_anchor() -> None:
    candidate = np.zeros((32, 32), dtype=bool)
    candidate[2:4, 2:4] = True  # remote 4-pixel false positive
    candidate[10:15, 10:15] = True  # supported object
    candidate[25, 25] = True  # one-pixel anchor must remain
    score = np.zeros((32, 32), dtype=np.float32)
    score[candidate] = 0.75
    votes = np.zeros((32, 32), dtype=np.uint8)
    votes[candidate] = 4
    persistence = np.zeros((32, 32), dtype=np.float32)
    persistence[candidate] = 0.8
    anchor = np.zeros((32, 32), dtype=bool)
    anchor[25, 25] = True
    near = np.zeros((32, 32), dtype=bool)
    near[8:18, 8:18] = True
    near[24:27, 24:27] = True
    filtered, rejection = _object_filter(candidate, score, votes, persistence, anchor, near, _cfg())
    assert not filtered[2:4, 2:4].any()
    assert filtered[10:15, 10:15].all()
    assert filtered[25, 25]
    assert int(rejection[2, 2]) == 4
