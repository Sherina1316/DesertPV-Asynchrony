from __future__ import annotations

import numpy as np

from pvchange.anchor_guided import AnchorPrototype, anchor_guided_score, anchor_similarity_matrix, classify_anchor_guided
from pvchange.mapping import _annual_layers


def _cfg() -> dict:
    return {
        "study": {"reference_year": 2024},
        "anchor_guided": {
            "probability_weight": 0.60,
            "similarity_weight": 0.35,
            "uncertainty_penalty_weight": 0.05,
            "near_anchor_score_threshold": 0.62,
            "medium_score_threshold": 0.52,
            "remote_score_threshold": 0.82,
            "remote_probability_threshold": 0.90,
            "remote_similarity_threshold": 0.70,
            "minimum_similarity": 0.20,
            "minimum_component_pixels": 1,
            "remote_minimum_component_pixels": 1,
            "minimum_component_bbox_fill": 0.0,
            "binary_closing_iterations": 0,
        },
    }


def test_anchor_similarity_prefers_prototype_pixels() -> None:
    proto = AnchorPrototype(
        feature_names=("A", "B"),
        center=np.array([1.0, 2.0], dtype=np.float32),
        scale=np.array([0.2, 0.2], dtype=np.float32),
        anchor_similarity_floor=0.5,
        background_similarity_ceiling=0.2,
        anchor_pixels=20,
        background_pixels=100,
    )
    matrix = np.array([[1.0, 2.0], [1.1, 1.9], [4.0, -3.0]], dtype=np.float32)
    sim = anchor_similarity_matrix(matrix, ["A", "B"], proto)
    assert sim[0] > sim[1] > sim[2]
    assert sim[0] > 0.99


def test_reference_anchor_is_forced_only_in_reference_year() -> None:
    cfg = _cfg()
    probability = np.full((4, 4), 0.1, dtype=np.float32)
    similarity = np.full((4, 4), 0.1, dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    near = np.ones((4, 4), dtype=bool)
    anchor = np.zeros((4, 4), dtype=bool)
    anchor[1:3, 1:3] = True
    high_ref, _, _ = classify_anchor_guided(probability, similarity, None, near, valid, anchor, True, cfg)
    high_old, _, _ = classify_anchor_guided(probability, similarity, None, near, valid, anchor, False, cfg)
    assert np.all(high_ref[anchor])
    assert not np.any(high_old)


def test_remote_candidate_requires_strong_joint_evidence() -> None:
    cfg = _cfg()
    probability = np.array([[0.95, 0.95], [0.7, 0.99]], dtype=np.float32)
    similarity = np.array([[0.8, 0.2], [0.95, 0.9]], dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    near = np.zeros((2, 2), dtype=bool)
    high, _, score = classify_anchor_guided(probability, similarity, None, near, valid, None, False, cfg)
    assert high[0, 0]
    assert not high[0, 1]
    assert not high[1, 0]
    assert high[1, 1]
    assert np.all((score >= 0) & (score <= 1))


def test_annual_layers_partition_change() -> None:
    previous = np.array([[1, 1], [0, 0]], dtype=bool)
    current = np.array([[1, 0], [1, 0]], dtype=bool)
    persistent, gain, loss = _annual_layers(current, previous)
    assert persistent.sum() == 1
    assert gain.sum() == 1
    assert loss.sum() == 1
    assert not np.any(persistent & gain)
