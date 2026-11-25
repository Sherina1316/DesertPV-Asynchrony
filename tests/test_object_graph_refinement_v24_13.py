from __future__ import annotations

import numpy as np

from pvchange.object_graph_refinement import (
    CLASS_ADJACENCY_RESCUE,
    CLASS_BACKGROUND,
    REJECT_HOLLOW_BOUNDARY,
    REJECT_SAND_TEXTURE,
    _filter_objects,
    _prototype_similarity,
    _temporal_object_graph,
)


def _settings() -> dict:
    return {
        "minimum_adjacent_component_pixels": 4,
        "minimum_remote_component_pixels": 12,
        "adjacent_minimum_mean_score": 0.50,
        "adjacent_minimum_positive_similarity": 0.34,
        "adjacent_maximum_sand_score": 0.64,
        "adjacent_minimum_solid_fill": 0.48,
        "adjacent_maximum_elongation": 28.0,
        "remote_minimum_mean_score": 0.64,
        "remote_minimum_q75_score": 0.68,
        "remote_minimum_positive_similarity": 0.48,
        "remote_maximum_sand_score": 0.46,
        "remote_minimum_persistence": 0.55,
        "remote_minimum_onset": 0.48,
        "remote_preexisting_minimum_stability": 0.76,
        "remote_minimum_solid_fill": 0.70,
        "remote_minimum_oriented_fill": 0.24,
        "remote_maximum_elongation": 18.0,
        "remote_minimum_boundary_contrast": -0.015,
        "sand_rejection_threshold": 0.54,
        "minimum_solid_fill": 0.55,
        "minimum_oriented_fill": 0.16,
        "maximum_object_elongation": 30.0,
        "minimum_object_persistence": 0.38,
        "minimum_positive_similarity": 0.30,
        "temporal_graph_link_radius_pixels": 2,
        "temporal_rescue_minimum_score": 0.50,
        "temporal_rescue_minimum_positive_similarity": 0.35,
        "temporal_rescue_maximum_sand_score": 0.56,
        "singleton_keep_minimum_onset": 0.72,
        "singleton_keep_minimum_stability": 0.78,
    }


def test_prototype_similarity_prefers_matching_spectrum() -> None:
    cube = np.zeros((2, 2, 2), dtype=np.float32)
    cube[:, 0, 0] = [0.2, 0.3]
    cube[:, 0, 1] = [0.22, 0.31]
    cube[:, 1, 0] = [0.8, 0.9]
    cube[:, 1, 1] = [0.0, 0.0]
    centers = np.array([[0.2, 0.3]], dtype=np.float32)
    scales = np.array([[0.05, 0.05]], dtype=np.float32)
    similarity = _prototype_similarity(cube, centers, scales)
    assert similarity[0, 0] > 0.99
    assert similarity[0, 1] > similarity[1, 0]


def test_adjacent_block_is_accepted_but_hollow_sand_ring_is_rejected() -> None:
    shape = (40, 40)
    candidate = np.zeros(shape, dtype=bool)
    candidate[5:12, 5:13] = True
    candidate[20:35, 20] = True
    candidate[20:35, 34] = True
    candidate[20, 20:35] = True
    candidate[34, 20:35] = True
    seed = np.zeros(shape, dtype=bool)
    seed[6:10, 6:9] = True
    adjacency = np.zeros(shape, dtype=bool)
    adjacency[5:12, 5:13] = True
    remote = candidate & ~adjacency
    anchor = np.zeros(shape, dtype=bool)
    score = np.full(shape, 0.72, dtype=np.float32)
    positive = np.full(shape, 0.65, dtype=np.float32)
    sand = np.full(shape, 0.20, dtype=np.float32)
    sand[20:35, 20:35] = 0.90
    onset = np.full(shape, 0.65, dtype=np.float32)
    persistence = np.full(shape, 0.80, dtype=np.float32)
    stable = np.full(shape, 0.80, dtype=np.float32)
    brightness = np.full(shape, 0.50, dtype=np.float32)
    brightness[5:12, 5:13] = 0.20

    accepted, classes, rejection, rows = _filter_objects(
        candidate, seed, adjacency, remote, anchor, score, positive, sand,
        onset, persistence, stable, brightness, _settings(), 2020, 6,
    )
    assert accepted[7, 10]
    assert classes[7, 10] != CLASS_BACKGROUND
    assert not accepted[20, 25]
    assert rejection[20, 25] in {REJECT_SAND_TEXTURE, REJECT_HOLLOW_BOUNDARY}
    assert any(row["accepted"] for row in rows)
    assert any(not row["accepted"] for row in rows)


def test_temporal_graph_rescues_adjacent_omission_and_removes_weak_singleton() -> None:
    masks = np.zeros((3, 30, 30), dtype=bool)
    masks[0, 10:15, 10:15] = True
    masks[2, 10:15, 10:15] = True
    masks[1, 3:5, 3:5] = True  # weak isolated artefact
    classes = np.zeros_like(masks, dtype=np.uint8)
    rejection = np.zeros_like(masks, dtype=np.uint8)
    growable = np.zeros_like(masks)
    growable[1, 10:15, 10:15] = True
    growable[1, 3:5, 3:5] = True
    score = np.full(masks.shape, 0.60, dtype=np.float32)
    positive = np.full(masks.shape, 0.50, dtype=np.float32)
    sand = np.full(masks.shape, 0.20, dtype=np.float32)
    onset = np.full(masks.shape, 0.40, dtype=np.float32)
    stable = np.full(masks.shape, 0.50, dtype=np.float32)
    anchor = np.zeros_like(masks)

    out_masks, out_classes, _ = _temporal_object_graph(
        masks, classes, rejection, growable, score, positive, sand, onset, stable, anchor, _settings()
    )
    assert out_masks[1, 12, 12]
    assert out_classes[1, 12, 12] == CLASS_ADJACENCY_RESCUE
    assert not out_masks[1, 3, 3]
