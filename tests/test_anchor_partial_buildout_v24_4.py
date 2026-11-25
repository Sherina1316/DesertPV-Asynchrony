from __future__ import annotations

import numpy as np

from pvchange.temporal_signature import (
    compute_anchor_reference_spectral_similarity,
    rescue_anchor_internal_onset_from_trajectory,
)


def _cfg():
    return {
        "study": {"start_year": 2011, "reference_year": 2024},
        "anchor_full_series_decoder": {
            "enabled": True,
            "probability_weight": 0.28,
            "fraction_weight": 0.22,
            "spectral_similarity_weight": 0.38,
            "low_confuser_weight": 0.12,
            "reference_scale_floor": 0.01,
            "reference_similarity_temperature": 2.8,
            "local_reference_window_pixels": 7,
            "minimum_local_reference_pixels": 3,
            "same_pixel_similarity_weight": 0.75,
            "global_similarity_weight": 0.25,
            "pixel_presence_weight": 0.70,
            "neighbourhood_presence_weight": 0.30,
            "neighbourhood_radius_pixels": 1,
            "annual_pixel_presence_threshold": 0.36,
            "annual_weak_presence_threshold": 0.28,
            "annual_strong_presence_threshold": 0.48,
            "pixel_spectral_similarity_threshold": 0.42,
            "strong_similarity_threshold": 0.56,
            "anchor_probability_floor": 0.14,
            "anchor_fraction_floor": 0.06,
            "strong_fraction_floor": 0.09,
            "anchor_maximum_confuser": 0.70,
            "minimum_supported_years_in_window": 2,
            "support_window_years": 3,
            "maximum_internal_gap_years": 1,
            "annual_minimum_component_pixels": 4,
            "annual_minimum_strong_pixels": 2,
            "annual_closing_iterations": 1,
            "backward_continuity_radius_pixels": 1,
        },
    }


def test_partial_2013_buildout_is_not_lost_by_large_2024_object_median():
    cfg = _cfg()
    t, h, w = 15, 30, 30
    reference_index = 13

    anchor = np.zeros((h, w), dtype=bool)
    anchor[4:26, 4:26] = True  # large 2024 station
    early = np.zeros_like(anchor)
    early[8:15, 7:15] = True  # only a small sub-object existed in 2013
    later = np.zeros_like(anchor)
    later[15:23, 15:24] = True  # expansion beginning in 2018

    valid = np.ones((t, h, w), dtype=bool)
    reflectance = np.full((t, 6, h, w), 0.28, dtype=np.float32)
    desert = np.asarray([0.20, 0.22, 0.25, 0.31, 0.34, 0.32], np.float32)
    pv = np.asarray([0.055, 0.065, 0.060, 0.070, 0.080, 0.075], np.float32)
    for year in range(t):
        reflectance[year][:, anchor] = desert[:, None]
    # Early sub-object visible from 2013 (index 2) onward.
    for year in range(2, reference_index + 1):
        reflectance[year][:, early] = pv[:, None]
    # Later expansion from 2018 (index 7) onward.
    for year in range(7, reference_index + 1):
        reflectance[year][:, later] = pv[:, None]
    # By 2024 all anchor pixels are PV.
    reflectance[reference_index][:, anchor] = pv[:, None]

    similarity = compute_anchor_reference_spectral_similarity(
        reflectance,
        anchor,
        valid,
        reference_index,
        cfg,
    )

    probability = np.full((t, h, w), 0.06, dtype=np.float32)
    fraction = np.full((t, h, w), 0.025, dtype=np.float32)
    confuser = np.full((t, h, w), 0.20, dtype=np.float32)
    probability[2:reference_index + 1, early] = 0.38
    fraction[2:reference_index + 1, early] = 0.12
    probability[7:reference_index + 1, later] = 0.44
    fraction[7:reference_index + 1, later] = 0.15
    probability[reference_index, anchor] = 0.92
    fraction[reference_index, anchor] = 0.50

    onset = np.full((h, w), -1, dtype=np.int16)
    qualified = np.zeros((h, w), dtype=bool)
    onset, qualified, rescued = rescue_anchor_internal_onset_from_trajectory(
        onset,
        qualified,
        anchor,
        probability,
        fraction,
        confuser,
        valid,
        cfg,
        spectral_similarity=similarity,
    )

    assert rescued[10, 10]
    assert int(onset[10, 10]) == 2  # 2013
    assert rescued[18, 18]
    assert int(onset[18, 18]) == 7  # 2018
    # Unbuilt 2013 portion must not be copied backward merely because it is in 2024 SHP.
    assert int(onset[24, 24]) != 2


def test_unsupported_2012_is_not_invented_before_2013():
    cfg = _cfg()
    t, h, w = 15, 12, 12
    anchor = np.zeros((h, w), dtype=bool)
    anchor[2:10, 2:10] = True
    valid = np.ones((t, h, w), dtype=bool)
    p = np.full((t, h, w), 0.05, dtype=np.float32)
    f = np.full((t, h, w), 0.02, dtype=np.float32)
    c = np.full((t, h, w), 0.15, dtype=np.float32)
    sim = np.full((t, h, w), 0.10, dtype=np.float32)
    p[2:5, 4:8, 4:8] = 0.45
    f[2:5, 4:8, 4:8] = 0.15
    sim[2:5, 4:8, 4:8] = 0.72
    p[13, anchor] = 0.90
    f[13, anchor] = 0.50
    sim[13, anchor] = 1.0

    onset = np.full((h, w), -1, dtype=np.int16)
    q = np.zeros((h, w), dtype=bool)
    onset, q, _ = rescue_anchor_internal_onset_from_trajectory(
        onset, q, anchor, p, f, c, valid, cfg, spectral_similarity=sim
    )
    assert int(onset[5, 5]) == 2
    assert not np.any(onset[4:8, 4:8] == 1)
