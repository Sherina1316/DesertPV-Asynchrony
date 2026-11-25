from __future__ import annotations

import numpy as np

from pvchange.temporal_signature import refine_anchor_onset_with_change_boundary_coherence


def _cube(event_year: int = 3, scattered: bool = False):
    t, h, w = 8, 28, 28
    anchor = np.zeros((h, w), dtype=bool)
    anchor[8:20, 8:20] = True
    obj = np.zeros_like(anchor)
    obj[10:18, 10:18] = True
    p = np.full((t, h, w), 0.05, dtype=np.float32)
    f = np.full((t, h, w), 0.02, dtype=np.float32)
    c = np.full((t, h, w), 0.20, dtype=np.float32)
    valid = np.ones((t, h, w), dtype=bool)
    refl = np.full((t, 6, h, w), 0.34, dtype=np.float32)
    # desert background with weak vegetation/brightness signal
    refl[:, 0] = 0.22
    refl[:, 1] = 0.24
    refl[:, 2] = 0.25
    refl[:, 3] = 0.33
    refl[:, 4] = 0.30
    refl[:, 5] = 0.26
    target = obj
    if scattered:
        target = np.zeros_like(anchor)
        target[10, 10] = True
        target[12, 18] = True
        target[18, 12] = True
        target[16, 16] = True
    for i in range(event_year, t):
        p[i][target] = 0.36
        f[i][target] = 0.14
        refl[i, 0][target] = 0.11
        refl[i, 1][target] = 0.12
        refl[i, 2][target] = 0.10
        refl[i, 3][target] = 0.10
        refl[i, 4][target] = 0.09
        refl[i, 5][target] = 0.08
    sim = np.zeros((t, h, w), dtype=np.float32)
    sim[:, anchor] = 0.25
    sim[event_year:, target] = 0.62
    return p, f, c, valid, refl, sim, anchor, obj


def _cfg():
    return {
        "study": {"start_year": 2011, "end_year": 2018, "reference_year": 2018},
        "change_cbc_sankey": {
            "enabled": True,
            "decode_until_year": 2018,
            "pre_window_years": 2,
            "post_window_years": 2,
            "future_support_window_years": 4,
            "future_support_minimum_years": 2,
            "future_presence_threshold": 0.28,
            "object_seed_threshold": 0.42,
            "object_growth_threshold": 0.26,
            "minimum_boundary_score": 0.12,
            "minimum_transition_score": 0.12,
            "minimum_future_support": 0.50,
            "minimum_object_pixels": 8,
            "minimum_seed_pixels": 2,
            "minimum_object_mean_score": 0.24,
            "minimum_object_q80_score": 0.32,
            "internal_gap_closing_iterations": 1,
            "weak_object_growth_iterations": 1,
            "maximum_anchor_confuser": 0.75,
            "minimum_current_probability": 0.10,
            "minimum_current_fraction": 0.04,
            "minimum_current_similarity": 0.36,
        },
    }


def test_change_cbc_sankey_recovers_early_coherent_object_without_pre_event_copy():
    p, f, c, valid, refl, sim, anchor, obj = _cube(event_year=3, scattered=False)
    onset = np.full(anchor.shape, -1, dtype=np.int16)
    qualified = np.zeros(anchor.shape, dtype=bool)
    onset2, q2, rescued, diag = refine_anchor_onset_with_change_boundary_coherence(
        onset, qualified, anchor, p, f, c, valid, refl, _cfg(), spectral_similarity=sim
    )
    assert rescued[obj].mean() > 0.80
    assert int(np.nanmedian(onset2[obj])) == 3
    assert not np.any((onset2[obj] >= 0) & (onset2[obj] < 3))
    assert diag[obj].max() > 0
    assert q2[obj].mean() > 0.80


def test_change_cbc_sankey_rejects_scattered_similar_bare_pixels():
    p, f, c, valid, refl, sim, anchor, _ = _cube(event_year=3, scattered=True)
    onset = np.full(anchor.shape, -1, dtype=np.int16)
    qualified = np.zeros(anchor.shape, dtype=bool)
    onset2, q2, rescued, _ = refine_anchor_onset_with_change_boundary_coherence(
        onset, qualified, anchor, p, f, c, valid, refl, _cfg(), spectral_similarity=sim
    )
    assert rescued.sum() == 0
    assert q2.sum() == 0
    assert np.all(onset2 < 0)
