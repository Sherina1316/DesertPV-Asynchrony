import numpy as np

from pvchange.study_area import study_ids, expected_counts
from pvchange.temporal_signature import refine_anchor_onset_with_landsat_object_pavg


def _cfg():
    return {
        "study": {"start_year": 2011, "end_year": 2025, "reference_year": 2024, "selected_unit_ids": [1, 6, 18, 28], "id_start": 1, "id_end": 142},
        "landsat_object_pavg": {
            "enabled": True,
            "anchor_internal_enabled": True,
            "local_window_pixels": 5,
            "seed_score_threshold": 0.40,
            "grow_score_threshold": 0.24,
            "minimum_object_pixels": 4,
            "minimum_seed_pixels": 1,
            "minimum_object_mean_score": 0.24,
            "minimum_object_q75_score": 0.24,
            "minimum_dark_contrast": 0.02,
            "minimum_texture_score": 0.01,
            "minimum_future_support": 0.30,
            "future_support_window_years": 4,
            "future_support_minimum_years": 2,
            "maximum_anchor_confuser": 0.90,
            "grow_iterations": 1,
            "gap_close_iterations": 1,
            "hole_fill_max_pixels": 20,
            "probability_weight": 0.18,
            "fraction_weight": 0.16,
            "similarity_weight": 0.16,
            "dark_contrast_weight": 0.22,
            "texture_weight": 0.12,
            "boundary_weight": 0.08,
            "future_weight": 0.08,
        },
    }


def test_selected_noncontiguous_units_contract():
    cfg = _cfg()
    assert study_ids(cfg) == [1, 6, 18, 28]
    assert expected_counts(cfg) == (4, 15, 60)


def test_landsat_object_pavg_recovers_coherent_early_object_not_scattered_pixels():
    cfg = _cfg()
    t, h, w = 15, 24, 24
    anchor = np.zeros((h, w), dtype=bool)
    anchor[8:16, 8:16] = True
    valid = np.ones((t, h, w), dtype=bool)
    p = np.full((t, h, w), 0.05, dtype=np.float32)
    f = np.full((t, h, w), 0.02, dtype=np.float32)
    c = np.full((t, h, w), 0.20, dtype=np.float32)
    sim = np.zeros((t, h, w), dtype=np.float32)
    refl = np.full((t, 6, h, w), 0.55, dtype=np.float32)

    # Continuous dark PV-like object appears from 2014 onward inside anchor.
    obj = np.zeros((h, w), dtype=bool)
    obj[10:15, 10:15] = True
    for i in range(3, t):
        refl[i, :, obj] = 0.18
        p[i, obj] = 0.16
        f[i, obj] = 0.06
        sim[i, obj] = 0.45

    # Scattered dark pixels outside the coherent object should not be accepted.
    scattered = [(3, 3), (5, 20), (20, 5), (19, 19)]
    for i in range(3, t):
        for y, x in scattered:
            refl[i, :, y, x] = 0.16
            p[i, y, x] = 0.15
            f[i, y, x] = 0.06
            sim[i, y, x] = 0.45

    onset = np.full((h, w), -1, dtype=np.int16)
    qualified = np.zeros((h, w), dtype=bool)
    onset2, qualified2, rescued, score = refine_anchor_onset_with_landsat_object_pavg(
        onset, qualified, anchor, p, f, c, valid, refl, cfg, spectral_similarity=sim
    )

    assert rescued[obj].sum() >= 16
    assert np.nanmedian(onset2[obj][qualified2[obj]]) <= 3
    for y, x in scattered:
        assert not qualified2[y, x]
    assert float(np.nanmax(score[obj])) > 0.0
