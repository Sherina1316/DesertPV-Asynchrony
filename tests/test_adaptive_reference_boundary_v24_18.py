import numpy as np

from pvchange.authoritative_reference import (
    _multiband_reference_break,
    _refine_manual_reference_boundary,
)


def _settings():
    return {
        "reference_core_erosion_pixels": 1,
        "reference_boundary_search_pixels": 2,
        "reference_core_fraction": 0.80,
        "boundary_addition_minimum_likelihood": 0.50,
        "boundary_addition_minimum_break": 0.35,
        "boundary_addition_minimum_support": 0.40,
        "boundary_addition_maximum_sand": 0.60,
        "boundary_addition_maximum_confuser": 0.50,
        "boundary_removal_maximum_likelihood": 0.40,
        "boundary_removal_maximum_break": 0.35,
        "boundary_removal_minimum_sand": 0.55,
        "boundary_removal_minimum_confuser": 0.55,
        "boundary_removal_maximum_pv_fraction": 0.15,
        "minimum_boundary_correction_pixels": 2,
        "maximum_reference_area_adjustment_fraction": 0.50,
    }


def test_adaptive_boundary_keeps_core_and_restricts_changes_to_band():
    anchor = np.zeros((20, 20), bool)
    anchor[6:14, 6:14] = True
    frac = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.55, np.float32)
    support = np.full(anchor.shape, 0.55, np.float32)
    change = np.full(anchor.shape, 0.45, np.float32)
    sand = np.zeros(anchor.shape, np.float32)
    conf = np.zeros(anchor.shape, np.float32)
    dark = np.full(anchor.shape, 0.5, np.float32)
    pvfrac = np.full(anchor.shape, 0.5, np.float32)

    refined, _, band, code, diag = _refine_manual_reference_boundary(
        anchor, frac, valid, score, support, change, sand, conf, dark, pvfrac, _settings()
    )
    core = code == 1
    assert np.all(refined[core])
    changed = refined ^ anchor
    assert not np.any(changed & ~band)
    assert diag["corrections_outside_boundary_band"] == 0
    assert diag["core_recall"] == 1.0


def test_strong_adjacent_boundary_is_added_but_remote_dark_patch_is_not():
    anchor = np.zeros((24, 24), bool)
    anchor[8:16, 8:16] = True
    frac = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.20, np.float32)
    support = np.full(anchor.shape, 0.20, np.float32)
    change = np.full(anchor.shape, 0.10, np.float32)
    sand = np.full(anchor.shape, 0.20, np.float32)
    conf = np.zeros(anchor.shape, np.float32)
    dark = np.full(anchor.shape, 0.20, np.float32)
    pvfrac = np.full(anchor.shape, 0.10, np.float32)

    # One-pixel adjacent strip with strong PV and change evidence.
    score[9:15, 16] = 0.95
    support[9:15, 16] = 0.90
    change[9:15, 16] = 0.90
    dark[9:15, 16] = 0.90
    pvfrac[9:15, 16] = 0.90
    # Remote dark patch has no installation support and lies outside the band.
    score[1:4, 1:4] = 0.95
    dark[1:4, 1:4] = 0.95

    refined, _, band, _, _ = _refine_manual_reference_boundary(
        anchor, frac, valid, score, support, change, sand, conf, dark, pvfrac, _settings()
    )
    assert refined[10:14, 16].all()
    assert not refined[1:4, 1:4].any()
    assert not band[1:4, 1:4].any()


def test_sand_like_manual_boundary_can_be_removed_but_core_cannot():
    anchor = np.zeros((20, 20), bool)
    anchor[5:15, 5:15] = True
    frac = anchor.astype(np.float32)
    valid = np.ones_like(anchor)
    score = np.full(anchor.shape, 0.80, np.float32)
    support = np.full(anchor.shape, 0.80, np.float32)
    change = np.full(anchor.shape, 0.70, np.float32)
    sand = np.full(anchor.shape, 0.10, np.float32)
    conf = np.zeros(anchor.shape, np.float32)
    dark = np.full(anchor.shape, 0.70, np.float32)
    pvfrac = np.full(anchor.shape, 0.70, np.float32)

    # Make top boundary row strongly sand-like with no break/PV fraction.
    score[5, 7:13] = 0.05
    support[5, 7:13] = 0.05
    change[5, 7:13] = 0.02
    sand[5, 7:13] = 0.95
    pvfrac[5, 7:13] = 0.01

    refined, _, _, code, _ = _refine_manual_reference_boundary(
        anchor, frac, valid, score, support, change, sand, conf, dark, pvfrac, _settings()
    )
    assert not refined[5, 8:12].any()
    assert np.all(refined[7:13, 7:13])
    assert np.any(code[5, 7:13] == 3)


def test_multiband_break_is_high_for_installation_and_low_for_stable_sand():
    years, h, w = 15, 4, 4
    ref = 13
    valid = np.ones((years, h, w), bool)
    base = {
        "positive_similarity": np.full((years, h, w), 0.15, np.float32),
        "pv_fraction": np.full((years, h, w), 0.10, np.float32),
        "local_dark": np.full((years, h, w), 0.10, np.float32),
        "brightness": np.full((years, h, w), 0.55, np.float32),
        "bsi": np.full((years, h, w), 0.55, np.float32),
        "ndvi": np.full((years, h, w), 0.35, np.float32),
        "ndbi": np.full((years, h, w), 0.05, np.float32),
    }
    # Pixel (1,1) undergoes a coherent PV-like break in 2024/2025.
    for name, value in {
        "positive_similarity": 0.90,
        "pv_fraction": 0.85,
        "local_dark": 0.80,
        "brightness": 0.20,
        "bsi": 0.10,
        "ndvi": 0.05,
        "ndbi": 0.50,
    }.items():
        base[name][ref:, 1, 1] = value
    out = _multiband_reference_break(base, valid, ref)
    assert out[1, 1] > 0.70
    assert out[0, 0] < 0.50
