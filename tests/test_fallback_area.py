from __future__ import annotations

import numpy as np

from pvchange.fallback import _best_positive_break, _weighted_quantile


def test_object_break_detects_positive_pv_transition() -> None:
    years = np.arange(2011, 2026)
    signal = np.array([0.05] * 7 + [0.45] * 8, dtype=float)
    year, score = _best_positive_break(years, signal, minimum_observations=6, min_seg=3)
    assert year == 2018
    assert score > 0


def test_weighted_quantile_respects_geographic_analog_weight() -> None:
    values = np.array([1.0, 4.0, 10.0])
    weights = np.array([0.8, 0.15, 0.05])
    assert _weighted_quantile(values, weights, 0.5) < 4.0
