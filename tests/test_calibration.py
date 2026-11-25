from __future__ import annotations

import numpy as np
import pandas as pd

from pvchange.calibration import Reservoir, _choose_threshold, _threshold_row


def test_vectorized_reservoir_is_bounded_and_tracks_population() -> None:
    reservoir = Reservoir(1000, np.random.default_rng(42))
    reservoir.update(np.arange(10000, dtype=np.float32))
    reservoir.update(np.arange(10000, 20000, dtype=np.float32))
    assert reservoir.seen == 20000
    assert len(reservoir.array()) == 1000
    assert 7000 < float(reservoir.array().mean()) < 13000


def test_threshold_selection_enforces_scene_constraints() -> None:
    y = np.array([1] * 100 + [0] * 900, dtype=np.uint8)
    p = np.array([0.9] * 80 + [0.4] * 20 + [0.7] * 20 + [0.1] * 880)
    w = np.ones_like(p, dtype=float)
    frame = pd.DataFrame([_threshold_row(y, p, w, t) for t in [0.3, 0.5, 0.8]])
    cfg = {"calibration": {
        "minimum_precision": 0.6,
        "maximum_negative_fpr": 0.05,
        "maximum_area_relative_error": 0.25,
        "selection_weights": {"iou": .35, "f1": .25, "precision": .15, "area_accuracy": .15, "fpr_accuracy": .10},
    }}
    selected, feasible = _choose_threshold(frame, cfg)
    assert feasible
    assert selected["precision"] >= 0.6
    assert selected["negative_control_fpr"] <= 0.05
