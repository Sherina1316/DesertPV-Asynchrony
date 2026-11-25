from __future__ import annotations

import numpy as np

from pvchange.spectral_mixture import pavg_fraction, unmix_lsma


def test_lsma_recovers_simple_endmember_mixture() -> None:
    endmembers = np.array([
        [0.40, 0.05, 0.06, 0.00],
        [0.45, 0.05, 0.10, 0.00],
        [0.50, 0.04, 0.05, 0.00],
        [0.55, 0.04, 0.50, 0.00],
        [0.48, 0.03, 0.24, 0.00],
        [0.42, 0.02, 0.14, 0.00],
    ], dtype=np.float32)
    true_fraction = np.array([0.25, 0.35, 0.30, 0.10], dtype=np.float32)
    pixel = (endmembers @ true_fraction).reshape(6, 1, 1)
    fraction, rmse = unmix_lsma(pixel, endmembers)
    assert np.allclose(fraction[:, 0, 0].sum(), 1.0, atol=1e-6)
    assert np.allclose(fraction[:, 0, 0], true_fraction, atol=0.03)
    assert float(rmse[0, 0]) < 0.01


def test_pavg_uses_panel_and_shadow_increment() -> None:
    vg = np.array([0.20], dtype=np.float32)
    la = np.array([0.30], dtype=np.float32)
    sh = np.array([0.20], dtype=np.float32)
    pavg = pavg_fraction(vg, la, sh, baseline_la=0.10, baseline_sh=0.10)
    # denominator = 1 - 0.2 - 0.1 = 0.7
    assert np.allclose(pavg, np.array([0.20 / 0.70], dtype=np.float32), atol=1e-6)
