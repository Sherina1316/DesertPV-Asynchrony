from __future__ import annotations

import numpy as np

from pvchange.temporal import forward_backward, first_persistent_year


def test_no_evidence_does_not_create_persistent_pv() -> None:
    years = np.arange(2011, 2026, dtype=np.int16)
    probability = np.full((len(years), 2), 0.5, dtype=np.float32)
    valid = np.zeros_like(probability, dtype=bool)
    posterior = forward_backward(probability, birth=0.002, retirement=0.005, initial_pv=0.001)
    decoded = (posterior >= 0.5) & valid
    onset = first_persistent_year(decoded, valid, years, persistence=2)
    assert np.array_equal(onset, np.zeros(2, dtype=np.int16))


def test_persistent_high_probability_has_onset() -> None:
    years = np.arange(2011, 2017, dtype=np.int16)
    p = np.array([[.01], [.02], [.10], [.85], [.92], [.95]], dtype=np.float32)
    valid = np.ones_like(p, dtype=bool)
    post = forward_backward(p, birth=.002, retirement=.005, initial_pv=.001)
    mask = (post >= .5) & valid
    onset = first_persistent_year(mask, valid, years, persistence=2)
    assert onset[0] >= 2014

import pandas as pd
from pvchange.temporal import build_endpoint_aware_synchrony_audit


def _sync_cfg() -> dict:
    return {
        "study": {"start_year": 2011, "reference_year": 2025},
        "temporal_decoding": {
            "maximum_synchronous_peak_fraction": 0.30,
            "synchrony_gate_roles": ["test"],
            "exclude_reference_year_from_synchrony_gate": True,
            "exclude_start_year_from_synchrony_gate": True,
            "synchrony_minimum_peak_gain_km2": 0.01,
            "minimum_synchrony_gate_units": 4,
            "minimum_synchronous_peak_units": 3,
            "synchrony_familywise_alpha": 0.05,
        },
    }


def test_reference_year_anchor_concentration_is_not_hard_gate() -> None:
    rows = []
    years = np.arange(2011, 2026, dtype=np.int16)
    # Train/calibration units all jump in 2025 due endpoint anchoring.
    for unit, role in [(1, "train"), (2, "train"), (3, "calibration"), (4, "calibration")]:
        for year in years:
            rows.append({"Id": unit, "year": int(year), "split_role": role,
                         "pv_area_km2": 10.0 if year == 2025 else 0.0})
    # Independent test units have staggered interior peaks.
    for unit, peak in [(5, 2016), (6, 2018), (7, 2020), (8, 2022)]:
        area = 0.0
        for year in years:
            if year >= peak:
                area += 1.0
            rows.append({"Id": unit, "year": int(year), "split_role": "test", "pv_area_km2": area})
    audit, diag = build_endpoint_aware_synchrony_audit(pd.DataFrame(rows), _sync_cfg(), years)
    assert diag["reference_year_peak_fraction_all_units_raw"] >= 0.5
    assert diag["synchrony_gate_evaluable"] is True
    assert diag["synchronous_anomaly"] is False
    gate = audit[audit["cohort"].eq("independent_unanchored_internal_year_gate")]
    assert 2025 not in gate.loc[gate["year_in_gate_window"], "year"].tolist()


def test_internal_test_cohort_synchrony_still_fails() -> None:
    rows = []
    years = np.arange(2011, 2026, dtype=np.int16)
    for unit in range(1, 11):
        area = 0.0
        for year in years:
            if year >= 2020:
                area += 1.0
            rows.append({"Id": unit, "year": int(year), "split_role": "test", "pv_area_km2": area})
    _, diag = build_endpoint_aware_synchrony_audit(pd.DataFrame(rows), _sync_cfg(), years)
    assert diag["synchrony_gate_evaluable"] is True
    assert diag["synchronous_anomaly"] is True
    assert diag["synchronous_anomaly_year"] == 2020
    assert diag["maximum_synchronous_peak_unit_count"] == 10


def test_sparse_active_units_do_not_create_pipeline_wide_anomaly() -> None:
    rows = []
    years = np.arange(2011, 2026, dtype=np.int16)
    # Twenty-one independent test units; only three have substantive gains and
    # two of those peak in 2021. Active-only concentration is 2/3, but the
    # pipeline-wide concentration is only 2/21 and must not hard fail.
    for unit in range(1, 22):
        area = 0.0
        peak = 2021 if unit in {1, 2} else (2019 if unit == 3 else None)
        for year in years:
            if peak is not None and year >= peak:
                area += 1.0
            rows.append({"Id": unit, "year": int(year), "split_role": "test", "pv_area_km2": area})
    _, diag = build_endpoint_aware_synchrony_audit(pd.DataFrame(rows), _sync_cfg(), years)
    assert diag["synchrony_gate_evaluable"] is True
    assert diag["synchrony_gate_active_gain_units"] == 3
    assert np.isclose(diag["maximum_active_only_peak_fraction_diagnostic"], 2 / 3)
    assert np.isclose(diag["maximum_synchronous_peak_fraction_observed"], 2 / 21)
    assert diag["synchronous_anomaly"] is False
