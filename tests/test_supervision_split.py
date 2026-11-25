from __future__ import annotations

import pandas as pd

from pvchange.preparation import _assign_supervision_roles


def test_independent_unit_roles_cover_all_positive_strata() -> None:
    registry = pd.DataFrame({
        "Id": range(1, 31),
        "has_mapped_pv": [True] * 24 + [False] * 6,
        "strong_positive_pixels": list(range(10, 250, 10)) + [0] * 6,
    })
    cfg = {
        "project": {"random_seed": 2027},
        "supervision": {"split": {
            "train_fraction": 0.70,
            "calibration_fraction": 0.15,
            "test_fraction": 0.15,
            "minimum_positive_train_units": 5,
            "minimum_positive_calibration_units": 3,
            "minimum_positive_test_units": 3,
        }},
    }
    result = _assign_supervision_roles(registry, cfg)
    assert result["Id"].nunique() == 30
    assert set(result["split_role"]) == {"train", "calibration", "test", "inference_only"}
    positive = result.loc[result["has_mapped_pv"]]
    assert positive.loc[positive.split_role.eq("train"), "Id"].nunique() >= 5
    assert positive.loc[positive.split_role.eq("calibration"), "Id"].nunique() >= 3
    assert positive.loc[positive.split_role.eq("test"), "Id"].nunique() >= 3
