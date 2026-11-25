from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import CORE_METRICS, TARGET_ALIASES


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[ok]
    y = values[ok]
    return float(np.polyfit(x, y, 1)[0])


def _within_region_rolling(
    shifted: pd.Series,
    region: pd.Series,
    window: int,
    min_periods: int,
    operation: str,
) -> pd.Series:
    """Rolling operation that cannot cross a region boundary."""
    grouped = shifted.groupby(region, sort=False)
    if operation == "mean":
        out = grouped.rolling(window, min_periods=min_periods).mean()
    elif operation == "std":
        out = grouped.rolling(window, min_periods=min_periods).std()
    elif operation == "slope":
        out = grouped.rolling(window, min_periods=min_periods).apply(_slope, raw=True)
    else:
        raise ValueError(f"Unsupported rolling operation: {operation}")
    return out.reset_index(level=0, drop=True).reindex(shifted.index)


def _fixed_baseline(df: pd.DataFrame, col: str, years: int) -> pd.Series:
    return df.groupby("region", sort=False)[col].transform(
        lambda s: pd.to_numeric(s.iloc[: min(years, len(s))], errors="coerce").median()
    )


def add_strict_lag_features(
    annual: pd.DataFrame,
    max_lag: int = 3,
    rolling_window: int = 3,
    baseline_years: int = 3,
    pv_threshold: float = 0.02,
) -> pd.DataFrame:
    """Create strict no-leakage, within-region temporal features.

    Rules
    -----
    1. Raw current-year metric values are never used as predictors.
    2. Target-derived predictors are lagged or fixed historical baselines.
    3. Differences, acceleration and rolling statistics use shifted history only.
    4. Rolling windows are computed separately within each region; values can never
       spill from the last year of one unit into the first year of another unit.
    """
    df = annual.copy().sort_values(["region", "year"]).reset_index(drop=True)
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    year_span = max(max_year - min_year, 1)
    region = df["region"]

    feature_data: dict[str, pd.Series | np.ndarray] = {
        "feat__year_index": df["year"] - min_year,
        "feat__year_norm": (df["year"] - min_year) / year_span,
        "feat__region_num": df["region"].astype(float),
    }

    metrics = [m for m in CORE_METRICS if m in df.columns]
    environmental = [
        c for c in df.columns
        if c.startswith("ENV_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    dynamic_exogenous = [
        c for c in [
            "Annual_Installed_Gain_km2",
            "Annual_Active_Gain_km2",
            "Annual_Active_Loss_km2",
            "Observed_Active_PV_Area_km2",
            "Observed_Active_PV_Ratio",
            "Inactive_or_Retired_Area_km2",
        ]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in metrics:
        numeric = pd.to_numeric(df[col], errors="coerce")
        g = numeric.groupby(region, sort=False)
        baseline = _fixed_baseline(df.assign(**{col: numeric}), col, baseline_years)
        feature_data[f"feat__{col}__baseline{baseline_years}"] = baseline

        lags: dict[int, pd.Series] = {}
        for lag in range(1, max_lag + 1):
            lags[lag] = g.shift(lag)
            feature_data[f"feat__{col}__lag{lag}"] = lags[lag]

        lag1 = lags.get(1, g.shift(1))
        lag2 = lags.get(2, g.shift(2))
        lag3 = lags.get(3, g.shift(3))
        feature_data[f"feat__{col}__prev_diff1"] = lag1 - lag2
        feature_data[f"feat__{col}__prev_diff2"] = lag2 - lag3
        feature_data[f"feat__{col}__prev_accel"] = (lag1 - lag2) - (lag2 - lag3)
        feature_data[f"feat__{col}__rolling{rolling_window}_mean"] = _within_region_rolling(
            lag1, region, rolling_window, 1, "mean"
        )
        feature_data[f"feat__{col}__rolling{rolling_window}_std"] = _within_region_rolling(
            lag1, region, rolling_window, 2, "std"
        )
        feature_data[f"feat__{col}__rolling{rolling_window}_slope"] = _within_region_rolling(
            lag1, region, rolling_window, 2, "slope"
        )
        feature_data[f"feat__{col}__lag1_anomaly_from_baseline"] = lag1 - baseline

    # Expansion dynamics distinguish genuine installation pulses from no-growth years.
    for col in dynamic_exogenous:
        numeric = pd.to_numeric(df[col], errors="coerce")
        g = numeric.groupby(region, sort=False)
        baseline = _fixed_baseline(df.assign(**{col: numeric}), col, baseline_years)
        lag1, lag2, lag3 = g.shift(1), g.shift(2), g.shift(3)
        feature_data[f"feat__{col}__baseline{baseline_years}"] = baseline
        feature_data[f"feat__{col}__lag1"] = lag1
        feature_data[f"feat__{col}__lag2"] = lag2
        feature_data[f"feat__{col}__lag3"] = lag3
        feature_data[f"feat__{col}__prev_diff1"] = lag1 - lag2
        feature_data[f"feat__{col}__rolling{rolling_window}_mean"] = _within_region_rolling(
            lag1, region, rolling_window, 1, "mean"
        )
        feature_data[f"feat__{col}__rolling{rolling_window}_slope"] = _within_region_rolling(
            lag1, region, rolling_window, 2, "slope"
        )

    # Environmental drivers are exogenous, but current-year climate is still excluded.
    for col in environmental:
        numeric = pd.to_numeric(df[col], errors="coerce")
        g = numeric.groupby(region, sort=False)
        baseline = _fixed_baseline(df.assign(**{col: numeric}), col, baseline_years)
        lag1, lag2 = g.shift(1), g.shift(2)
        feature_data[f"feat__{col}__baseline{baseline_years}"] = baseline
        feature_data[f"feat__{col}__lag1"] = lag1
        feature_data[f"feat__{col}__lag2"] = lag2
        feature_data[f"feat__{col}__prev_diff1"] = lag1 - lag2
        feature_data[f"feat__{col}__rolling{rolling_window}_mean"] = _within_region_rolling(
            lag1, region, rolling_window, 1, "mean"
        )
        feature_data[f"feat__{col}__rolling{rolling_window}_slope"] = _within_region_rolling(
            lag1, region, rolling_window, 2, "slope"
        )

    if "PV_Cover_Ratio" in df.columns:
        ratio = pd.to_numeric(df["PV_Cover_Ratio"], errors="coerce")
        ratio_lag1 = ratio.groupby(region, sort=False).shift(1)
        feature_data["feat__pv_ratio_lag1_stage"] = pd.Series(
            np.select(
                [
                    ratio_lag1 <= pv_threshold,
                    (ratio_lag1 > pv_threshold) & (ratio_lag1 < 0.10),
                    (ratio_lag1 >= 0.10) & (ratio_lag1 < 0.18),
                    ratio_lag1 >= 0.18,
                ],
                [0, 1, 2, 3],
                default=np.nan,
            ),
            index=df.index,
        )

        age = pd.Series(0.0, index=df.index, dtype=float)
        for _, idx in df.groupby("region", sort=False).groups.items():
            idx = list(idx)
            years = df.loc[idx, "year"].to_numpy(int)
            vals = ratio.loc[idx].to_numpy(float)
            positive = np.flatnonzero(np.isfinite(vals) & (vals > pv_threshold))
            if len(positive):
                first_year = int(years[positive[0]])
                age.loc[idx] = np.maximum(0, years - first_year - 1)
        feature_data["feat__years_since_first_installation_lagged"] = age
    else:
        feature_data["feat__pv_ratio_lag1_stage"] = pd.Series(0.0, index=df.index)
        feature_data["feat__years_since_first_installation_lagged"] = pd.Series(0.0, index=df.index)

    features = pd.DataFrame(feature_data, index=df.index)
    result = pd.concat([df, features], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def feature_columns_by_group(df: pd.DataFrame, group: str = "full") -> list[str]:
    cols = [c for c in df.columns if c.startswith("feat__") and pd.api.types.is_numeric_dtype(df[c])]

    if group == "base":
        keys = ["year_", "region_", "baseline", "__lag1", "__lag2", "__lag3"]
        return [c for c in cols if any(k in c for k in keys) and "PV_Cover" not in c and "pv_" not in c]
    if group == "base_pv":
        keys = ["year_", "region_", "baseline", "__lag1", "__lag2", "__lag3", "PV_Cover", "pv_", "installation"]
        return [c for c in cols if any(k in c for k in keys)]
    if group == "base_pv_rolling":
        keys = [
            "year_", "region_", "baseline", "__lag", "PV_Cover", "pv_", "installation",
            "rolling", "prev_diff", "prev_accel", "lag1_anomaly",
        ]
        return [c for c in cols if any(k in c for k in keys)]
    return cols


def audit_no_leakage(feature_cols: list[str], target: str) -> tuple[bool, list[str]]:
    """Check whether target-derived predictors are explicitly historical."""
    aliases = set(TARGET_ALIASES.get(target, [target]))
    suspicious = []
    allowed_tokens = ["__lag", "rolling", "prev_diff", "prev_accel", "baseline", "lag1_anomaly"]
    for feat in feature_cols:
        for alias in aliases:
            if alias.lower() in feat.lower() and not any(tok in feat for tok in allowed_tokens):
                suspicious.append(feat)
    return len(suspicious) == 0, suspicious


def feature_manifest(df: pd.DataFrame, targets: list[str], groups: list[str]) -> pd.DataFrame:
    rows = []
    for group in groups:
        cols = feature_columns_by_group(df, group)
        for target in targets:
            ok, suspicious = audit_no_leakage(cols, target)
            rows.append({
                "target": target,
                "feature_group": group,
                "n_features": len(cols),
                "leakage_audit_passed": bool(ok),
                "suspicious_features": ";".join(suspicious[:20]),
            })
    return pd.DataFrame(rows)
