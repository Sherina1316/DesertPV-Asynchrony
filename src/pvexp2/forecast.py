from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import add_strict_lag_features, audit_no_leakage, feature_columns_by_group
from .metrics import summarize
from .models import available_model_builders, fit_model, predict_model


PV_AREA_TARGET = "PV_Cover_Area_km2"
PV_RATIO_TARGET = "PV_Cover_Ratio"
PV_GAIN_TARGET = "Annual_Installed_Gain_km2"


@dataclass
class HurdleGrowthModel:
    classifiers: list[tuple[str, object]]
    magnitude_models: list[tuple[str, object]]
    constant_occurrence_probability: float
    fallback_positive_gain: float
    occurrence_threshold: float
    diagnostics: dict


def _bounds_for_target(target: str, hist_values: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(hist_values, errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return -np.inf, np.inf
    mn = float(np.nanmin(v))
    mx = float(np.nanmax(v))
    if "Ratio" in target or "EIPAVG" in target or "PAVG" in target or "ECRI" in target or "Vegetation" in target:
        return max(0.0, mn - 0.10), min(1.0, mx + 0.15)
    if "Area" in target:
        return 0.0, max(mx * 1.35, mx + 1.0)
    return mn - abs(mx - mn), mx + abs(mx - mn)


def _carry_forward_environment(working: pd.DataFrame, hist: pd.DataFrame, regions: list[int], year: int) -> None:
    """Project environmental drivers conservatively using bounded recent trends.

    These projected covariates are used only by the future scenario module. Historical
    PV mapping never lets environmental suitability create a PV detection by itself.
    """
    env_cols = [c for c in hist.columns if c.startswith("ENV_") and pd.api.types.is_numeric_dtype(hist[c])]
    for rid in regions:
        mask = (working["region"] == int(rid)) & (working["year"] == int(year))
        unit_hist = working[(working["region"] == int(rid)) & (working["year"] < int(year))].sort_values("year")
        for col in env_cols:
            values = pd.to_numeric(unit_hist[col], errors="coerce").dropna()
            if values.empty:
                continue
            recent = values.iloc[-5:].to_numpy(float)
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 3 else 0.0
            projected = float(recent[-1] + slope)
            historical = pd.to_numeric(hist.loc[hist["region"].eq(int(rid)), col], errors="coerce").dropna()
            if len(historical):
                lo = float(historical.quantile(0.02))
                hi = float(historical.quantile(0.98))
                margin = max((hi - lo) * 0.15, 1e-6)
                projected = float(np.clip(projected, lo - margin, hi + margin))
            working.loc[mask, col] = projected


def _ensure_growth_columns(annual: pd.DataFrame) -> pd.DataFrame:
    df = annual.copy().sort_values(["region", "year"]).reset_index(drop=True)
    if PV_AREA_TARGET not in df.columns:
        raise KeyError(f"{PV_AREA_TARGET} is required for the growth forecast")

    area = pd.to_numeric(df[PV_AREA_TARGET], errors="coerce").clip(lower=0.0)
    df[PV_AREA_TARGET] = area

    derived_gain = df.groupby("region")[PV_AREA_TARGET].diff().clip(lower=0.0)
    if PV_GAIN_TARGET in df.columns:
        supplied = pd.to_numeric(df[PV_GAIN_TARGET], errors="coerce").clip(lower=0.0)
        # Prefer the explicit dynamics table, but use cumulative-area differences where
        # the explicit value is missing or internally inconsistent.
        mismatch = (supplied - derived_gain).abs() > np.maximum(0.02, 0.25 * derived_gain.fillna(0.0))
        supplied = supplied.where(~mismatch, derived_gain)
        df[PV_GAIN_TARGET] = supplied.where(supplied.notna(), derived_gain).fillna(0.0)
    else:
        df[PV_GAIN_TARGET] = derived_gain.fillna(0.0)

    if "Valid_Area_km2" not in df.columns:
        valid = pd.Series(np.nan, index=df.index, dtype=float)
        if PV_RATIO_TARGET in df.columns:
            ratio = pd.to_numeric(df[PV_RATIO_TARGET], errors="coerce")
            valid = area / ratio.where(ratio > 1e-8)
        df["Valid_Area_km2"] = valid

    df["Valid_Area_km2"] = pd.to_numeric(df["Valid_Area_km2"], errors="coerce")
    df["Valid_Area_km2"] = df["Valid_Area_km2"].where(df["Valid_Area_km2"] > 0)
    df["Valid_Area_km2"] = df["Valid_Area_km2"].fillna(
        df.groupby("region")["Valid_Area_km2"].transform("median")
    )

    if PV_RATIO_TARGET not in df.columns:
        df[PV_RATIO_TARGET] = area / df["Valid_Area_km2"].where(df["Valid_Area_km2"] > 0)
    df[PV_RATIO_TARGET] = pd.to_numeric(df[PV_RATIO_TARGET], errors="coerce").clip(0.0, 1.0)
    return df


def _classifier_builders(random_state: int) -> dict[str, callable]:
    def linear_classifier() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            )),
        ])

    def tree_classifier(estimator: object) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    return {
        "logistic": linear_classifier,
        "random_forest_classifier": lambda: tree_classifier(RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )),
        "extra_trees_classifier": lambda: tree_classifier(ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
        "gradient_boosting_classifier": lambda: tree_classifier(GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=random_state,
        )),
    }


def _positive_class_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X), dtype=float)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
        if hasattr(model, "decision_function"):
            z = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.asarray(model.predict(X), dtype=float).clip(0.0, 1.0)


def _fit_hurdle_growth_model(
    train_features: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
    random_state: int,
    occurrence_threshold: float,
) -> HurdleGrowthModel:
    train = train_features.copy()
    gain = pd.to_numeric(train[PV_GAIN_TARGET], errors="coerce").clip(lower=0.0)
    usable = gain.notna()
    train = train.loc[usable].copy()
    gain = gain.loc[usable]

    positive_cutoff = max(float(occurrence_threshold), 1e-9)
    occurrence = (gain > positive_cutoff).astype(int)
    constant_probability = float(occurrence.mean()) if len(occurrence) else 0.0
    positive_gain = gain[gain > positive_cutoff]
    fallback_positive_gain = float(positive_gain.median()) if len(positive_gain) else 0.0

    classifiers: list[tuple[str, object]] = []
    if len(train) >= 20 and occurrence.nunique() >= 2 and occurrence.sum() >= 3:
        for name, builder in _classifier_builders(random_state).items():
            try:
                model = builder()
                model.fit(train[feature_cols], occurrence)
                classifiers.append((name, model))
            except Exception as exc:
                print(f"[WARN] growth occurrence fit failed: {name}: {exc}")

    regressors: list[tuple[str, object]] = []
    pos_mask = gain > positive_cutoff
    positive_train = train.loc[pos_mask].copy()
    positive_y = np.log1p(gain.loc[pos_mask].astype(float))
    regression_builders = available_model_builders(random_state)
    magnitude_names = [
        name for name in model_names
        if name in regression_builders and name in {
            "ridge", "huber", "random_forest", "extra_trees", "hgb", "hgb_l1",
            "xgboost", "lightgbm", "lightgbm_l1", "catboost", "ngboost",
        }
    ]
    if not magnitude_names:
        magnitude_names = [name for name in ["ridge", "extra_trees", "hgb"] if name in regression_builders]

    if len(positive_train) >= 8:
        for name in magnitude_names:
            try:
                model = regression_builders[name]()
                model.fit(positive_train[feature_cols], positive_y)
                regressors.append((name, model))
            except Exception as exc:
                print(f"[WARN] positive-growth magnitude fit failed: {name}: {exc}")

    diagnostics = {
        "n_training_rows": int(len(train)),
        "n_positive_growth_rows": int(pos_mask.sum()),
        "positive_growth_fraction": constant_probability,
        "n_occurrence_models": int(len(classifiers)),
        "n_magnitude_models": int(len(regressors)),
        "fallback_positive_gain_km2": fallback_positive_gain,
        "occurrence_threshold_km2": positive_cutoff,
    }
    return HurdleGrowthModel(
        classifiers=classifiers,
        magnitude_models=regressors,
        constant_occurrence_probability=constant_probability,
        fallback_positive_gain=fallback_positive_gain,
        occurrence_threshold=positive_cutoff,
        diagnostics=diagnostics,
    )


def _predict_hurdle_growth(model: HurdleGrowthModel, X: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(X)
    if model.classifiers:
        prob_arr = np.vstack([_positive_class_probability(est, X) for _, est in model.classifiers])
    else:
        prob_arr = np.full((1, n), model.constant_occurrence_probability, dtype=float)

    if model.magnitude_models:
        mag_arr = np.vstack([
            np.expm1(np.clip(predict_model(est, X), -20, 20))
            for _, est in model.magnitude_models
        ])
        mag_arr = np.clip(mag_arr, 0.0, np.inf)
    else:
        mag_arr = np.full((1, n), model.fallback_positive_gain, dtype=float)

    p_med = np.nanmedian(prob_arr, axis=0)
    p_low = np.nanquantile(prob_arr, 0.10, axis=0)
    p_high = np.nanquantile(prob_arr, 0.90, axis=0)
    m_med = np.nanmedian(mag_arr, axis=0)
    m_low = np.nanquantile(mag_arr, 0.10, axis=0)
    m_high = np.nanquantile(mag_arr, 0.90, axis=0)

    # Prevent a continuous drizzle of tiny false growth. The thresholds are applied
    # to the occurrence probability, not to the remotely sensed historical masks.
    median_gain = np.where(p_med >= 0.45, p_med * m_med, 0.0)
    conservative_gain = np.where(p_low >= 0.55, p_low * m_low, 0.0)
    rapid_gain = np.where(p_high >= 0.25, p_high * m_high, 0.0)

    return {
        "occurrence_probability": np.clip(p_med, 0.0, 1.0),
        "occurrence_probability_q10": np.clip(p_low, 0.0, 1.0),
        "occurrence_probability_q90": np.clip(p_high, 0.0, 1.0),
        "median_gain": np.clip(median_gain, 0.0, np.inf),
        "conservative_gain": np.clip(conservative_gain, 0.0, np.inf),
        "rapid_gain": np.clip(rapid_gain, 0.0, np.inf),
    }


def _region_growth_limits(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_positive = pd.to_numeric(hist[PV_GAIN_TARGET], errors="coerce")
    global_positive = global_positive[global_positive > 0]
    global_q95 = float(global_positive.quantile(0.95)) if len(global_positive) else 0.0

    for rid, grp in hist.groupby("region"):
        g = pd.to_numeric(grp[PV_GAIN_TARGET], errors="coerce").clip(lower=0.0)
        pos = g[g > 0]
        region_q95 = float(pos.quantile(0.95)) if len(pos) else 0.0
        historical_area = pd.to_numeric(grp[PV_AREA_TARGET], errors="coerce").clip(lower=0.0)
        historical_ratio = pd.to_numeric(grp.get(PV_RATIO_TARGET, np.nan), errors="coerce").clip(0.0, 1.0)
        valid_area = pd.to_numeric(grp.get("Valid_Area_km2", np.nan), errors="coerce")
        valid_area = valid_area[valid_area > 0]
        valid = float(valid_area.median()) if len(valid_area) else np.nan
        max_ratio = float(historical_ratio.max()) if historical_ratio.notna().any() else np.nan

        annual_cap = max(region_q95 * 2.5, global_q95, 0.0009)
        if np.isfinite(valid):
            annual_cap = min(annual_cap, max(valid * 0.20, 0.0009))
            capacity_ratio = min(0.98, max(0.08, (max_ratio if np.isfinite(max_ratio) else 0.0) * 1.75 + 0.02))
            capacity_area = valid * capacity_ratio
        else:
            capacity_area = max(float(historical_area.max()) * 2.0, float(historical_area.max()) + annual_cap * 10.0)

        rows.append({
            "region": int(rid),
            "valid_area_km2": valid,
            "annual_gain_cap_km2": float(annual_cap),
            "capacity_area_km2": float(max(capacity_area, float(historical_area.max()))),
        })
    return pd.DataFrame(rows).set_index("region")


def _constrain_gain(
    proposed: np.ndarray,
    previous_area: np.ndarray,
    regions: np.ndarray,
    limits: pd.DataFrame,
) -> np.ndarray:
    out = np.asarray(proposed, dtype=float).copy()
    for i, (rid, prev) in enumerate(zip(regions, previous_area)):
        if int(rid) not in limits.index:
            out[i] = max(0.0, out[i])
            continue
        row = limits.loc[int(rid)]
        remaining = max(0.0, float(row["capacity_area_km2"]) - float(prev))
        out[i] = float(np.clip(out[i], 0.0, min(float(row["annual_gain_cap_km2"]), remaining)))
    return out


def _valid_area_for_regions(working: pd.DataFrame, regions: np.ndarray, year: int) -> np.ndarray:
    vals = []
    for rid in regions:
        unit = working[(working["region"] == int(rid)) & (working["year"] <= int(year))]
        valid = pd.to_numeric(unit.get("Valid_Area_km2", np.nan), errors="coerce")
        valid = valid[valid > 0]
        vals.append(float(valid.median()) if len(valid) else np.nan)
    return np.asarray(vals, dtype=float)


def rolling_growth_hindcast(
    annual: pd.DataFrame,
    model_names: list[str] | None = None,
    min_history: int = 4,
    random_state: int = 42,
    occurrence_threshold_km2: float = 0.0018,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest the two-part PV expansion model with strict rolling-origin splits."""
    hist = _ensure_growth_columns(annual)
    feat = add_strict_lag_features(hist)
    feature_cols = feature_columns_by_group(feat, "full")
    ok, suspicious = audit_no_leakage(feature_cols, PV_GAIN_TARGET)
    audit_rows = [{
        "target": PV_GAIN_TARGET,
        "stage": "hurdle_growth_hindcast",
        "audit_passed": bool(ok),
        "n_features": len(feature_cols),
        "suspicious_features": ";".join(suspicious[:30]),
    }]
    if not ok:
        raise RuntimeError(f"Growth-model leakage audit failed: {suspicious[:10]}")

    model_names = list(model_names or [])
    years = sorted(feat["year"].unique().astype(int))
    limits = _region_growth_limits(hist)
    records: list[dict] = []

    for test_year in years[min_history:]:
        train = feat[feat["year"] < int(test_year)].copy()
        test = feat[feat["year"] == int(test_year)].copy()
        if len(train) < 20 or test.empty:
            continue
        hurdle = _fit_hurdle_growth_model(
            train_features=train,
            feature_cols=feature_cols,
            model_names=model_names,
            random_state=random_state,
            occurrence_threshold=occurrence_threshold_km2,
        )
        pred = _predict_hurdle_growth(hurdle, test[feature_cols])
        regions = test["region"].to_numpy(int)
        prev_lookup = hist[hist["year"] == int(test_year) - 1].set_index("region")[PV_AREA_TARGET]
        prev_area = np.asarray([
            float(prev_lookup.get(int(rid), 0.0)) if np.isfinite(prev_lookup.get(int(rid), np.nan)) else 0.0
            for rid in regions
        ])
        gain = _constrain_gain(pred["median_gain"], prev_area, regions, limits)
        area_pred = prev_area + gain
        valid_area = _valid_area_for_regions(hist, regions, test_year)
        ratio_pred = np.divide(area_pred, valid_area, out=np.full_like(area_pred, np.nan), where=valid_area > 0)
        ratio_pred = np.clip(ratio_pred, 0.0, 1.0)

        observed_gain = pd.to_numeric(test[PV_GAIN_TARGET], errors="coerce").to_numpy(float)
        observed_area = pd.to_numeric(test[PV_AREA_TARGET], errors="coerce").to_numpy(float)
        observed_ratio = pd.to_numeric(test[PV_RATIO_TARGET], errors="coerce").to_numpy(float)
        for i, rid in enumerate(regions):
            common = {
                "model": "two_part_multisource_growth",
                "feature_group": "full_lagged_environment",
                "year": int(test_year),
                "region": int(rid),
                "occurrence_probability": float(pred["occurrence_probability"][i]),
            }
            records.append({**common, "target": PV_GAIN_TARGET, "observed": float(observed_gain[i]), "predicted": float(gain[i])})
            records.append({**common, "target": PV_AREA_TARGET, "observed": float(observed_area[i]), "predicted": float(area_pred[i])})
            records.append({**common, "target": PV_RATIO_TARGET, "observed": float(observed_ratio[i]), "predicted": float(ratio_pred[i])})

    rec = pd.DataFrame(records)
    metrics = summarize(rec, by=("target", "model", "feature_group"))
    return rec, metrics, pd.DataFrame(audit_rows)


def recursive_projection(
    annual: pd.DataFrame,
    targets: list[str],
    forecast_year: int = 2050,
    model_names: list[str] | None = None,
    random_state: int = 42,
    occurrence_threshold_km2: float = 0.0018,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project PV expansion and ecological response to 2050.

    PV area and ratio are generated by a two-part growth process:
    occurrence probability × positive gain magnitude, followed by cumulative-state,
    annual-rate and available-space constraints. Ecological targets remain direct
    strict-lag ensemble projections conditioned on the recursively updated PV path.
    """
    builders = available_model_builders(random_state)
    if model_names:
        model_names = [m for m in model_names if m in builders]
    else:
        prefer = [
            "ridge", "huber", "random_forest", "extra_trees", "hgb", "hgb_l1",
            "xgboost", "lightgbm", "lightgbm_l1", "catboost", "ngboost",
        ]
        model_names = [m for m in prefer if m in builders]

    hist = _ensure_growth_columns(annual)
    max_year = int(hist["year"].max())
    regions = sorted(hist["region"].dropna().unique().astype(int))
    feat_hist = add_strict_lag_features(hist)
    feature_cols = feature_columns_by_group(feat_hist, "full")
    audit_rows: list[dict] = []

    # Fit the new two-part expansion model once on all historical rows. Recursive
    # future values only enter lagged features; models are never refitted on predictions.
    growth_ok, growth_suspicious = audit_no_leakage(feature_cols, PV_GAIN_TARGET)
    audit_rows.append({
        "target": PV_GAIN_TARGET,
        "stage": "two_part_growth_projection_fit",
        "audit_passed": bool(growth_ok),
        "n_features": len(feature_cols),
        "suspicious_features": ";".join(growth_suspicious[:30]),
    })
    if not growth_ok:
        raise RuntimeError(f"Projection leakage audit failed for {PV_GAIN_TARGET}: {growth_suspicious[:10]}")

    growth_model = _fit_hurdle_growth_model(
        train_features=feat_hist,
        feature_cols=feature_cols,
        model_names=model_names,
        random_state=random_state,
        occurrence_threshold=occurrence_threshold_km2,
    )
    audit_rows.append({
        "target": PV_GAIN_TARGET,
        "stage": "two_part_growth_model_diagnostics",
        "audit_passed": bool(
            growth_model.diagnostics["n_training_rows"] >= 20
            and growth_model.diagnostics["n_positive_growth_rows"] >= 1
        ),
        "n_features": len(feature_cols),
        **growth_model.diagnostics,
    })
    limits = _region_growth_limits(hist)

    # Direct models are retained only for ecology/uncertainty targets. Cumulative PV
    # area and ratio are explicitly excluded from ordinary regression.
    direct_targets = [t for t in targets if t not in {PV_AREA_TARGET, PV_RATIO_TARGET, PV_GAIN_TARGET}]
    fitted: dict[str, list[tuple[str, object]]] = {}
    for target in direct_targets:
        ok, suspicious = audit_no_leakage(feature_cols, target)
        audit_rows.append({
            "target": target,
            "stage": "ecological_projection_fit",
            "audit_passed": bool(ok),
            "n_features": len(feature_cols),
            "suspicious_features": ";".join(suspicious[:30]),
        })
        if not ok:
            raise RuntimeError(f"Projection leakage audit failed for {target}: {suspicious[:10]}")

        train = feat_hist[feat_hist[target].notna()].copy()
        if len(train) < 20:
            continue
        fitted[target] = []
        X = train[feature_cols]
        y = train[target].astype(float)
        for name in model_names:
            try:
                m = builders[name]()
                fitted[target].append((name, fit_model(m, X, y)))
            except Exception as exc:
                print(f"[WARN] final fit failed: {target}/{name}: {exc}")

    working = hist.copy()
    outputs: list[dict] = []
    growth_model_names = ",".join(
        [name for name, _ in growth_model.classifiers]
        + [name for name, _ in growth_model.magnitude_models]
    ) or "empirical_hurdle_fallback"

    for year in range(max_year + 1, int(forecast_year) + 1):
        new_rows = []
        for rid in regions:
            row = {"region": rid, "year": year}
            for col in hist.columns:
                if col in {"region", "year"}:
                    continue
                row[col] = np.nan
            new_rows.append(row)
        working = pd.concat([working, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
        _carry_forward_environment(working, hist, regions, year)

        feat_work = add_strict_lag_features(working)
        idx_future = feat_work["year"] == year
        X_future = feat_work.loc[idx_future, feature_cols]
        rid_list = feat_work.loc[idx_future, "region"].to_numpy(int)

        # Two-part PV growth path.
        growth = _predict_hurdle_growth(growth_model, X_future)
        prev_lookup = working[working["year"] == year - 1].set_index("region")[PV_AREA_TARGET]
        previous_area = np.asarray([
            float(prev_lookup.get(int(rid), 0.0)) if np.isfinite(prev_lookup.get(int(rid), np.nan)) else 0.0
            for rid in rid_list
        ])
        gain_median = _constrain_gain(growth["median_gain"], previous_area, rid_list, limits)
        gain_low = _constrain_gain(growth["conservative_gain"], previous_area, rid_list, limits)
        gain_high = _constrain_gain(growth["rapid_gain"], previous_area, rid_list, limits)
        gain_low = np.minimum(gain_low, gain_median)
        gain_high = np.maximum(gain_high, gain_median)

        area_median = previous_area + gain_median
        area_low = previous_area + gain_low
        area_high = previous_area + gain_high
        valid_area = _valid_area_for_regions(working, rid_list, year)
        ratio_median = np.divide(area_median, valid_area, out=np.full_like(area_median, np.nan), where=valid_area > 0)
        ratio_low = np.divide(area_low, valid_area, out=np.full_like(area_low, np.nan), where=valid_area > 0)
        ratio_high = np.divide(area_high, valid_area, out=np.full_like(area_high, np.nan), where=valid_area > 0)
        ratio_median = np.clip(ratio_median, 0.0, 1.0)
        ratio_low = np.clip(ratio_low, 0.0, 1.0)
        ratio_high = np.clip(ratio_high, 0.0, 1.0)

        for i, rid in enumerate(rid_list):
            mask = (working["region"] == int(rid)) & (working["year"] == year)
            working.loc[mask, PV_GAIN_TARGET] = float(gain_median[i])
            working.loc[mask, PV_AREA_TARGET] = float(area_median[i])
            working.loc[mask, PV_RATIO_TARGET] = float(ratio_median[i])
            if np.isfinite(valid_area[i]):
                working.loc[mask, "Valid_Area_km2"] = float(valid_area[i])

            common = {
                "year": int(year),
                "region": int(rid),
                "models": growth_model_names,
                "occurrence_probability": float(growth["occurrence_probability"][i]),
                "projection_method": "two_part_multisource_growth",
            }
            for target, med, low, high in [
                (PV_GAIN_TARGET, gain_median[i], gain_low[i], gain_high[i]),
                (PV_AREA_TARGET, area_median[i], area_low[i], area_high[i]),
                (PV_RATIO_TARGET, ratio_median[i], ratio_low[i], ratio_high[i]),
            ]:
                if target not in targets and target != PV_GAIN_TARGET:
                    continue
                outputs.extend([
                    {**common, "target": target, "scenario": "median", "projected": float(med), "q10": float(low), "q90": float(high)},
                    {**common, "target": target, "scenario": "conservative", "projected": float(low), "q10": float(low), "q90": float(high)},
                    {**common, "target": target, "scenario": "rapid", "projected": float(high), "q10": float(low), "q90": float(high)},
                ])

        # Recompute features after updating the recursive PV path, so ecology models use
        # the new cumulative state and annual-growth history rather than the old direct fit.
        feat_work = add_strict_lag_features(working)
        idx_future = feat_work["year"] == year
        X_future = feat_work.loc[idx_future, feature_cols]
        rid_list = feat_work.loc[idx_future, "region"].to_numpy(int)

        for target in direct_targets:
            if target not in fitted or len(fitted[target]) == 0:
                continue
            preds = []
            names = []
            for name, model in fitted[target]:
                try:
                    preds.append(predict_model(model, X_future))
                    names.append(name)
                except Exception as exc:
                    print(f"[WARN] projection failed {target}/{name}/{year}: {exc}")
            if not preds:
                continue

            arr = np.vstack(preds)
            median = np.nanmedian(arr, axis=0)
            q10 = np.nanquantile(arr, 0.10, axis=0)
            q90 = np.nanquantile(arr, 0.90, axis=0)
            lo, hi = _bounds_for_target(target, hist[target]) if target in hist.columns else (-np.inf, np.inf)
            median = np.clip(median, lo, hi)
            q10 = np.clip(q10, lo, hi)
            q90 = np.clip(q90, lo, hi)

            for rid, med, low, high in zip(rid_list, median, q10, q90):
                low, high = min(float(low), float(med), float(high)), max(float(low), float(med), float(high))
                outputs.extend([
                    {
                        "target": target, "year": int(year), "region": int(rid),
                        "scenario": "median", "projected": float(med), "q10": low, "q90": high,
                        "models": ",".join(names), "projection_method": "strict_lag_ecological_ensemble",
                    },
                    {
                        "target": target, "year": int(year), "region": int(rid),
                        "scenario": "conservative", "projected": low, "q10": low, "q90": high,
                        "models": ",".join(names), "projection_method": "strict_lag_ecological_ensemble",
                    },
                    {
                        "target": target, "year": int(year), "region": int(rid),
                        "scenario": "rapid", "projected": high, "q10": low, "q90": high,
                        "models": ",".join(names), "projection_method": "strict_lag_ecological_ensemble",
                    },
                ])
                mask = (working["region"] == int(rid)) & (working["year"] == year)
                working.loc[mask, target] = float(med)

    return pd.DataFrame(outputs), pd.DataFrame(audit_rows)
