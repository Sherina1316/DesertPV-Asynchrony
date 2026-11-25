
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import add_strict_lag_features, feature_columns_by_group, audit_no_leakage
from .models import available_model_builders, fit_predict
from .metrics import summarize


def rolling_hindcast_validation(
    annual: pd.DataFrame,
    targets: list[str],
    model_names: list[str] | None = None,
    min_history: int = 4,
    feature_group: str = "full",
    random_state: int = 42,
    add_ensemble: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Historical rolling-origin hindcast.

    Train years are strictly < test_year, and all predictors are lag-only/fixed-baseline.
    """
    builders = available_model_builders(random_state)
    if model_names:
        model_names = [m for m in model_names if m in builders]
    else:
        prefer = ["ridge", "huber", "random_forest", "extra_trees", "hgb", "hgb_l1",
                  "xgboost", "lightgbm", "lightgbm_l1", "catboost", "ngboost"]
        model_names = [m for m in prefer if m in builders]

    if not model_names:
        raise RuntimeError("No available models.")

    feat = add_strict_lag_features(annual)
    years = sorted(feat["year"].unique().astype(int))
    records = []
    audit_rows = []

    for target in tqdm(targets, desc=f"Rolling hindcast [{feature_group}]"):
        if target not in feat.columns:
            continue
        feature_cols = feature_columns_by_group(feat, feature_group)
        ok, suspicious = audit_no_leakage(feature_cols, target)
        audit_rows.append({
            "target": target,
            "feature_group": feature_group,
            "audit_passed": bool(ok),
            "n_features": len(feature_cols),
            "suspicious_features": ";".join(suspicious[:30]),
        })
        if not ok:
            raise RuntimeError(f"Leakage audit failed for {target}: {suspicious[:10]}")

        for test_year in years[min_history:]:
            train = feat[(feat["year"] < test_year) & feat[target].notna()].copy()
            test = feat[(feat["year"] == test_year) & feat[target].notna()].copy()
            if len(train) < 20 or len(test) == 0:
                continue

            X_train = train[feature_cols]
            y_train = train[target].astype(float)
            X_test = test[feature_cols]

            fold_predictions = []
            for name in model_names:
                try:
                    model = builders[name]()
                    pred = fit_predict(model, X_train, y_train, X_test)
                    fold_predictions.append((name, pred))
                    for rid, obs, pp in zip(test["region"].values, test[target].values, pred):
                        records.append({
                            "target": target,
                            "model": name,
                            "feature_group": feature_group,
                            "year": int(test_year),
                            "region": int(rid),
                            "observed": float(obs),
                            "predicted": float(pp),
                        })
                except Exception as exc:
                    print(f"[WARN] {target}/{name}/{test_year} failed: {exc}")

            if add_ensemble and len(fold_predictions) >= 2:
                arr = np.vstack([p for _, p in fold_predictions])
                ens = np.nanmedian(arr, axis=0)
                q10 = np.nanquantile(arr, 0.10, axis=0)
                q90 = np.nanquantile(arr, 0.90, axis=0)
                for rid, obs, pp, lo, hi in zip(test["region"].values, test[target].values, ens, q10, q90):
                    records.append({
                        "target": target,
                        "model": "ensemble_median",
                        "feature_group": feature_group,
                        "year": int(test_year),
                        "region": int(rid),
                        "observed": float(obs),
                        "predicted": float(pp),
                        "q10": float(lo),
                        "q90": float(hi),
                    })

    rec = pd.DataFrame(records)
    met = summarize(rec, by=("target", "model", "feature_group"))
    audit = pd.DataFrame(audit_rows)
    return rec, met, audit


def ablation_validation(
    annual: pd.DataFrame,
    targets: list[str],
    model_name: str = "lightgbm",
    min_history: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    builders = available_model_builders(random_state)
    if model_name not in builders:
        model_name = "hgb" if "hgb" in builders else list(builders.keys())[0]
        print(f"[WARN] Requested ablation model unavailable; using {model_name}")

    groups = ["base", "base_pv", "base_pv_rolling", "full"]
    recs, mets, audits = [], [], []

    for group in groups:
        rec, met, audit = rolling_hindcast_validation(
            annual=annual,
            targets=targets,
            model_names=[model_name],
            min_history=min_history,
            feature_group=group,
            random_state=random_state,
            add_ensemble=False,
        )
        if not rec.empty:
            recs.append(rec)
        if not met.empty:
            met["ablation_model"] = model_name
            mets.append(met)
        if not audit.empty:
            audit["ablation_model"] = model_name
            audits.append(audit)

    return (
        pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(),
        pd.concat(mets, ignore_index=True) if mets else pd.DataFrame(),
        pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
    )
