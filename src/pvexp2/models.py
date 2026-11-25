
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _linear(est):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", est),
    ])


def _tree(est):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", est),
    ])


def available_model_builders(random_state: int = 42) -> dict:
    builders = {
        "ridge": lambda: _linear(Ridge(alpha=10.0)),
        "elasticnet": lambda: _linear(ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=random_state, max_iter=5000)),
        "huber": lambda: _linear(HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=1000)),
        "random_forest": lambda: _tree(RandomForestRegressor(
            n_estimators=160, min_samples_leaf=3, max_features=0.8, random_state=random_state, n_jobs=-1
        )),
        "extra_trees": lambda: _tree(ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=3, max_features=0.8, random_state=random_state, n_jobs=-1
        )),
        # The names hgb/hgb_l1 are retained for backward compatibility with v2 configs.
        # GradientBoostingRegressor is used here instead of HistGradientBoostingRegressor
        # because it is more predictable on small rolling-origin folds.
        "hgb": lambda: _tree(GradientBoostingRegressor(
            n_estimators=140, learning_rate=0.04, max_depth=2,
            min_samples_leaf=3, subsample=0.85, random_state=random_state,
            loss="squared_error"
        )),
        "hgb_l1": lambda: _tree(GradientBoostingRegressor(
            n_estimators=140, learning_rate=0.04, max_depth=2,
            min_samples_leaf=3, subsample=0.85, random_state=random_state,
            loss="absolute_error"
        )),
    }

    try:
        from xgboost import XGBRegressor
        builders["xgboost"] = lambda: _tree(XGBRegressor(
            n_estimators=180, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, reg_alpha=0.10,
            objective="reg:squarederror", random_state=random_state, n_jobs=1
        ))
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor
        builders["lightgbm"] = lambda: _tree(LGBMRegressor(
            n_estimators=200, learning_rate=0.035, num_leaves=15,
            min_child_samples=15, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, random_state=random_state, n_jobs=1, verbose=-1
        ))
        builders["lightgbm_l1"] = lambda: _tree(LGBMRegressor(
            n_estimators=200, learning_rate=0.035, num_leaves=15,
            min_child_samples=15, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, random_state=random_state, n_jobs=1, verbose=-1,
            objective="regression_l1"
        ))
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor
        builders["catboost"] = lambda: _tree(CatBoostRegressor(
            iterations=180, learning_rate=0.035, depth=4,
            l2_leaf_reg=5.0, random_seed=random_state,
            loss_function="RMSE", verbose=False, allow_writing_files=False
        ))
    except Exception:
        pass

    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        builders["ngboost"] = lambda: _tree(NGBRegressor(
            Dist=Normal, n_estimators=160, learning_rate=0.035,
            random_state=random_state, verbose=False
        ))
    except Exception:
        pass

    return builders


def fit_predict(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_test), dtype=float)


def fit_model(model, X: pd.DataFrame, y: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(X), dtype=float)
