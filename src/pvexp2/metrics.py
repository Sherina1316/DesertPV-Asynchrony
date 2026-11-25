
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) == 0:
        return {
            "n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan,
            "bias": np.nan, "normalized_bias": np.nan,
            "nrmse": np.nan, "nmae": np.nan,
            "observed_range": np.nan, "observed_mean": np.nan,
        }
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    rng = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(rng) or rng <= 1e-12:
        rng = float(np.nanmean(np.abs(y))) if np.nanmean(np.abs(y)) > 1e-12 else np.nan
    nrmse = float(rmse / rng) if np.isfinite(rng) and rng > 0 else np.nan
    nmae = float(mae / rng) if np.isfinite(rng) and rng > 0 else np.nan
    nbias = float(bias / rng) if np.isfinite(rng) and rng > 0 else np.nan
    return {
        "n": int(len(y)), "rmse": rmse, "mae": mae, "r2": r2,
        "bias": bias, "normalized_bias": nbias,
        "nrmse": nrmse, "nmae": nmae,
        "observed_range": rng, "observed_mean": float(np.mean(y)),
    }


def summarize(records: pd.DataFrame, by=("target", "model")) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in records.groupby(list(by)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        row.update(regression_metrics(grp["observed"], grp["predicted"]))
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_residuals(records: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame()
    df = records.copy()
    df["residual"] = df["predicted"] - df["observed"]
    df["abs_residual"] = df["residual"].abs()
    outs = []
    for (target, model), grp in df.groupby(["target", "model"]):
        if grp.empty:
            continue
        thr = grp["abs_residual"].quantile(quantile)
        tmp = grp[grp["abs_residual"] >= thr].copy()
        tmp["extreme_threshold"] = thr
        outs.append(tmp)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
