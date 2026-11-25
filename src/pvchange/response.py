from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

from .utils import atomic_json, ensure_dir, save_table, utc_now


def _metric_series(ecology: pd.DataFrame, region_id: int, metric: str) -> pd.DataFrame:
    out = ecology.loc[
        ecology["Id"].eq(region_id) & ecology["metric"].eq(metric),
        ["year", "external_anomaly", "quality"],
    ].copy().sort_values("year")
    out = out.loc[out["quality"].eq("valid")].rename(columns={"external_anomaly": "response"})
    return out


def _detect_onset_and_stability(series: pd.DataFrame, event_year: int, cfg: dict[str, Any]) -> dict[str, Any]:
    pre = series.loc[series.year < event_year, "response"].dropna().to_numpy(dtype=float)
    post = series.loc[series.year >= event_year].dropna(subset=["response"]).copy()
    if len(pre) < int(cfg["ecology"]["minimum_pre_years"]) or post.empty:
        return {"onset_year": np.nan, "onset_lag_years": np.nan, "direction": "insufficient", "onset_right_censored": True, "stabilization_year": np.nan, "stabilization_lag_years": np.nan, "stabilization_right_censored": True}
    baseline = float(np.median(pre))
    mad = float(np.median(np.abs(pre - baseline)))
    threshold = max(float(cfg["response_dynamics"]["onset_mad_multiplier"]) * 1.4826 * mad, 1e-4)
    post["delta"] = post.response - baseline
    post["signal"] = np.abs(post.delta) >= threshold
    onset_year = np.nan
    direction = "none_detected"
    values = post.signal.to_numpy(dtype=bool)
    years = post.year.to_numpy(dtype=int)
    for i in range(len(values) - 1):
        if values[i] and values[i + 1] and np.sign(post.delta.iloc[i]) == np.sign(post.delta.iloc[i + 1]):
            onset_year = int(years[i]); direction = "increase" if post.delta.iloc[i] > 0 else "decline"; break
    onset_censored = not np.isfinite(onset_year)
    stability_year = np.nan
    if np.isfinite(onset_year):
        window = int(cfg["response_dynamics"]["stabilization_window_years"])
        tolerance = float(cfg["response_dynamics"]["stabilization_slope_tolerance"])
        cv_tol = float(cfg["response_dynamics"]["stabilization_cv_tolerance"])
        after = post.loc[post.year >= onset_year].copy()
        for i in range(0, len(after) - window + 1):
            w = after.iloc[i:i + window]
            slope = linregress(w.year, w.response).slope if len(w) >= 2 else np.inf
            cv = float(np.std(w.response) / max(abs(np.mean(w.response)), 1e-4))
            retained = np.all(np.sign(w.delta.to_numpy()) == (1 if direction == "increase" else -1))
            if abs(slope) <= tolerance and cv <= cv_tol and retained:
                stability_year = int(w.year.iloc[-1]); break
    return {
        "onset_year": onset_year,
        "onset_lag_years": float(onset_year - event_year) if np.isfinite(onset_year) else np.nan,
        "direction": direction,
        "onset_right_censored": onset_censored,
        "stabilization_year": stability_year,
        "stabilization_lag_years": float(stability_year - event_year) if np.isfinite(stability_year) else np.nan,
        "adjustment_duration_years": float(stability_year - onset_year) if np.isfinite(stability_year) and np.isfinite(onset_year) else np.nan,
        "stabilization_right_censored": not np.isfinite(stability_year),
        "baseline_median": baseline,
        "baseline_mad": mad,
        "onset_threshold": threshold,
    }



def _finite_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None

def analyze_response_dynamics(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "15_response_dynamics")
    ecology = pd.read_csv(root / "12_dynamic_ecology" / "external_ecology_anomaly_panel.csv")
    events = pd.read_csv(root / "13_pv_events" / "pv_expansion_events.csv")
    events = events.loc[events.qualified.eq(True)].copy()
    metrics = list(cfg["ecology"]["metrics"])
    metric_rows, consensus_rows = [], []
    if events.empty:
        metric_df = pd.DataFrame(columns=[
            "Id", "metric", "event_year", "onset_year", "onset_lag_years", "direction",
            "onset_right_censored", "stabilization_year", "stabilization_lag_years",
            "adjustment_duration_years", "stabilization_right_censored", "baseline_median",
            "baseline_mad", "onset_threshold",
        ])
        consensus = pd.DataFrame(columns=[
            "Id", "event_year", "consensus_direction", "supporting_metric_count",
            "consensus_onset_year", "consensus_onset_lag_years", "onset_right_censored",
            "consensus_stabilization_year", "consensus_stabilization_lag_years",
            "stabilization_right_censored", "consensus_status",
        ])
        save_table(metric_df, out / "metric_response_dynamics.csv")
        save_table(consensus, out / "consensus_response_dynamics.csv")
        atomic_json({
            "status": "completed_no_events", "created_utc": utc_now(), "qualified_events": 0,
            "consensus_onset_observed_units": 0, "consensus_stabilization_observed_units": 0,
            "right_censoring_retained": True,
        }, out / "response_dynamics_status.json")
        return consensus
    for event in tqdm(list(events.itertuples(index=False)), desc="Ecological response timing", unit="event", disable=not cfg["processing"].get("show_progress", True)):
        detections = []
        for metric in metrics:
            series = _metric_series(ecology, int(event.Id), metric)
            result = _detect_onset_and_stability(series, int(event.event_year), cfg)
            row = {"Id": int(event.Id), "metric": metric, "event_year": int(event.event_year), **result}
            metric_rows.append(row)
            if np.isfinite(result["onset_year"]):
                detections.append(row)
        support_required = int(cfg["response_dynamics"]["onset_minimum_supporting_metrics"])
        if detections:
            directions = pd.Series([r["direction"] for r in detections]).value_counts()
            direction = str(directions.index[0])
            agreeing = [r for r in detections if r["direction"] == direction]
        else:
            direction, agreeing = "none_detected", []
        qualified = len(agreeing) >= support_required
        onset = float(np.median([r["onset_year"] for r in agreeing])) if qualified else np.nan
        stabs = [r["stabilization_year"] for r in agreeing if np.isfinite(r["stabilization_year"])]
        stab = float(np.median(stabs)) if len(stabs) >= support_required else np.nan
        consensus_rows.append({
            "Id": int(event.Id), "event_year": int(event.event_year), "consensus_direction": direction,
            "supporting_metric_count": len(agreeing), "consensus_onset_year": onset,
            "consensus_onset_lag_years": onset - int(event.event_year) if np.isfinite(onset) else np.nan,
            "onset_right_censored": not qualified, "consensus_stabilization_year": stab,
            "consensus_stabilization_lag_years": stab - int(event.event_year) if np.isfinite(stab) else np.nan,
            "stabilization_right_censored": not np.isfinite(stab),
            "consensus_status": "qualified_consensus" if qualified else "right_censored_or_insufficient_consensus",
        })
    metric_df = pd.DataFrame(metric_rows)
    consensus = pd.DataFrame(consensus_rows)
    save_table(metric_df, out / "metric_response_dynamics.csv")
    save_table(consensus, out / "consensus_response_dynamics.csv")
    atomic_json({
        "status": "passed", "created_utc": utc_now(), "qualified_events": len(events),
        "consensus_onset_observed_units": int((~consensus.onset_right_censored).sum()) if not consensus.empty else 0,
        "median_consensus_onset_lag_years": _finite_median(consensus.consensus_onset_lag_years) if not consensus.empty else None,
        "consensus_stabilization_observed_units": int((~consensus.stabilization_right_censored).sum()) if not consensus.empty else 0,
        "median_consensus_stabilization_lag_years": _finite_median(consensus.consensus_stabilization_lag_years) if not consensus.empty else None,
        "right_censoring_retained": True,
    }, out / "response_dynamics_status.json")
    return consensus
