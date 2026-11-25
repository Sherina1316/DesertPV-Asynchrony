from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import atomic_json, ensure_dir, save_table, utc_now


def _bic(years: np.ndarray, values: np.ndarray, event_year: int | None = None) -> tuple[float, np.ndarray]:
    if event_year is None:
        matrix = np.column_stack([np.ones(len(years)), years - years.mean()])
    else:
        time = years - event_year
        post = (years >= event_year).astype(float)
        matrix = np.column_stack([np.ones(len(years)), time, post, post * time])
    beta, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    residual = values - matrix @ beta
    rss = max(float(np.sum(residual**2)), 1e-12)
    return len(values) * np.log(rss / len(values)) + matrix.shape[1] * np.log(len(values)), beta


def detect_pv_expansion_events(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"]); out = ensure_dir(root / "13_pv_events")
    annual = pd.read_csv(root / "11_pv_dynamics" / "annual_pv_dynamics.csv")
    uncertainty = pd.read_csv(root / "10_uncertainty" / "change_year_uncertainty.csv")
    validation = json.loads((root / "09_accuracy_validation" / "accuracy_validation_status.json").read_text(encoding="utf-8"))
    temporal_status = json.loads((root / "08_temporal_pv_change" / "temporal_decoding_status.json").read_text(encoding="utf-8"))
    gate_pass = bool(validation.get("segmentation_gate_pass", False))
    temporal_ok = not bool(temporal_status.get("synchronous_anomaly", False))
    event_cfg = cfg["events"]
    rows: list[dict[str, Any]] = []

    for region_id, group in tqdm(list(annual.groupby("Id")), desc="Object-level PV transition detection", unit="region", disable=not cfg["processing"].get("show_progress", True)):
        g = group.sort_values("year")
        years = g["year"].to_numpy(int)
        area_column = "cumulative_installed_area_km2" if "cumulative_installed_area_km2" in g else "pv_area_km2"
        area = pd.to_numeric(g[area_column], errors="coerce").fillna(0).to_numpy(float)
        route = str(g.iloc[0].get("estimation_route", "direct_pixel"))
        mode = str(g.iloc[0].get("estimation_mode", "direct_pixel"))
        values = np.log1p(np.maximum(area, 0))
        if len(years) < int(event_cfg["minimum_pre_years"]) + int(event_cfg["minimum_post_years"]) + 1:
            rows.append({"Id": int(region_id), "estimation_mode": mode, "estimation_route": route, "diagnostic_candidate": False, "qualified": False, "reason": "insufficient_years"})
            continue
        null_bic, _ = _bic(years, values)
        candidates: list[tuple[float, int, np.ndarray]] = []
        pre_min = int(event_cfg["minimum_pre_years"]); post_min = int(event_cfg["minimum_post_years"])
        for event_year in years[pre_min : len(years) - post_min + 1]:
            bic, beta = _bic(years, values, int(event_year)); candidates.append((bic, int(event_year), beta))
        best_bic, best_year, _ = min(candidates, key=lambda item: item[0])
        delta_bic = float(null_bic - best_bic)
        pre = g.loc[g["year"] < best_year]; post = g.loc[g["year"] >= best_year]
        pre_median = float(pre[area_column].median()) if len(pre) else np.nan
        post_median = float(post[area_column].median()) if len(post) else np.nan
        level_change = post_median - pre_median
        gain_column = "installed_gain_km2" if "installed_gain_km2" in post else "gross_gain_km2"
        peak_gain = float(post[gain_column].max()) if len(post) else np.nan
        relative_gain = level_change / max(pre_median, float(event_cfg["minimum_absolute_gain_km2"])) if np.isfinite(level_change) else np.nan
        post_threshold = pre_median + 0.5 * max(level_change, 0)
        persistent_years = int((post[area_column] >= post_threshold).sum()) if len(post) else 0
        candidate = bool(
            delta_bic >= float(event_cfg["minimum_delta_bic"])
            and level_change >= float(event_cfg["minimum_absolute_gain_km2"])
            and relative_gain >= float(event_cfg["minimum_relative_gain"])
            and persistent_years >= int(event_cfg["minimum_post_persistence_years"])
        )
        inferred = pd.to_numeric(g.get("inferred_deployment_year"), errors="coerce").dropna() if "inferred_deployment_year" in g else pd.Series(dtype=float)
        object_anchor_year = int(round(float(inferred.iloc[0]))) if len(inferred) else np.nan
        rows.append({
            "Id": int(region_id), "estimation_mode": mode, "estimation_route": route,
            "event_year": best_year, "object_anchor_break_year": object_anchor_year,
            "delta_bic": delta_bic, "pre_median_area_km2": pre_median, "post_median_area_km2": post_median,
            "level_change_km2": level_change, "relative_level_change": relative_gain,
            "peak_post_event_gain_km2": peak_gain, "post_persistent_years": persistent_years,
            "diagnostic_candidate": candidate, "qualified": False,
            "reason": "candidate_pending_global_gates" if candidate else "failed_bic_growth_or_persistence_gate",
            "pre_years": int(len(pre)), "post_years": int(len(post)),
            "event_interpretation": "object-level persistent footprint transition; not commissioning date",
        })

    events = pd.DataFrame(rows).sort_values("Id").merge(uncertainty, on=["Id", "estimation_mode"], how="left", suffixes=("", "_uncertainty"))
    if "timing_identified" not in events:
        events["timing_identified"] = False
    events["timing_ci_width"] = pd.to_numeric(events.get("change_year_ci_high"), errors="coerce") - pd.to_numeric(events.get("change_year_ci_low"), errors="coerce")
    events["timing_stable"] = events["timing_identified"].fillna(False).astype(bool) & events["timing_ci_width"].le(float(event_cfg["maximum_change_year_ci_width"]))
    # Area-only anchor-break estimates can be retained as diagnostic events even if threshold sensitivity is unavailable.
    fallback_stable = events["estimation_mode"].eq("area_only_fallback") & pd.to_numeric(events["object_anchor_break_year"], errors="coerce").notna()
    events.loc[fallback_stable, "timing_stable"] = True
    candidates = events.loc[events["diagnostic_candidate"].fillna(False).astype(bool)]
    if len(candidates):
        counts = candidates["event_year"].value_counts(); sync_year = int(counts.idxmax()); sync_count = int(counts.max()); sync_fraction = float(sync_count / max(len(events), 1))
    else:
        sync_year = -1; sync_count = 0; sync_fraction = 0.0
    synchrony_ok = sync_fraction <= float(event_cfg["maximum_synchronous_peak_fraction"])
    direct = events["estimation_mode"].eq("direct_pixel")
    events["formal_qualified"] = events["diagnostic_candidate"].fillna(False).astype(bool) & events["timing_stable"].fillna(False).astype(bool) & direct & gate_pass & temporal_ok & synchrony_ok
    events["qualified"] = events["formal_qualified"]
    events["diagnostic_qualified"] = events["diagnostic_candidate"].fillna(False).astype(bool) & events["timing_stable"].fillna(False).astype(bool)
    events.loc[events["diagnostic_candidate"].fillna(False).astype(bool) & ~events["timing_stable"].fillna(False).astype(bool), "reason"] = "unstable_event_timing"
    events.loc[events["diagnostic_qualified"] & events["estimation_mode"].eq("area_only_fallback"), "reason"] = "area_only_anchor_analog_event_diagnostic"
    events.loc[events["diagnostic_qualified"] & events["estimation_mode"].eq("anchor_guided_candidate"), "reason"] = "anchor_guided_pixel_event_diagnostic"
    events.loc[events["formal_qualified"], "reason"] = "qualified_direct_pixel_event"
    if not gate_pass:
        events.loc[events["diagnostic_qualified"] & direct, "reason"] = "positive_unlabelled_scientific_gate_failed"
    elif not temporal_ok:
        events.loc[events["diagnostic_qualified"] & direct, "reason"] = "temporal_synchrony_warning"
    elif not synchrony_ok:
        events.loc[events["diagnostic_qualified"] & direct, "reason"] = f"cross_unit_synchrony_warning_{sync_year}"

    save_table(events, out / "pv_expansion_events.csv")
    save_table(events.loc[events["formal_qualified"]], out / "qualified_pv_expansion_events.csv")
    years = sorted(pd.to_numeric(events.get("event_year"), errors="coerce").dropna().astype(int).unique())
    synchrony = pd.DataFrame({
        "year": years,
        "diagnostic_candidate_units": [int(((events["event_year"] == year) & events["diagnostic_candidate"].fillna(False)).sum()) for year in years],
        "formal_direct_units": [int(((events["event_year"] == year) & events["formal_qualified"].fillna(False)).sum()) for year in years],
    })
    save_table(synchrony, out / "event_synchrony_audit.csv")
    atomic_json({
        "status": "passed", "created_utc": utc_now(), "units": len(events),
        "diagnostic_candidates": int(events["diagnostic_candidate"].fillna(False).sum()),
        "diagnostic_qualified_events": int(events["diagnostic_qualified"].fillna(False).sum()),
        "formal_qualified_events": int(events["formal_qualified"].fillna(False).sum()),
        "direct_pixel_event_units": int(events.estimation_mode.eq("direct_pixel").sum()),
        "anchor_guided_candidate_event_units": int(events.estimation_mode.eq("anchor_guided_candidate").sum()),
        "area_only_event_units": int(events.estimation_mode.eq("area_only_fallback").sum()),
        "scientific_gate_pass": gate_pass, "temporal_synchrony_pass": temporal_ok,
        "cross_unit_synchrony_pass": synchrony_ok, "maximum_event_year_fraction": sync_fraction,
        "maximum_event_year_count": sync_count, "synchronous_event_year": sync_year if sync_year >= 0 else None,
        "formal_event_claims_allowed": bool(gate_pass and temporal_ok and synchrony_ok),
        "interpretation": "Event years are object-level persistent remote-sensing transitions. Fallback units retain area/timing inference but no fabricated footprint maps.",
    }, out / "pv_event_status.json")
    return events
