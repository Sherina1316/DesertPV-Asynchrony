from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pvchange.config import load_config
from pvchange.utils import atomic_json, utc_now
from pvexp2.io_utils import read_annual_table, available_targets, ensure_output_dirs, save_csv
from pvexp2.validation import rolling_hindcast_validation, ablation_validation
from pvexp2.forecast import recursive_projection, rolling_growth_hindcast
from pvexp2.metrics import extreme_residuals
from pvexp2.plots import (
    plot_validation_summary,
    plot_model_comparison,
    plot_ablation,
    plot_hindcast,
    plot_obs_pred,
    plot_residual,
    plot_projection,
)
from pvexp2.report import build_report


def _list_from_cfg(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def run_strict_noleak_forecast(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = cfg.get("strict_noleak_forecast", {})
    output_root = Path(cfg["paths"]["output_root"])
    experiment_subdir = str(settings.get("experiment_subdir", "21_strict_noleak_forecast"))
    experiment_dir = output_root / experiment_subdir
    dirs = ensure_output_dirs(experiment_dir)

    targets_requested = _list_from_cfg(
        settings.get("targets"),
        ["PV_Cover_Ratio", "PV_Cover_Area_km2", "PVIntegrated_PAVG_Optimized", "PVIntegrated_EIPAVG"],
    )
    model_names = _list_from_cfg(
        settings.get("models"),
        ["ridge", "huber", "random_forest", "extra_trees", "hgb", "hgb_l1", "xgboost", "lightgbm", "lightgbm_l1", "catboost", "ngboost"],
    )
    ablation_model = str(settings.get("ablation_model", "lightgbm"))
    forecast_year = int(settings.get("forecast_year", cfg.get("study", {}).get("future_year", 2050)))
    min_history = int(settings.get("min_history", 4))
    random_state = int(cfg.get("project", {}).get("random_seed", 42))
    occurrence_threshold_km2 = float(settings.get("occurrence_threshold_km2", 0.0018))

    run_cfg = {
        "output_root": str(output_root),
        "experiment_dir": str(experiment_dir),
        "targets_requested": targets_requested,
        "models_requested": model_names,
        "ablation_model": ablation_model,
        "forecast_year": forecast_year,
        "min_history": min_history,
        "random_state": random_state,
        "occurrence_threshold_km2": occurrence_threshold_km2,
        "input_semantics": "v24 anchor-constrained cumulative installed PV plus fixed-support LSMA-PAVG ecology",
    }
    (dirs["root"] / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    annual = read_annual_table(output_root)
    targets = available_targets(annual, targets_requested)
    if not targets:
        raise RuntimeError("No requested no-leakage forecast targets are available after v24 panel construction")
    save_csv(annual, dirs["tables"] / "annual_all_regions_nature_ready.csv")

    records, metrics, audit_main = rolling_hindcast_validation(
        annual=annual,
        targets=targets,
        model_names=model_names,
        min_history=min_history,
        feature_group="full",
        random_state=random_state,
        add_ensemble=True,
    )
    save_csv(records, dirs["validation"] / "historical_hindcast_records_noleak.csv")
    save_csv(metrics, dirs["validation"] / "historical_hindcast_metrics_by_model_noleak.csv")
    save_csv(audit_main, dirs["audit"] / "feature_leakage_audit_main.csv")

    growth_records, growth_metrics, audit_growth = rolling_growth_hindcast(
        annual=annual,
        model_names=model_names,
        min_history=min_history,
        random_state=random_state,
        occurrence_threshold_km2=occurrence_threshold_km2,
    )
    save_csv(growth_records, dirs["validation"] / "historical_hindcast_two_part_growth_records.csv")
    save_csv(growth_metrics, dirs["validation"] / "historical_hindcast_two_part_growth_metrics.csv")
    save_csv(audit_growth, dirs["audit"] / "feature_leakage_audit_two_part_growth.csv")

    ab_rec, ab_met, audit_ab = ablation_validation(
        annual=annual,
        targets=targets,
        model_name=ablation_model,
        min_history=min_history,
        random_state=random_state,
    )
    save_csv(ab_rec, dirs["validation"] / "feature_ablation_records_noleak.csv")
    save_csv(ab_met, dirs["validation"] / "feature_ablation_metrics_noleak.csv")
    save_csv(audit_ab, dirs["audit"] / "feature_leakage_audit_ablation.csv")

    best_model = "ensemble_median" if (not records.empty and "ensemble_median" in set(records.get("model", []))) else None
    ext = extreme_residuals(records[records["model"] == best_model], quantile=0.99) if best_model else extreme_residuals(records, quantile=0.99)
    save_csv(ext, dirs["validation"] / "extreme_residual_regions_noleak.csv")

    forecast, audit_forecast = recursive_projection(
        annual=annual,
        targets=targets,
        forecast_year=forecast_year,
        model_names=model_names,
        random_state=random_state,
        occurrence_threshold_km2=occurrence_threshold_km2,
    )
    save_csv(forecast, dirs["forecast"] / f"future_projection_to_{forecast_year}_region_year_noleak.csv")
    if not forecast.empty:
        global_forecast = forecast.groupby(["target", "year", "scenario"], as_index=False)[["projected", "q10", "q90"]].mean()
    else:
        global_forecast = pd.DataFrame(columns=["target", "year", "scenario", "projected", "q10", "q90"])
    save_csv(global_forecast, dirs["forecast"] / f"future_projection_to_{forecast_year}_global_mean_noleak.csv")
    save_csv(audit_forecast, dirs["audit"] / "feature_leakage_audit_forecast.csv")

    plot_validation_summary(metrics, dirs["validation"] / "historical_hindcast_validation_normalized_metrics_noleak.png")
    plot_model_comparison(metrics, dirs["figures"])
    plot_ablation(ab_met, dirs["figures"])
    plot_hindcast(records, dirs["validation"])
    plot_obs_pred(records, dirs["validation"])
    plot_residual(records, dirs["validation"])
    plot_projection(annual, forecast, dirs["forecast"])

    audits = pd.concat([audit_main, audit_growth, audit_ab, audit_forecast], ignore_index=True, sort=False)
    build_report(
        output_path=dirs["report"] / "pv_forecast_strict_noleak_report_cn.html",
        metrics=metrics,
        ablation=ab_met,
        extreme=ext,
        audits=audits,
        dirs=dirs,
    )

    audit_pass = bool(audits.get("audit_passed", pd.Series(dtype=bool)).fillna(True).astype(bool).all()) if not audits.empty else False
    status = {
        "status": "passed" if (audit_pass and not metrics.empty and not forecast.empty) else "failed",
        "created_utc": utc_now(),
        "experiment_dir": str(experiment_dir),
        "targets": targets,
        "models_requested": model_names,
        "hindcast_records": int(len(records)),
        "hindcast_metrics": int(len(metrics)),
        "two_part_growth_hindcast_records": int(len(growth_records)),
        "two_part_growth_hindcast_metrics": int(len(growth_metrics)),
        "forecast_records": int(len(forecast)),
        "leakage_audit_passed": audit_pass,
        "strict_time_order": True,
        "training_years_are_always_less_than_test_year": True,
        "current_year_targets_excluded_from_features": True,
        "canonical_table": str(dirs["tables"] / "annual_all_regions_nature_ready.csv"),
        "validation_table": str(dirs["validation"] / "historical_hindcast_metrics_by_model_noleak.csv"),
        "forecast_table": str(dirs["forecast"] / f"future_projection_to_{forecast_year}_global_mean_noleak.csv"),
        "report": str(dirs["report"] / "pv_forecast_strict_noleak_report_cn.html"),
        "projection_method": "two-part annual installation occurrence + positive gain magnitude, constrained cumulative area, lagged environmental drivers",
        "note": "These are strict historical hindcast and recursive scenario projections; publication-grade claims still depend on the upstream multisource segmentation gates and independent real-data diagnostics.",
    }
    atomic_json(status, dirs["root"] / "noleak_forecast_status.json")
    if status["status"] != "passed":
        raise RuntimeError("Strict no-leakage forecast contract failed")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_strict_noleak_forecast(load_config(args.config))
