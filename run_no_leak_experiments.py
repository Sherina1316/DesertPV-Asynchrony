
from __future__ import annotations

from pathlib import Path as _PVPath
import sys as _PVSys
_PV_SRC_DIR = _PVPath(__file__).resolve().parent / "src"
if str(_PV_SRC_DIR) not in _PVSys.path:
    _PVSys.path.insert(0, str(_PV_SRC_DIR))

import argparse
from pathlib import Path
import json
import pandas as pd

from pvexp2.io_utils import read_annual_table, available_targets, ensure_output_dirs, save_csv
from pvexp2.validation import rolling_hindcast_validation, ablation_validation
from pvexp2.forecast import recursive_projection
from pvexp2.metrics import extreme_residuals
from pvexp2.plots import (
    plot_validation_summary, plot_model_comparison, plot_ablation,
    plot_hindcast, plot_obs_pred, plot_residual, plot_projection
)
from pvexp2.report import build_report


def parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Strict no-leakage PV forecasting experiments.")
    ap.add_argument("--output-root", default="/home/wsy/pv_nature_experiment/output_v24_22_novelty_gated_remote_change")
    ap.add_argument("--experiment-dir", default="")
    ap.add_argument("--forecast-year", type=int, default=2050)
    ap.add_argument("--targets", default="PV_Cover_Ratio,PV_Cover_Area_km2,PVIntegrated_PAVG_Optimized,PVIntegrated_EIPAVG")
    ap.add_argument("--models", default="ridge,huber,random_forest,extra_trees,hgb,hgb_l1,xgboost,lightgbm,lightgbm_l1,catboost,ngboost")
    ap.add_argument("--ablation-model", default="lightgbm")
    ap.add_argument("--min-history", type=int, default=4)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else output_root / "21_strict_noleak_forecast_standalone"
    dirs = ensure_output_dirs(experiment_dir)

    cfg = vars(args).copy()
    cfg["experiment_dir"] = str(experiment_dir)
    (dirs["root"] / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] experiment_dir={experiment_dir}")

    annual = read_annual_table(output_root)
    requested = parse_csv_list(args.targets)
    targets = available_targets(annual, requested)
    if not targets:
        raise RuntimeError("No requested target columns are available.")
    print(f"[INFO] targets={targets}")
    save_csv(annual, dirs["tables"] / "annual_all_regions_canonical.csv")

    model_names = parse_csv_list(args.models)

    print("[STEP 1] Strict no-leakage rolling-origin historical hindcast")
    records, metrics, audit_main = rolling_hindcast_validation(
        annual=annual, targets=targets, model_names=model_names,
        min_history=args.min_history, feature_group="full",
        random_state=args.random_state, add_ensemble=True
    )
    save_csv(records, dirs["validation"] / "historical_hindcast_records_noleak.csv")
    save_csv(metrics, dirs["validation"] / "historical_hindcast_metrics_by_model_noleak.csv")
    save_csv(audit_main, dirs["audit"] / "feature_leakage_audit_main.csv")

    print("[STEP 2] Strict no-leakage feature ablation")
    ab_rec, ab_met, audit_ab = ablation_validation(
        annual=annual, targets=targets, model_name=args.ablation_model,
        min_history=args.min_history, random_state=args.random_state
    )
    save_csv(ab_rec, dirs["validation"] / "feature_ablation_records_noleak.csv")
    save_csv(ab_met, dirs["validation"] / "feature_ablation_metrics_noleak.csv")
    save_csv(audit_ab, dirs["audit"] / "feature_leakage_audit_ablation.csv")

    print("[STEP 3] Extreme residual diagnosis")
    best_model = "ensemble_median" if (not records.empty and "ensemble_median" in set(records["model"])) else None
    if best_model:
        ext = extreme_residuals(records[records["model"] == best_model], quantile=0.99)
    else:
        ext = extreme_residuals(records, quantile=0.99)
    save_csv(ext, dirs["validation"] / "extreme_residual_regions_noleak.csv")

    print("[STEP 4] Recursive future projection to 2050")
    forecast, audit_forecast = recursive_projection(
        annual=annual, targets=targets, forecast_year=args.forecast_year,
        model_names=model_names, random_state=args.random_state
    )
    save_csv(forecast, dirs["forecast"] / f"future_projection_to_{args.forecast_year}_region_year_noleak.csv")
    if not forecast.empty:
        global_forecast = forecast.groupby(["target", "year", "scenario"], as_index=False)[["projected", "q10", "q90"]].mean()
        save_csv(global_forecast, dirs["forecast"] / f"future_projection_to_{args.forecast_year}_global_mean_noleak.csv")
    save_csv(audit_forecast, dirs["audit"] / "feature_leakage_audit_forecast.csv")

    print("[STEP 5] Plotting")
    plot_validation_summary(metrics, dirs["validation"] / "historical_hindcast_validation_normalized_metrics_noleak.png")
    plot_model_comparison(metrics, dirs["figures"])
    plot_ablation(ab_met, dirs["figures"])
    plot_hindcast(records, dirs["validation"])
    plot_obs_pred(records, dirs["validation"])
    plot_residual(records, dirs["validation"])
    plot_projection(annual, forecast, dirs["forecast"])

    print("[STEP 6] HTML report")
    audits = pd.concat([audit_main, audit_ab, audit_forecast], ignore_index=True, sort=False)
    build_report(
        output_path=dirs["report"] / "pv_forecast_strict_noleak_report_cn.html",
        metrics=metrics, ablation=ab_met, extreme=ext, audits=audits, dirs=dirs
    )

    print("[DONE]")
    print(f"[OUT] {experiment_dir}")
    print(f"[REPORT] {dirs['report'] / 'pv_forecast_strict_noleak_report_cn.html'}")


if __name__ == "__main__":
    main()
