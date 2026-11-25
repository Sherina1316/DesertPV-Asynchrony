
from pathlib import Path
import pandas as pd
from pvexp2.report import build_report

base = Path("/home/wsy/pv_nature_experiment/output_v24_22_novelty_gated_remote_change/21_strict_noleak_forecast")
dirs = {
    "root": base,
    "tables": base / "tables",
    "validation": base / "validation",
    "forecast": base / "forecast",
    "figures": base / "figures",
    "audit": base / "audit",
    "report": base / "report",
}

def read(name):
    return pd.read_csv(name) if Path(name).exists() else pd.DataFrame()

metrics = read(dirs["validation"] / "historical_hindcast_metrics_by_model_noleak.csv")
ablation = read(dirs["validation"] / "feature_ablation_metrics_noleak.csv")
extreme = read(dirs["validation"] / "extreme_residual_regions_noleak.csv")

audits = []
for p in [dirs["audit"] / "feature_leakage_audit_main.csv",
          dirs["audit"] / "feature_leakage_audit_ablation.csv",
          dirs["audit"] / "feature_leakage_audit_forecast.csv"]:
    if p.exists():
        audits.append(pd.read_csv(p))
audits = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()

build_report(
    output_path=dirs["report"] / "pv_forecast_strict_noleak_report_cn.html",
    metrics=metrics, ablation=ablation, extreme=extreme, audits=audits, dirs=dirs
)
print("[OK] report rebuilt")
print(dirs["report"] / "pv_forecast_strict_noleak_report_cn.html")
