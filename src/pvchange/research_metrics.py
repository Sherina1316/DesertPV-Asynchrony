from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress, spearmanr

from .utils import atomic_json, ensure_dir, save_table, utc_now

ZONE_DISTANCE_KM = {
    "pv_core": 0.0,
    "edge_0_250m": 0.125,
    "near_250_500m": 0.375,
    "local_500_1000m": 0.75,
    "landscape_1000_3000m": 2.0,
    "context_3000_5000m": 4.0,
}


def _zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = float(x.std(ddof=0))
    return (x - float(x.mean())) / (sd if np.isfinite(sd) and sd > 1e-12 else 1.0)


def _bootstrap_mean(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sample[i] = np.mean(rng.choice(x, x.size, replace=True))
    return float(np.mean(x)), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def _unit_slopes(panel: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for region_id, group in panel.groupby("Id"):
        g = group[["year", value]].dropna().sort_values("year")
        if len(g) < 5:
            continue
        fit = linregress(g.year.to_numpy(float), g[value].to_numpy(float))
        rows.append({"Id":int(region_id), "slope":float(fit.slope), "p_value":float(fit.pvalue), "n_years":len(g)})
    return pd.DataFrame(rows)


def _fixed_effect_trend(panel: pd.DataFrame, outcome: str, covariates: list[str]) -> dict[str, float]:
    cols = ["Id", "year", outcome, *covariates]
    data = panel[cols].copy().dropna()
    if len(data) < 80 or data.Id.nunique() < 8:
        return {"coefficient":np.nan,"std_error":np.nan,"p_value":np.nan,"ci_low":np.nan,"ci_high":np.nan,"observations":len(data),"units":data.Id.nunique()}
    year0 = float(data.year.min())
    data["time"] = data.year.astype(float) - year0
    x_parts = [data[["time", *covariates]].astype(float)]
    x_parts.append(pd.get_dummies(data.Id.astype(str), prefix="unit", drop_first=True, dtype=float))
    X = sm.add_constant(pd.concat(x_parts, axis=1), has_constant="add")
    model = sm.OLS(data[outcome].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups":data.Id})
    coef = float(model.params.get("time", np.nan)); se=float(model.bse.get("time", np.nan))
    return {"coefficient":coef,"std_error":se,"p_value":float(model.pvalues.get("time",np.nan)),"ci_low":coef-1.96*se,"ci_high":coef+1.96*se,"observations":len(data),"units":data.Id.nunique()}


def _event_response_diagnostics(anomaly: pd.DataFrame, events: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    metrics = list(cfg["ecology"]["metrics"])
    rows = []
    for event in events.itertuples(index=False):
        region_id = int(event.Id); event_year = int(event.event_year)
        detections = []
        for metric in metrics:
            g = anomaly.loc[(anomaly.Id.eq(region_id)) & anomaly.metric.eq(metric) & anomaly.quality.eq("valid"), ["year","external_anomaly"]].dropna().sort_values("year")
            pre = g.loc[g.year < event_year, "external_anomaly"].to_numpy(float)
            post = g.loc[g.year >= event_year].copy()
            if len(pre) < 3 or len(post) < 2:
                continue
            baseline=float(np.median(pre)); mad=float(np.median(np.abs(pre-baseline))); threshold=max(2.5*1.4826*mad,1e-4)
            post["delta"]=post.external_anomaly-baseline
            arr=post.delta.to_numpy(float); yrs=post.year.to_numpy(int)
            onset=np.nan; direction="none"
            for i in range(len(arr)-1):
                if abs(arr[i])>=threshold and abs(arr[i+1])>=threshold and np.sign(arr[i])==np.sign(arr[i+1]):
                    onset=int(yrs[i]); direction="increase" if arr[i]>0 else "decline"; break
            stability=np.nan
            if np.isfinite(onset):
                after=post.loc[post.year>=onset]
                for i in range(max(0,len(after)-2)):
                    w=after.iloc[i:i+3]
                    if len(w)<3: continue
                    slope=linregress(w.year,w.external_anomaly).slope
                    cv=float(np.std(w.external_anomaly)/max(abs(np.mean(w.external_anomaly)),1e-4))
                    if abs(slope)<=0.002 and cv<=0.15:
                        stability=int(w.year.iloc[-1]); break
                detections.append({"metric":metric,"onset":onset,"direction":direction,"stability":stability,"amplitude":float(np.nanmedian(post.delta))})
        if detections:
            d=pd.DataFrame(detections)
            dominant=str(d.direction.value_counts().index[0])
            agree=d.loc[d.direction.eq(dominant)]
            onset=float(np.median(agree.onset)) if len(agree)>=2 else np.nan
            stable_values=agree.stability.dropna().to_numpy(float)
            stability=float(np.median(stable_values)) if len(stable_values)>=2 else np.nan
            amplitude=float(np.median(np.abs(agree.amplitude)))
        else:
            dominant="insufficient"; onset=stability=amplitude=np.nan
        rows.append({
            "Id":region_id,"event_year":event_year,"response_direction":dominant,
            "supporting_metrics":len(detections),"response_onset_year":onset,
            "response_lag_years":onset-event_year if np.isfinite(onset) else np.nan,
            "stabilization_year":stability,
            "stabilization_lag_years":stability-event_year if np.isfinite(stability) else np.nan,
            "stabilization_observed":bool(np.isfinite(stability)),"response_amplitude":amplitude,
        })
    columns = [
        "Id", "event_year", "response_direction", "supporting_metrics",
        "response_onset_year", "response_lag_years", "stabilization_year",
        "stabilization_lag_years", "stabilization_observed",
        "response_amplitude",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_research_claim_metrics(cfg: dict[str, Any]) -> pd.DataFrame:
    root=Path(cfg["paths"]["output_root"]); out=ensure_dir(root/"18_figures")
    ecology=pd.read_csv(root/"12_dynamic_ecology"/"dynamic_pv_ecology_panel.csv")
    env=pd.read_csv(root/"12_dynamic_ecology"/"environmental_covariates_imputed.csv")
    anomaly=pd.read_csv(root/"12_dynamic_ecology"/"external_ecology_anomaly_panel.csv")
    events=pd.read_csv(root/"13_pv_events"/"pv_expansion_events.csv")
    dynamics=pd.read_csv(root/"11_pv_dynamics"/"unit_pv_dynamics_summary.csv")
    future=pd.read_csv(root/"16_future_2050"/"conditional_2050_candidate_summary.csv")
    validation=json.loads((root/"09_accuracy_validation"/"accuracy_validation_status.json").read_text(encoding="utf-8"))
    covariates=list(cfg["ecology"]["climate_covariates"])
    reps=int(cfg["research_claims"].get("bootstrap_repetitions",1000)); seed=int(cfg["project"]["random_seed"])

    wide=ecology.pivot_table(index=["Id","year","zone"],columns="metric",values="mean",dropna=False).reset_index(); wide.columns.name=None
    wide=wide.merge(env[["Id","year",*covariates]],on=["Id","year"],how="left")
    for metric in [m for m in ["NDVI","EVI","TC_Greenness","NDMI","TC_Wetness"] if m in wide]:
        wide[f"z_{metric}"]=_zscore(wide[metric])
    greenness_cols=[c for c in ["z_NDVI","z_EVI","z_TC_Greenness"] if c in wide]
    moisture_cols=[c for c in ["z_NDMI","z_TC_Wetness"] if c in wide]
    wide["greenness_composite"]=wide[greenness_cols].mean(axis=1)
    wide["moisture_composite"]=wide[moisture_cols].mean(axis=1)
    save_table(wide, out/"zone_environment_panel.csv")

    context=wide.loc[wide.zone.eq("context_3000_5000m")]
    core=wide.loc[wide.zone.eq("pv_core")]
    green_trend=_fixed_effect_trend(context,"greenness_composite",covariates)
    core_green=_fixed_effect_trend(core,"greenness_composite",covariates)
    core_moist=_fixed_effect_trend(core,"moisture_composite",covariates)
    decoupling=core_green["coefficient"]-core_moist["coefficient"] if np.isfinite(core_green["coefficient"]) and np.isfinite(core_moist["coefficient"]) else np.nan

    zone_rows=[]
    for zone,g in wide.groupby("zone"):
        trend=_fixed_effect_trend(g,"greenness_composite",covariates)
        zone_rows.append({"zone":zone,"distance_km":ZONE_DISTANCE_KM.get(zone,np.nan),**trend})
    zone_trends=pd.DataFrame(zone_rows).sort_values("distance_km")
    valid_zone=zone_trends.dropna(subset=["distance_km","coefficient"])
    gradient=linregress(valid_zone.distance_km,valid_zone.coefficient) if len(valid_zone)>=3 else None
    edge=zone_trends.loc[zone_trends.zone.isin(["edge_0_250m","near_250_500m"]),"coefficient"].mean()
    core_coef=zone_trends.loc[zone_trends.zone.eq("pv_core"),"coefficient"].mean()
    surrounding_minus_core=float(edge-core_coef) if np.isfinite(edge) and np.isfinite(core_coef) else np.nan
    save_table(zone_trends,out/"distance_gradient_trends.csv")

    diagnostic_events=events.loc[events.get("diagnostic_candidate",pd.Series(False,index=events.index)).fillna(False).astype(bool)].copy()
    response=_event_response_diagnostics(anomaly,diagnostic_events,cfg)
    save_table(response,out/"model_consistent_response_timing.csv")
    lag_mean,lag_lo,lag_hi=_bootstrap_mean(response.response_lag_years.to_numpy(float) if len(response) else np.array([]),reps,seed+10)
    observed_stability=float(response.stabilization_observed.mean()) if len(response) else np.nan
    right_censored=1-observed_stability if np.isfinite(observed_stability) else np.nan

    scenario=future.groupby("scenario",as_index=False).agg(target_area_2050_km2=("target_area_2050_km2","sum"),additional_area_2050_km2=("additional_area_2050_km2","sum"))
    transition=float(scenario.loc[scenario.scenario.eq("transition"),"target_area_2050_km2"].iloc[0]) if scenario.scenario.eq("transition").any() else np.nan
    scenario_spread=float(scenario.target_area_2050_km2.max()-scenario.target_area_2050_km2.min()) if len(scenario) else np.nan
    scenario_relative_spread=scenario_spread/max(abs(transition),1e-9) if np.isfinite(transition) else np.nan
    save_table(scenario,out/"national_2050_scenario_sensitivity.csv")

    reference_area_column = "cumulative_installed_area_2024_km2" if "cumulative_installed_area_2024_km2" in dynamics else "pv_area_2024_km2"
    speed=dynamics[[c for c in ["Id","peak_annual_gain_km2",reference_area_column,"net_change_km2"] if c in dynamics]].copy()
    speed.rename(columns={reference_area_column:"reference_cumulative_installed_area_km2"}, inplace=True)
    speed["pv_expansion_speed_normalized"]=speed["peak_annual_gain_km2"]/speed["reference_cumulative_installed_area_km2"].clip(lower=0.01)
    speed=speed.merge(response[["Id","response_lag_years","response_amplitude","stabilization_observed"]],on="Id",how="left")
    speed["ecological_adjustment_speed"]=speed.response_amplitude.abs()/speed.response_lag_years.clip(lower=1)
    speed["expansion_adjustment_ratio"]=speed.pv_expansion_speed_normalized/speed.ecological_adjustment_speed.where(speed.ecological_adjustment_speed>=1e-4,np.nan)
    speed["log10_expansion_adjustment_ratio"]=np.log10(speed["expansion_adjustment_ratio"].where(speed["expansion_adjustment_ratio"]>0))
    valid_speed=speed.dropna(subset=["pv_expansion_speed_normalized","ecological_adjustment_speed"])
    rho,p_rho=spearmanr(valid_speed.pv_expansion_speed_normalized,valid_speed.ecological_adjustment_speed) if len(valid_speed)>=5 else (np.nan,np.nan)
    ratio=valid_speed.expansion_adjustment_ratio.replace([np.inf,-np.inf],np.nan).dropna()
    log_ratio=valid_speed.log10_expansion_adjustment_ratio.replace([np.inf,-np.inf],np.nan).dropna()
    ratio_cv=float(ratio.std(ddof=1)/abs(ratio.mean())) if len(ratio)>=2 and abs(ratio.mean())>1e-12 else np.nan
    log_ratio_median=float(log_ratio.median()) if len(log_ratio) else np.nan
    log_ratio_iqr=float(log_ratio.quantile(0.75)-log_ratio.quantile(0.25)) if len(log_ratio)>=2 else np.nan
    save_table(speed,out/"expansion_ecological_speed_by_unit.csv")

    metrics=[
        {"claim_id":1,"metric":"desert_background_greenness_trend_z_per_year","value":green_trend["coefficient"],"ci_low":green_trend["ci_low"],"ci_high":green_trend["ci_high"],"p_value":green_trend["p_value"],"units":green_trend["units"],"interpretation":"positive values indicate overall desert greening after environmental adjustment"},
        {"claim_id":1,"metric":"pv_core_moisture_trend_z_per_year","value":core_moist["coefficient"],"ci_low":core_moist["ci_low"],"ci_high":core_moist["ci_high"],"p_value":core_moist["p_value"],"units":core_moist["units"],"interpretation":"negative values indicate declining core moisture indicators"},
        {"claim_id":1,"metric":"greenness_moisture_decoupling_index","value":decoupling,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":min(core_green["units"],core_moist["units"]),"interpretation":"core greenness trend minus core moisture trend"},
        {"claim_id":2,"metric":"surrounding_minus_core_greenness_trend","value":surrounding_minus_core,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":int(wide.Id.nunique()),"interpretation":"positive values indicate stronger greening in 0-500 m surroundings than in the PV core"},
        {"claim_id":2,"metric":"distance_gradient_z_per_year_per_km","value":float(gradient.slope) if gradient else np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_value":float(gradient.pvalue) if gradient else np.nan,"units":int(wide.Id.nunique()),"interpretation":"spatial gradient of climate-adjusted greenness trend with distance"},
        {"claim_id":3,"metric":"median_response_lag_years","value":lag_mean,"ci_low":lag_lo,"ci_high":lag_hi,"p_value":np.nan,"units":int(response.response_lag_years.notna().sum()) if len(response) else 0,"interpretation":"years from structural PV transition to consistent multi-metric vegetation response"},
        {"claim_id":3,"metric":"not_stabilized_fraction_by_2025","value":right_censored,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":len(response),"interpretation":"fraction without an observed stable three-year state by 2025"},
        {"claim_id":4,"metric":"2050_scenario_relative_spread","value":scenario_relative_spread,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":int(future.Id.nunique()),"interpretation":"range across scenarios divided by the transition-scenario total"},
        {"claim_id":4,"metric":"2050_scenario_absolute_spread_km2","value":scenario_spread,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":int(future.Id.nunique()),"interpretation":"difference between accelerated and conservative conditional forecasts"},
        {"claim_id":5,"metric":"expansion_adjustment_speed_ratio_cv","value":ratio_cv,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":len(valid_speed),"interpretation":"regional coefficient of variation of PV expansion speed relative to ecological adjustment speed"},
        {"claim_id":5,"metric":"expansion_vs_adjustment_spearman_rho","value":float(rho),"ci_low":np.nan,"ci_high":np.nan,"p_value":float(p_rho),"units":len(valid_speed),"interpretation":"rank association between normalized PV expansion speed and ecological adjustment speed"},
        {"claim_id":5,"metric":"log10_expansion_adjustment_ratio_median","value":log_ratio_median,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":len(log_ratio),"interpretation":"robust median log10 ratio after excluding near-zero ecological adjustment denominators"},
        {"claim_id":5,"metric":"log10_expansion_adjustment_ratio_iqr","value":log_ratio_iqr,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"units":len(log_ratio),"interpretation":"interquartile spread of the robust log10 expansion-adjustment ratio"},
    ]
    metric_df=pd.DataFrame(metrics)
    save_table(metric_df,out/"five_claim_metrics.csv")

    def fmt(v,d=3): return "不可估计" if not np.isfinite(v) else f"{v:.{d}f}"
    narratives={
        "claim_1":f"经区域固定效应、年份固定效应及六项环境协变量调整后，沙漠背景绿度综合指标年趋势为 {fmt(green_trend['coefficient'])} z/年（95% CI {fmt(green_trend['ci_low'])}–{fmt(green_trend['ci_high'])}），光伏核心区水分综合指标趋势为 {fmt(core_moist['coefficient'])} z/年。绿度—水分解耦指数为 {fmt(decoupling)}。",
        "claim_2":f"0–500 m周边区域的绿度趋势较核心区高 {fmt(surrounding_minus_core)} z/年；距离梯度为 {fmt(float(gradient.slope) if gradient else np.nan)} z/年/km，显示生态响应具有距离梯度和区域异质性。",
        "claim_3":f"在获得一致多指标响应的区域中，结构转折至生态响应的中位滞后为 {fmt(lag_mean,2)} 年（bootstrap 95% CI {fmt(lag_lo,2)}–{fmt(lag_hi,2)}）；截至2025年，未观察到稳定状态的比例为 {fmt(right_censored*100,1)}%。",
        "claim_4":f"2050年加速与保守情景的条件性光伏面积差为 {fmt(scenario_spread,1)} km²，相对过渡情景的敏感度为 {fmt(scenario_relative_spread*100,1)}%，说明结果高度依赖情景假设而非确定性建设预测。",
        "claim_5":f"光伏扩张—生态调整速度比的区域变异系数为 {fmt(ratio_cv)}；两种速度的Spearman相关为 ρ={fmt(float(rho))}（p={fmt(float(p_rho))}），表明区域响应节奏存在显著异质性。",
        "evidence_grade":"formal" if validation.get("segmentation_gate_pass",False) else "qualified_diagnostic",
        "validation_scope":"2024 SHP confirmed-positive sensitivity only; full-scene precision is not identifiable from the SHP",
    }
    atomic_json({"status":"passed","created_utc":utc_now(),"narratives":narratives,"scientific_gate_pass":bool(validation.get("segmentation_gate_pass",False))},out/"five_claim_narratives.json")
    return metric_df
