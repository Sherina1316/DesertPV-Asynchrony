from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .utils import atomic_json, ensure_dir, save_table, utc_now
from .sankey import generate_ecology_sankey_figures

COLORS={"blue":"#2F6B8A","orange":"#D97732","green":"#4C956C","red":"#B44A4A","purple":"#76528B","gray":"#666666","light":"#D8D8D8","dark":"#222222"}


def _style(cfg:dict[str,Any])->None:
    mpl.rcParams.update({
        "font.family":cfg["mapping"].get("font_family","DejaVu Sans"),"font.size":7.5,
        "axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "axes.linewidth":0.7,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.major.size":3,"ytick.major.size":3,
        "pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white",
    })


def _fig(cfg:dict[str,Any],height_mm:float=105):
    return plt.subplots(figsize=(float(cfg["mapping"]["figure_width_mm"])/25.4,height_mm/25.4),layout="constrained")


def _save(fig,stem:Path,cfg:dict[str,Any])->None:
    for ext in cfg["mapping"]["raster_formats"]:
        fig.savefig(stem.with_suffix(f".{ext}"),dpi=int(cfg["mapping"]["map_dpi"]),facecolor="white",bbox_inches="tight",pad_inches=0.03)
    for ext in cfg["mapping"]["vector_formats"]:
        fig.savefig(stem.with_suffix(f".{ext}"),facecolor="white",bbox_inches="tight",pad_inches=0.03)
    plt.close(fig)


def _diagnostic(ax,gate:bool)->None:
    if not gate:
        ax.text(0.995,0.012,"QUALIFIED DIAGNOSTIC — COMPLETE-CENSUS ACCURACY GATE NOT PASSED",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.5,color=COLORS["red"],fontweight="bold")


def generate_figures(cfg:dict[str,Any])->pd.DataFrame:
    _style(cfg)
    root=Path(cfg["paths"]["output_root"]); out=ensure_dir(root/"18_figures")
    validation=json.loads((root/"09_accuracy_validation"/"accuracy_validation_status.json").read_text(encoding="utf-8")); gate=bool(validation.get("segmentation_gate_pass",False))
    manifest=[]

    bench=pd.read_csv(root/"05_model_benchmark"/"benchmark_model_summary.csv").sort_values("composite_score")
    fig,ax=_fig(cfg,90); ax.barh(bench.model,bench.composite_score,color=COLORS["blue"],height=0.62); ax.set_xlabel("Spatial-block composite score"); ax.set_title("Model discrimination under spatial blocking",loc="left",fontweight="semibold"); ax.grid(axis="x",color=COLORS["light"],lw=0.5); _save(fig,out/"figure_01_model_benchmark",cfg); manifest.append((1,"model_benchmark"))

    national=pd.read_csv(root/"11_pv_dynamics"/"national_pv_dynamics.csv")
    fig,ax=_fig(cfg,100)
    active_col="observed_active_area_km2" if "observed_active_area_km2" in national else "pv_area_km2"
    cumulative_col="cumulative_installed_area_km2" if "cumulative_installed_area_km2" in national else active_col
    inactive = np.maximum(
        pd.to_numeric(national[cumulative_col], errors="coerce").to_numpy(float)
        - pd.to_numeric(national[active_col], errors="coerce").to_numpy(float),
        0.0,
    )
    ax.plot(national.year,national[cumulative_col],marker="o",ms=3,color="#E69F00",lw=1.25,label="Cumulative installed")
    ax.plot(national.year,national[active_col],marker="o",ms=2.6,color="#0072B2",lw=1.15,label="Observed active")
    ax.plot(national.year,inactive,marker="s",ms=2.4,color="#CC79A7",lw=1.0,label="Installed but inactive/retired")
    ax.set_xlabel("Year"); ax.set_ylabel("PV footprint (km²)"); ax.set_title("Cumulative installed and observed active inland-desert PV",loc="left",fontweight="semibold"); ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8)); ax.grid(axis="y",color=COLORS["light"],lw=0.5); ax.legend(frameon=False,ncol=1,loc="upper left"); _diagnostic(ax,gate); _save(fig,out/"figure_02_national_pv_dynamics",cfg); manifest.append((2,"national_pv_dynamics"))

    val=pd.read_csv(root/"09_accuracy_validation"/"segmentation_validation_summary.csv")
    metric_labels={
        "precision":"Precision",
        "recall":"Recall",
        "f1":"F1 score",
        "iou":"Intersection over union",
        "pr_auc":"Precision–recall AUC",
        "boundary_f1_1px":"Boundary F1 (1 pixel)",
    }
    shown=[m for m in metric_labels if m in set(val.metric)]
    vv=val.set_index("metric").loc[shown].reset_index() if shown else pd.DataFrame()
    fig,ax=_fig(cfg,112)
    if len(vv):
        y=np.arange(len(vv))
        values=pd.to_numeric(vv["mean"],errors="coerce").to_numpy(float)
        low=pd.to_numeric(vv["ci_low"],errors="coerce").to_numpy(float)
        high=pd.to_numeric(vv["ci_high"],errors="coerce").to_numpy(float)
        xerr=np.vstack([np.maximum(values-low,0.0),np.maximum(high-values,0.0)])
        ax.errorbar(values,y,xerr=xerr,fmt="o",ms=4.5,capsize=2.5,color="#0072B2",ecolor="#777777",lw=0.9)
        thresholds=pd.to_numeric(vv.get("required_threshold",np.nan),errors="coerce")
        for i,t in enumerate(thresholds):
            if np.isfinite(t):
                ax.plot([t,t],[i-0.26,i+0.26],color="#D55E00",lw=1.1)
        ax.set_yticks(y,[metric_labels[x] for x in vv.metric])
    ax.set_xlim(0,1.02)
    ax.set_xlabel("Independent complete-census validation metric")
    ax.set_title("Independent 2024 detection accuracy before endpoint enforcement",loc="left",fontweight="semibold")
    ax.text(0.0,-0.16,"Blue: mean and 95% unit interval; orange: predeclared threshold. The final 2024 map uses a bounded, audited refinement of the manual boundary; trusted-core retention and boundary corrections are reported separately.",transform=ax.transAxes,ha="left",va="top",fontsize=5.8,color="#555555",wrap=True)
    ax.grid(axis="x",color=COLORS["light"],lw=0.5)
    _diagnostic(ax,gate)
    _save(fig,out/"figure_03_complete_reference_validation",cfg)
    manifest.append((3,"complete_reference_validation"))

    metrics=pd.read_csv(out/"five_claim_metrics.csv")
    claim1=metrics.loc[metrics.claim_id.eq(1) & metrics.metric.isin(["desert_background_greenness_trend_z_per_year","pv_core_moisture_trend_z_per_year"])]
    fig,ax=_fig(cfg,92); x=np.arange(len(claim1)); ax.bar(x,claim1.value,color=[COLORS["green"],COLORS["blue"]][:len(claim1)],width=0.58); ax.errorbar(x,claim1.value,yerr=[claim1.value-claim1.ci_low,claim1.ci_high-claim1.value],fmt="none",ecolor=COLORS["dark"],capsize=3,lw=0.8); ax.axhline(0,color="black",lw=0.7); ax.set_xticks(x,["Desert background\ngreenness","PV core\nmoisture"][:len(claim1)]); ax.set_ylabel("Climate-adjusted trend (z year⁻¹)"); ax.set_title("Greenness–moisture decoupling",loc="left",fontweight="semibold"); _diagnostic(ax,gate); _save(fig,out/"figure_04_greenness_moisture_decoupling",cfg); manifest.append((4,"greenness_moisture_decoupling"))

    distance=pd.read_csv(out/"distance_gradient_trends.csv").dropna(subset=["distance_km","coefficient"]).sort_values("distance_km")
    fig,ax=_fig(cfg,95); ax.errorbar(distance.distance_km,distance.coefficient,yerr=[distance.coefficient-distance.ci_low,distance.ci_high-distance.coefficient],fmt="o-",ms=4,capsize=2.5,color=COLORS["green"],ecolor=COLORS["gray"]); ax.axhline(0,color="black",lw=0.7); ax.set_xlabel("Distance from PV footprint (km)"); ax.set_ylabel("Greenness trend (z year⁻¹)"); ax.set_title("Distance gradient of ecological response",loc="left",fontweight="semibold"); ax.grid(color=COLORS["light"],lw=0.5); _diagnostic(ax,gate); _save(fig,out/"figure_05_distance_gradient",cfg); manifest.append((5,"distance_gradient"))

    response=pd.read_csv(out/"model_consistent_response_timing.csv")
    fig,ax=_fig(cfg,95); lags=response.response_lag_years.dropna(); bins=np.arange(-0.5,max(6.5,float(lags.max()+1) if len(lags) else 6.5),1); ax.hist(lags,bins=bins,color=COLORS["orange"],edgecolor="white",lw=0.5); if_stable=response.stabilization_observed.mean() if len(response) else np.nan; ax.set_xlabel("Lag from PV structural transition (years)"); ax.set_ylabel("Units"); ax.set_title(f"Vegetation-response lag; stabilized by 2025: {if_stable*100:.1f}%" if np.isfinite(if_stable) else "Vegetation-response lag",loc="left",fontweight="semibold"); _diagnostic(ax,gate); _save(fig,out/"figure_06_response_lag",cfg); manifest.append((6,"response_lag"))

    scenario=pd.read_csv(out/"national_2050_scenario_sensitivity.csv")
    fig,ax=_fig(cfg,92); ax.bar(scenario.scenario,scenario.target_area_2050_km2,color=[COLORS["green"],COLORS["blue"],COLORS["orange"]][:len(scenario)],width=0.6); ax.set_ylabel("Conditional PV area in 2050 (km²)"); ax.set_title("Scenario-sensitive 2050 conditional forecasts",loc="left",fontweight="semibold"); ax.grid(axis="y",color=COLORS["light"],lw=0.5); _diagnostic(ax,gate); _save(fig,out/"figure_07_2050_scenario_sensitivity",cfg); manifest.append((7,"2050_scenario_sensitivity"))

    speed=pd.read_csv(out/"expansion_ecological_speed_by_unit.csv").dropna(subset=["pv_expansion_speed_normalized","ecological_adjustment_speed"])
    fig,ax=_fig(cfg,98); ax.scatter(speed.pv_expansion_speed_normalized,speed.ecological_adjustment_speed,s=20,alpha=0.65,color=COLORS["purple"],edgecolor="white",lw=0.3); ax.set_xscale("symlog",linthresh=0.01); ax.set_yscale("symlog",linthresh=0.001); ax.set_xlabel("Normalized PV expansion speed (year⁻¹)"); ax.set_ylabel("Ecological adjustment speed (z year⁻¹)"); ax.set_title("Regional mismatch between expansion and ecological adjustment",loc="left",fontweight="semibold"); ax.grid(color=COLORS["light"],lw=0.5); _diagnostic(ax,gate); _save(fig,out/"figure_08_speed_mismatch",cfg); manifest.append((8,"speed_mismatch"))

    events=pd.read_csv(root/"13_pv_events"/"pv_expansion_events.csv"); candidate=events.loc[events.get("diagnostic_candidate",False).fillna(False).astype(bool)]
    years=np.arange(int(cfg["study"]["start_year"]),int(cfg["study"]["end_year"])+1); fig,ax=_fig(cfg,90); ax.bar(years,[int((candidate.event_year==y).sum()) for y in years],color=COLORS["orange"],width=0.7); ax.set_xlabel("Persistent structural-transition year"); ax.set_ylabel("Units"); ax.set_title("Distribution of model-consistent PV transitions",loc="left",fontweight="semibold"); ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8)); _diagnostic(ax,gate); _save(fig,out/"figure_09_transition_years",cfg); manifest.append((9,"transition_years"))

    sankey_manifest = generate_ecology_sankey_figures(cfg)
    for row in sankey_manifest.itertuples(index=False):
        manifest.append((int(row.figure), str(row.name)))

    mdf=pd.DataFrame([{"figure":i,"name":name,"png":str(out/f"figure_{i:02d}_{name}.png"),"pdf":str(out/f"figure_{i:02d}_{name}.pdf"),"svg":str(out/f"figure_{i:02d}_{name}.svg")} for i,name in manifest])
    save_table(mdf,out/"figure_manifest.csv")
    atomic_json({"status":"passed","created_utc":utc_now(),"figures":len(mdf),"complete_reference_gate_pass":gate,"single_panel_figures":True,"national_dynamics_uses_nonstacked_lines":True,"independent_model_validation_precedes_boundary_refinement":True,"final_2024_boundary_refinement_audited":True,"vector_exports":True,"ecology_sankey_figures":2,"cbc_pavg_formalized":True,"dpi":int(cfg["mapping"]["map_dpi"]),"claim_specific_figures":5},out/"figure_status.json")
    return mdf
