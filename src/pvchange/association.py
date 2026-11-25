from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .utils import atomic_json, ensure_dir, save_table, utc_now


def _standardize(data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    result=data.copy()
    for covariate in covariates:
        x=pd.to_numeric(result.get(covariate,np.nan),errors="coerce"); x=x.fillna(x.median() if x.notna().any() else 0.0); sd=float(x.std(ddof=0)); result[f"z_{covariate}"]=(x-float(x.mean()))/(sd if sd>1e-12 else 1.0)
    return result


def _fit(formula: str, panel: pd.DataFrame):
    if panel["Id"].nunique()>=8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning); model=smf.ols(formula,data=panel).fit(cov_type="cluster",cov_kwds={"groups":panel["Id"]})
        if np.isfinite(np.asarray(model.bse,float)).all(): return model,"cluster_by_unit"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning); model=smf.ols(formula,data=panel).fit(cov_type="HC3")
    return model,"HC3_fallback"


def _fdr(pvalues: pd.Series) -> pd.Series:
    p=pd.to_numeric(pvalues,errors="coerce").to_numpy(float); result=np.full(len(p),np.nan); valid=np.flatnonzero(np.isfinite(p))
    if not len(valid): return pd.Series(result,index=pvalues.index)
    order=valid[np.argsort(p[valid])]; q=p[order]*len(order)/np.arange(1,len(order)+1); q=np.minimum.accumulate(q[::-1])[::-1]; result[order]=np.clip(q,0,1); return pd.Series(result,index=pvalues.index)


def _metric_panel(anomaly: pd.DataFrame, events: pd.DataFrame, environment: pd.DataFrame, metric: str, cfg: dict[str,Any]) -> pd.DataFrame:
    panel=anomaly.loc[anomaly["metric"].eq(metric),["Id","year","external_anomaly","quality"]].rename(columns={"external_anomaly":"outcome"})
    event_year=events.loc[events["qualified"].fillna(False).astype(bool),["Id","event_year"]]
    panel=panel.merge(event_year,on="Id",how="left").merge(environment,on=["Id","year"],how="left")
    panel=panel.loc[panel["quality"].eq("valid")&panel["outcome"].notna()].copy()
    panel["ever_treated"]=panel["event_year"].notna().astype(int)
    panel["treated"]=(panel["event_year"].notna()&(panel["year"]>=panel["event_year"])).astype(int)
    panel["not_yet_or_never_control"]=(panel["event_year"].isna()|(panel["year"]<panel["event_year"])).astype(int)
    panel["event_time"]=panel["year"]-panel["event_year"]
    return _standardize(panel,list(cfg["ecology"]["climate_covariates"]))


def _same_year_matches(panel: pd.DataFrame,covariates:list[str],metric:str,cfg:dict[str,Any]) -> tuple[pd.DataFrame,dict[str,Any]]:
    feature=[f"z_{c}" for c in covariates]; rows=[]
    for year,year_data in panel.groupby("year"):
        treated=year_data.loc[year_data["treated"].eq(1)].copy(); controls=year_data.loc[year_data["not_yet_or_never_control"].eq(1)].copy()
        if treated.empty or controls.empty: continue
        combined=pd.concat([treated.assign(_t=1),controls.assign(_t=0)],ignore_index=True)
        if combined["_t"].nunique()<2: continue
        model=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=int(cfg["project"]["random_seed"])); model.fit(combined[feature],combined["_t"])
        propensity=np.clip(model.predict_proba(combined[feature])[:,1],1e-6,1-1e-6); combined["logit"]=np.log(propensity/(1-propensity)); treated=combined.loc[combined["_t"].eq(1)]; controls=combined.loc[combined["_t"].eq(0)].reset_index(drop=True)
        k=min(int(cfg["ecology"]["matched_control_candidates_per_treated"]),len(controls)); nbr=NearestNeighbors(n_neighbors=k).fit(controls[["logit"]]); distances,indices=nbr.kneighbors(treated[["logit"]]); caliper=float(cfg["ecology"]["propensity_caliper"])*max(float(combined["logit"].std(ddof=0)),1e-6)
        for position,(distance_row,index_row) in enumerate(zip(distances,indices)):
            tr=treated.iloc[position]; accepted=[controls.iloc[int(i)] for d,i in zip(distance_row,index_row) if float(d)<=caliper and int(controls.iloc[int(i)].Id)!=int(tr.Id)]
            if not accepted: continue
            control_mean=float(np.mean([float(x.outcome) for x in accepted])); rows.append({"metric":metric,"treated_Id":int(tr.Id),"treated_year":int(year),"treated_outcome":float(tr.outcome),"matched_control_count":len(accepted),"matched_control_outcome_mean":control_mean,"matched_difference":float(tr.outcome)-control_mean})
    matches=pd.DataFrame(rows)
    if matches.empty: return matches,{"metric":metric,"status":"no_same_year_matches","matches":0}
    differences=matches["matched_difference"].to_numpy(float); se=float(np.std(differences,ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else np.nan; minimum=int(cfg["ecology"]["minimum_valid_control_matches"])
    return matches,{"metric":metric,"status":"valid" if len(matches)>=minimum else "insufficient_match_support","matches":len(matches),"minimum_required_matches":minimum,"matched_effect":float(np.mean(differences)),"matched_effect_se":se,"ci_low":float(np.mean(differences)-1.96*se) if np.isfinite(se) else np.nan,"ci_high":float(np.mean(differences)+1.96*se) if np.isfinite(se) else np.nan}


def run_conditional_association(cfg: dict[str, Any]) -> pd.DataFrame:
    root=Path(cfg["paths"]["output_root"]); out=ensure_dir(root/"14_conditional_association")
    anomaly=pd.read_csv(root/"12_dynamic_ecology"/"external_ecology_anomaly_panel.csv"); environment=pd.read_csv(root/"12_dynamic_ecology"/"environmental_covariates_imputed.csv"); events=pd.read_csv(root/"13_pv_events"/"pv_expansion_events.csv")
    metrics=list(cfg["ecology"]["metrics"]); covariates=list(cfg["ecology"]["climate_covariates"]); qualified=int(events["qualified"].fillna(False).sum())
    if qualified==0:
        fixed=pd.DataFrame([{"metric":m,"model_status":"no_qualified_events","regions":0,"observations":0,"coefficient":np.nan,"std_error":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"fdr_q_value":np.nan,"fdr_significant":False} for m in metrics]); event=pd.DataFrame(columns=["metric","event_time","coefficient","std_error","ci_low","ci_high","reference"]); pretrend=pd.DataFrame([{"metric":m,"status":"no_qualified_events","pretrend_p_value":np.nan} for m in metrics]); matches=pd.DataFrame(); match_summary=pd.DataFrame([{"metric":m,"status":"no_qualified_events","matches":0} for m in metrics])
        save_table(fixed,out/"fixed_effect_associations.csv"); save_table(event,out/"event_study_coefficients.csv"); save_table(pretrend,out/"event_study_pretrend_tests.csv"); save_table(matches,out/"propensity_matched_pairs.csv"); save_table(match_summary,out/"propensity_matching_summary.csv")
        atomic_json({"status":"completed_no_events","created_utc":utc_now(),"qualified_events":0,"valid_adjusted_models":0,"valid_propensity_match_models":0,"identification_design":"staggered event study with not-yet-treated and never-treated controls","causal_claim_allowed":False},out/"conditional_association_status.json"); return fixed
    fixed_rows=[]; event_rows=[]; pretrend_rows=[]; all_matches=[]; matching_summaries=[]
    for metric in tqdm(metrics,desc="Staggered ecological event study",unit="metric",disable=not cfg["processing"].get("show_progress",True)):
        panel=_metric_panel(anomaly,events,environment,metric,cfg); minimum_units=int(cfg["association"]["minimum_treated_units"]); minimum_obs=int(cfg["association"]["minimum_observations"])
        if panel.loc[panel["ever_treated"].eq(1),"Id"].nunique()<minimum_units or len(panel)<minimum_obs or panel["treated"].nunique()<2:
            fixed_rows.append({"metric":metric,"model_status":"insufficient_data","regions":int(panel.Id.nunique()),"observations":len(panel)}); matching_summaries.append({"metric":metric,"status":"insufficient_data","matches":0}); continue
        cov_terms=" + ".join(f"z_{c}" for c in covariates); formula=f"outcome ~ treated + {cov_terms} + C(Id) + C(year)"
        try:
            fit,covariance=_fit(formula,panel); coefficient=float(fit.params.get("treated",np.nan)); se=float(fit.bse.get("treated",np.nan)); pvalue=float(fit.pvalues.get("treated",np.nan)); fixed_rows.append({"metric":metric,"coefficient":coefficient,"std_error":se,"ci_low":coefficient-1.96*se,"ci_high":coefficient+1.96*se,"p_value":pvalue,"regions":int(panel.Id.nunique()),"treated_regions":int(panel.loc[panel.ever_treated.eq(1),"Id"].nunique()),"observations":len(panel),"covariance_method":covariance,"model_status":"valid"})
            lo=int(cfg["association"]["relative_year_min"]); hi=int(cfg["association"]["relative_year_max"]); ref=int(cfg["association"]["reference_relative_year"])
            panel_event=panel.copy(); names=[]
            for relative in range(lo,hi+1):
                if relative==ref: continue
                name=f"event_{'m'+str(abs(relative)) if relative<0 else 'p'+str(relative)}"; panel_event[name]=((panel_event["ever_treated"]==1)&(panel_event["event_time"]==relative)).astype(int); names.append((relative,name))
            event_formula=f"outcome ~ {' + '.join(name for _,name in names)} + {cov_terms} + C(Id) + C(year)"; event_fit,event_covariance=_fit(event_formula,panel_event)
            event_rows.append({"metric":metric,"event_time":ref,"coefficient":0.0,"std_error":0.0,"ci_low":0.0,"ci_high":0.0,"reference":True,"covariance_method":event_covariance})
            for relative,name in names:
                value=float(event_fit.params.get(name,np.nan)); event_se=float(event_fit.bse.get(name,np.nan)); event_rows.append({"metric":metric,"event_time":relative,"coefficient":value,"std_error":event_se,"ci_low":value-1.96*event_se if np.isfinite(event_se) else np.nan,"ci_high":value+1.96*event_se if np.isfinite(event_se) else np.nan,"reference":False,"covariance_method":event_covariance})
            pre_names=[name for relative,name in names if relative<ref]
            if pre_names:
                restriction=" = 0, ".join(pre_names)+" = 0"; test=event_fit.f_test(restriction); pre_p=float(np.asarray(test.pvalue).reshape(-1)[0])
            else: pre_p=np.nan
            pretrend_rows.append({"metric":metric,"status":"valid","pretrend_p_value":pre_p,"pretrend_pass":bool(np.isfinite(pre_p) and pre_p>=float(cfg["association"]["require_pretrend_p_value"])),"tested_leads":";".join(pre_names)})
            matches,summary=_same_year_matches(panel,covariates,metric,cfg); matching_summaries.append(summary); all_matches.append(matches)
        except Exception as exc:
            fixed_rows.append({"metric":metric,"model_status":f"failed: {type(exc).__name__}: {exc}","regions":int(panel.Id.nunique()),"observations":len(panel)}); matching_summaries.append({"metric":metric,"status":"model_failed","matches":0,"reason":str(exc)})
    fixed=pd.DataFrame(fixed_rows); fixed["fdr_q_value"]=_fdr(fixed.get("p_value",pd.Series(np.nan,index=fixed.index))); fixed["fdr_significant"]=fixed["fdr_q_value"]<=float(cfg["association"]["fdr_alpha"]); event=pd.DataFrame(event_rows); pretrend=pd.DataFrame(pretrend_rows); matches=pd.concat([x for x in all_matches if not x.empty],ignore_index=True) if any(not x.empty for x in all_matches) else pd.DataFrame(); matching_summary=pd.DataFrame(matching_summaries)
    save_table(fixed,out/"fixed_effect_associations.csv"); save_table(event,out/"event_study_coefficients.csv"); save_table(pretrend,out/"event_study_pretrend_tests.csv"); save_table(matches,out/"propensity_matched_pairs.csv"); save_table(matching_summary,out/"propensity_matching_summary.csv")
    valid_models=int(fixed.get("model_status",pd.Series(dtype=str)).eq("valid").sum()); valid_matches=int(matching_summary.get("status",pd.Series(dtype=str)).eq("valid").sum()); pretrend_pass=int(pretrend.get("pretrend_pass",pd.Series(dtype=bool)).fillna(False).sum())
    status="passed" if valid_models>0 else "completed_insufficient_models"; atomic_json({"status":status,"created_utc":utc_now(),"qualified_events":qualified,"valid_adjusted_models":valid_models,"valid_propensity_match_models":valid_matches,"pretrend_pass_models":pretrend_pass,"identification_design":"staggered event study using not-yet-treated and never-treated controls in the same calendar year","fixed_spatial_support":True,"environmental_covariates":covariates,"causal_claim_allowed":False},out/"conditional_association_status.json")
    return fixed
