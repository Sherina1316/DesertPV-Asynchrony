
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import safe_metric_label

COL = {
    "blue": "#0072B2", "green": "#009E73", "orange": "#D55E00",
    "purple": "#6A3D9A", "pink": "#CC79A7", "sky": "#56B4E9",
    "dark": "#333333", "grey": "#8F8F8F"
}

TARGET_COL = {
    "PV_Cover_Ratio": COL["orange"],
    "PV_Cover_Area_km2": COL["orange"],
    "PVIntegrated_PAVG_Optimized": COL["green"],
    "PVIntegrated_EIPAVG": COL["blue"],
    "PVIntegrated_ECRI": COL["purple"],
    "PVIntegrated_Vegetation_Abundance": COL["sky"],
}


def _ax(ax):
    ax.grid(True, ls="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def plot_validation_summary(metrics: pd.DataFrame, out: Path, model: str = "ensemble_median"):
    if metrics.empty:
        return
    df = metrics[metrics["model"] == model].copy()
    if df.empty:
        df = metrics.sort_values("nrmse").groupby("target", as_index=False).first()
    order = ["PVIntegrated_EIPAVG", "PVIntegrated_PAVG_Optimized", "PV_Cover_Area_km2", "PV_Cover_Ratio"]
    df["target"] = pd.Categorical(df["target"], categories=[x for x in order if x in df["target"].unique()], ordered=True)
    df = df.sort_values("target")
    labels = [str(x).replace("_", "\n") for x in df["target"]]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    panels = [
        ("nrmse", "Historical validation NRMSE", COL["blue"]),
        ("nmae", "Historical validation NMAE", COL["green"]),
        ("r2", "Historical validation R²", COL["purple"]),
        ("normalized_bias", "Historical validation normalized bias", COL["orange"]),
    ]
    for ax, (metric, title, color) in zip(axes, panels):
        vals = df[metric].astype(float).to_numpy() if metric in df.columns else np.full(len(df), np.nan)
        ax.bar(range(len(df)), vals, color=color, alpha=0.86)
        ax.axhline(0, color=COL["dark"], lw=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        _ax(ax)
    _save(out)


def plot_model_comparison(metrics: pd.DataFrame, out_dir: Path):
    if metrics.empty:
        return
    for target, grp in metrics.groupby("target"):
        g = grp.sort_values("nrmse").copy()
        labels = g["model"].astype(str).values
        plt.figure(figsize=(max(7.0, len(labels)*0.62), 4.2))
        ax = plt.gca()
        ax.bar(range(len(g)), g["nrmse"].astype(float).values, color=COL["blue"], alpha=0.86)
        ax.set_title(f"Strict no-leakage model comparison by NRMSE: {target}", fontsize=13)
        ax.set_ylabel("NRMSE")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        _ax(ax)
        _save(out_dir / f"model_comparison_nrmse_{safe_metric_label(target)}.png")


def plot_ablation(ablation: pd.DataFrame, out_dir: Path):
    if ablation.empty:
        return
    order = ["base", "base_pv", "base_pv_rolling", "full"]
    for target, grp in ablation.groupby("target"):
        g = grp.copy()
        g["feature_group"] = pd.Categorical(g["feature_group"], categories=order, ordered=True)
        g = g.sort_values("feature_group")
        labels = g["feature_group"].astype(str).values
        plt.figure(figsize=(6.6, 4.2))
        ax = plt.gca()
        ax.bar(range(len(g)), g["nrmse"].astype(float).values, color=COL["green"], alpha=0.86)
        ax.set_title(f"Strict no-leakage feature ablation by NRMSE: {target}", fontsize=13)
        ax.set_ylabel("NRMSE")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        _ax(ax)
        _save(out_dir / f"ablation_nrmse_{safe_metric_label(target)}.png")


def plot_hindcast(records: pd.DataFrame, out_dir: Path, model: str = "ensemble_median"):
    if records.empty:
        return
    df = records[records["model"] == model].copy()
    if df.empty:
        return
    for target, grp in df.groupby("target"):
        g = grp.groupby("year", as_index=False)[["observed", "predicted"]].mean()
        plt.figure(figsize=(7.4, 4.5))
        ax = plt.gca()
        color = TARGET_COL.get(target, COL["blue"])
        ax.plot(g["year"], g["observed"], marker="o", color=color, lw=2.2, label="Observed")
        ax.plot(g["year"], g["predicted"], marker="o", color=COL["pink"], lw=2.2, ls="--", label="Historical hindcast")
        ax.set_title(f"Historical hindcast validation: {target}", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean value")
        ax.legend(frameon=False, loc="best")
        _ax(ax)
        _save(out_dir / f"historical_hindcast_{safe_metric_label(target)}.png")


def plot_obs_pred(records: pd.DataFrame, out_dir: Path, model: str = "ensemble_median"):
    df = records[records["model"] == model].copy()
    if df.empty:
        return
    for target, grp in df.groupby("target"):
        x = grp["observed"].to_numpy(float)
        y = grp["predicted"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) == 0:
            continue
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        plt.figure(figsize=(5.4, 5.1))
        ax = plt.gca()
        ax.scatter(x, y, s=24, alpha=0.62, color="#4C78A8", edgecolors="white", linewidth=0.25)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "--", color=COL["dark"], lw=1.1)
        ax.set_title(f"Observed vs predicted: {target}", fontsize=13)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        _ax(ax)
        _save(out_dir / f"obs_pred_{safe_metric_label(target)}.png")


def plot_residual(records: pd.DataFrame, out_dir: Path, model: str = "ensemble_median"):
    df = records[records["model"] == model].copy()
    if df.empty:
        return
    df["residual"] = df["predicted"] - df["observed"]
    for target, grp in df.groupby("target"):
        x = grp["observed"].to_numpy(float)
        y = grp["residual"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) == 0:
            continue
        thr = np.nanquantile(np.abs(y), 0.99)
        ext = np.abs(y) >= thr
        plt.figure(figsize=(6.4, 4.8))
        ax = plt.gca()
        ax.scatter(x[~ext], y[~ext], s=22, alpha=0.55, color="#4C78A8", edgecolors="white", linewidth=0.25)
        if ext.any():
            ax.scatter(x[ext], y[ext], s=42, alpha=0.88, color=COL["orange"], edgecolors="white", linewidth=0.35, label="Extreme residuals")
            ax.legend(frameon=False, loc="best")
        ax.axhline(0, ls="--", color=COL["dark"], lw=1.1)
        ax.set_title(f"Residual diagnosis: {target}", fontsize=13)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Residual (predicted - observed)")
        _ax(ax)
        _save(out_dir / f"residual_diagnosis_{safe_metric_label(target)}.png")


def plot_projection(annual: pd.DataFrame, forecast: pd.DataFrame, out_dir: Path):
    if forecast.empty:
        return
    med = forecast[forecast["scenario"] == "median"].copy()
    for target, grp in med.groupby("target"):
        obs = annual[["year", target]].dropna() if target in annual.columns else pd.DataFrame()
        if not obs.empty:
            obs_g = obs.groupby("year", as_index=False)[target].mean()
        else:
            obs_g = pd.DataFrame(columns=["year", target])
        f = grp.groupby("year", as_index=False)[["projected", "q10", "q90"]].mean()
        plt.figure(figsize=(7.7, 4.7))
        ax = plt.gca()
        color = TARGET_COL.get(target, COL["blue"])
        if not obs_g.empty:
            ax.plot(obs_g["year"], obs_g[target], marker="o", color=color, lw=2.2, label="Observed")
            ax.axvline(int(obs_g["year"].max()), color=COL["dark"], lw=1.0, ls="--")
        ax.plot(f["year"], f["projected"], marker="o", color=COL["pink"], lw=2.2, ls="--", label="Future projection")
        if {"q10", "q90"}.issubset(f.columns):
            ax.fill_between(f["year"], f["q10"], f["q90"], color=COL["pink"], alpha=0.14, linewidth=0)
        ax.set_title(f"Future projection to 2050: {target}", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean value")
        ax.legend(frameon=False, loc="best")
        _ax(ax)
        _save(out_dir / f"future_projection_to_{int(f['year'].max())}_{safe_metric_label(target)}.png")
