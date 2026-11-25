from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


TARGET_ALIASES = {
    "PV_Cover_Ratio": [
        "PV_Cover_Ratio", "Mean_PV_Cover_Ratio", "pv_cover_ratio", "PV cover ratio"
    ],
    "PV_Cover_Area_km2": [
        "PV_Cover_Area_km2", "Mean_PV_Cover_Area_km2", "pv_cover_area_km2", "PV cover area",
        "cumulative_installed_area_km2", "observed_active_area_km2", "pv_area_km2",
    ],
    "PVIntegrated_PAVG_Optimized": [
        "PVIntegrated_PAVG_Optimized", "Mean_PVIntegrated_PAVG_Optimized",
        "Mean_PAVG_Optimized", "PAVG_Optimized", "Mean_PAVG", "PAVG",
        "PAVG_climate_adjusted", "PAVG_fraction_raw",
    ],
    "PVIntegrated_EIPAVG": [
        "PVIntegrated_EIPAVG", "Mean_PVIntegrated_EIPAVG", "Mean_EIPAVG", "EIPAVG"
    ],
    "PVIntegrated_ECRI": [
        "PVIntegrated_ECRI", "Mean_PVIntegrated_ECRI", "Mean_ECRI", "ECRI"
    ],
    "PVIntegrated_Vegetation_Abundance": [
        "PVIntegrated_Vegetation_Abundance", "Mean_PVIntegrated_Vegetation_Abundance",
        "Mean_Vegetation_Abundance", "Vegetation_Abundance", "vegetation_area_km2",
    ],
    "Uncertainty": ["Uncertainty", "Mean_Uncertainty", "uncertainty", "area_relative_uncertainty"],
}

DEFAULT_TARGETS = [
    "PV_Cover_Ratio",
    "PV_Cover_Area_km2",
    "PVIntegrated_PAVG_Optimized",
    "PVIntegrated_EIPAVG",
    "PVIntegrated_ECRI",
    "PVIntegrated_Vegetation_Abundance",
]

CORE_METRICS = list(TARGET_ALIASES.keys())

ENVIRONMENTAL_COVARIATES = [
    "precipitation", "temperature", "potential_evapotranspiration",
    "soil_moisture", "radiation", "wind_speed",
]



def parse_region_id(path: Path) -> int | None:
    for part in path.parts:
        m = re.match(r"region_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def _coalesce_numeric(df: pd.DataFrame, columns: list[str], default: float | None = np.nan) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            result = result.where(result.notna(), values)
    if default is not None:
        result = result.fillna(float(default))
    return result


def _robust01(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    vals = x.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return pd.Series(np.nan, index=x.index, dtype=float)
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index, dtype=float).where(x.notna(), np.nan)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def _bounded_fraction(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    return x.clip(0.0, 1.0)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce").astype(float)
    d = pd.to_numeric(den, errors="coerce").astype(float)
    out = n / d.where(d > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _merge_if_exists(left: pd.DataFrame, path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return left
    right = pd.read_csv(path)
    if "Id" in right.columns and "region" not in right.columns:
        right = right.rename(columns={"Id": "region"})
    if "Year" in right.columns and "year" not in right.columns:
        right = right.rename(columns={"Year": "year"})
    if usecols is not None:
        keep = [c for c in usecols if c in right.columns]
        right = right[["region", "year", *[c for c in keep if c not in {"region", "year"}]]]
    right["region"] = pd.to_numeric(right["region"], errors="coerce").astype("Int64")
    right["year"] = pd.to_numeric(right["year"], errors="coerce").astype("Int64")
    right = right.dropna(subset=["region", "year"]).copy()
    right["region"] = right["region"].astype(int)
    right["year"] = right["year"].astype(int)
    return left.merge(right, on=["region", "year"], how="left", suffixes=("", "_extra"))


def _build_from_v24_output(output_root: Path) -> pd.DataFrame:
    """Build a strict forecasting panel from v24.9/v24.11 pipeline outputs.

    The v24 segmentation/ecology pipeline stores the scientifically important
    quantities across several stage folders rather than in the older
    `global_summary/annual_all_regions.csv` layout. This builder constructs the
    canonical region-year table used by the no-leakage hindcast module while
    preserving the v24 semantics:

    * the corrected 2024 SHP is the complete binary census: inside is PV and
      outside is non-PV in 2024;
    * historical years are not copied backward from 2024 without temporal
      spectral evidence;
    * PV_Cover_Area_km2 is the cumulative installed footprint so the 2050
      extrapolation obeys the irreversible installation state;
    * observed-active area and annual gains remain available as lagged features.
    """
    dynamics_path = output_root / "11_pv_dynamics" / "annual_pv_dynamics.csv"
    if not dynamics_path.exists():
        raise FileNotFoundError(f"v24 dynamics table not found: {dynamics_path}")
    dyn = pd.read_csv(dynamics_path)
    if "Id" in dyn.columns:
        dyn = dyn.rename(columns={"Id": "region"})
    if "Year" in dyn.columns and "year" not in dyn.columns:
        dyn = dyn.rename(columns={"Year": "year"})
    if "region" not in dyn.columns or "year" not in dyn.columns:
        raise KeyError("annual_pv_dynamics.csv must contain Id/region and year columns")

    df = dyn.copy()
    df["region"] = pd.to_numeric(df["region"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["region", "year"]).copy()
    df["region"] = df["region"].astype(int)
    df["year"] = df["year"].astype(int)

    cumulative = _coalesce_numeric(df, ["cumulative_installed_area_km2", "observed_active_area_km2", "pv_area_km2"])
    active = _coalesce_numeric(df, ["observed_active_area_km2", "pv_area_km2", "cumulative_installed_area_km2"])
    valid_area = _coalesce_numeric(df, ["valid_area_km2"], default=np.nan)
    valid_area = valid_area.where(valid_area > 0)
    valid_area = valid_area.fillna(df.groupby("region")["valid_area_km2"].transform("max") if "valid_area_km2" in df else np.nan)
    valid_area = valid_area.where(valid_area > 0)

    df["PV_Cover_Area_km2"] = cumulative.clip(lower=0.0)
    df["Observed_Active_PV_Area_km2"] = active.clip(lower=0.0)
    df["PV_Cover_Ratio"] = _safe_ratio(df["PV_Cover_Area_km2"], valid_area).clip(0.0, 1.0)
    df["Observed_Active_PV_Ratio"] = _safe_ratio(df["Observed_Active_PV_Area_km2"], valid_area).clip(0.0, 1.0)
    df["Valid_Area_km2"] = valid_area
    df["Annual_Installed_Gain_km2"] = _coalesce_numeric(df, ["installed_gain_km2", "gross_gain_km2"], default=0.0).clip(lower=0.0)
    df["Annual_Active_Gain_km2"] = _coalesce_numeric(df, ["gross_gain_km2"], default=0.0).clip(lower=0.0)
    df["Annual_Active_Loss_km2"] = _coalesce_numeric(df, ["gross_loss_km2"], default=0.0).clip(lower=0.0)
    df["Inactive_or_Retired_Area_km2"] = _coalesce_numeric(df, ["inactive_or_retired_area_km2"], default=0.0).clip(lower=0.0)

    low = _coalesce_numeric(df, ["cumulative_installed_area_ci_low_km2", "observed_active_area_ci_low_km2", "pv_area_ci_low_km2"], default=np.nan)
    high = _coalesce_numeric(df, ["cumulative_installed_area_ci_high_km2", "observed_active_area_ci_high_km2", "pv_area_ci_high_km2"], default=np.nan)
    half_width = ((high - low).abs() / 2.0).where(high.notna() & low.notna())
    df["Uncertainty"] = _safe_ratio(half_width, df["PV_Cover_Area_km2"].clip(lower=0.01)).clip(0.0, 1.0).fillna(0.0)

    env_path = output_root / "12_dynamic_ecology" / "environmental_covariates_imputed.csv"
    if not env_path.exists():
        env_path = output_root / "01_input_validation" / "environmental_covariates.csv"
    if env_path.exists():
        env = pd.read_csv(env_path)
        if "Id" in env.columns:
            env = env.rename(columns={"Id": "region"})
        if "Year" in env.columns and "year" not in env.columns:
            env = env.rename(columns={"Year": "year"})
        keep = [c for c in ENVIRONMENTAL_COVARIATES if c in env.columns]
        if {"region", "year"}.issubset(env.columns) and keep:
            env = env[["region", "year", *keep]].copy()
            env["region"] = pd.to_numeric(env["region"], errors="coerce")
            env["year"] = pd.to_numeric(env["year"], errors="coerce")
            env = env.dropna(subset=["region", "year"])
            env["region"] = env["region"].astype(int)
            env["year"] = env["year"].astype(int)
            env = env.groupby(["region", "year"], as_index=False)[keep].mean()
            env = env.rename(columns={c: f"ENV_{c}" for c in keep})
            df = df.merge(env, on=["region", "year"], how="left")

    pavg_path = output_root / "12_dynamic_ecology" / "pavg_annual_panel.csv"
    if pavg_path.exists():
        pavg = pd.read_csv(pavg_path)
        if "Id" in pavg.columns:
            pavg = pavg.rename(columns={"Id": "region"})
        pavg_keep = [
            "region", "year", "PAVG", "PAVG_climate_adjusted", "PAVG_fraction_raw",
            "PAVG_minus_baseline_VG", "vegetation_area_km2", "PAVG_quality",
            "LSMA_RMSE", "HA", "LA", "VG", "SH",
        ]
        pavg = pavg[[c for c in pavg_keep if c in pavg.columns]].copy()
        df = df.merge(pavg, on=["region", "year"], how="left")

    pavg_core = _coalesce_numeric(df, ["PAVG_climate_adjusted", "PAVG", "PAVG_fraction_raw"], default=np.nan)
    # PAVG is an LSMA fraction; retain a bounded target for recursive forecast stability.
    df["PVIntegrated_PAVG_Optimized"] = _bounded_fraction(pavg_core)
    raw_pavg = _coalesce_numeric(df, ["PAVG_fraction_raw", "PAVG", "PAVG_climate_adjusted"], default=np.nan)
    df["PAVG_Fraction_Raw"] = _bounded_fraction(raw_pavg)

    veg_area = _coalesce_numeric(df, ["vegetation_area_km2"], default=np.nan)
    veg_abundance = _safe_ratio(veg_area, df["PV_Cover_Area_km2"].clip(lower=0.0009))
    df["PVIntegrated_Vegetation_Abundance"] = veg_abundance.where(veg_abundance.notna(), df["PVIntegrated_PAVG_Optimized"]).clip(0.0, 1.0)

    anomaly_path = output_root / "12_dynamic_ecology" / "external_ecology_anomaly_panel.csv"
    if anomaly_path.exists():
        anomaly = pd.read_csv(anomaly_path)
        if "Id" in anomaly.columns:
            anomaly = anomaly.rename(columns={"Id": "region"})
        if {"region", "year", "metric", "external_anomaly"}.issubset(anomaly.columns):
            if "quality" in anomaly.columns:
                anomaly = anomaly.loc[anomaly["quality"].astype(str).eq("valid")].copy()
            wide = anomaly.pivot_table(index=["region", "year"], columns="metric", values="external_anomaly", aggfunc="mean").reset_index()
            wide.columns.name = None
            rename = {c: f"External_Anomaly_{c}" for c in wide.columns if c not in {"region", "year"}}
            wide = wide.rename(columns=rename)
            df = df.merge(wide, on=["region", "year"], how="left")

    green_cols = [c for c in ["External_Anomaly_NDVI", "External_Anomaly_EVI", "External_Anomaly_TC_Greenness"] if c in df.columns]
    moisture_cols = [c for c in ["External_Anomaly_NDMI", "External_Anomaly_TC_Wetness"] if c in df.columns]
    if green_cols:
        df["External_Greenness_Anomaly"] = df[green_cols].mean(axis=1)
    else:
        df["External_Greenness_Anomaly"] = np.nan
    if moisture_cols:
        df["External_Moisture_Anomaly"] = df[moisture_cols].mean(axis=1)
    else:
        df["External_Moisture_Anomaly"] = np.nan

    # Transparent bounded composite indices for forecasting and figure synthesis.
    pavg01 = df["PVIntegrated_PAVG_Optimized"]
    green01 = _robust01(df["External_Greenness_Anomaly"]).where(df["External_Greenness_Anomaly"].notna(), pavg01)
    moist01 = _robust01(df["External_Moisture_Anomaly"]).where(df["External_Moisture_Anomaly"].notna(), pavg01)
    gain01 = _robust01(df["Annual_Installed_Gain_km2"]).fillna(0.0)
    unc_penalty = 1.0 - df["Uncertainty"].fillna(0.0).clip(0.0, 1.0)

    df["PVIntegrated_EIPAVG"] = (0.50 * pavg01 + 0.28 * green01 + 0.12 * moist01 + 0.10 * unc_penalty).clip(0.0, 1.0)

    df = df.sort_values(["region", "year"]).reset_index(drop=True)
    pavg_lag = df.groupby("region")["PVIntegrated_PAVG_Optimized"].shift(1)
    pavg_diff = df["PVIntegrated_PAVG_Optimized"] - pavg_lag
    recovery01 = _robust01(pavg_diff).where(pavg_diff.notna(), 0.5)
    expansion_penalty = 1.0 - gain01
    df["PVIntegrated_ECRI"] = (0.45 * pavg01 + 0.25 * recovery01 + 0.20 * unc_penalty + 0.10 * expansion_penalty).clip(0.0, 1.0)

    # Keep naming compatible with the older v2 module while retaining v24 diagnostics.
    df["region"] = df["region"].astype(int)
    df["year"] = df["year"].astype(int)
    for col in CORE_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["region", "year"]).reset_index(drop=True)


def canonicalize_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Year" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"Year": "year"})
    if "Region" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"Region": "region"})
    if "Id" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"Id": "region"})

    if "year" not in df.columns:
        year_candidates = [c for c in df.columns if c.lower() == "year"]
        if year_candidates:
            df = df.rename(columns={year_candidates[0]: "year"})
        else:
            raise KeyError("No year column found.")

    if "region" not in df.columns:
        df["region"] = 0

    for canonical, aliases in TARGET_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = pd.to_numeric(df[alias], errors="coerce")
                break

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["region"] = pd.to_numeric(df["region"], errors="coerce")
    df = df.dropna(subset=["year", "region"]).copy()
    df["year"] = df["year"].astype(int)
    df["region"] = df["region"].astype(int)

    for c in CORE_METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["region", "year"]).reset_index(drop=True)
    return df


def read_annual_table(output_root: Path) -> pd.DataFrame:
    output_root = Path(output_root)
    global_csv = output_root / "global_summary" / "annual_all_regions.csv"
    if global_csv.exists():
        df = pd.read_csv(global_csv)
        return canonicalize_annual_table(df)

    # v24.9/v24.11 native output layout.
    if (output_root / "11_pv_dynamics" / "annual_pv_dynamics.csv").exists():
        return _build_from_v24_output(output_root)

    files = sorted(output_root.glob("region_*/**/annual_summary.csv"))
    if not files:
        raise FileNotFoundError(
            f"No global_summary/annual_all_regions.csv, v24 annual_pv_dynamics.csv, or region_*/annual_summary.csv found under {output_root}"
        )

    frames = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            rid = parse_region_id(f)
            if rid is not None and "region" not in tmp.columns and "Region" not in tmp.columns and "Id" not in tmp.columns:
                tmp["region"] = rid
            frames.append(tmp)
        except Exception as exc:
            print(f"[WARN] Failed to read {f}: {exc}")

    if not frames:
        raise RuntimeError("No readable annual summary files.")
    return canonicalize_annual_table(pd.concat(frames, ignore_index=True))


def available_targets(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    targets = requested or DEFAULT_TARGETS
    min_n = min(20, max(5, int(0.05 * len(df)))) if len(df) else 20
    return [t for t in targets if t in df.columns and df[t].notna().sum() >= min_n]


def ensure_output_dirs(base: Path) -> dict[str, Path]:
    base = Path(base)
    dirs = {
        "root": base,
        "tables": base / "tables",
        "validation": base / "validation",
        "forecast": base / "forecast",
        "figures": base / "figures",
        "audit": base / "audit",
        "report": base / "report",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_metric_label(x: str) -> str:
    return (
        str(x).strip().lower()
        .replace(" ", "_").replace("/", "_").replace("-", "_")
        .replace("__", "_")
    )
