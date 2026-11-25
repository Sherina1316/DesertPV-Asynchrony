from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from sklearn.linear_model import Ridge
from tqdm import tqdm

from .features import compute_features
from .spectral_mixture import load_endmember_matrix, pavg_fraction
from .routes import is_pixel_mode
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now

ZONE_NAMES = ["pv_core", "edge_0_250m", "near_250_500m", "local_500_1000m", "landscape_1000_3000m", "context_3000_5000m"]
LSMA_NAMES = ["LSMA_HA", "LSMA_LA", "LSMA_VG", "LSMA_SH", "LSMA_RMSE"]


@dataclass
class StreamingStats:
    reservoir_size: int
    rng: np.random.Generator
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    reservoir: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    priorities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        self.count += int(x.size)
        self.total += float(np.sum(x, dtype=np.float64))
        self.total_sq += float(np.sum(x.astype(np.float64) ** 2))
        keys = self.rng.random(x.size)
        values_all = x if self.reservoir.size == 0 else np.concatenate([self.reservoir, x])
        keys_all = keys if self.priorities.size == 0 else np.concatenate([self.priorities, keys])
        if len(values_all) > self.reservoir_size:
            keep = np.argpartition(keys_all, self.reservoir_size - 1)[: self.reservoir_size]
            values_all, keys_all = values_all[keep], keys_all[keep]
        self.reservoir = values_all.astype(np.float32, copy=False)
        self.priorities = keys_all.astype(np.float64, copy=False)

    def summarize(self, minimum_pixels: int) -> dict[str, Any]:
        if self.count < minimum_pixels or self.reservoir.size == 0:
            return {"n_pixels": self.count, "mean": np.nan, "median": np.nan, "std": np.nan, "q25": np.nan, "q75": np.nan, "quality": "insufficient"}
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean**2, 0.0)
        q25, median, q75 = np.quantile(self.reservoir, [0.25, 0.5, 0.75])
        return {"n_pixels": self.count, "mean": float(mean), "median": float(median), "std": float(math.sqrt(variance)), "q25": float(q25), "q75": float(q75), "quality": "valid"}


def _core_windows(width: int, height: int, tile_size: int) -> Iterator[Window]:
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            yield Window(col, row, min(tile_size, width - col), min(tile_size, height - row))


def _expanded_window(core: Window, width: int, height: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    col0 = max(0, int(core.col_off) - halo); row0 = max(0, int(core.row_off) - halo)
    col1 = min(width, int(core.col_off + core.width) + halo); row1 = min(height, int(core.row_off + core.height) + halo)
    expanded = Window(col0, row0, col1 - col0, row1 - row0)
    r0 = int(core.row_off) - row0; c0 = int(core.col_off) - col0
    return expanded, (slice(r0, r0 + int(core.height)), slice(c0, c0 + int(core.width)))


def _zones(mask: np.ndarray, valid: np.ndarray, resolution: float, distances: list[float]) -> np.ndarray:
    zones = np.full(mask.shape, -1, dtype=np.int8)
    footprint = mask & valid
    if not np.any(footprint):
        return zones
    distance = ndimage.distance_transform_edt(~footprint) * resolution
    zones[footprint] = 0
    for i in range(1, len(distances)):
        lo, hi = distances[i - 1], distances[i]
        zones[valid & ~footprint & (distance > lo) & (distance <= hi)] = i
    return zones


def _impute_environment(panel: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.copy().sort_values(["Id", "year"]); rows = []
    for covariate in covariates:
        source = data[covariate] if covariate in data.columns else pd.Series(np.nan, index=data.index)
        original = pd.to_numeric(source, errors="coerce")
        data[f"{covariate}_observed"] = original.notna().astype(np.uint8)
        data[covariate] = original
        interpolated = data.groupby("Id", group_keys=False)[covariate].apply(lambda x: pd.to_numeric(x, errors="coerce").interpolate(limit_direction="both"))
        year_median = data.assign(_value=interpolated).groupby("year")["_value"].transform("median")
        global_median = float(np.nanmedian(interpolated)) if np.isfinite(interpolated).any() else 0.0
        data[covariate] = interpolated.fillna(year_median).fillna(global_median).astype(float)
        rows.append({"covariate": covariate, "records": len(data), "observed_records": int(original.notna().sum()), "imputed_records": int(original.isna().sum()), "observed_fraction": float(original.notna().mean()), "global_median_fallback": global_median})
    return data, pd.DataFrame(rows)


def _climate_residualize(panel: pd.DataFrame, covariates: list[str], outcome: str, seed: int) -> tuple[pd.Series, dict[str, Any]]:
    valid = panel[outcome].notna()
    meta = {"requested": True, "implemented": False, "valid_records": int(valid.sum()), "covariates": covariates, "method": "ridge with unit and year fixed effects"}
    if valid.sum() < max(30, len(covariates) * 4) or panel.loc[valid, "Id"].nunique() < 3:
        meta["reason"] = "insufficient records"; return panel[outcome].copy(), meta
    design = panel.loc[valid, ["Id", "year", *covariates]].copy()
    for column in covariates:
        x = pd.to_numeric(design[column], errors="coerce")
        x = x.fillna(x.median() if x.notna().any() else 0.0)
        sd = float(x.std(ddof=0)); design[column] = (x - float(x.mean())) / (sd if sd > 1e-12 else 1)
    fixed = pd.get_dummies(design[["Id", "year"]].astype(str), drop_first=True, dtype=float)
    matrix = pd.concat([design[covariates].reset_index(drop=True), fixed.reset_index(drop=True)], axis=1).to_numpy(float)
    y = panel.loc[valid, outcome].to_numpy(float)
    model = Ridge(alpha=1e-6, fit_intercept=True, random_state=seed); model.fit(matrix, y)
    adjusted = pd.Series(np.nan, index=panel.index, dtype=float)
    adjusted.loc[valid] = y - model.predict(matrix) + float(np.mean(y))
    meta.update({"implemented": True, "r_squared": float(model.score(matrix, y)), "design_columns": int(matrix.shape[1]), "reason": None})
    return adjusted, meta


def _deployment_years(temporal: pd.DataFrame, reference_year: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for region_id, group in temporal.groupby("Id"):
        inferred = pd.to_numeric(group.get("inferred_deployment_year"), errors="coerce").dropna()
        if len(inferred):
            result[int(region_id)] = int(round(float(inferred.iloc[0]))); continue
        group = group.sort_values("year")
        area_column = "cumulative_installed_area_km2" if "cumulative_installed_area_km2" in group else "pv_area_km2"
        area = pd.to_numeric(group[area_column], errors="coerce").fillna(0).to_numpy(float)
        years = group["year"].to_numpy(int)
        reference_area = float(group.loc[group.year.eq(reference_year), area_column].iloc[0]) if group.year.eq(reference_year).any() else float(np.nanmax(area))
        threshold = max(0.01, 0.05 * reference_area)
        present = np.where(area >= threshold)[0]
        result[int(region_id)] = int(years[present[0]]) if present.size else reference_year
    return result


def extract_dynamic_ecology(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"]); out = ensure_dir(root / "12_dynamic_ecology")
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv")
    temporal = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv")
    reference_year = int(cfg["study"]["reference_year"])
    reference = temporal.loc[temporal.year.eq(reference_year), ["Id", "cumulative_mask_path", "estimation_mode", "estimation_route"]].copy()
    reference = reference.merge(supervision[["Id", "pv_fraction_path"]], on="Id", how="left").merge(quality[["Id", "endmember_path"]], on="Id", how="left")
    data = aligned.merge(reference, on="Id", how="inner").sort_values(["Id", "year"])
    metrics = list(cfg["ecology"]["metrics"])
    requested = list(dict.fromkeys(metrics + LSMA_NAMES))
    feature_indices = {name: i for i, name in enumerate(requested)}
    distances = list(map(float, cfg["ecology"]["distance_zones_m"]))
    minimum = int(cfg["ecology"]["minimum_pixels_per_zone"])
    tile_size = int(cfg["ecology"].get("zonal_tile_size", 384))
    reservoir = int(cfg["ecology"].get("quantile_reservoir_size", 20000))
    mask_nodata = int(cfg["temporal_decoding"].get("mask_nodata", 255))
    strong_fraction = float(cfg["supervision"]["strong_positive_fraction"])
    rows: list[dict[str, Any]] = []
    lsma_rows: list[dict[str, Any]] = []

    for row in tqdm(list(data.itertuples(index=False)), desc="Fixed-support LSMA/PAVG ecology", unit="image", disable=not cfg["processing"].get("show_progress", True)):
        seed = int(cfg["project"]["random_seed"]) + int(row.Id) * 100 + int(row.year)
        aggregators = {(z, m): StreamingStats(reservoir, np.random.default_rng(seed + z * 31 + i)) for z in range(len(ZONE_NAMES)) for i, m in enumerate(requested)}
        endmembers = load_endmember_matrix(row.endmember_path)
        footprint_mode = "pixel_mapped_reference_year_cumulative_installed_footprint" if is_pixel_mode(row.estimation_mode) and str(row.cumulative_mask_path) else "confirmed_shp_anchor_footprint"
        footprint_path = str(row.cumulative_mask_path) if footprint_mode.startswith("pixel_mapped") else str(row.pv_fraction_path)
        with rasterio.open(row.path) as src, rasterio.open(footprint_path) as footprint_src:
            resolution = abs(float(src.transform.a)); halo = int(math.ceil(max(distances) / resolution)) + 2
            for core in _core_windows(src.width, src.height, tile_size):
                expanded, crop = _expanded_window(core, src.width, src.height, halo)
                encoded = footprint_src.read(1, window=expanded)
                if footprint_mode.startswith("pixel_mapped"):
                    fixed_mask = encoded == 1; valid_support = encoded != mask_nodata
                else:
                    fixed_mask = encoded >= strong_fraction; valid_support = np.isfinite(encoded)
                zone_core = _zones(fixed_mask, valid_support, resolution, distances)[crop]
                reflectance = src.read([1, 2, 3, 4, 5, 6], window=core).astype(np.float32)
                reflectance[reflectance <= -9990] = np.nan
                features = compute_features(reflectance, requested, False, endmembers=endmembers)
                spectral_valid = np.isfinite(reflectance).all(axis=0)
                for z in range(len(ZONE_NAMES)):
                    selected = (zone_core == z) & spectral_valid
                    if not selected.any():
                        continue
                    for metric in requested:
                        aggregators[(z, metric)].update(features[feature_indices[metric]][selected])
        for z, name in enumerate(ZONE_NAMES):
            for metric in metrics:
                rows.append({"Id": int(row.Id), "year": int(row.year), "zone": name, "metric": metric, "spatial_support": footprint_mode, "estimation_route": str(row.estimation_route), **aggregators[(z, metric)].summarize(minimum)})
        core_lsma = {metric: aggregators[(0, metric)].summarize(minimum) for metric in LSMA_NAMES}
        lsma_rows.append({
            "Id": int(row.Id), "year": int(row.year), "spatial_support": footprint_mode,
            "estimation_route": str(row.estimation_route),
            "core_pixels": int(core_lsma["LSMA_VG"]["n_pixels"]),
            "HA": core_lsma["LSMA_HA"]["mean"], "LA": core_lsma["LSMA_LA"]["mean"],
            "VG": core_lsma["LSMA_VG"]["mean"], "SH": core_lsma["LSMA_SH"]["mean"],
            "LSMA_RMSE": core_lsma["LSMA_RMSE"]["mean"],
            "quality": "valid" if core_lsma["LSMA_VG"]["quality"] == "valid" else "insufficient",
        })

    result = pd.DataFrame(rows).sort_values(["Id", "year", "zone", "metric"])
    save_table(result, out / "dynamic_pv_ecology_panel.csv")
    wide = result.pivot_table(index=["Id", "year", "zone"], columns="metric", values="mean", dropna=False).reset_index(); wide.columns.name = None
    save_table(wide, out / "dynamic_pv_ecology_panel_wide.csv")
    lsma = pd.DataFrame(lsma_rows).sort_values(["Id", "year"])
    save_table(lsma, out / "lsma_core_fraction_panel.csv")

    primary = list(cfg["ecology"]["primary_effect_zones"]); control_zone = str(cfg["ecology"]["control_zone"])
    anomaly_rows: list[dict[str, Any]] = []
    for (region_id, year, metric), group in result.groupby(["Id", "year", "metric"]):
        treated = group.loc[group["zone"].isin(primary) & group["mean"].notna()]
        control = group.loc[group["zone"].eq(control_zone) & group["mean"].notna()]
        core = group.loc[group["zone"].eq("pv_core") & group["mean"].notna()]
        treated_pixels = int(treated["n_pixels"].sum()); control_pixels = int(control["n_pixels"].sum())
        treated_mean = float(np.average(treated["mean"], weights=treated["n_pixels"])) if treated_pixels else np.nan
        control_mean = float(np.average(control["mean"], weights=control["n_pixels"])) if control_pixels else np.nan
        core_mean = float(core.iloc[0]["mean"]) if len(core) else np.nan
        anomaly_rows.append({
            "Id": int(region_id), "year": int(year), "metric": metric,
            "treated_external_mean": treated_mean, "control_mean": control_mean,
            "external_anomaly": treated_mean - control_mean if np.isfinite(treated_mean) and np.isfinite(control_mean) else np.nan,
            "core_spectral_mean": core_mean, "treated_pixels": treated_pixels, "control_pixels": control_pixels,
            "quality": "valid" if treated_pixels >= minimum and control_pixels >= minimum else "insufficient",
        })
    anomaly = pd.DataFrame(anomaly_rows)
    anomaly.loc[anomaly["quality"] != "valid", "external_anomaly"] = np.nan
    save_table(anomaly, out / "external_ecology_anomaly_panel.csv")

    deployment = _deployment_years(temporal, reference_year)
    pavg_rows: list[dict[str, Any]] = []
    threshold = float(cfg.get("spectral_mixture", {}).get("vegetation_change_threshold", 0.15))
    for region_id, group in lsma.groupby("Id"):
        group = group.sort_values("year").copy()
        event_year = deployment[int(region_id)]
        valid_pre = group.loc[(group.year < event_year) & group.quality.eq("valid")]
        if len(valid_pre) < 2:
            valid_pre = group.loc[group.quality.eq("valid")].head(int(cfg.get("spectral_mixture", {}).get("baseline_year_count", 3)))
        baseline_la = float(valid_pre.LA.median()) if len(valid_pre) else np.nan
        baseline_sh = float(valid_pre.SH.median()) if len(valid_pre) else np.nan
        baseline_vg = float(valid_pre.VG.median()) if len(valid_pre) else np.nan
        for r in group.itertuples(index=False):
            if r.quality == "valid" and np.all(np.isfinite([r.VG, r.LA, r.SH, baseline_la, baseline_sh])):
                pavg = float(pavg_fraction(np.asarray(r.VG), np.asarray(r.LA), np.asarray(r.SH), baseline_la, baseline_sh))
                difference = pavg - baseline_vg if np.isfinite(baseline_vg) else np.nan
                category = "greening" if difference >= threshold else ("degradation" if difference <= -threshold else "non_change")
                vegetation_area = pavg * int(r.core_pixels) * 0.0009
            else:
                pavg = difference = vegetation_area = np.nan; category = "insufficient"
            pavg_rows.append({
                "Id": int(region_id), "year": int(r.year), "inferred_deployment_year": event_year,
                "spatial_support": r.spatial_support, "estimation_route": r.estimation_route,
                "core_pixels": int(r.core_pixels), "HA": r.HA, "LA": r.LA, "VG": r.VG, "SH": r.SH,
                "LSMA_RMSE": r.LSMA_RMSE, "baseline_LA": baseline_la, "baseline_SH": baseline_sh,
                "baseline_VG": baseline_vg, "PAVG_fraction_raw": pavg,
                "PAVG_minus_baseline_VG": difference, "vegetation_change_class": category,
                "vegetation_area_km2": vegetation_area,
                "PAVG_quality": "valid" if np.isfinite(pavg) else "insufficient",
            })
    pavg = pd.DataFrame(pavg_rows).sort_values(["Id", "year"])
    keys = aligned[["Id", "year"]].drop_duplicates().sort_values(["Id", "year"])
    env = pd.read_csv(root / "01_input_validation" / "environmental_covariates.csv")
    covariates = list(cfg["ecology"]["climate_covariates"])
    env_full = keys.merge(env, on=["Id", "year"], how="left")
    env_imputed, imputation = _impute_environment(env_full, covariates)
    save_table(env_imputed, out / "environmental_covariates_imputed.csv"); save_table(imputation, out / "environmental_imputation_report.csv")
    pavg = pavg.merge(env_imputed, on=["Id", "year"], how="left")
    adjusted, residualization = _climate_residualize(pavg, covariates, "PAVG_fraction_raw", int(cfg["project"]["random_seed"]))
    pavg["PAVG_climate_adjusted"] = adjusted
    pavg["PAVG"] = adjusted
    pavg["interpretation"] = "Xia-style LSMA PAVG fraction within a fixed confirmed footprint; external buffers are analysed separately. Landsat cannot resolve vegetation directly beneath individual panels."
    save_table(pavg, out / "pavg_annual_panel.csv")
    save_table(pavg[["Id", "year", "vegetation_area_km2", "PAVG_fraction_raw", "PAVG_climate_adjusted", "vegetation_change_class", "PAVG_quality"]], out / "vegetation_area_panel.csv")
    class_summary = pavg.groupby(["year", "vegetation_change_class"], as_index=False).agg(units=("Id", "nunique"), vegetation_area_km2=("vegetation_area_km2", "sum"))
    save_table(class_summary, out / "vegetation_change_class_summary.csv")

    validation = json.loads((root / "09_accuracy_validation" / "accuracy_validation_status.json").read_text(encoding="utf-8"))
    expected_units = expected_counts(cfg)[0]
    expected_years = int(cfg["study"]["end_year"]) - int(cfg["study"]["start_year"]) + 1
    expected_records = expected_units * expected_years * len(ZONE_NAMES) * len(metrics)
    coverage = len(result) == expected_records and len(pavg) == expected_units * expected_years
    valid_fraction = float(pavg["PAVG"].notna().mean())
    formal = bool(validation.get("segmentation_gate_pass", False)) and valid_fraction >= float(cfg["ecology"]["minimum_valid_pavg_fraction_for_formal_inference"])
    atomic_json({
        "status": "passed" if coverage else "failed", "created_utc": utc_now(),
        "records": len(result), "expected_records": expected_records, "units": int(result.Id.nunique()),
        "zones": ZONE_NAMES, "metrics": metrics, "fixed_spatial_support": True,
        "direct_support": "reference-year stable model footprint", "fallback_support": "confirmed SHP anchor only",
        "low_quality_units_retained_without_fabricated_pixel_history": True,
        "pavg_definition": "VG/(1-deltaSH-deltaLA)", "pavg_method_synced_to_xia_2022": True,
        "vegetation_change_threshold": threshold, "valid_pavg_records": int(pavg["PAVG"].notna().sum()),
        "valid_pavg_fraction": valid_fraction, "formal_ecological_inference_eligible": formal,
        "scientific_segmentation_gate_pass": bool(validation.get("segmentation_gate_pass", False)),
        "core_interpretation": "fractional surface composition within fixed footprint, not direct under-panel observation",
        "external_interpretation": "0-500 m external anomaly relative to fixed 3-5 km controls",
        "pavg_climate_residualization": residualization,
    }, out / "dynamic_ecology_status.json")
    if not coverage:
        raise RuntimeError("Fixed-support ecology output coverage incomplete")
    return result
