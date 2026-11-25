from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from .features import compute_features
from .spectral_mixture import load_endmember_matrix
from .utils import ensure_dir, save_table


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cumulative, values))


def _best_positive_break(years: np.ndarray, signal: np.ndarray, minimum_observations: int, min_seg: int) -> tuple[int | None, float]:
    """Detect the strongest persistent positive object-level level shift.

    The score combines robust positive level change with reduction in within-
    segment squared error. This avoids the legacy tie in which any early split
    whose post segment happened to contain a majority of high observations was
    assigned the same median difference as the true transition.
    """
    years = np.asarray(years, dtype=int)
    signal = np.asarray(signal, dtype=float)
    valid = np.isfinite(signal)
    if int(valid.sum()) < minimum_observations:
        return None, float("nan")
    overall = signal[valid]
    global_center = float(np.median(overall))
    total_sse = float(np.sum((overall - global_center) ** 2)) + 1e-8
    robust_scale = float(np.median(np.abs(overall - global_center)) * 1.4826 + 1e-4)
    best_year: int | None = None
    best_score = -np.inf
    for split in range(min_seg, len(years) - min_seg + 1):
        pre = signal[:split]
        post = signal[split:]
        pre = pre[np.isfinite(pre)]
        post = post[np.isfinite(post)]
        if len(pre) < min_seg or len(post) < min_seg:
            continue
        pre_level = float(np.median(pre))
        post_level = float(np.median(post))
        magnitude = post_level - pre_level
        if magnitude <= 0:
            continue
        segmented_sse = float(np.sum((pre - pre_level) ** 2) + np.sum((post - post_level) ** 2))
        reduction = max(total_sse - segmented_sse, 0.0) / total_sse
        standardized = magnitude / robust_scale
        # The reduction term determines the correct split; magnitude confirms
        # that the shift is PV-like (positive LA/SH evidence).
        score = reduction * max(standardized, 0.0)
        if score > best_score + 1e-12:
            best_score = score
            best_year = int(years[split])
    return best_year, float(best_score) if best_year is not None else float("nan")


def anchor_object_timeseries(cfg: dict[str, Any], region_id: int) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv")
    group = aligned.loc[aligned["Id"].eq(region_id)].sort_values("year")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv").set_index("Id")
    if region_id not in supervision.index or region_id not in quality.index:
        return pd.DataFrame()
    fraction_path = str(supervision.loc[region_id, "pv_fraction_path"])
    endmembers = load_endmember_matrix(quality.loc[region_id, "endmember_path"])
    rows: list[dict[str, Any]] = []
    with rasterio.open(fraction_path) as frac_src:
        for row in group.itertuples(index=False):
            values: list[np.ndarray] = []
            with rasterio.open(row.path) as src:
                for _, window in src.block_windows(1):
                    fraction = frac_src.read(1, window=window)
                    anchor = fraction >= float(cfg["supervision"].get("weak_positive_fraction", 0.2))
                    if not np.any(anchor):
                        continue
                    refl = src.read([1, 2, 3, 4, 5, 6], window=window).astype(np.float32)
                    refl[refl <= -9990] = np.nan
                    feature = compute_features(
                        refl,
                        ["LSMA_HA", "LSMA_LA", "LSMA_VG", "LSMA_SH", "LSMA_RMSE", "PV_Spectral_Evidence"],
                        False,
                        endmembers=endmembers,
                    )
                    selected = feature[:, anchor & np.all(np.isfinite(refl), axis=0)]
                    if selected.size:
                        values.append(selected)
            if values:
                stack = np.concatenate(values, axis=1)
                med = np.nanmedian(stack, axis=1)
                pv_signal = float(med[1] + med[3] - med[0] + 0.25 * med[5])
                rows.append({
                    "Id": region_id,
                    "year": int(row.year),
                    "HA": float(med[0]), "LA": float(med[1]), "VG": float(med[2]),
                    "SH": float(med[3]), "LSMA_RMSE": float(med[4]),
                    "PV_Spectral_Evidence": float(med[5]), "anchor_pv_signal": pv_signal,
                    "anchor_valid_pixels": int(stack.shape[1]),
                })
            else:
                rows.append({"Id": region_id, "year": int(row.year), "anchor_pv_signal": np.nan, "anchor_valid_pixels": 0})
    return pd.DataFrame(rows)


def _feature_table(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv")
    grids = pd.read_csv(root / "02_aligned_30m" / "grid_registry.csv")
    env = pd.read_csv(root / "01_input_validation" / "environmental_covariates.csv")
    env_cols = [c for c in cfg.get("ecology", {}).get("climate_covariates", []) if c in env.columns]
    env_summary = env.groupby("Id", as_index=False)[env_cols].median(numeric_only=True) if env_cols else env[["Id"]].drop_duplicates()
    table = quality.merge(grids[["Id", "left", "bottom", "right", "top", "width", "height"]], on="Id", how="left").merge(env_summary, on="Id", how="left")
    table["centroid_x"] = (table["left"] + table["right"]) / 2
    table["centroid_y"] = (table["bottom"] + table["top"]) / 2
    table["unit_area_km2"] = table["width"] * table["height"] * 0.0009
    table["log_anchor_area"] = np.log1p(table["confirmed_anchor_area_km2"].clip(lower=0))
    return table


def _analog_weights(target: pd.Series, donors: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    columns = ["centroid_x", "centroid_y", "log_anchor_area"] + [
        c for c in cfg.get("ecology", {}).get("climate_covariates", []) if c in donors.columns
    ]
    donor_values = donors[columns].astype(float).copy()
    target_values = target[columns].astype(float).copy()
    med = donor_values.median()
    scale = donor_values.std(ddof=0).replace(0, 1).fillna(1)
    donor_z = (donor_values.fillna(med) - med) / scale
    target_z = (target_values.fillna(med) - med) / scale
    # Geographic location is intentionally weighted more strongly.
    weights_by_column = np.ones(len(columns), dtype=float)
    for name in ["centroid_x", "centroid_y"]:
        if name in columns:
            weights_by_column[columns.index(name)] = float(cfg.get("fallback_area", {}).get("geographic_weight", 2.0))
    distance = np.sqrt(np.sum(((donor_z.to_numpy() - target_z.to_numpy()) * weights_by_column) ** 2, axis=1))
    k = min(int(cfg.get("fallback_area", {}).get("analog_count", 8)), len(donors))
    chosen = np.argsort(distance)[:k]
    result = pd.Series(0.0, index=donors.index)
    result.iloc[chosen] = 1.0 / np.maximum(distance[chosen], 0.15) ** 2
    result = result / max(float(result.sum()), 1e-12)
    return result


def infer_area_only_units(cfg: dict[str, Any], direct_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer annual PV area for low-quality units without fabricating pixel maps."""
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "08_temporal_pv_change" / "area_only_fallback")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv")[["Id", "confirmed_anchor_area_km2", "has_mapped_pv"]]
    features = _feature_table(cfg).merge(supervision, on=["Id", "confirmed_anchor_area_km2"], how="left")
    reference_year = int(cfg["study"]["reference_year"])
    complete_inventory = bool(cfg.get("supervision", {}).get("reference_year_complete_inventory_assumed", False))
    years = np.arange(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1)
    direct_ids = sorted(direct_registry.loc[direct_registry["estimation_mode"].eq("direct_pixel"), "Id"].unique())
    quality_fallback_ids = quality.loc[
        quality["estimation_route"].isin(["anchor_analog_area", "unanchored_analog_area"]),
        "Id",
    ].astype(int).tolist()
    canonical_registry_path = root / "07_probability_calibration" / "calibrated_inference_registry.csv"
    explicit_fallback_ids: list[int] = []
    if canonical_registry_path.exists():
        canonical_registry = pd.read_csv(canonical_registry_path)
        if {"Id", "inference_mode"}.issubset(canonical_registry.columns):
            explicit_fallback_ids = (
                pd.to_numeric(
                    canonical_registry.loc[
                        canonical_registry["inference_mode"].astype(str).eq("area_only_fallback"),
                        "Id",
                    ],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
    fallback_ids = sorted(set(quality_fallback_ids) | set(explicit_fallback_ids))
    direct_areas = direct_registry.loc[direct_registry["Id"].isin(direct_ids)].pivot(index="Id", columns="year", values="pv_area_km2").reindex(columns=years)
    donor_features = features.set_index("Id").loc[direct_ids].copy() if direct_ids else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    seed = int(cfg["project"]["random_seed"])
    bootstrap = int(cfg.get("fallback_area", {}).get("bootstrap_repetitions", 500))

    for region_id in fallback_ids:
        target = features.set_index("Id").loc[region_id]
        anchor_area = float(target["confirmed_anchor_area_km2"])
        anchor_ts = anchor_object_timeseries(cfg, region_id)
        if not anchor_ts.empty:
            deployment_year, break_score = _best_positive_break(
                anchor_ts["year"].to_numpy(int),
                anchor_ts["anchor_pv_signal"].to_numpy(float),
                int(cfg["ccdc"].get("minimum_observations", 6)),
                int(cfg["ccdc"].get("break_minimum_segment_years", 3)),
            )
            save_table(anchor_ts, out / f"unit_{region_id:03d}_anchor_fraction_timeseries.csv")
        else:
            deployment_year, break_score = None, float("nan")

        if donor_features.empty or direct_areas.empty:
            point_2024 = anchor_area
            curve = np.zeros(len(years), dtype=float)
            curve[years >= (deployment_year or reference_year)] = point_2024
            lower = curve.copy(); upper = curve.copy()
            analog_ids: list[int] = []
            capture_ratio = 1.0
            method = "anchor_only_lower_bound"
        else:
            weights = _analog_weights(target, donor_features, cfg)
            active = weights[weights > 0]
            analog_ids = [int(x) for x in active.index.tolist()]
            donor = direct_areas.loc[analog_ids].copy()
            donor_2024 = donor[reference_year].replace(0, np.nan)
            donor_anchor = features.set_index("Id").loc[analog_ids, "confirmed_anchor_area_km2"].astype(float)
            ratios = (donor_anchor / donor_2024).clip(lower=float(cfg.get("fallback_area", {}).get("minimum_anchor_capture_ratio", 0.05)), upper=1.0)
            w = active.reindex(analog_ids).to_numpy(float)
            capture_ratio = _weighted_quantile(ratios.to_numpy(float), w, 0.5)
            if not np.isfinite(capture_ratio):
                capture_ratio = 1.0
            if complete_inventory:
                # The corrected 2024 census fixes the reference-year area exactly.
                # Analog donors may shape the historical/future trajectory but may
                # not inflate, reduce or supplement the 2024 endpoint.
                point_2024 = anchor_area
                method = "authoritative_2024_area_geographic_analog" if anchor_area > 0 else "authoritative_zero_2024_area"
            elif anchor_area > 0:
                point_2024 = anchor_area / max(capture_ratio, 0.05)
                method = "anchor_geographic_analog"
            else:
                density = donor_2024.to_numpy(float) / donor_features.loc[analog_ids, "unit_area_km2"].to_numpy(float)
                point_2024 = max(0.0, _weighted_quantile(density, w, 0.5) * float(target["unit_area_km2"]))
                method = "unanchored_geographic_analog"
            normalized = donor.div(donor_2024, axis=0).replace([np.inf, -np.inf], np.nan)
            curve = np.asarray([
                point_2024 * _weighted_quantile(normalized[year].to_numpy(float), w, 0.5)
                for year in years
            ], dtype=float)
            ref_idx = int(np.where(years == reference_year)[0][0])
            curve[ref_idx] = point_2024
            curve = np.maximum.accumulate(np.nan_to_num(curve, nan=0.0))
            if deployment_year is not None:
                curve[years < deployment_year] = 0.0
            rng = np.random.default_rng(seed + region_id)
            simulations = np.empty((bootstrap, len(years)), dtype=float)
            probabilities = w / max(w.sum(), 1e-12)
            for b in range(bootstrap):
                sample = rng.choice(len(analog_ids), size=max(3, len(analog_ids)), replace=True, p=probabilities)
                sampled_ids = [analog_ids[i] for i in sample]
                sample_donor = direct_areas.loc[sampled_ids]
                sample_ref = sample_donor[reference_year].replace(0, np.nan)
                sample_anchor = features.set_index("Id").loc[sampled_ids, "confirmed_anchor_area_km2"].astype(float)
                sample_capture = np.nanmedian((sample_anchor / sample_ref).clip(0.05, 1.0))
                sample_point = anchor_area / max(sample_capture, 0.05) if anchor_area > 0 else point_2024
                sample_norm = sample_donor.div(sample_ref, axis=0)
                sim = np.nanmedian(sample_norm.to_numpy(float), axis=0) * sample_point
                sim[ref_idx] = sample_point
                sim = np.maximum.accumulate(np.nan_to_num(sim, nan=0.0))
                if deployment_year is not None:
                    sim[years < deployment_year] = 0.0
                simulations[b] = sim
            lower = np.nanquantile(simulations, 0.025, axis=0)
            upper = np.nanquantile(simulations, 0.975, axis=0)

        if complete_inventory and anchor_area <= 0:
            # With no 2024 PV and no defensible pixel history, an area-only route
            # cannot identify a historical retired site. Retain zero rather than
            # inventing a trajectory from analog regions.
            curve[:] = 0.0
            lower[:] = 0.0
            upper[:] = 0.0
            method = "authoritative_zero_2024_no_fabricated_history"

        maximum_fraction = float(cfg.get("fallback_area", {}).get("maximum_unit_pv_fraction", 0.35))
        cap = max(float(target["unit_area_km2"]) * maximum_fraction, anchor_area if complete_inventory else 0.0)
        curve = np.clip(curve, 0, cap); lower = np.clip(lower, 0, cap); upper = np.clip(upper, 0, cap)
        if complete_inventory:
            ref_idx = int(np.where(years == reference_year)[0][0])
            curve[ref_idx] = anchor_area
            lower[ref_idx] = anchor_area
            upper[ref_idx] = anchor_area
        elif anchor_area > 0:
            ref_idx = int(np.where(years == reference_year)[0][0])
            curve[ref_idx] = max(curve[ref_idx], anchor_area)
            lower[ref_idx] = max(lower[ref_idx], anchor_area)
            upper[ref_idx] = max(upper[ref_idx], curve[ref_idx])
        explicit_stage7_fallback = region_id in set(explicit_fallback_ids)
        effective_route = (
            "explicit_stage7_area_only"
            if explicit_stage7_fallback
            else str(target["estimation_route"])
        )
        for i, year in enumerate(years):
            point = float(curve[i])
            rows.append({
                "Id": region_id,
                "year": int(year),
                "split_role": str(target["split_role"]),
                "posterior_path": "",
                "mask_path": "",
                "active_mask_path": "",
                "cumulative_mask_path": "",
                "model_only_cumulative_mask_path": "",
                "pv_pixels": np.nan,
                "active_pv_pixels": np.nan,
                "cumulative_installed_pixels": np.nan,
                "model_only_cumulative_installed_pixels": np.nan,
                "valid_pixels": np.nan,
                "pv_area_km2": point,
                "observed_active_area_km2": point,
                "cumulative_installed_area_km2": point,
                "model_only_cumulative_installed_area_km2": point,
                "anchor_constrained_added_area_km2": 0.0,
                "anchor_constrained_removed_area_km2": 0.0,
                "pv_area_latent_persistent_km2": point,
                "inactive_or_retired_area_km2": 0.0,
                "observation_coverage_fraction": np.nan,
                "pv_area_ci_low_km2": float(lower[i]),
                "pv_area_ci_high_km2": float(upper[i]),
                "change_year_path": "",
                "model_only_change_year_path": "",
                "retirement_year_path": "",
                "transition_score_path": "",
                "confuser_penalty_path": "",
                "temporal_candidate_path": "",
                "morphology_removed_path": "",
                "valid_year_count_path": "",
                "temporal_exclusion_code_path": "",
                "anchor_backcast_path": "",
                "anchor_forced_reference_path": "",
                "anchor_backcast_object_fill_path": "",
                "anchor_reference_path": "",
                "historical_growth_front_rescue_path": "",
                "terminal_2025_expansion_source_path": "",
                "calibrated_threshold": np.nan,
                "posterior_threshold": np.nan,
                "minimum_observation_probability": np.nan,
                "effective_birth_probability": np.nan,
                "estimation_mode": "area_only_fallback",
                "estimation_route": effective_route,
                "confirmed_anchor_area_km2": anchor_area,
                "manual_reference_area_km2": anchor_area,
                "refined_reference_area_km2": anchor_area,
                "inferred_deployment_year": deployment_year,
                "deployment_break_score": break_score,
                "fallback_method": method,
                "explicit_stage7_area_only": explicit_stage7_fallback,
                "anchor_constraint_applied": False,
                "annual_probability_bias": 0.0,
                "area_state_definition": "area_only_cumulative_estimate",
                "temporal_decoder": "area_only_fallback_v24_22",
                "pixel_map_available": False,
            })
        audits.append({
            "Id": region_id, "estimation_route": effective_route, "fallback_method": method,
            "explicit_stage7_area_only": explicit_stage7_fallback,
            "confirmed_anchor_area_km2": anchor_area, "estimated_total_area_reference_year_km2": float(curve[years == reference_year][0]),
            "predicted_anchor_capture_ratio": capture_ratio, "inferred_deployment_year": deployment_year,
            "deployment_break_score": break_score, "analog_unit_ids": ";".join(map(str, analog_ids)),
            "pixel_map_available": False, "scientific_interpretation": "unit-level area inference only; no fabricated pixel footprint",
        })
    frame = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    save_table(frame, out / "area_only_annual_registry.csv")
    save_table(audit, out / "area_only_inference_audit.csv")
    return frame, audit
