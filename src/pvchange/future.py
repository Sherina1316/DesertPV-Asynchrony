from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.stats import theilslopes
from tqdm import tqdm

from .features import compute_features
from .routes import is_pixel_mode
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _windows(width: int, height: int, size: int) -> Iterator[Window]:
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def _expanded(core: Window, width: int, height: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    c0 = max(0, int(core.col_off) - halo); r0 = max(0, int(core.row_off) - halo)
    c1 = min(width, int(core.col_off + core.width) + halo); r1 = min(height, int(core.row_off + core.height) + halo)
    win = Window(c0, r0, c1 - c0, r1 - r0)
    return win, (slice(int(core.row_off) - r0, int(core.row_off) - r0 + int(core.height)), slice(int(core.col_off) - c0, int(core.col_off) - c0 + int(core.width)))


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _clean(mask: np.ndarray, minimum: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel()); keep = sizes >= minimum; keep[0] = False
    return keep[labels]


def _sample(values: np.ndarray, rng: np.random.Generator, n: int = 2500) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1); x = x[np.isfinite(x)]
    if len(x) > n:
        x = rng.choice(x, n, replace=False)
    return x


def _constraint_paths(cfg: dict[str, Any], region_id: int) -> dict[str, Path]:
    root = Path(cfg["paths"]["future_constraint_root"])
    return {name: root / pattern.format(Id=region_id) for name, pattern in cfg["future_2050"]["constraint_rasters"].items()}


def _quantile_pair(values: list[np.ndarray]) -> tuple[float, float]:
    valid = [x for x in values if len(x)]
    if not valid:
        return 0.0, 1.0
    x = np.concatenate(valid)
    lo, hi = np.quantile(x, [0.02, 0.98])
    return float(lo), float(hi if hi > lo else lo + 1e-6)


def _forecast_candidates(years: np.ndarray, area: np.ndarray, target_year: int) -> dict[str, float]:
    valid = np.isfinite(area)
    years = years[valid].astype(float); area = np.maximum(area[valid].astype(float), 0.0)
    if len(years) < 3:
        base = float(area[-1]) if len(area) else 0.0
        return {"recent_linear": base, "theil_sen": base, "log_linear": base, "damped_gain": base}
    horizon = max(float(target_year - years[-1]), 0.0)
    gains = np.diff(np.maximum.accumulate(area))
    recent_gain = float(np.nanmedian(gains[-min(5, len(gains)):])) if len(gains) else 0.0
    recent_linear = area[-1] + max(recent_gain, 0.0) * horizon
    slope = float(theilslopes(area, years).slope) if len(years) >= 3 else recent_gain
    theil = area[-1] + max(slope, 0.0) * horizon
    positive = area > 0
    if positive.sum() >= 3:
        coef = np.polyfit(years[positive], np.log1p(area[positive]), 1)
        log_linear = float(np.expm1(np.polyval(coef, target_year)))
    else:
        log_linear = recent_linear
    damping_years = 12.0
    damped = area[-1] + max(recent_gain, 0.0) * damping_years * (1.0 - np.exp(-horizon / damping_years))
    return {"recent_linear": max(0.0, recent_linear), "theil_sen": max(0.0, theil), "log_linear": max(0.0, log_linear), "damped_gain": max(0.0, damped)}


def _select_forecast_model(years: np.ndarray, area: np.ndarray, holdout: int = 3) -> tuple[str, dict[str, float]]:
    if len(years) <= holdout + 2:
        predictions = _forecast_candidates(years, area, int(years[-1]))
        return "damped_gain", {name: np.nan for name in predictions}
    train_years = years[:-holdout]; train_area = area[:-holdout]
    errors: dict[str, list[float]] = {}
    for target_year, observed in zip(years[-holdout:], area[-holdout:]):
        candidates = _forecast_candidates(train_years, train_area, int(target_year))
        for name, prediction in candidates.items():
            errors.setdefault(name, []).append(abs(float(prediction) - float(observed)))
        train_years = np.append(train_years, target_year); train_area = np.append(train_area, observed)
    mae = {name: float(np.mean(values)) for name, values in errors.items()}
    return min(mae, key=mae.get), mae


def _dynamic_forecast(group: pd.DataFrame, cfg: dict[str, Any], unit_area_km2: float, environmental_modifier: float = 1.0) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    group = group.sort_values("year")
    years = group["year"].to_numpy(int)
    area_column = "cumulative_installed_area_km2" if "cumulative_installed_area_km2" in group else "pv_area_km2"
    area = pd.to_numeric(group[area_column], errors="coerce").fillna(0).to_numpy(float)
    target_year = int(cfg["study"]["future_year"])
    model, backtest_mae = _select_forecast_model(years, area, int(cfg["future_2050"].get("backtest_holdout_years", 3)))
    candidates = _forecast_candidates(years, area, target_year)
    base = float(candidates[model])
    multipliers = cfg["future_2050"].get("scenario_growth_multipliers", {"conservative": 0.7, "transition": 1.0, "accelerated": 1.3})
    current = float(area[-1])
    cap = float(unit_area_km2) * float(cfg["future_2050"].get("maximum_total_pv_fraction_per_unit", 0.35))
    forecasts: dict[str, dict[str, float]] = {}
    for scenario in cfg["future_2050"]["scenarios"]:
        multiplier = float(multipliers.get(scenario, 1.0))
        target = current + max(base - current, 0.0) * multiplier * float(environmental_modifier)
        target = float(np.clip(target, current, cap))
        forecasts[scenario] = {"target_area_km2": target, "additional_area_km2": max(target - current, 0.0)}
    diagnostics = {"selected_model": model, "model_predictions_2050": candidates, "backtest_mae": backtest_mae, "current_area_km2": current, "unit_area_cap_km2": cap, "environmental_modifier": float(environmental_modifier)}
    return forecasts, diagnostics


def _spectral_components(reflectance: np.ndarray, current: np.ndarray, resolution: float, constraints: dict[str, np.ndarray], quantiles: dict[str, tuple[float, float]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    requested = ["NDVI", "BSI", "NDMI", "MNDWI", "TC_Brightness"]
    f = compute_features(reflectance, requested, False)
    idx = {name: i for i, name in enumerate(requested)}
    valid = np.all(np.isfinite(reflectance), axis=0)
    distance = ndimage.distance_transform_edt(~current) * resolution if current.any() else np.full(current.shape, 10000.0, dtype=np.float32)
    pressure = np.exp(-distance / 5000.0).astype(np.float32)
    spectral = (
        0.24 * (1 - _normalize(f[idx["NDVI"]], *quantiles["NDVI"]))
        + 0.20 * _normalize(f[idx["BSI"]], *quantiles["BSI"])
        + 0.15 * (1 - _normalize(f[idx["NDMI"]], *quantiles["NDMI"]))
        + 0.16 * (1 - _normalize(f[idx["MNDWI"]], *quantiles["MNDWI"]))
        + 0.10 * _normalize(f[idx["TC_Brightness"]], *quantiles["TC_Brightness"])
        + 0.15 * pressure
    ).astype(np.float32)
    result = {"spectral": spectral}; hard_exclusion = np.zeros(current.shape, dtype=bool)
    for name, array in constraints.items():
        arr = np.asarray(array, dtype=np.float32); valid &= np.isfinite(arr) & (arr > -9990)
        if name == "protected_area":
            hard_exclusion |= arr > 0; result[name] = np.where(arr > 0, 0.0, 1.0).astype(np.float32)
        elif name == "solar_irradiance":
            result[name] = _normalize(arr, *quantiles[name])
        elif name in {"slope", "water_stress", "grid_distance", "road_distance"}:
            result[name] = 1 - _normalize(arr, *quantiles[name])
    valid &= ~current
    return result, valid & ~hard_exclusion


def _weighted_score(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    available = {name: float(weights[name]) for name in components if name in weights}
    total = sum(available.values())
    score = np.zeros_like(next(iter(components.values())), dtype=np.float32)
    for name, weight in available.items():
        score += (weight / max(total, 1e-12)) * components[name]
    return score


def generate_future_envelopes(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"]); out = ensure_dir(root / "16_future_2050"); mask_root = ensure_dir(out / "candidate_masks_30m")
    dynamics = pd.read_csv(root / "11_pv_dynamics" / "annual_pv_dynamics.csv")
    grids = pd.read_csv(root / "02_aligned_30m" / "grid_registry.csv")
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv")
    reference_year = int(cfg["study"]["reference_year"])
    aligned = aligned.loc[aligned.year.eq(reference_year), ["Id", "path"]]
    temporal = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv")
    reference = temporal.loc[temporal.year.eq(reference_year), ["Id", "cumulative_mask_path", "estimation_mode", "estimation_route"]]
    units = grids[["Id", "width", "height"]].merge(aligned, on="Id", how="left").merge(reference, on="Id", how="left")
    units["unit_area_km2"] = units.width * units.height * 0.0009
    validation = json.loads((root / "09_accuracy_validation" / "accuracy_validation_status.json").read_text(encoding="utf-8"))
    formal_detection = bool(validation.get("segmentation_gate_pass", False))
    env_path = root / "12_dynamic_ecology" / "environmental_covariates_imputed.csv"
    environmental_modifier_by_unit: dict[int, float] = {}
    if env_path.exists():
        env = pd.read_csv(env_path)
        covariates = ["precipitation", "temperature", "potential_evapotranspiration", "soil_moisture", "radiation", "wind_speed"]
        recent = env.loc[env.year >= max(int(cfg["study"]["reference_year"]) - 5, int(cfg["study"]["start_year"])), ["Id", *covariates]].groupby("Id", as_index=False).mean()
        z = recent.copy()
        for column in covariates:
            values = pd.to_numeric(z[column], errors="coerce")
            sd = float(values.std(ddof=0))
            z[column] = (values - float(values.mean())) / (sd if np.isfinite(sd) and sd > 1e-12 else 1.0)
        aridity = z["potential_evapotranspiration"] - z["precipitation"] - z["soil_moisture"]
        thermal_penalty = np.abs(z["temperature"])
        feasibility = 0.45 * z["radiation"] + 0.15 * z["wind_speed"] - 0.25 * aridity - 0.15 * thermal_penalty
        modifier = np.clip(np.exp(0.12 * feasibility), 0.80, 1.20)
        environmental_modifier_by_unit = dict(zip(recent["Id"].astype(int), modifier.astype(float)))
    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
    base_weights = dict(cfg["future_2050"]["constraint_weights"])
    rows: list[dict[str, Any]] = []; mc_rows: list[dict[str, Any]] = []; coverage_rows: list[dict[str, Any]] = []; forecast_rows: list[dict[str, Any]] = []
    mask_nodata = int(cfg["temporal_decoding"].get("mask_nodata", 255))

    for unit in tqdm(list(units.itertuples(index=False)), desc="Change-modelled 2050 area and spatial allocation", unit="region", disable=not cfg["processing"].get("show_progress", True)):
        group = dynamics.loc[dynamics.Id.eq(int(unit.Id))].sort_values("year")
        environmental_modifier = float(environmental_modifier_by_unit.get(int(unit.Id), 1.0))
        forecasts, diagnostic = _dynamic_forecast(group, cfg, float(unit.unit_area_km2), environmental_modifier)
        for model_name, value in diagnostic["model_predictions_2050"].items():
            forecast_rows.append({"Id": int(unit.Id), "model": model_name, "prediction_2050_km2": value, "selected": model_name == diagnostic["selected_model"], "backtest_mae_km2": diagnostic["backtest_mae"].get(model_name, np.nan)})
        direct = is_pixel_mode(unit.estimation_mode) and str(unit.cumulative_mask_path) and Path(str(unit.cumulative_mask_path)).exists()
        constraint_sources: dict[str, rasterio.DatasetReader] = {}
        available_constraints: list[str] = []
        if not direct:
            for scenario, forecast in forecasts.items():
                rows.append({
                    "Id": int(unit.Id), "scenario": scenario, "estimation_mode": "area_only_fallback",
                    "selected_change_model": diagnostic["selected_model"], "current_area_km2": diagnostic["current_area_km2"],
                    "target_area_2050_km2": forecast["target_area_km2"], "additional_area_2050_km2": forecast["additional_area_km2"],
                    "candidate_pixels": np.nan, "candidate_area_km2": forecast["additional_area_km2"],
                    "median_suitability": np.nan, "effective_threshold": np.nan, "candidate_path": "",
                    "spatial_map_available": False, "formal_result": False, "available_constraints": "",
                    "interpretation": "area-only change forecast; no fabricated 2050 footprint map",
                })
                series_for_residual = group["cumulative_installed_area_km2"] if "cumulative_installed_area_km2" in group else group["pv_area_km2"]
                residual_scale = max(float(np.nanmedian(np.abs(np.diff(series_for_residual)))), 0.05 * max(forecast["additional_area_km2"], 0.01))
                simulations = np.clip(rng.normal(forecast["target_area_km2"], residual_scale * math.sqrt(max(int(cfg["study"]["future_year"]) - reference_year, 1)), int(cfg["future_2050"]["monte_carlo_repetitions"])), diagnostic["current_area_km2"], diagnostic["unit_area_cap_km2"])
                mc_rows.append({"Id": int(unit.Id), "scenario": scenario, "target_area_median_km2": float(np.median(simulations)), "target_area_ci_low_km2": float(np.quantile(simulations, 0.025)), "target_area_ci_high_km2": float(np.quantile(simulations, 0.975)), "candidate_area_median_km2": float(np.median(simulations - diagnostic["current_area_km2"])), "candidate_area_ci_low_km2": float(np.quantile(simulations - diagnostic["current_area_km2"], 0.025)), "candidate_area_ci_high_km2": float(np.quantile(simulations - diagnostic["current_area_km2"], 0.975)), "monte_carlo_method": "change-model residual propagation"})
            coverage_rows.append({"Id": int(unit.Id), "estimation_mode": "area_only_fallback", "available_constraint_count": 0, "available_constraints": "", "formal_multisource_complete": False, "spatial_map_available": False})
            continue

        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open(unit.path)); mask_src = stack.enter_context(rasterio.open(unit.cumulative_mask_path))
            paths = _constraint_paths(cfg, int(unit.Id))
            for name, path in paths.items():
                if path.exists():
                    candidate = stack.enter_context(rasterio.open(path))
                    if candidate.shape == src.shape and candidate.crs == src.crs and candidate.transform.almost_equals(src.transform):
                        constraint_sources[name] = candidate; available_constraints.append(name)
            # Quantiles from bounded samples.
            samples: dict[str, list[np.ndarray]] = {name: [] for name in ["NDVI", "BSI", "NDMI", "MNDWI", "TC_Brightness", *available_constraints]}
            for _, window in list(src.block_windows(1))[: min(250, sum(1 for _ in src.block_windows(1)))]:
                refl = src.read([1, 2, 3, 4, 5, 6], window=window).astype(np.float32); refl[refl <= -9990] = np.nan
                f = compute_features(refl, ["NDVI", "BSI", "NDMI", "MNDWI", "TC_Brightness"], False)
                for i, name in enumerate(["NDVI", "BSI", "NDMI", "MNDWI", "TC_Brightness"]):
                    samples[name].append(_sample(f[i], rng, 1000))
                for name, source in constraint_sources.items():
                    samples[name].append(_sample(source.read(1, window=window), rng, 1000))
            quantiles = {name: _quantile_pair(values) for name, values in samples.items()}
            resolution = abs(float(src.transform.a)); halo = int(math.ceil(5000 / resolution)) + 4
            score_samples: list[np.ndarray] = []; eligible_pixels = 0
            for core in _windows(src.width, src.height, 384):
                expanded, crop = _expanded(core, src.width, src.height, halo)
                refl = src.read([1, 2, 3, 4, 5, 6], window=expanded).astype(np.float32); refl[refl <= -9990] = np.nan
                encoded = mask_src.read(1, window=expanded); current = encoded == 1
                constraints = {name: source.read(1, window=expanded).astype(np.float32) for name, source in constraint_sources.items()}
                comp, eligible = _spectral_components(refl, current, resolution, constraints, quantiles)
                score = _weighted_score(comp, base_weights)[crop]; eligible = eligible[crop]
                eligible_pixels += int(eligible.sum()); score_samples.append(_sample(score[eligible], rng, 1500))
            sampled = np.concatenate([x for x in score_samples if len(x)]) if any(len(x) for x in score_samples) else np.array([], dtype=np.float32)
            profile = src.profile.copy(); profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            thresholds: dict[str, float] = {}
            for scenario, forecast in forecasts.items():
                target_pixels = int(round(forecast["additional_area_km2"] / 0.0009))
                target_fraction = min(max(target_pixels / max(eligible_pixels, 1), 0.0), 1.0)
                thresholds[scenario] = float(np.quantile(sampled, max(0.0, 1 - target_fraction))) if len(sampled) and target_pixels > 0 else 1.1
            destinations: dict[str, tuple[Path, rasterio.DatasetWriter]] = {}
            counts = {scenario: 0 for scenario in thresholds}; score_sums = {scenario: 0.0 for scenario in thresholds}
            try:
                for scenario in thresholds:
                    target = mask_root / f"unit_{int(unit.Id):03d}_{scenario}_candidate_envelope_30m.tif"
                    destinations[scenario] = (target, rasterio.open(target, "w", **profile))
                for core in _windows(src.width, src.height, 384):
                    expanded, crop = _expanded(core, src.width, src.height, halo)
                    refl = src.read([1, 2, 3, 4, 5, 6], window=expanded).astype(np.float32); refl[refl <= -9990] = np.nan
                    encoded = mask_src.read(1, window=expanded); current = encoded == 1
                    constraints = {name: source.read(1, window=expanded).astype(np.float32) for name, source in constraint_sources.items()}
                    comp, eligible = _spectral_components(refl, current, resolution, constraints, quantiles); score = _weighted_score(comp, base_weights)
                    for scenario, threshold in thresholds.items():
                        candidate = _clean((score >= threshold) & eligible, int(cfg["future_2050"]["minimum_component_pixels"]))
                        core_candidate = candidate[crop]; core_eligible = eligible[crop]
                        encoded_out = np.full(core_candidate.shape, mask_nodata, dtype=np.uint8); encoded_out[core_eligible] = core_candidate[core_eligible].astype(np.uint8)
                        destinations[scenario][1].write(encoded_out, 1, window=core)
                        counts[scenario] += int(core_candidate.sum()); score_sums[scenario] += float(np.sum(score[crop][core_candidate]))
            finally:
                for _, dst in destinations.values():
                    dst.close()
            formal_constraints = set(cfg["future_2050"]["constraint_rasters"]).issubset(constraint_sources)
            for scenario, forecast in forecasts.items():
                count = counts[scenario]
                rows.append({
                    "Id": int(unit.Id), "scenario": scenario, "estimation_mode": str(unit.estimation_mode),
                    "selected_change_model": diagnostic["selected_model"], "current_area_km2": diagnostic["current_area_km2"],
                    "target_area_2050_km2": forecast["target_area_km2"], "additional_area_2050_km2": forecast["additional_area_km2"],
                    "candidate_pixels": count, "candidate_area_km2": count * 0.0009,
                    "allocation_target_error_km2": count * 0.0009 - forecast["additional_area_km2"],
                    "median_suitability": score_sums[scenario] / count if count else np.nan,
                    "effective_threshold": thresholds[scenario], "candidate_path": str(destinations[scenario][0]),
                    "spatial_map_available": True,
                    "formal_result": bool(str(unit.estimation_mode) == "direct_pixel" and formal_detection and (formal_constraints if cfg["future_2050"]["formal_multisource_required"] else True)),
                    "available_constraints": ";".join(sorted(available_constraints)),
                    "environmental_modifier": float(diagnostic.get("environmental_modifier", 1.0)),
                    "interpretation": "change-modelled target area adjusted by all six unit-level environmental covariates and allocated to constrained high-suitability pixels; reference footprint is an initial state, not the forecast",
                })
                repetitions = int(cfg["future_2050"]["monte_carlo_repetitions"])
                residual_scale = max(float(diagnostic["backtest_mae"].get(diagnostic["selected_model"], 0.0) or 0.0), 0.05 * max(forecast["additional_area_km2"], 0.01))
                simulations = np.clip(rng.normal(forecast["target_area_km2"], residual_scale, repetitions), diagnostic["current_area_km2"], diagnostic["unit_area_cap_km2"])
                mc_rows.append({"Id": int(unit.Id), "scenario": scenario, "target_area_median_km2": float(np.median(simulations)), "target_area_ci_low_km2": float(np.quantile(simulations, 0.025)), "target_area_ci_high_km2": float(np.quantile(simulations, 0.975)), "candidate_area_median_km2": float(np.median(simulations - diagnostic["current_area_km2"])), "candidate_area_ci_low_km2": float(np.quantile(simulations - diagnostic["current_area_km2"], 0.025)), "candidate_area_ci_high_km2": float(np.quantile(simulations - diagnostic["current_area_km2"], 0.975)), "monte_carlo_method": "backtest residual+scenario growth propagation"})
            coverage_rows.append({"Id": int(unit.Id), "estimation_mode": str(unit.estimation_mode), "available_constraint_count": len(available_constraints), "available_constraints": ";".join(sorted(available_constraints)), "formal_multisource_complete": formal_constraints, "spatial_map_available": True})

    summary = pd.DataFrame(rows).sort_values(["scenario", "Id"])
    monte = pd.DataFrame(mc_rows).sort_values(["scenario", "Id"])
    coverage = pd.DataFrame(coverage_rows).sort_values("Id")
    diagnostics = pd.DataFrame(forecast_rows).sort_values(["Id", "model"])
    save_table(summary, out / "conditional_2050_candidate_summary.csv")
    save_table(monte, out / "conditional_2050_monte_carlo.csv")
    save_table(coverage, out / "future_constraint_coverage.csv")
    save_table(diagnostics, out / "area_forecast_model_diagnostics.csv")
    expected = expected_counts(cfg)[0] * len(cfg["future_2050"]["scenarios"])
    coverage_ok = len(summary) == expected
    formal_eligible = bool(formal_detection and summary.loc[summary.estimation_mode.eq("direct_pixel"), "formal_result"].all()) if len(summary.loc[summary.estimation_mode.eq("direct_pixel")]) else False
    atomic_json({
        "status": "passed" if coverage_ok else "failed", "created_utc": utc_now(),
        "records": len(summary), "expected_records": expected, "scientific_gate_pass": formal_detection,
        "formal_forecast_eligible": formal_eligible, "deterministic_forecast": False,
        "area_forecast_basis": "observed/reconstructed 2011-reference-year change with backtested trend selection and a bounded six-covariate environmental feasibility modifier",
        "spatial_allocation_basis": "multisource constrained suitability for direct and anchor-guided pixel-mapped units; only direct units can be formally qualified",
        "reference_shp_role": "certain positive initial-state evidence, not directly copied to 2050",
        "area_only_units_have_spatial_maps": False,
        "constraint_coverage_units": int((coverage.available_constraint_count > 0).sum()),
    }, out / "future_2050_status.json")
    if not coverage_ok:
        raise RuntimeError("Future forecast coverage incomplete")
    return summary
