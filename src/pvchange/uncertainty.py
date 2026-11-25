from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from tqdm import tqdm

from .routes import PIXEL_MODES, is_pixel_mode
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _windows(width: int, height: int, size: int = 512) -> Iterator[Window]:
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def _expanded(core: Window, width: int, height: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    col0 = max(0, int(core.col_off) - halo)
    row0 = max(0, int(core.row_off) - halo)
    col1 = min(width, int(core.col_off + core.width) + halo)
    row1 = min(height, int(core.row_off + core.height) + halo)
    win = Window(col0, row0, col1 - col0, row1 - row0)
    r0 = int(core.row_off) - row0
    c0 = int(core.col_off) - col0
    return win, (slice(r0, r0 + int(core.height)), slice(c0, c0 + int(core.width)))


def _direct_uncertainty(row: Any, offsets: list[float], base_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_pixels = variance_pixels = std_sum = entropy_sum = 0.0
    std_count = entropy_count = uncertain_count = posterior_valid_count = 0
    area_counts = {offset: 0 for offset in offsets}
    with rasterio.open(row.calibrated_probability_path) as probability_src, rasterio.open(row.epistemic_std_path) as std_src, rasterio.open(row.posterior_path) as posterior_src:
        for _, win in probability_src.block_windows(1):
            probability = probability_src.read(1, window=win).astype(np.float32)
            std = std_src.read(1, window=win).astype(np.float32)
            posterior = posterior_src.read(1, window=win).astype(np.float32)
            probability[probability <= -9990] = np.nan
            std[std <= -9990] = np.nan
            posterior[posterior <= -9990] = np.nan
            valid_probability = np.isfinite(probability)
            valid_posterior = np.isfinite(posterior)
            clipped = np.clip(probability[valid_probability], 1e-6, 1 - 1e-6)
            entropy_sum += float(np.sum(-(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)), dtype=np.float64))
            entropy_count += int(clipped.size)
            valid_std = std[np.isfinite(std)]
            std_sum += float(np.sum(valid_std, dtype=np.float64)); std_count += int(valid_std.size)
            post_values = posterior[valid_posterior]
            expected_pixels += float(np.sum(post_values, dtype=np.float64))
            variance_pixels += float(np.sum(post_values * (1 - post_values), dtype=np.float64))
            if std.shape == posterior.shape:
                variance_pixels += float(np.sum(np.nan_to_num(std[valid_posterior], nan=0.0) ** 2, dtype=np.float64))
            uncertain_count += int(np.sum((post_values >= 0.35) & (post_values <= 0.65)))
            posterior_valid_count += int(post_values.size)
            for offset in offsets:
                threshold = float(np.clip(base_threshold + offset, 0.05, 0.95))
                area_counts[offset] += int(np.sum(post_values >= threshold))
    expected_area = expected_pixels * 0.0009
    sd_area = np.sqrt(max(variance_pixels, 0.0)) * 0.0009
    areas = [count * 0.0009 for count in area_counts.values()]
    summary = {
        "Id": int(row.Id),
        "year": int(row.year),
        "estimation_mode": str(getattr(row, "estimation_mode", "direct_pixel")),
        "expected_pv_area_km2": expected_area,
        "area_ci_low_km2": max(0.0, expected_area - 1.96 * sd_area),
        "area_ci_high_km2": expected_area + 1.96 * sd_area,
        "threshold_area_min_km2": min(areas) if areas else np.nan,
        "threshold_area_max_km2": max(areas) if areas else np.nan,
        "mean_epistemic_std": std_sum / std_count if std_count else np.nan,
        "mean_predictive_entropy": entropy_sum / entropy_count if entropy_count else np.nan,
        "uncertain_pixel_fraction": uncertain_count / posterior_valid_count if posterior_valid_count else np.nan,
        "uncertainty_source": "ensemble+posterior+threshold",
    }
    sensitivity = [
        {"Id": int(row.Id), "year": int(row.year), "estimation_mode": str(getattr(row, "estimation_mode", "direct_pixel")), "threshold_offset": float(offset), "area_pixels": int(count), "area_km2": count * 0.0009}
        for offset, count in area_counts.items()
    ]
    return summary, sensitivity


def quantify_uncertainty(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "10_uncertainty")
    inference = pd.read_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv")
    temporal = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv")
    offsets = list(map(float, cfg["uncertainty"]["threshold_offsets"]))
    base_threshold = float(pd.to_numeric(inference.get("calibrated_threshold"), errors="coerce").median())
    if not np.isfinite(base_threshold):
        base_threshold = 0.5
    direct = temporal.loc[temporal["estimation_mode"].isin(PIXEL_MODES)].merge(
        inference[["Id", "year", "calibrated_probability_path", "epistemic_std_path", "calibrated_threshold"]],
        on=["Id", "year"], how="left",
    )
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for row in tqdm(
        list(direct.itertuples(index=False)),
        desc="Streaming direct-pixel uncertainty",
        unit="image",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        summary, sensitivity = _direct_uncertainty(row, offsets, base_threshold)
        rows.append(summary); threshold_rows.extend(sensitivity)

    fallback = temporal.loc[~temporal["estimation_mode"].isin(PIXEL_MODES)].copy()
    for row in fallback.itertuples(index=False):
        point = float(row.pv_area_km2)
        low = float(getattr(row, "pv_area_ci_low_km2", np.nan))
        high = float(getattr(row, "pv_area_ci_high_km2", np.nan))
        rows.append({
            "Id": int(row.Id), "year": int(row.year), "estimation_mode": "area_only_fallback",
            "expected_pv_area_km2": point, "area_ci_low_km2": low, "area_ci_high_km2": high,
            "threshold_area_min_km2": np.nan, "threshold_area_max_km2": np.nan,
            "mean_epistemic_std": np.nan, "mean_predictive_entropy": np.nan,
            "uncertain_pixel_fraction": np.nan,
            "uncertainty_source": "geographic_analog_bootstrap+anchor_break",
        })
        for offset in offsets:
            threshold_rows.append({
                "Id": int(row.Id), "year": int(row.year), "estimation_mode": "area_only_fallback",
                "threshold_offset": float(offset), "area_pixels": np.nan, "area_km2": point,
            })

    summary = pd.DataFrame(rows).sort_values(["Id", "year"])
    threshold_area = pd.DataFrame(threshold_rows).sort_values(["Id", "threshold_offset", "year"])
    save_table(summary, out / "annual_area_uncertainty.csv")
    save_table(threshold_area, out / "threshold_area_sensitivity.csv")

    timing_rows: list[dict[str, Any]] = []
    for region_id, group in threshold_area.groupby("Id"):
        if not is_pixel_mode(group.estimation_mode.iloc[0]):
            source = temporal.loc[temporal.Id.eq(region_id)]
            inferred = pd.to_numeric(source.get("inferred_deployment_year"), errors="coerce").dropna()
            year = float(inferred.iloc[0]) if len(inferred) else np.nan
            timing_rows.append({
                "Id": int(region_id), "change_year_median": year, "change_year_ci_low": year,
                "change_year_ci_high": year, "threshold_scenarios": 0,
                "timing_identified": bool(np.isfinite(year)), "estimation_mode": "area_only_fallback",
            })
            continue
        onset_values: list[float] = []
        for _, scenario in group.groupby("threshold_offset"):
            scenario = scenario.sort_values("year")
            years = scenario["year"].to_numpy(dtype=int)
            areas = scenario["area_km2"].to_numpy(dtype=float)
            gains = np.diff(areas, prepend=areas[0])
            positive = np.where(gains > max(0.0036, 0.01 * max(float(np.nanmax(areas)), 1e-6)))[0]
            onset_values.append(float(years[positive[0]]) if positive.size else np.nan)
        finite = np.asarray(onset_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        timing_rows.append({
            "Id": int(region_id),
            "change_year_median": float(np.median(finite)) if finite.size else np.nan,
            "change_year_ci_low": float(np.min(finite)) if finite.size else np.nan,
            "change_year_ci_high": float(np.max(finite)) if finite.size else np.nan,
            "threshold_scenarios": len(onset_values),
            "timing_identified": bool(finite.size),
            "estimation_mode": str(group.estimation_mode.iloc[0]),
        })
    timing = pd.DataFrame(timing_rows).sort_values("Id")
    save_table(timing, out / "change_year_uncertainty.csv")

    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv")
    perturbations = [int(value) for value in cfg["uncertainty"].get("boundary_perturbation_pixels", [-2, -1, 0, 1, 2])]
    halo = max(abs(value) for value in perturbations)
    boundary_rows: list[dict[str, Any]] = []
    for row in supervision.itertuples(index=False):
        totals = {value: {"base": 0, "perturbed": 0, "union": 0, "intersection": 0} for value in perturbations}
        with rasterio.open(row.pv_fraction_path) as src:
            for core in _windows(src.width, src.height, 512):
                expanded, crop = _expanded(core, src.width, src.height, halo)
                fraction = src.read(1, window=expanded).astype(np.float32)
                baseline = fraction >= float(cfg["supervision"]["strong_positive_fraction"])
                base_core = baseline[crop]
                for pixels in perturbations:
                    if pixels > 0:
                        perturbed = ndimage.binary_dilation(baseline, iterations=pixels); direction = "dilation"
                    elif pixels < 0:
                        perturbed = ndimage.binary_erosion(baseline, iterations=abs(pixels)); direction = "erosion"
                    else:
                        perturbed = baseline; direction = "baseline"
                    perturbed_core = perturbed[crop]
                    totals[pixels]["base"] += int(base_core.sum())
                    totals[pixels]["perturbed"] += int(perturbed_core.sum())
                    totals[pixels]["union"] += int(np.count_nonzero(base_core | perturbed_core))
                    totals[pixels]["intersection"] += int(np.count_nonzero(base_core & perturbed_core))
                    totals[pixels]["direction"] = direction
        for pixels, values in totals.items():
            boundary_rows.append({
                "Id": int(row.Id), "pixels": pixels, "direction": values["direction"],
                "baseline_pixels": values["base"], "perturbed_pixels": values["perturbed"],
                "area_change_fraction": (values["perturbed"] - values["base"]) / max(values["base"], 1),
                "iou": values["intersection"] / max(values["union"], 1),
            })
    boundary = pd.DataFrame(boundary_rows)
    save_table(boundary, out / "shp_boundary_sensitivity.csv")

    expected_units = expected_counts(cfg)[0]
    expected_years = int(cfg["study"]["end_year"]) - int(cfg["study"]["start_year"]) + 1
    expected_records = expected_units * expected_years
    coverage_ok = len(summary) == expected_records and summary.Id.nunique() == expected_units
    atomic_json({
        "status": "passed" if coverage_ok else "failed",
        "created_utc": utc_now(), "records": len(summary), "expected_records": expected_records,
        "direct_pixel_records": int(summary.estimation_mode.eq("direct_pixel").sum()),
        "anchor_guided_candidate_records": int(summary.estimation_mode.eq("anchor_guided_candidate").sum()),
        "pixel_mapping_records": int(summary.estimation_mode.isin(PIXEL_MODES).sum()),
        "area_only_records": int((~summary.estimation_mode.isin(PIXEL_MODES)).sum()),
        "confidence_level": cfg["uncertainty"]["confidence_level"], "bounded_memory": True,
        "sources": ["model ensemble", "posterior variance", "threshold sensitivity", "SHP boundary", "analog bootstrap"],
        "area_only_pixel_uncertainty_claimed": False,
    }, out / "uncertainty_status.json")
    if not coverage_ok:
        raise RuntimeError("Uncertainty coverage incomplete")
    return summary
