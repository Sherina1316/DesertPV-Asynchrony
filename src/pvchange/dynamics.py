from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from tqdm import tqdm

from .routes import PIXEL_MODES, is_pixel_mode
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _tile_fragmentation(mask: np.ndarray) -> tuple[int, int, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0, 0, 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), int(sizes.sum()), int(sizes.max())


def _direct_unit_rows(group: pd.DataFrame, cfg: dict[str, Any], change_root: Path) -> list[dict[str, Any]]:
    mask_nodata = int(cfg["temporal_decoding"].get("mask_nodata", 255))
    region_id = int(group.iloc[0]["Id"])
    rows: list[dict[str, Any]] = []
    previous_active_path: Path | None = None
    previous_cumulative_path: Path | None = None
    previous_active_area = 0.0
    previous_centroid = (np.nan, np.nan)
    for record in group.sort_values("year").itertuples(index=False):
        active_path = Path(str(getattr(record, "active_mask_path", record.mask_path)))
        cumulative_path = Path(str(getattr(record, "cumulative_mask_path", record.mask_path)))
        if not active_path.exists() or not cumulative_path.exists():
            raise RuntimeError(f"Id {region_id}, year {int(record.year)}: dual-state masks missing")
        target = ensure_dir(change_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{int(record.year)}_pv_change_class_30m.tif"
        with rasterio.open(active_path) as active_src, rasterio.open(cumulative_path) as cumulative_src:
            previous_active_src = rasterio.open(previous_active_path) if previous_active_path is not None else None
            previous_cumulative_src = rasterio.open(previous_cumulative_path) if previous_cumulative_path is not None else None
            try:
                profile = active_src.profile.copy()
                profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
                active_pixels = cumulative_pixels = valid_pixels = 0
                active_gain_pixels = active_loss_pixels = active_stable_pixels = 0
                installed_gain_pixels = inactive_pixels = 0
                x_sum = y_sum = 0.0
                fragment_count = fragment_pixel_sum = largest_fragment_pixels = 0
                with rasterio.open(target, "w", **profile) as dst:
                    for _, window in active_src.block_windows(1):
                        active_encoded = active_src.read(1, window=window)
                        cumulative_encoded = cumulative_src.read(1, window=window)
                        current_valid = active_encoded != mask_nodata
                        active = active_encoded == 1
                        cumulative = cumulative_encoded == 1
                        if previous_active_src is None:
                            previous_active = np.zeros_like(active)
                            previous_active_valid = current_valid.copy()
                            previous_cumulative = np.zeros_like(cumulative)
                        else:
                            previous_active_encoded = previous_active_src.read(1, window=window)
                            previous_active_valid = previous_active_encoded != mask_nodata
                            previous_active = previous_active_encoded == 1
                            previous_cumulative = previous_cumulative_src.read(1, window=window) == 1
                        comparable = current_valid & previous_active_valid
                        active_gain = active & ~previous_active & comparable
                        active_loss = previous_active & ~active & comparable
                        active_stable = active & previous_active & comparable
                        installed_gain = cumulative & ~previous_cumulative
                        inactive = cumulative & ~active
                        change = np.full(active.shape, mask_nodata, dtype=np.uint8)
                        change[current_valid] = 0
                        change[active_stable] = 1
                        change[installed_gain & active] = 2
                        change[inactive] = 3
                        dst.write(change, 1, window=window)

                        active_mask = active & current_valid
                        active_count = int(active_mask.sum())
                        active_pixels += active_count
                        cumulative_pixels += int(cumulative.sum())
                        valid_pixels += int(current_valid.sum())
                        active_gain_pixels += int(active_gain.sum())
                        active_loss_pixels += int(active_loss.sum())
                        active_stable_pixels += int(active_stable.sum())
                        installed_gain_pixels += int(installed_gain.sum())
                        inactive_pixels += int(inactive.sum())
                        if active_count:
                            rr, cc = np.where(active_mask)
                            gr = rr + int(window.row_off)
                            gc = cc + int(window.col_off)
                            xs = active_src.transform.c + (gc + 0.5) * active_src.transform.a + (gr + 0.5) * active_src.transform.b
                            ys = active_src.transform.f + (gc + 0.5) * active_src.transform.d + (gr + 0.5) * active_src.transform.e
                            x_sum += float(np.sum(xs)); y_sum += float(np.sum(ys))
                        count, pixel_sum, largest = _tile_fragmentation(active_mask)
                        fragment_count += count
                        fragment_pixel_sum += pixel_sum
                        largest_fragment_pixels = max(largest_fragment_pixels, largest)
                active_area = active_pixels * 0.0009
                cumulative_area = cumulative_pixels * 0.0009
                centroid = (x_sum / active_pixels, y_sum / active_pixels) if active_pixels else (np.nan, np.nan)
                shift = float(np.hypot(centroid[0] - previous_centroid[0], centroid[1] - previous_centroid[1])) if np.all(np.isfinite([*centroid, *previous_centroid])) else np.nan
                rows.append({
                    "Id": region_id,
                    "year": int(record.year),
                    "estimation_mode": str(getattr(record, "estimation_mode", "direct_pixel")),
                    "estimation_route": str(getattr(record, "estimation_route", "direct_pixel")),
                    "pv_area_km2": active_area,
                    "observed_active_area_km2": active_area,
                    "cumulative_installed_area_km2": cumulative_area,
                    "inactive_or_retired_area_km2": inactive_pixels * 0.0009,
                    "pv_area_ci_low_km2": float(getattr(record, "pv_area_ci_low_km2", np.nan)),
                    "pv_area_ci_high_km2": float(getattr(record, "pv_area_ci_high_km2", np.nan)),
                    "valid_area_km2": valid_pixels * 0.0009,
                    "gross_gain_km2": active_gain_pixels * 0.0009,
                    "gross_loss_km2": active_loss_pixels * 0.0009,
                    "installed_gain_km2": installed_gain_pixels * 0.0009,
                    "persistent_area_km2": active_stable_pixels * 0.0009,
                    "net_change_km2": active_area - previous_active_area,
                    "annual_growth_rate": active_area / previous_active_area - 1 if previous_active_area > 0 else np.nan,
                    "fragment_count": fragment_count,
                    "mean_patch_area_km2": fragment_pixel_sum / fragment_count * 0.0009 if fragment_count else 0.0,
                    "largest_patch_area_km2": largest_fragment_pixels * 0.0009,
                    "centroid_easting_m": centroid[0],
                    "centroid_northing_m": centroid[1],
                    "centroid_shift_m": shift,
                    "change_class_path": str(target),
                    "mask_path": str(active_path),
                    "active_mask_path": str(active_path),
                    "cumulative_mask_path": str(cumulative_path),
                    "pixel_map_available": True,
                    "nodata_safe": True,
                })
                previous_active_area = active_area
                previous_centroid = centroid
            finally:
                if previous_active_src is not None:
                    previous_active_src.close()
                if previous_cumulative_src is not None:
                    previous_cumulative_src.close()
        previous_active_path = active_path
        previous_cumulative_path = cumulative_path
    return rows

def _area_only_rows(group: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = 0.0
    for record in group.sort_values("year").itertuples(index=False):
        area = float(record.pv_area_km2)
        gain = max(area - previous, 0.0)
        loss = max(previous - area, 0.0)
        rows.append({
            "Id": int(record.Id),
            "year": int(record.year),
            "estimation_mode": "area_only_fallback",
            "estimation_route": str(getattr(record, "estimation_route", "area_only_fallback")),
            "fallback_method": str(getattr(record, "fallback_method", "geographic_analog")),
            "pv_area_km2": area,
            "observed_active_area_km2": area,
            "cumulative_installed_area_km2": area,
            "inactive_or_retired_area_km2": 0.0,
            "pv_area_ci_low_km2": float(getattr(record, "pv_area_ci_low_km2", np.nan)),
            "pv_area_ci_high_km2": float(getattr(record, "pv_area_ci_high_km2", np.nan)),
            "valid_area_km2": np.nan,
            "gross_gain_km2": gain,
            "gross_loss_km2": loss,
            "installed_gain_km2": gain,
            "persistent_area_km2": min(area, previous),
            "net_change_km2": area - previous,
            "annual_growth_rate": area / previous - 1 if previous > 0 else np.nan,
            "fragment_count": np.nan,
            "mean_patch_area_km2": np.nan,
            "largest_patch_area_km2": np.nan,
            "centroid_easting_m": np.nan,
            "centroid_northing_m": np.nan,
            "centroid_shift_m": np.nan,
            "change_class_path": "",
            "mask_path": "",
            "active_mask_path": "",
            "cumulative_mask_path": "",
            "pixel_map_available": False,
            "nodata_safe": True,
        })
        previous = area
    return rows


def summarize_pv_dynamics(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "11_pv_dynamics")
    change_root = ensure_dir(out / "annual_change_class_30m")
    registry = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv")
    rows: list[dict[str, Any]] = []
    for _, group in tqdm(
        list(registry.groupby("Id")),
        desc="PV dynamics across direct and area-only routes",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        mode = str(group.iloc[0].get("estimation_mode", "direct_pixel"))
        rows.extend(_direct_unit_rows(group, cfg, change_root) if is_pixel_mode(mode) else _area_only_rows(group))

    annual = pd.DataFrame(rows).sort_values(["Id", "year"])
    uncertainty_path = root / "10_uncertainty" / "annual_area_uncertainty.csv"
    if uncertainty_path.exists():
        uncertainty = pd.read_csv(uncertainty_path)[["Id", "year", "expected_pv_area_km2", "area_ci_low_km2", "area_ci_high_km2"]]
        annual = annual.merge(uncertainty, on=["Id", "year"], how="left", suffixes=("", "_model"))
        expected = pd.to_numeric(annual["expected_pv_area_km2"], errors="coerce")
        model_low = pd.to_numeric(annual["area_ci_low_km2"], errors="coerce")
        model_high = pd.to_numeric(annual["area_ci_high_km2"], errors="coerce")
        half_width = pd.concat([(expected-model_low).abs(), (model_high-expected).abs()], axis=1).max(axis=1).fillna(0.0)
        active_point = pd.to_numeric(annual["observed_active_area_km2"], errors="coerce").fillna(0.0)
        cumulative_point = pd.to_numeric(annual["cumulative_installed_area_km2"], errors="coerce").fillna(active_point)
        direct = annual["estimation_mode"].isin(PIXEL_MODES)
        annual.loc[direct, "pv_area_ci_low_km2"] = np.maximum(active_point[direct]-half_width[direct], 0.0)
        annual.loc[direct, "pv_area_ci_high_km2"] = active_point[direct]+half_width[direct]
        annual["observed_active_area_ci_low_km2"] = np.maximum(active_point-half_width, 0.0)
        annual["observed_active_area_ci_high_km2"] = active_point+half_width
        annual["cumulative_installed_area_ci_low_km2"] = np.maximum(cumulative_point-half_width, 0.0)
        annual["cumulative_installed_area_ci_high_km2"] = cumulative_point+half_width
        annual.drop(columns=[c for c in ["expected_pv_area_km2", "area_ci_low_km2", "area_ci_high_km2"] if c in annual], inplace=True)
    else:
        annual["observed_active_area_ci_low_km2"] = annual["pv_area_ci_low_km2"]
        annual["observed_active_area_ci_high_km2"] = annual["pv_area_ci_high_km2"]
        annual["cumulative_installed_area_ci_low_km2"] = annual["pv_area_ci_low_km2"]
        annual["cumulative_installed_area_ci_high_km2"] = annual["pv_area_ci_high_km2"]
    save_table(annual, out / "annual_pv_dynamics.csv")
    reference_year = int(cfg["study"]["reference_year"])
    end_year = int(cfg["study"]["end_year"])
    unit_rows: list[dict[str, Any]] = []
    for region_id, group in annual.groupby("Id"):
        group = group.sort_values("year")
        first = float(group.iloc[0].pv_area_km2)
        last = float(group.iloc[-1].pv_area_km2)
        span = max(int(group.iloc[-1].year - group.iloc[0].year), 1)
        ref = group.loc[group.year.eq(reference_year), "pv_area_km2"]
        unit_rows.append({
            "Id": int(region_id),
            "estimation_mode": str(group.iloc[0].estimation_mode),
            "estimation_route": str(group.iloc[0].estimation_route),
            "pv_area_start_km2": first,
            f"pv_area_{reference_year}_km2": float(ref.iloc[0]) if len(ref) else np.nan,
            f"pv_area_{end_year}_km2": last,
            "net_change_km2": last - first,
            "cumulative_gain_km2": float(group.gross_gain_km2.sum()),
            "cumulative_loss_km2": float(group.gross_loss_km2.sum()),
            "cagr": (last / first) ** (1 / span) - 1 if first > 0 and last > 0 else np.nan,
            "peak_annual_gain_km2": float(group.installed_gain_km2.max()),
            "peak_gain_year": int(group.loc[group.installed_gain_km2.idxmax(), "year"]),
            f"cumulative_installed_area_{reference_year}_km2": float(group.loc[group.year.eq(reference_year), "cumulative_installed_area_km2"].iloc[0]) if group.year.eq(reference_year).any() else np.nan,
            f"cumulative_installed_area_{end_year}_km2": float(group.iloc[-1].cumulative_installed_area_km2),
            "centroid_displacement_m": float(np.nansum(group.centroid_shift_m)) if group.pixel_map_available.any() else np.nan,
            "pixel_map_available": bool(group.pixel_map_available.any()),
        })
    unit = pd.DataFrame(unit_rows).sort_values("Id")
    save_table(unit, out / "unit_pv_dynamics_summary.csv")
    national = annual.groupby("year", as_index=False).agg(
        pv_area_km2=("pv_area_km2", "sum"),
        observed_active_area_km2=("observed_active_area_km2", "sum"),
        cumulative_installed_area_km2=("cumulative_installed_area_km2", "sum"),
        inactive_or_retired_area_km2=("inactive_or_retired_area_km2", "sum"),
        pv_area_ci_low_km2=("pv_area_ci_low_km2", "sum"),
        pv_area_ci_high_km2=("pv_area_ci_high_km2", "sum"),
        observed_active_area_ci_low_km2=("observed_active_area_ci_low_km2", "sum"),
        observed_active_area_ci_high_km2=("observed_active_area_ci_high_km2", "sum"),
        cumulative_installed_area_ci_low_km2=("cumulative_installed_area_ci_low_km2", "sum"),
        cumulative_installed_area_ci_high_km2=("cumulative_installed_area_ci_high_km2", "sum"),
        gross_gain_km2=("gross_gain_km2", "sum"),
        gross_loss_km2=("gross_loss_km2", "sum"),
        installed_gain_km2=("installed_gain_km2", "sum"),
        net_change_km2=("net_change_km2", "sum"),
        direct_pixel_area_km2=("pv_area_km2", lambda x: float(x[annual.loc[x.index, "estimation_mode"].eq("direct_pixel")].sum())),
        anchor_guided_area_km2=("pv_area_km2", lambda x: float(x[annual.loc[x.index, "estimation_mode"].eq("anchor_guided_candidate")].sum())),
        area_only_inferred_km2=("pv_area_km2", lambda x: float(x[~annual.loc[x.index, "estimation_mode"].isin(PIXEL_MODES)].sum())),
    )
    tolerance = 1e-6
    if ((national["observed_active_area_km2"] - national["cumulative_installed_area_km2"]) > tolerance).any():
        raise RuntimeError("National observed active area exceeds cumulative installed area")
    if not ((national["pv_area_ci_low_km2"]-tolerance <= national["pv_area_km2"]) & (national["pv_area_km2"] <= national["pv_area_ci_high_km2"]+tolerance)).all():
        raise RuntimeError("National active-area confidence interval does not contain the point estimate")
    save_table(national, out / "national_pv_dynamics.csv")
    expected_units = expected_counts(cfg)[0]
    expected_years = int(cfg["study"]["end_year"]) - int(cfg["study"]["start_year"]) + 1
    coverage_ok = len(annual) == expected_units * expected_years and annual.Id.nunique() == expected_units
    atomic_json({
        "status": "passed" if coverage_ok else "failed",
        "created_utc": utc_now(),
        "records": len(annual),
        "units": int(annual.Id.nunique()),
        "direct_pixel_units": int(unit.estimation_mode.eq("direct_pixel").sum()),
        "anchor_guided_candidate_units": int(unit.estimation_mode.eq("anchor_guided_candidate").sum()),
        "pixel_mapping_units": int(unit.pixel_map_available.sum()),
        "area_only_units": int((~unit.pixel_map_available).sum()),
        "area_only_pixel_maps_fabricated": False,
        "resolution_m": 30.0,
        "reference_year": reference_year,
    }, out / "pv_dynamics_status.json")
    if not coverage_ok:
        raise RuntimeError("PV dynamics coverage incomplete")
    return annual
