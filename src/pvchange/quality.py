from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from .features import compute_features
from .spectral_mixture import (
    derive_endmembers,
    save_endmember_bundle,
    stream_reference_samples,
    summarize_fraction_cube,
    unmix_lsma,
)
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _overview(path: str | Path, size: int = 320) -> np.ndarray:
    with rasterio.open(path) as src:
        h = min(size, src.height)
        w = min(size, src.width)
        data = src.read([1, 2, 3, 4, 5, 6], out_shape=(6, h, w), masked=True).filled(np.nan).astype(np.float32)
    data[data <= -9990] = np.nan
    return data


def _annual_quality(group: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    valid_fractions: list[float] = []
    band_medians: list[np.ndarray] = []
    for row in group.sort_values("year").itertuples(index=False):
        data = _overview(row.path)
        valid = np.all(np.isfinite(data), axis=0)
        valid_fractions.append(float(valid.mean()))
        if np.any(valid):
            band_medians.append(np.nanmedian(data.reshape(6, -1), axis=1))
        else:
            band_medians.append(np.full(6, np.nan, dtype=np.float32))
    medians = np.asarray(band_medians, dtype=float)
    jumps = np.linalg.norm(np.diff(medians, axis=0), axis=1) if len(medians) > 1 else np.array([0.0])
    finite_jumps = jumps[np.isfinite(jumps)]
    return {
        "median_valid_fraction": float(np.nanmedian(valid_fractions)),
        "minimum_valid_fraction": float(np.nanmin(valid_fractions)),
        "valid_year_count_50pct": int(np.sum(np.asarray(valid_fractions) >= 0.5)),
        "maximum_annual_median_spectral_jump": float(np.nanmax(finite_jumps)) if finite_jumps.size else float("nan"),
    }


def _separation(fractions: np.ndarray, anchor_fraction: np.ndarray) -> float:
    anchor = anchor_fraction >= 0.7
    background = anchor_fraction <= 0
    score = fractions[1] + fractions[3] - fractions[0]  # LA + SH - HA
    a = score[anchor & np.isfinite(score)]
    b = score[background & np.isfinite(score)]
    if a.size < 5 or b.size < 20:
        return float("nan")
    return float(np.nanmedian(a) - np.nanmedian(b))


def _quality_score(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, list[str]]:
    qc = cfg.get("quality_routing", {})
    score = 0.0
    reasons: list[str] = []
    valid = float(row["median_valid_fraction"])
    years = int(row["valid_year_count_50pct"])
    rmse = float(row["reference_lsma_rmse_median"])
    sep = float(row["anchor_spectral_separation"])
    jump = float(row["maximum_annual_median_spectral_jump"])
    condition = float(row["endmember_condition_number"])

    score += 0.30 * np.clip(valid / max(float(qc.get("target_median_valid_fraction", 0.75)), 1e-6), 0, 1)
    score += 0.20 * np.clip(years / max(float(qc.get("target_valid_years", 12)), 1), 0, 1)
    score += 0.20 * (1.0 - np.clip(rmse / max(float(qc.get("maximum_lsma_rmse", 0.15)), 1e-6), 0, 1)) if np.isfinite(rmse) else 0.0
    score += 0.15 * np.clip(sep / max(float(qc.get("target_anchor_separation", 0.08)), 1e-6), 0, 1) if np.isfinite(sep) else 0.05
    score += 0.10 * (1.0 - np.clip(jump / max(float(qc.get("maximum_annual_spectral_jump", 0.25)), 1e-6), 0, 1)) if np.isfinite(jump) else 0.0
    score += 0.05 * (1.0 - np.clip(np.log10(max(condition, 1.0)) / 5.0, 0, 1)) if np.isfinite(condition) else 0.0

    if valid < float(qc.get("minimum_median_valid_fraction", 0.35)):
        reasons.append("low_median_valid_fraction")
    if years < int(qc.get("minimum_valid_years", 8)):
        reasons.append("too_few_valid_years")
    if np.isfinite(rmse) and rmse > float(qc.get("hard_maximum_lsma_rmse", 0.25)):
        reasons.append("high_lsma_residual")
    if np.isfinite(jump) and jump > float(qc.get("hard_maximum_annual_spectral_jump", 0.45)):
        reasons.append("large_cross_year_spectral_jump")
    if np.isfinite(condition) and condition > float(qc.get("maximum_endmember_condition_number", 2e4)):
        reasons.append("unstable_endmember_matrix")
    return float(np.clip(score, 0, 1)), reasons



def _mixed_pixel_anchor_rescue_eligible(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Keep an anchored unit on a spatial route when its parcel is mixed-pixel.

    This route is intentionally weaker than formal direct detection but stronger
    than area-only inference.  It requires a real 2024 anchor, multiple usable
    annual images and at least a few valid anchor samples from which a local
    prototype can be constructed.
    """
    routing = cfg.get("quality_routing", {})
    if not bool(routing.get("mixed_pixel_anchor_rescue_enabled", True)):
        return False
    if not bool(row.get("has_confirmed_anchor", False)):
        return False
    valid_years = int(row.get("valid_year_count_50pct", 0))
    median_valid = float(row.get("median_valid_fraction", 0.0))
    anchor_pixels = int(
        row.get(
            "mixed_anchor_sample_pixels",
            row.get("anchor_sample_pixels", 0),
        )
    )
    rmse = float(row.get("reference_lsma_rmse_median", np.nan))
    jump = float(row.get("maximum_annual_median_spectral_jump", np.nan))
    condition = float(row.get("endmember_condition_number", np.nan))
    return bool(
        valid_years >= int(routing.get("mixed_pixel_anchor_minimum_valid_years", 4))
        and median_valid >= float(routing.get("mixed_pixel_anchor_minimum_median_valid_fraction", 0.20))
        and anchor_pixels >= int(routing.get("mixed_pixel_anchor_minimum_samples", 3))
        and (not np.isfinite(rmse) or rmse <= float(routing.get("mixed_pixel_anchor_maximum_lsma_rmse", 0.55)))
        and (not np.isfinite(jump) or jump <= float(routing.get("mixed_pixel_anchor_maximum_annual_spectral_jump", 0.85)))
        and (not np.isfinite(condition) or condition <= float(routing.get("mixed_pixel_anchor_maximum_condition_number", 1.0e6)))
    )



def _reference_internal_analysis_eligible(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Retain a spatial route for a 2024-authoritative parcel without pure pixels.

    This is not a training-label relaxation.  The unit remains excluded from
    formal supervised roles unless it satisfies the direct detector gate.  The
    route only guarantees that annual Landsat imagery and the authoritative
    2024 parcel can be analysed by the independent internal-change branch.
    """
    routing = cfg.get("quality_routing", {})
    if not bool(routing.get("reference_internal_analysis_enabled", True)):
        return False
    if not bool(row.get("has_confirmed_anchor", False)):
        return False
    valid_years = int(row.get("valid_year_count_50pct", 0))
    median_valid = float(row.get("median_valid_fraction", 0.0))
    rmse = float(row.get("reference_lsma_rmse_median", np.nan))
    jump = float(row.get("maximum_annual_median_spectral_jump", np.nan))
    return bool(
        (
            valid_years >= int(routing.get("reference_internal_minimum_valid_years", 2))
            or median_valid >= float(routing.get("reference_internal_minimum_median_valid_fraction", 0.12))
        )
        and (
            not np.isfinite(rmse)
            or rmse <= float(routing.get("reference_internal_maximum_lsma_rmse", 0.80))
        )
        and (
            not np.isfinite(jump)
            or jump <= float(routing.get("reference_internal_maximum_annual_spectral_jump", 1.20))
        )
    )

def build_unit_quality_and_endmembers(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "04_unit_quality")
    endmember_root = ensure_dir(out / "endmembers")
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv")
    reference_year = int(cfg["study"]["reference_year"])
    reference = aligned.loc[aligned["year"].eq(reference_year)].set_index("Id")
    sup = supervision.set_index("Id")
    rows: list[dict[str, Any]] = []
    for region_id, group in tqdm(
        list(aligned.groupby("Id")),
        desc="Unit data quality and adaptive LSMA",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        region_id = int(region_id)
        annual = _annual_quality(group, cfg)
        if region_id not in reference.index or region_id not in sup.index:
            raise RuntimeError(f"Id {region_id}: reference-year or supervision row missing")
        raster_path = Path(str(reference.loc[region_id, "path"]))
        fraction_path = Path(str(sup.loc[region_id, "pv_fraction_path"]))
        reflectance, anchor_fraction = stream_reference_samples(raster_path, fraction_path, cfg, int(cfg["project"]["random_seed"]) + region_id)
        bundle = derive_endmembers(reflectance, anchor_fraction, cfg, int(cfg["project"]["random_seed"]) + region_id)
        endmember_path = endmember_root / f"unit_{region_id:03d}_endmembers.npz"
        save_endmember_bundle(bundle, endmember_path)
        fractions, rmse = unmix_lsma(reflectance, bundle.matrix)
        valid = np.all(np.isfinite(reflectance), axis=0)
        soft_anchor_floor = float(
            cfg.get("anchor_guided", {}).get("mixed_pixel_prototype_fraction_floor", 0.02)
        )
        mixed_anchor_sample_pixels = int(
            np.count_nonzero(valid & (anchor_fraction >= soft_anchor_floor))
        )
        summary = summarize_fraction_cube(fractions, rmse, valid)
        separation = _separation(fractions, anchor_fraction)
        row: dict[str, Any] = {
            "Id": region_id,
            "endmember_path": str(endmember_path),
            "endmember_source": bundle.source,
            "endmember_condition_number": bundle.condition_number,
            "minimum_spectral_angle_deg": bundle.minimum_spectral_angle_deg,
            "anchor_sample_pixels": bundle.anchor_pixels,
            "mixed_anchor_sample_pixels": mixed_anchor_sample_pixels,
            "background_sample_pixels": bundle.background_pixels,
            "anchor_spectral_separation": separation,
            "reference_lsma_rmse_median": summary["LSMA_RMSE_median"],
            "reference_lsma_rmse_p90": summary["LSMA_RMSE_p90"],
            "confirmed_anchor_area_km2": float(sup.loc[region_id, "confirmed_anchor_area_km2"]),
            "has_confirmed_anchor": bool(sup.loc[region_id].get("has_authoritative_2024_pv", sup.loc[region_id, "has_mapped_pv"])),
            "has_strong_training_core": bool(sup.loc[region_id].get("has_strong_training_core", sup.loc[region_id, "has_mapped_pv"])),
            "split_role": str(sup.loc[region_id, "split_role"]),
            **annual,
        }
        score, reasons = _quality_score(row, cfg)
        routing = cfg.get("quality_routing", {})
        threshold = float(routing.get("direct_detection_minimum_score", 0.55))
        direct = score >= threshold and not reasons
        standard_guided = bool(
            cfg.get("anchor_guided", {}).get("enabled", True)
            and row["has_confirmed_anchor"]
            and not direct
            and score >= float(routing.get("anchor_guided_minimum_score", 0.38))
            and int(row["valid_year_count_50pct"]) >= int(routing.get("anchor_guided_minimum_valid_years", 6))
            and (
                not np.isfinite(float(row["reference_lsma_rmse_median"]))
                or float(row["reference_lsma_rmse_median"]) <= float(routing.get("anchor_guided_maximum_lsma_rmse", 0.35))
            )
            and (
                not np.isfinite(float(row["maximum_annual_median_spectral_jump"]))
                or float(row["maximum_annual_median_spectral_jump"]) <= float(routing.get("anchor_guided_maximum_annual_spectral_jump", 0.60))
            )
            and int(row["anchor_sample_pixels"]) >= int(cfg.get("anchor_guided", {}).get("minimum_anchor_prototype_pixels", 20))
        )
        mixed_pixel_rescue = bool(not direct and _mixed_pixel_anchor_rescue_eligible(row, cfg))
        reference_internal_analysis = bool(
            not direct
            and not standard_guided
            and not mixed_pixel_rescue
            and _reference_internal_analysis_eligible(row, cfg)
        )
        guided = bool(standard_guided or mixed_pixel_rescue or reference_internal_analysis)
        # High-quality units use unrestricted full-scene inference. Moderate-quality
        # anchored units use conservative anchor-guided candidate mapping. Only units
        # without enough spatial evidence fall back to area-only inference.
        row["quality_score"] = score
        row["quality_reasons"] = ";".join(reasons)
        row["estimation_route"] = (
            "direct_pixel" if direct else
            "anchor_guided_candidate" if guided else
            "anchor_analog_area" if row["has_confirmed_anchor"] else
            "unanchored_analog_area"
        )
        row["direct_detection_eligible"] = bool(direct)
        row["anchor_guided_eligible"] = bool(guided)
        row["mixed_pixel_anchor_rescue"] = bool(mixed_pixel_rescue)
        row["reference_internal_analysis"] = bool(reference_internal_analysis)
        row["anchor_guided_route_reason"] = (
            "standard_anchor_guided" if standard_guided else
            "mixed_pixel_anchor_rescue" if mixed_pixel_rescue else
            "reference_internal_change_analysis" if reference_internal_analysis else
            "not_eligible"
        )
        row["pixel_mapping_eligible"] = bool(direct or guided)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("Id")

    # Quality-aware role rebalancing. The initial stage-3 split is based only on
    # confirmed anchors because quality has not yet been measured. After LSMA
    # quality routing, direct positive units are deterministically reallocated
    # to train/calibration/test so that calibration and independent validation
    # remain independent.  Anchor-guided candidates retain spatial inference but
    # are not promoted into formal supervised training or validation roles.
    from .preparation import _assign_supervision_roles
    frame["initial_split_role"] = frame["split_role"].astype(str)
    role_input = frame[["Id", "confirmed_anchor_area_km2"]].copy()
    # Reassign formal train/calibration/test roles only among units that have
    # both a genuine strong training core and pass the direct detector gate.
    # Authoritative mixed-pixel parcels remain inference-only but keep their
    # spatial dual-stage analysis route.
    role_input["has_mapped_pv"] = frame["has_strong_training_core"].astype(bool) & frame["direct_detection_eligible"].astype(bool)
    role_input["has_strong_training_core"] = role_input["has_mapped_pv"]
    role_input["strong_positive_pixels"] = frame["anchor_sample_pixels"].fillna(0).astype(int)
    effective_roles = _assign_supervision_roles(role_input, cfg)[["Id", "split_role", "is_training_unit", "is_calibration_unit", "is_independent_test_unit"]]
    frame = frame.drop(columns=["split_role"], errors="ignore").merge(effective_roles, on="Id", how="left")

    supervision_path = root / "03_supervision" / "supervision_registry.csv"
    supervision_frame = pd.read_csv(supervision_path)
    supervision_frame = supervision_frame.drop(columns=["split_role", "is_training_unit", "is_calibration_unit", "is_independent_test_unit"], errors="ignore").merge(effective_roles, on="Id", how="left")
    save_table(supervision_frame.sort_values("Id"), supervision_path)
    split_columns = [
        "Id", "split_role", "has_mapped_pv", "strong_positive_pixels", "confirmed_anchor_area_km2",
        "hard_negative_pixels", "overlap_area_m2", "is_training_unit", "is_calibration_unit", "is_independent_test_unit",
    ]
    save_table(supervision_frame[split_columns].sort_values("Id"), root / "03_supervision" / "supervision_split.csv")

    save_table(frame, out / "unit_quality_registry.csv")
    save_table(frame[[
        "Id", "initial_split_role", "split_role", "quality_score", "estimation_route", "quality_reasons", "median_valid_fraction",
        "valid_year_count_50pct", "reference_lsma_rmse_median", "anchor_spectral_separation",
        "anchor_sample_pixels", "mixed_anchor_sample_pixels",
        "has_confirmed_anchor", "has_strong_training_core", "confirmed_anchor_area_km2", "endmember_path", "direct_detection_eligible",
        "anchor_guided_eligible", "mixed_pixel_anchor_rescue", "reference_internal_analysis", "anchor_guided_route_reason", "pixel_mapping_eligible",
    ]], out / "unit_routing_audit.csv")
    expected = expected_counts(cfg)[0]
    direct_units = int(frame["direct_detection_eligible"].sum())
    guided_units = int(frame["anchor_guided_eligible"].sum())
    pixel_units = int(frame["pixel_mapping_eligible"].sum())
    selected_units = cfg.get("study", {}).get("selected_unit_ids") or []
    relaxed_selected_test = bool(
        selected_units
        and cfg.get("test_mode", {}).get("relax_formal_production_contracts", False)
    )
    production_pass = bool(
        len(frame) == expected
        and direct_units >= int(cfg.get("quality_routing", {}).get("minimum_direct_units", 10))
    )
    selected_spatial_pass = bool(
        relaxed_selected_test
        and len(frame) == expected
        and pixel_units == len(frame)
    )
    status = "passed" if production_pass or selected_spatial_pass else "failed"
    atomic_json({
        "status": status,
        "created_utc": utc_now(),
        "units": len(frame),
        "expected_units": expected,
        "direct_pixel_units": direct_units,
        "direct_positive_anchor_units": int((frame["direct_detection_eligible"].astype(bool) & frame["has_confirmed_anchor"].astype(bool)).sum()),
        "anchor_guided_candidate_units": guided_units,
        "pixel_mapping_units": pixel_units,
        "effective_split_counts": frame["split_role"].value_counts().to_dict(),
        "area_only_units": int((~frame["pixel_mapping_eligible"]).sum()),
        "reference_year": reference_year,
        "shp_role": "complete-reference sufficient condition",
        "low_quality_policy": "all usable 2024-authoritative parcels retain a spatial analysis route; pure-pixel quality controls only formal training and validation eligibility",
        "mixed_pixel_anchor_rescue_units": int(frame.get("mixed_pixel_anchor_rescue", pd.Series(False, index=frame.index)).astype(bool).sum()),
        "reference_internal_analysis_units": int(frame.get("reference_internal_analysis", pd.Series(False, index=frame.index)).astype(bool).sum()),
        "selected_unit_spatial_contract_used": bool(selected_spatial_pass),
        "formal_direct_unit_gate_relaxed": False,
    }, out / "unit_quality_status.json")
    if selected_spatial_pass and not production_pass:
        print(
            "[test-units] All selected units retain a spatial route "
            "(direct or anchor-guided); accepted for the selected-unit "
            "smoke test without relaxing the full-production direct-unit gate."
        )
    if status == "failed":
        raise RuntimeError(
            f"Unit quality routing failed: direct units={direct_units}, "
            f"pixel-mapping units={pixel_units}, total={len(frame)}"
        )
    return frame
