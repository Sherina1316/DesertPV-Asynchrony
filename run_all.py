from __future__ import annotations
# --- v24.8 self-contained import path injection start ---
from pathlib import Path as _PVPath
import sys as _PVSys

_PV_SRC_DIR = _PVPath(__file__).resolve().parent / "src"
if str(_PV_SRC_DIR) not in _PVSys.path:
    _PVSys.path.insert(0, str(_PV_SRC_DIR))
# --- v24.8 self-contained import path injection end ---

import argparse
import fcntl
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pvchange.utils import atomic_json, sha256_file, utc_now
from pvchange.study_area import expected_counts, test_mode_enabled


STAGES: list[tuple[int, str, str, list[str]]] = [
    (1, "Strict 142-unit × 15-year input and environmental validation", "01_validate_inputs.py", [
        "01_input_validation/input_validation_status.json",
        "01_input_validation/raster_inventory.csv",
        "01_input_validation/environmental_covariates.csv",
    ]),
    (2, "Canonical projected 30 m grid alignment", "02_build_aligned_30m_grids.py", [
        "02_aligned_30m/aligned_grid_status.json",
        "02_aligned_30m/grid_registry.csv",
        "02_aligned_30m/aligned_raster_registry.csv",
    ]),
    (3, "Complete 2024 PV census rasterization and independent unit split", "03_build_shp_supervision.py", [
        "03_supervision/supervision_status.json",
        "03_supervision/supervision_registry.csv",
        "03_supervision/supervision_split.csv",
    ]),
    (4, "Adaptive LSMA quality routing and one-pass causal Annual-CCDC cache", "04_build_patch_dataset.py", [
        "04_unit_quality/unit_quality_status.json",
        "04_unit_quality/unit_quality_registry.csv",
        "04_unit_quality/unit_routing_audit.csv",
        "04_patch_dataset/patch_dataset_status.json",
        "04_patch_dataset/patch_index.csv",
        "04_patch_dataset/training_feature_cache_registry.csv",
        "04_patch_dataset/patch_fold_summary.csv",
        "04_patch_dataset/training_unit_audit.csv",
    ]),
    (5, "Cached-feature complete-reference supervised CPU model benchmark", "05_benchmark_segmentation_models.py", [
        "05_model_benchmark/model_benchmark_status.json",
        "05_model_benchmark/benchmark_model_summary.csv",
        "05_model_benchmark/benchmark_fold_metrics.csv",
    ]),
    (6, "Final weighted CPU ensemble", "06_train_final_ensemble.py", [
        "06_final_models/final_model_status.json",
        "06_final_models/final_ensemble_manifest.csv",
    ]),
    (7, "Annual inference, boundary-aware calibration and adaptive 2024 reference construction", "07_infer_annual_pv_probabilities.py", [
        "07_annual_pv_inference/annual_inference_status.json",
        "07_annual_pv_inference/annual_inference_registry.csv",
        "07_probability_calibration/probability_calibration_status.json",
        "07_probability_calibration/calibrated_inference_registry.csv",
        "07_probability_calibration/calibration_method_comparison.csv",
        "07_probability_calibration/panoramic_threshold_scan.csv",
        "07_authoritative_reference/authoritative_reference_status.json",
        "07_authoritative_reference/authoritative_reference_registry.csv",
    ]),
    (8, "Adaptive-reference installation, retirement and 2025 expansion reconstruction", "08_temporal_decode_pv_change.py", [
        "08_temporal_pv_change/temporal_decoding_status.json",
        "08_temporal_pv_change/temporal_mask_registry.csv",
        "08_temporal_pv_change/temporal_synchrony_audit.csv",
    ]),
    (9, "Boundary-aware 2024 validation and temporal endpoint verification", "09_validate_segmentation_and_change.py", [
        "09_accuracy_validation/accuracy_validation_status.json",
        "09_accuracy_validation/segmentation_validation_summary.csv",
        "09_accuracy_validation/complete_reference_validation_by_unit.csv",
    ]),
    (10, "Route-aware full-chain uncertainty", "10_quantify_full_chain_uncertainty.py", [
        "10_uncertainty/uncertainty_status.json",
        "10_uncertainty/annual_area_uncertainty.csv",
    ]),
    (11, "Hybrid pixel/area PV dynamics", "11_summarize_pv_dynamics.py", [
        "11_pv_dynamics/pv_dynamics_status.json",
        "11_pv_dynamics/annual_pv_dynamics.csv",
        "11_pv_dynamics/national_pv_dynamics.csv",
        "11_pv_dynamics/unit_pv_dynamics_summary.csv",
    ]),
    (12, "Fixed-support LSMA-PAVG ecology and climate integration", "12_extract_dynamic_ecology.py", [
        "12_dynamic_ecology/dynamic_ecology_status.json",
        "12_dynamic_ecology/dynamic_pv_ecology_panel.csv",
        "12_dynamic_ecology/external_ecology_anomaly_panel.csv",
        "12_dynamic_ecology/pavg_annual_panel.csv",
        "12_dynamic_ecology/vegetation_area_panel.csv",
        "12_dynamic_ecology/vegetation_change_class_summary.csv",
        "12_dynamic_ecology/environmental_covariates_imputed.csv",
    ]),
    (13, "Object-level PV deployment transitions and qualification", "13_detect_pv_expansion_events.py", [
        "13_pv_events/pv_event_status.json",
        "13_pv_events/pv_expansion_events.csv",
    ]),
    (14, "Staggered event study and same-year controls", "14_run_conditional_association.py", [
        "14_conditional_association/conditional_association_status.json",
        "14_conditional_association/fixed_effect_associations.csv",
        "14_conditional_association/event_study_coefficients.csv",
        "14_conditional_association/event_study_pretrend_tests.csv",
        "14_conditional_association/propensity_matching_summary.csv",
    ]),
    (15, "Right-censored ecological response timing", "15_analyze_response_dynamics.py", [
        "15_response_dynamics/response_dynamics_status.json",
        "15_response_dynamics/consensus_response_dynamics.csv",
    ]),
    (16, "Backtested multisource 2050 growth forecasts and constrained spatial allocation", "16_generate_2050_envelopes.py", [
        "16_future_2050/future_2050_status.json",
        "16_future_2050/conditional_2050_candidate_summary.csv",
        "16_future_2050/conditional_2050_monte_carlo.csv",
        "16_future_2050/future_constraint_coverage.csv",
        "16_future_2050/area_forecast_model_diagnostics.csv",
    ]),
    (17, "Annual PV distribution maps and 15-year Nature-scale atlases", "17_generate_publication_maps.py", [
        "17_maps/map_status.json",
        "17_maps/map_manifest.csv",
        "17_maps/map_rendering_audit.csv",
    ]),
    (18, "Nature-scale publication figures", "18_generate_publication_figures.py", [
        "18_figures/figure_status.json",
        "18_figures/figure_manifest.csv",
        "18_figures/five_claim_metrics.csv",
        "18_figures/five_claim_narratives.json",
    ]),
    (19, "Spatial, semantic and claim qualification audit", "19_audit_outputs_and_claims.py", [
        "19_audit/audit_status.json",
        "19_audit/spatial_output_audit.csv",
        "19_audit/claim_qualification_audit.csv",
    ]),
    (20, "Chinese final report and completion manifest", "20_generate_chinese_report.py", [
        "20_report/report_status.json",
        "20_report/complete_research_report_CN.md",
        "20_report/complete_research_report_CN.html",
        "20_report/chinese_technical_route.png",
        "20_report/chinese_technical_route.pdf",
        "20_report/chinese_technical_route.svg",
        "20_report/chinese_technical_route_explanation.md",
        "20_report/all_output_sha256_manifest.csv",
    ]),
    (21, "Environment-augmented strict no-leakage hindcast and 2050 recursive projection", "21_run_strict_noleak_forecast.py", [
        "21_strict_noleak_forecast/noleak_forecast_status.json",
        "21_strict_noleak_forecast/tables/annual_all_regions_nature_ready.csv",
        "21_strict_noleak_forecast/validation/historical_hindcast_metrics_by_model_noleak.csv",
        "21_strict_noleak_forecast/audit/feature_leakage_audit_main.csv",
        "21_strict_noleak_forecast/forecast/future_projection_to_2050_global_mean_noleak.csv",
        "21_strict_noleak_forecast/report/pv_forecast_strict_noleak_report_cn.html",
    ]),
]


ALLOWED_STATUSES = {
    "passed", "ok", "completed", "completed_no_events",
    "completed_insufficient_models", "completed_diagnostic",
    "completed_with_scientific_gate_failure",
}


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expected(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return expected_counts(cfg)


def read_status(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_json(path: Path, _cfg: dict[str, Any]) -> bool:
    try:
        return str(read_status(path).get("status", "")).lower() in ALLOWED_STATUSES
    except Exception:
        return False


def _table_contract(path: Path, expected_rows: int | None, expected_units: int | None) -> list[str]:
    problems: list[str] = []
    try:
        frame = pd.read_csv(path)
        if expected_rows is not None and len(frame) != expected_rows:
            problems.append(f"{path.name}: rows {len(frame)} != {expected_rows}")
        if expected_units is not None:
            present = int(frame["Id"].nunique()) if "Id" in frame else 0
            if present != expected_units:
                problems.append(f"{path.name}: units {present} != {expected_units}")
    except Exception as exc:
        problems.append(f"{path.name}: coverage read failed: {exc}")
    return problems


def coverage_valid(step: int, output_root: Path, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    units, years, records = expected(cfg)
    scenarios = len(cfg["future_2050"]["scenarios"])
    problems: list[str] = []
    checks: dict[int, tuple[str, int | None, int | None]] = {
        1: ("01_input_validation/raster_inventory.csv", records, units),
        2: ("02_aligned_30m/aligned_raster_registry.csv", records, units),
        3: ("03_supervision/supervision_registry.csv", units, units),
        4: ("04_unit_quality/unit_quality_registry.csv", units, units),
        7: ("07_probability_calibration/calibrated_inference_registry.csv", records, units),
        8: ("08_temporal_pv_change/temporal_mask_registry.csv", records, units),
        # Stage 9 contains only independent positive-bearing test units; its
        # cohort size is checked explicitly below rather than forced to 142.
        10: ("10_uncertainty/annual_area_uncertainty.csv", records, units),
        11: ("11_pv_dynamics/annual_pv_dynamics.csv", records, units),
        12: ("12_dynamic_ecology/pavg_annual_panel.csv", records, units),
        13: ("13_pv_events/pv_expansion_events.csv", units, units),
        16: ("16_future_2050/conditional_2050_candidate_summary.csv", units * scenarios, units),
    }
    if step in checks:
        relative, nrows, nunits = checks[step]
        problems.extend(_table_contract(output_root / relative, nrows, nunits))

    if step == 3:
        try:
            registry = pd.read_csv(output_root / "03_supervision/supervision_registry.csv")
            split = pd.read_csv(output_root / "03_supervision/supervision_split.csv")
            if len(split) != units or split["Id"].nunique() != units:
                problems.append(f"Supervision split does not cover all configured units ({units})")
            roles = set(registry["split_role"].astype(str))
            smoke_relax = bool(
                test_mode_enabled(cfg)
                and cfg.get("test_mode", {}).get("relax_formal_production_contracts", False)
            )
            # Stage 4 legitimately rewrites Stage-3 roles after quality routing.
            # A tiny selected-unit smoke test may therefore have no formal
            # calibration role even though its existing supervision outputs are
            # valid. Calibration applies a selected-unit-only fallback. The
            # formal 142-unit contract remains unchanged.
            required_roles = (
                {"train"}
                if smoke_relax
                else ({"train", "calibration"} if test_mode_enabled(cfg) else {"train", "calibration", "test"})
            )
            missing = required_roles - roles
            if missing:
                problems.append(f"Missing positive-anchor split roles: {sorted(missing)}")
            supervision_status = read_status(output_root / "03_supervision/supervision_status.json")
            boundary_mode = bool(cfg["supervision"].get("boundary_uncertainty_enabled", False))
            complete_inventory = bool(cfg["supervision"].get("reference_year_complete_inventory_assumed", False))
            semantics_values = set(registry.get("outside_shp_semantics", pd.Series(dtype=str)).astype(str))
            if boundary_mode:
                if not bool(supervision_status.get("boundary_band_is_unlabelled", False)):
                    problems.append("Reference boundary-uncertainty pixels were not certified as unlabelled")
                if not bool(supervision_status.get("far_background_is_negative", False)):
                    problems.append("Far-background pixels were not certified as negative in complete-inventory mode")
                if semantics_values and semantics_values != {"boundary_uncertain_far_background_negative"}:
                    problems.append(f"Unexpected boundary-aware outside-SHP semantics: {sorted(semantics_values)}")
            elif complete_inventory:
                if not bool(supervision_status.get("far_background_is_negative", False)):
                    problems.append("Outside-SHP pixels were not certified as exact negatives in complete-inventory mode")
                if semantics_values and semantics_values != {"exact_negative_in_2024"}:
                    problems.append(f"Unexpected complete-inventory outside-SHP semantics: {sorted(semantics_values)}")
            else:
                if not bool(supervision_status.get("outside_is_fully_unlabelled", False)):
                    problems.append("Outside-SHP pixels were not certified as unlabelled in positive-only mode")
                if semantics_values and semantics_values != {"unlabelled_not_negative"}:
                    problems.append(f"Unexpected positive-only outside-SHP semantics: {sorted(semantics_values)}")
            positive = registry.loc[registry["has_mapped_pv"].astype(bool)]
            minima = cfg["supervision"]["split"]
            role_checks = [("train", "minimum_positive_train_units")]
            if not smoke_relax:
                role_checks.append(("calibration", "minimum_positive_calibration_units"))
                if not test_mode_enabled(cfg):
                    role_checks.append(("test", "minimum_positive_test_units"))
            for role, key in role_checks:
                present = int(positive.loc[positive["split_role"].eq(role), "Id"].nunique())
                required = 1 if test_mode_enabled(cfg) else int(minima[key])
                if present < required:
                    problems.append(f"Positive {role} units {present} < required {required}")
        except Exception as exc:
            problems.append(f"Supervision diagnostics failed: {exc}")

    if step == 4:
        try:
            quality = pd.read_csv(output_root / "04_unit_quality/unit_quality_registry.csv")
            direct = int(quality["direct_detection_eligible"].astype(bool).sum())
            pixel_mapping = int(
                quality.get(
                    "pixel_mapping_eligible",
                    quality["estimation_route"].astype(str).isin({"direct_pixel", "anchor_guided_candidate"}),
                ).astype(bool).sum()
            )
            smoke_relax = bool(
                test_mode_enabled(cfg)
                and cfg.get("test_mode", {}).get("relax_formal_production_contracts", False)
            )
            if direct < int(cfg["quality_routing"]["minimum_direct_units"]):
                if not (smoke_relax and pixel_mapping == units):
                    problems.append(
                        f"Direct-quality units {direct} below minimum; "
                        f"pixel-mapping units={pixel_mapping}/{units}"
                    )
            if quality["estimation_route"].isna().any():
                problems.append("One or more units have no estimation route")
            folds = pd.read_csv(output_root / "04_patch_dataset/patch_fold_summary.csv")
            if folds.empty or not folds["has_both_classes"].astype(bool).all():
                problems.append("One or more training spatial folds lack a class")
            audit = pd.read_csv(output_root / "04_patch_dataset/training_unit_audit.csv")
            if audit["Id"].nunique() != units:
                problems.append("Training audit does not cover all units")
        except Exception as exc:
            problems.append(f"Quality/training diagnostics failed: {exc}")

    if step == 7:
        try:
            raw = pd.read_csv(output_root / "07_annual_pv_inference/annual_inference_registry.csv")
            if len(raw) != records or raw["Id"].nunique() != units:
                problems.append("Raw annual inference coverage incomplete")
            calibration = read_status(output_root / "07_probability_calibration/probability_calibration_status.json")
            if bool(calibration.get("test_units_used", False)):
                problems.append("Independent test units leaked into calibration")
            if not bool(calibration.get("boundary_band_is_unlabelled", False)):
                problems.append("Boundary-aware calibration did not preserve an unlabelled reference boundary band")
            if not bool(calibration.get("far_outside_is_negative", False)):
                problems.append("Boundary-aware calibration did not retain far-background negatives")
            reconstruction = read_status(output_root / "07_authoritative_reference/authoritative_reference_status.json")
            if reconstruction.get("status") != "passed":
                problems.append("Authoritative-reference reconstruction did not pass")
            semantics = str(reconstruction.get("reference_semantics", "")).lower()
            bounded_certified = bool(
                reconstruction.get(
                    "bounded_boundary_refinement_certified",
                    "bounded boundary band" in semantics,
                )
            )
            multiband_certified = bool(
                reconstruction.get(
                    "multiband_break_support_certified",
                    "multiband break" in semantics,
                )
            )
            terminal_growth_certified = bool(
                reconstruction.get(
                    "terminal_2025_growth_front_certified",
                    "2025 expansion" in semantics,
                )
            )
            if not bounded_certified:
                problems.append("Bounded 2024 boundary-refinement semantics were not certified")
            if not multiband_certified:
                problems.append("Multiband change support was not certified for 2024 boundary refinement")
            if not terminal_growth_certified:
                problems.append("The 2025 terminal growth-front detector was not certified")
            if not bool(reconstruction.get("causal_reference_freeze_certified", False)):
                problems.append("The 2024 reference inventory was not certified as causally frozen before 2025")
            if bool(reconstruction.get("reference_refinement_uses_2025", True)):
                problems.append("The 2024 reference refinement used 2025 evidence")
            if not bool(reconstruction.get("terminal_2025_intrasite_perimeter_completion_enabled", False)):
                problems.append("The component-conditioned 2025 perimeter-extension pathway was not certified")
            if not bool(reconstruction.get("terminal_2025_novelty_gated_remote_change_enabled", False)):
                problems.append("The full-domain 2025 remote-census pathway was not certified")
            if str(reconstruction.get("terminal_2025_remote_census_distance_constraint", "")) != "none":
                problems.append("The remote-census pathway still depends on distance to the 2024 inventory")
            if bool(reconstruction.get("historical_anchor_copy", False)):
                problems.append("Reference geometry was copied into historical years")
            reference_registry = pd.read_csv(output_root / "07_authoritative_reference/authoritative_reference_registry.csv")
            if len(reference_registry) != records or reference_registry["Id"].nunique() != units:
                problems.append("Reference-constrained annual evidence coverage incomplete")
            terminal_columns = {
                "terminal_2025_expansion_source_path",
                "terminal_2025_probability_gain_path",
                "terminal_2025_site_likelihood_path",
                "terminal_2025_component_similarity_path",
                "terminal_2025_multicue_change_path",
                "terminal_2025_front_proximity_path",
                "terminal_2025_remote_census_likelihood_path",
                "terminal_2025_remote_change_support_path",
                "terminal_2025_remote_candidate_density_path",
                "terminal_2025_absolute_darkness_path",
            }
            missing_terminal_columns = sorted(terminal_columns - set(reference_registry.columns))
            if missing_terminal_columns:
                problems.append(
                    f"Reference registry missing terminal-extension diagnostics: {missing_terminal_columns}"
                )
            else:
                reference_rows = reference_registry.loc[
                    pd.to_numeric(reference_registry["year"], errors="coerce").eq(2024)
                ].copy()
                if "inference_mode" not in reference_rows.columns:
                    problems.append(
                        "Reference registry is missing inference_mode for pixel/area-only contract separation"
                    )
                    pixel_reference_rows = reference_rows.iloc[0:0]
                    area_only_reference_rows = reference_rows.iloc[0:0]
                else:
                    pixel_mask = reference_rows["inference_mode"].astype(str).isin(
                        {"direct_pixel", "anchor_guided_candidate"}
                    )
                    pixel_reference_rows = reference_rows.loc[pixel_mask]
                    area_only_reference_rows = reference_rows.loc[
                        reference_rows["inference_mode"].astype(str).eq("area_only_fallback")
                    ]
                    if len(pixel_reference_rows) + len(area_only_reference_rows) != len(reference_rows):
                        problems.append("Stage-7 registry contains an unknown inference route")
                for column in sorted(terminal_columns):
                    missing_paths = pixel_reference_rows[column].map(
                        lambda value: not Path(str(value)).is_file()
                    )
                    if missing_paths.any():
                        missing_ids = sorted(
                            pd.to_numeric(
                                pixel_reference_rows.loc[missing_paths, "Id"], errors="coerce"
                            ).dropna().astype(int).unique().tolist()
                        )
                        problems.append(
                            f"Missing Stage-7 terminal diagnostic rasters in column {column} "
                            f"for pixel-route units {missing_ids[:20]}"
                        )
                    fabricated = area_only_reference_rows[column].map(
                        lambda value: bool(str(value).strip()) and Path(str(value)).is_file()
                    )
                    if fabricated.any():
                        problems.append(
                            f"Area-only units unexpectedly contain fabricated Stage-7 rasters in {column}"
                        )
        except Exception as exc:
            problems.append(f"Inference/calibration diagnostics failed: {exc}")

    if step == 8:
        try:
            status = read_status(output_root / "08_temporal_pv_change/temporal_decoding_status.json")
            if not bool(status.get("no_data_is_never_decoded_as_pv", False)):
                problems.append("Temporal decoder did not certify NoData exclusion")
            if bool(status.get("test_units_anchored", False)):
                problems.append("Independent test units were anchored")
            if bool(status.get("area_only_pixel_maps_fabricated", True)):
                problems.append("Area-only fallback fabricated pixel maps")
            if (
                int(status.get("direct_pixel_units", 0))
                + int(status.get("anchor_guided_candidate_units", 0))
                + int(status.get("area_only_units", 0))
            ) != units:
                problems.append("Temporal routes do not cover all units")
            if not bool(status.get("cumulative_installed_state_is_persistent", False)):
                problems.append("Cumulative installed footprint was not certified as persistent")
            retirement_contract = bool(
                status.get(
                    "observed_active_state_allows_consecutive_negative_retirement",
                    status.get(
                        "observed_active_state_allows_only_multiyear_negative_retirement",
                        False,
                    ),
                )
            )
            if not retirement_contract:
                problems.append(
                    "Observed active footprint does not support "
                    "evidence-based retirement"
                )
            if not bool(status.get("image_quality_gaps_do_not_create_retirement", False)):
                problems.append("NoData/image-quality gaps can create false retirement")
            if not bool(status.get("preexisting_start_of_series_facilities_supported", False)):
                problems.append("Pre-series or start-of-series PV facilities are not supported")
            refined_endpoint_used = bool(
                status.get(
                    "refined_reference_inventory_used_in_reference_year",
                    status.get("reference_anchor_forced_in_reference_year", False),
                )
            )
            boundary_restricted = bool(
                status.get(
                    "reference_year_corrections_restricted_to_boundary_band",
                    status.get("reference_anchor_forced_in_reference_year", False),
                )
            )
            if not refined_endpoint_used:
                problems.append("The audited boundary-refined 2024 inventory was not used as the reference-year endpoint")
            if not boundary_restricted:
                problems.append("2024 corrections were not certified as restricted to the boundary band")
            if bool(status.get("reference_anchor_copied_to_historical_years_without_change_evidence", True)):
                problems.append("Reference anchor was copied backwards without temporal change evidence")
            if not bool(status.get("model_only_diagnostic_product_retained", False)):
                problems.append("Model-only diagnostic footprint was not retained for unbiased validation")
            growth_front_contract = str(status.get("schema_version", "")).startswith(("24.19", "24.20", "24.21", "24.22"))
            if growth_front_contract and not bool(status.get("historical_growth_front_completion_enabled", False)):
                problems.append("Historical full-extent growth-front completion was not certified")
            if growth_front_contract and not bool(status.get("terminal_2025_gain_aware_detection_enabled", False)):
                problems.append("Gain-aware 2025 expansion detection was not certified")
            if str(status.get("schema_version", "")).startswith(("24.20", "24.21", "24.22")):
                if not bool(status.get("terminal_2025_intrasite_perimeter_completion_enabled", False)):
                    problems.append("2025 intra-site perimeter completion was not certified")
                if not bool(status.get("terminal_2025_stage7_source_preserved", False)):
                    problems.append("Stage 8 did not preserve the exact Stage-7 2025 expansion source")
                if str(status.get("schema_version", "")).startswith(("24.21", "24.22")) and not bool(status.get("terminal_2025_novelty_gated_remote_change_preserved", False)):
                    problems.append("Stage 8 did not preserve full-domain remote-census source code 4")
                if not bool(status.get("reference_refinement_is_causal_to_2024", False)):
                    problems.append("Temporal decoding did not certify causal 2024 reference refinement")
            registry = pd.read_csv(output_root / "08_temporal_pv_change/temporal_mask_registry.csv")
            required_columns = {
                "active_mask_path", "cumulative_mask_path", "model_only_cumulative_mask_path",
                "observed_active_area_km2", "cumulative_installed_area_km2",
                "model_only_cumulative_installed_area_km2", "anchor_constrained_added_area_km2",
                "temporal_candidate_path", "morphology_removed_path",
                "valid_year_count_path", "temporal_exclusion_code_path",
                "anchor_backcast_path", "anchor_forced_reference_path",
                "anchor_backcast_object_fill_path", "anchor_reference_path",
            }
            if growth_front_contract:
                required_columns |= {
                    "historical_growth_front_rescue_path",
                    "terminal_2025_expansion_source_path",
                }
            missing_columns = sorted(required_columns - set(registry.columns))
            if missing_columns:
                problems.append(f"Temporal registry missing dual-state/diagnostic columns: {missing_columns}")
            else:
                active = pd.to_numeric(registry["observed_active_area_km2"], errors="coerce")
                cumulative = pd.to_numeric(registry["cumulative_installed_area_km2"], errors="coerce")
                model_only = pd.to_numeric(registry["model_only_cumulative_installed_area_km2"], errors="coerce")
                constrained_added = pd.to_numeric(registry["anchor_constrained_added_area_km2"], errors="coerce")
                if "anchor_constrained_removed_area_km2" in registry.columns:
                    constrained_removed = pd.to_numeric(
                        registry["anchor_constrained_removed_area_km2"],
                        errors="coerce",
                    )
                else:
                    constrained_removed = (model_only - cumulative).clip(lower=0.0)

                if ((active - cumulative) > 1e-9).any():
                    problems.append("Observed active area exceeds cumulative installed area")
                if (constrained_added < -1e-9).any():
                    problems.append("Anchor-constrained added area contains negative values")
                if (constrained_removed < -1e-9).any():
                    problems.append("Anchor-constrained removed area contains negative values")

                both_directions = (
                    (constrained_added > 1e-9)
                    & (constrained_removed > 1e-9)
                )
                if both_directions.any():
                    problems.append(
                        "Anchor reconciliation adds and removes area in "
                        "the same Id-year record"
                    )

                reconciled = (
                    model_only
                    + constrained_added
                    - constrained_removed
                )
                finite_reconciliation = cumulative.notna() & reconciled.notna()
                if (
                    (
                        reconciled.loc[finite_reconciliation]
                        - cumulative.loc[finite_reconciliation]
                    ).abs()
                    > 1e-8
                ).any():
                    problems.append(
                        "Anchor reconciliation identity is inconsistent "
                        "with final cumulative area"
                    )
                pixel = registry["estimation_mode"].astype(str).isin({"direct_pixel", "anchor_guided_candidate"})
                area_only = registry["estimation_mode"].astype(str).eq("area_only_fallback")
                if int(pixel.sum() + area_only.sum()) != len(registry):
                    problems.append("Temporal registry contains an unknown estimation mode")
                for column in ["active_mask_path", "cumulative_mask_path", "model_only_cumulative_mask_path"]:
                    missing_paths = registry.loc[pixel, column].map(lambda value: not Path(str(value)).exists())
                    if missing_paths.any():
                        problems.append(f"{column} missing for {int(missing_paths.sum())} pixel-mapped records")
                    fabricated = registry.loc[area_only, column].map(
                        lambda value: bool(str(value).strip()) and Path(str(value)).exists()
                    )
                    if fabricated.any():
                        problems.append(f"{column} was fabricated for {int(fabricated.sum())} area-only records")
                expected_area_only_ids = {
                    int(value)
                    for value in status.get("explicit_stage7_area_only_unit_ids", [])
                }
                actual_area_only_ids = set(
                    pd.to_numeric(registry.loc[area_only, "Id"], errors="coerce")
                    .dropna().astype(int).unique().tolist()
                )
                if expected_area_only_ids and not expected_area_only_ids.issubset(actual_area_only_ids):
                    problems.append(
                        f"Explicit area-only units missing after Stage 8: "
                        f"{sorted(expected_area_only_ids - actual_area_only_ids)}"
                    )
        except Exception as exc:
            problems.append(f"Temporal diagnostics failed: {exc}")

    if step == 9:
        try:
            status = read_status(output_root / "09_accuracy_validation/accuracy_validation_status.json")
            if not bool(status.get("boundary_excluded_model_metrics_identifiable", False)):
                problems.append("Boundary-aware model validation metrics were not certified")
            if not bool(status.get("final_refined_inventory_exact_by_contract", False)):
                problems.append("Final 2024 product did not certify exact agreement with the audited refined inventory")
            if not bool(status.get("reference_boundary_refinement_audited", False)):
                problems.append("2024 reference boundary corrections were not audited")
            frame = pd.read_csv(output_root / "09_accuracy_validation/complete_reference_validation_by_unit.csv")
            minimum_units = int(cfg["validation"].get("minimum_test_positive_units", 3))
            if frame.empty or frame["Id"].nunique() < minimum_units:
                problems.append("Too few independent complete-census validation units")
            expected_metrics = {
                "precision", "recall", "f1", "iou", "pr_auc", "roc_auc",
                "boundary_f1_1px", "false_positive_rate", "area_relative_error",
                "final_2024_exact_match", "final_2024_iou", "trusted_core_recall",
                "reference_adjustment_fraction", "corrections_outside_boundary_band",
            }
            missing_metrics = sorted(expected_metrics - set(frame.columns))
            if missing_metrics:
                problems.append(f"Complete-reference validation metrics missing: {missing_metrics}")
            else:
                exact = pd.to_numeric(frame["final_2024_exact_match"], errors="coerce").dropna()
                exact_iou = pd.to_numeric(frame["final_2024_iou"], errors="coerce").dropna()
                if exact.empty or not np.allclose(exact.to_numpy(), 1.0, atol=1e-12):
                    problems.append("Final 2024 map is not an exact pixelwise match to the audited refined inventory")
                if exact_iou.empty or not np.allclose(exact_iou.to_numpy(), 1.0, atol=1e-12):
                    problems.append("Final 2024 map IoU against the audited refined inventory is below 1.000")
        except Exception as exc:
            problems.append(f"Scientific qualification unreadable: {exc}")

    if step == 12:
        try:
            status = read_status(output_root / "12_dynamic_ecology/dynamic_ecology_status.json")
            if not bool(status.get("fixed_spatial_support", False)):
                problems.append("Ecological analysis did not use fixed spatial support")
            if not bool(status.get("pavg_method_synced_to_xia_2022", False)):
                problems.append("Physical LSMA-PAVG method was not certified")
            if not bool(status.get("low_quality_units_retained_without_fabricated_pixel_history", False)):
                problems.append("Low-quality units were not safely retained")
        except Exception as exc:
            problems.append(f"Ecology status unreadable: {exc}")

    if step == 14:
        try:
            status = str(read_status(output_root / "14_conditional_association/conditional_association_status.json").get("status", "")).lower()
            if status not in ALLOWED_STATUSES:
                problems.append(f"Unexpected association status: {status}")
        except Exception as exc:
            problems.append(f"Association status unreadable: {exc}")

    if step == 15:
        try:
            status = read_status(output_root / "15_response_dynamics/response_dynamics_status.json")
            frame = pd.read_csv(output_root / "15_response_dynamics/consensus_response_dynamics.csv")
            if str(status.get("status", "")).lower() == "completed_no_events" and not frame.empty:
                problems.append("Response table should be empty when no formal events exist")
        except Exception as exc:
            problems.append(f"Response status unreadable: {exc}")

    if step == 16:
        try:
            status = read_status(output_root / "16_future_2050/future_2050_status.json")
            if bool(status.get("area_only_units_have_spatial_maps", True)):
                problems.append("2050 spatial maps were fabricated for area-only units")
            if "not directly copied" not in str(status.get("reference_shp_role", "")):
                problems.append("2050 status does not certify change-modelled SHP use")
        except Exception as exc:
            problems.append(f"2050 forecast status unreadable: {exc}")

    if step == 21:
        try:
            status = read_status(output_root / "21_strict_noleak_forecast/noleak_forecast_status.json")
            if not bool(status.get("leakage_audit_passed", False)):
                problems.append("Strict no-leakage audit did not pass")
            if int(status.get("hindcast_records", 0)) <= 0:
                problems.append("Strict no-leakage hindcast produced no records")
            if int(status.get("forecast_records", 0)) <= 0:
                problems.append("Strict no-leakage 2050 forecast produced no records")
            metrics = pd.read_csv(output_root / "21_strict_noleak_forecast/validation/historical_hindcast_metrics_by_model_noleak.csv")
            if metrics.empty or "nrmse" not in metrics.columns:
                problems.append("No-leakage metrics are empty or missing NRMSE")
            audit = pd.read_csv(output_root / "21_strict_noleak_forecast/audit/feature_leakage_audit_main.csv")
            if audit.empty or not audit.get("audit_passed", pd.Series(False, index=audit.index)).astype(bool).all():
                problems.append("Main no-leakage feature audit failed")
        except Exception as exc:
            problems.append(f"Strict no-leakage forecast status unreadable: {exc}")

    if step == 17:
        try:
            manifest = pd.read_csv(output_root / "17_maps/map_manifest.csv")
            audit = pd.read_csv(output_root / "17_maps/map_rendering_audit.csv")
            if manifest.empty:
                problems.append("Map manifest is empty")
            expected_units = units
            expected_year_maps = units * years if bool(cfg.get("annual_visualization", {}).get("write_individual_year_maps", True)) else 0
            expected_atlases = units if bool(cfg.get("annual_visualization", {}).get("write_annual_atlas", True)) else 0
            actual_year_maps = int(manifest.get("map_type", pd.Series(dtype=str)).eq("annual_pv_change").sum())
            actual_atlases = int(manifest.get("map_type", pd.Series(dtype=str)).eq("annual_pv_change_atlas").sum())
            if actual_year_maps != expected_year_maps:
                problems.append(f"Expected {expected_year_maps} annual maps, found {actual_year_maps}")
            if actual_atlases != expected_atlases:
                problems.append(f"Expected {expected_atlases} annual atlases, found {actual_atlases}")
            if audit.empty or int(audit.get("Id", pd.Series(dtype=int)).nunique()) != expected_units or not audit.get("contract_pass", pd.Series(False, index=audit.index)).astype(bool).all():
                problems.append("One or more map rendering contracts failed")
            if not audit.empty:
                if audit.get("reference_shp_drawn", pd.Series(True, index=audit.index)).astype(bool).any():
                    problems.append("Reference SHP was drawn on one or more publication maps")
                if not audit.get("dual_state_visualization", pd.Series(False, index=audit.index)).astype(bool).all():
                    problems.append("Dual cumulative/active state visualization is incomplete")
                if not audit.get("colourblind_safe_high_contrast", pd.Series(False, index=audit.index)).astype(bool).all():
                    problems.append("High-contrast colourblind-safe map palette was not certified")
                if not audit.get("single_exclusive_class_layer", pd.Series(False, index=audit.index)).astype(bool).all():
                    problems.append("Annual maps do not use one mutually exclusive PV class layer")
                if not audit.get("no_stacked_pv_layers", pd.Series(False, index=audit.index)).astype(bool).all():
                    problems.append("One or more annual maps still stack PV overlays")
                if not audit.get("area_annotations_outside_map_axes", pd.Series(False, index=audit.index)).astype(bool).all():
                    problems.append("Annual area labels were not placed outside the map axes")
        except Exception as exc:
            problems.append(f"Map publication audit failed: {exc}")

    if step == 18:
        try:
            manifest = pd.read_csv(output_root / "18_figures/figure_manifest.csv")
            if len(manifest) < 8:
                problems.append(f"Expected at least 8 publication figures, found {len(manifest)}")
            for column in ["png", "pdf", "svg"]:
                if column not in manifest or not manifest[column].map(lambda value: Path(str(value)).exists()).all():
                    problems.append(f"Figure {column} exports are incomplete")
        except Exception as exc:
            problems.append(f"Figure manifest unreadable: {exc}")

    return not problems, problems


def contract_valid(output_root: Path, required: list[str], step: int, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for relative in required:
        path = output_root / relative
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {relative}")
        elif path.suffix == ".json" and not valid_json(path, cfg):
            problems.append(f"semantic status failed: {relative}")
    _, coverage_problems = coverage_valid(step, output_root, cfg)
    problems.extend(coverage_problems)
    return not problems, problems


def marker_path(output_root: Path, step: int) -> Path:
    directory = output_root / ".step_markers"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"step_{step:02d}_completed.json"


def write_effective_config(cfg: dict[str, Any], output_root: Path) -> Path:
    path = output_root / "effective_analysis_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="v24.11 anchor-constrained Landsat Object-PAVG and strict no-leakage inland-desert PV analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--strict-from-step", action="store_true")
    parser.add_argument("--allow-scientific-gate-failure", action="store_true")
    args = parser.parse_args()

    source_config = Path(args.config).resolve()
    cfg = load_cfg(source_config)
    if args.allow_scientific_gate_failure:
        cfg["processing"]["allow_scientific_gate_failure"] = True
    output_root = Path(cfg["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["log_root"]).mkdir(parents=True, exist_ok=True)
    effective_config = write_effective_config(cfg, output_root)
    code_root = Path(__file__).resolve().parent
    requested = max(1, min(len(STAGES), int(args.from_step)))

    lock_path = Path(cfg["paths"]["project_root"]) / "pv_anchor_backcast_temporal_ccdc_sma_v24_10.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Another v24 pipeline process holds the execution lock") from exc
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()

    earliest_missing: int | None = None
    for step, _, _, required in STAGES[: requested - 1]:
        ok, _ = contract_valid(output_root, required, step, cfg)
        if not ok:
            earliest_missing = step
            break
    if earliest_missing is not None:
        if args.strict_from_step:
            raise SystemExit(f"Requested step {requested}, but upstream step {earliest_missing} is incomplete")
        effective = earliest_missing
        print(f"Requested step {requested}; effective start changed to {effective} because upstream outputs are incomplete", flush=True)
    else:
        effective = requested

    status_path = output_root / "pipeline_status.json"
    status: dict[str, Any] = {
        "status": "running", "requested_from_step": requested, "effective_from_step": effective,
        "current_step": None, "started_utc": utc_now(), "pid": os.getpid(),
        "host": platform.node(), "python": sys.executable,
        "pipeline_version": cfg["project"]["version"], "run_mode": cfg["processing"].get("run_mode"),
        "reference_year": int(cfg["study"]["reference_year"]),
        "reference_shp_semantics": "pv_desert_fix.shp is a high-confidence 2024 manual prior; only a narrow audited boundary band can be changed by causal 2011-2024 multiband and temporal-break evidence; 2025 is excluded from reference refinement and analysed only after the 2024 endpoint is frozen",
    }
    atomic_json(status, status_path)

    try:
        upstream_changed = False
        for step, title, script, required in STAGES:
            if step < effective:
                continue
            ok, _ = contract_valid(output_root, required, step, cfg)
            should_execute = bool(args.rerun_completed or upstream_changed or not ok)
            if not should_execute:
                marker = marker_path(output_root, step)
                if not marker.exists():
                    atomic_json({"step": step, "title": title, "completed_utc": utc_now(), "elapsed_seconds": 0.0, "required_outputs": required, "mode": "validated_existing_outputs"}, marker)
                print(f"[{step}/{len(STAGES)}] {title}: existing outputs passed contracts", flush=True)
                continue

            marker_path(output_root, step).unlink(missing_ok=True)
            (output_root / "pipeline_completion_receipt.json").unlink(missing_ok=True)
            print("\n" + "─" * 100, flush=True)
            print(f"[{step}/{len(STAGES)}] {title}\nScript: {script}", flush=True)
            print("─" * 100, flush=True)
            status.update({"current_step": step, "current_title": title, "status": "running", "last_update_utc": utc_now()})
            status.pop("failed_step", None); status.pop("contract_problems", None)
            atomic_json(status, status_path)

            start = time.time()
            command = [sys.executable, "-u", str(code_root / "workflows" / script), "--config", str(effective_config)]
            workers = str(max(1, int(cfg["processing"].get("workers", 1))))
            environment = {
                **os.environ,
                "PYTHONPATH": str(code_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "PYTHONUNBUFFERED": "1",
                "GDAL_CACHEMAX": os.environ.get("GDAL_CACHEMAX", str(int(cfg.get("resource_control", {}).get("gdal_cache_mb", 32)))),
                "VSI_CACHE": "FALSE",
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", workers),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", workers),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", workers),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", workers),
                "MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX", "2"),
                "MALLOC_TRIM_THRESHOLD_": os.environ.get("MALLOC_TRIM_THRESHOLD_", "65536"),
            }
            return_code = subprocess.call(command, env=environment)
            if return_code != 0:
                status.update({"status": "failed", "failed_step": step, "return_code": return_code, "failed_utc": utc_now()})
                atomic_json(status, status_path)
                raise subprocess.CalledProcessError(return_code, command)
            ok, problems = contract_valid(output_root, required, step, cfg)
            if not ok:
                status.update({"status": "failed", "failed_step": step, "contract_problems": problems, "failed_utc": utc_now()})
                atomic_json(status, status_path)
                raise RuntimeError(f"Step {step} completed but output contract failed: {problems}")
            atomic_json({"step": step, "title": title, "completed_utc": utc_now(), "elapsed_seconds": time.time() - start, "required_outputs": required, "mode": "executed"}, marker_path(output_root, step))
            upstream_changed = True
            print(f"Completed in {(time.time() - start) / 60:.2f} min", flush=True)

        final_problems: list[str] = []
        for step, title, _, required in STAGES:
            ok, problems = contract_valid(output_root, required, step, cfg)
            if not ok:
                final_problems.extend([f"step {step} ({title}): {problem}" for problem in problems])
        expected_markers = {f"step_{step:02d}_completed.json" for step, *_ in STAGES}
        actual_markers = {path.name for path in (output_root / ".step_markers").glob("step_*_completed.json")}
        missing_markers = sorted(expected_markers - actual_markers)
        if missing_markers:
            final_problems.append(f"missing completion markers: {missing_markers}")
        if final_problems:
            status.update({"status": "failed", "failed_step": "final_audit", "contract_problems": final_problems, "failed_utc": utc_now()})
            atomic_json(status, status_path)
            raise RuntimeError("Final multi-stage audit failed: " + "; ".join(final_problems[:20]))

        validation = read_status(output_root / "09_accuracy_validation/accuracy_validation_status.json")
        audit = read_status(output_root / "19_audit/audit_status.json")
        calibration = read_status(output_root / "07_probability_calibration/probability_calibration_status.json")
        temporal = read_status(output_root / "08_temporal_pv_change/temporal_decoding_status.json")
        quality = read_status(output_root / "04_unit_quality/unit_quality_status.json")
        receipt = {
            "status": "completed", "pipeline_version": cfg["project"]["version"], "completed_utc": utc_now(),
            "units": expected(cfg)[0], "years": expected(cfg)[1], "expected_unit_years": expected(cfg)[2],
            "completion_markers": len(STAGES), "all_stage_contracts_valid": True,
            "reference_year": int(cfg["study"]["reference_year"]),
            "reference_shp_semantics": "pv_desert_fix.shp provides the trusted 2024 site prior; the final endpoint uses bounded image-driven boundary refinement and historical years are reconstructed independently",
            "direct_pixel_units": int(quality.get("direct_pixel_units", 0)),
            "anchor_guided_candidate_units": int(quality.get("anchor_guided_candidate_units", 0)),
            "pixel_mapping_units": int(quality.get("pixel_mapping_units", 0)),
            "area_only_units": int(quality.get("area_only_units", 0)),
            "area_only_pixel_maps_fabricated": False,
            "scientific_gate_pass": bool(validation.get("segmentation_gate_pass", False)),
            "publication_eligible_core_detection": bool(audit.get("publication_eligible_core_detection", False)),
            "calibration_constraints_feasible": bool(calibration.get("threshold_constraints_feasible", False)),
            "temporal_synchrony_anomaly": bool(temporal.get("synchronous_anomaly", False)),
            "reference_shp_drawn_on_maps": False,
            "reference_anchor_preserved_in_2024_product": False,
            "reference_core_preserved_in_2024_product": True,
            "reference_boundary_refined_in_2024_product": True,
            "model_only_validation_product_retained": True,
            "temporal_signature_decoder": temporal.get("decoder"),
            "effective_config": str(effective_config), "effective_config_sha256": sha256_file(effective_config),
        }
        atomic_json(receipt, output_root / "pipeline_completion_receipt.json")
        status.update({
            "status": "completed", "current_step": len(STAGES), "completed_utc": utc_now(),
            "completion_markers": len(STAGES), "all_stage_contracts_valid": True,
            "scientific_gate_pass": receipt["scientific_gate_pass"],
            "publication_eligible_core_detection": receipt["publication_eligible_core_detection"],
            "direct_pixel_units": receipt["direct_pixel_units"],
            "anchor_guided_candidate_units": receipt["anchor_guided_candidate_units"],
            "pixel_mapping_units": receipt["pixel_mapping_units"],
            "area_only_units": receipt["area_only_units"],
        })
        atomic_json(status, status_path)
        verdict = "COMPLETED_AND_PUBLICATION_ELIGIBLE" if receipt["publication_eligible_core_detection"] else "COMPLETED_WITH_QUALIFIED_DIAGNOSTIC_RESULTS"
        print(f"\nPipeline completed: {len(STAGES)}/{len(STAGES)} stages, {expected(cfg)[0]} units, {expected(cfg)[2]} unit-years. Verdict: {verdict}", flush=True)
    except KeyboardInterrupt:
        status.update({"status": "interrupted", "interrupted_step": status.get("current_step"), "interrupted_utc": utc_now()})
        atomic_json(status, status_path)
        raise
    except Exception as exc:
        if status.get("status") != "failed":
            status.update({"status": "failed", "failed_step": status.get("current_step", "preflight"), "error_type": type(exc).__name__, "error": str(exc), "failed_utc": utc_now()})
            atomic_json(status, status_path)
        raise
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


if __name__ == "__main__":
    main()
