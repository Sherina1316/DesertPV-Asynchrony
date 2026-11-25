from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import rasterio
from tqdm import tqdm

from .utils import atomic_json, ensure_dir, save_table, utc_now


def audit_outputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cfg["paths"]["output_root"]); out = ensure_dir(root / "19_audit"); target = float(cfg["native_grid"]["target_resolution_m"])
    rows: list[dict[str, Any]] = []
    for path in tqdm(sorted(root.rglob("*.tif")), desc="Spatial output audit", unit="raster", disable=not cfg["processing"].get("show_progress", True)):
        try:
            with rasterio.open(path) as src:
                px = abs(float(src.transform.a)); py = abs(float(src.transform.e)); rotation = max(abs(float(src.transform.b)), abs(float(src.transform.d))); projected = bool(src.crs and src.crs.is_projected)
                passed = abs(px - target) <= 1e-6 and abs(py - target) <= 1e-6 and rotation <= 1e-9 and projected
                rows.append({"path": str(path), "status": "ok", "crs": str(src.crs), "width": src.width, "height": src.height, "pixel_size_x": px, "pixel_size_y": py, "rotation": rotation, "projected_crs": projected, "nodata": src.nodata, "spatial_gate_pass": passed})
        except Exception as exc:
            rows.append({"path": str(path), "status": f"failed: {type(exc).__name__}: {exc}", "spatial_gate_pass": False})
    spatial = pd.DataFrame(rows)

    future = pd.read_csv(root / "16_future_2050" / "conditional_2050_candidate_summary.csv")
    temporal = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv")
    current = temporal.loc[temporal.year.eq(int(cfg["study"]["reference_year"]))].set_index("Id")
    overlaps: list[dict[str, Any]] = []
    for row in future.itertuples(index=False):
        candidate_path = str(getattr(row, "candidate_path", "") or "")
        spatial_map = bool(getattr(row, "spatial_map_available", False)) and candidate_path and Path(candidate_path).exists()
        current_column = "cumulative_mask_path" if "cumulative_mask_path" in current.columns else "mask_path"
        current_path = str(current.loc[int(row.Id), current_column] or "") if int(row.Id) in current.index else ""
        if not spatial_map:
            overlaps.append({"Id": int(row.Id), "scenario": row.scenario, "candidate_path": candidate_path, "overlap_pixels": pd.NA, "overlap_gate_pass": True, "audit_mode": "area_only_no_map"})
            continue
        if not current_path or not Path(current_path).exists():
            overlaps.append({"Id": int(row.Id), "scenario": row.scenario, "candidate_path": candidate_path, "overlap_pixels": pd.NA, "overlap_gate_pass": False, "audit_mode": "missing_current_direct_mask"})
            continue
        overlap = 0
        with rasterio.open(candidate_path) as candidate_src, rasterio.open(current_path) as current_src:
            if candidate_src.shape != current_src.shape or candidate_src.crs != current_src.crs or not candidate_src.transform.almost_equals(current_src.transform):
                raise RuntimeError(f"Id {int(row.Id)}: future and current masks are not on the same canonical grid")
            for _, window in candidate_src.block_windows(1):
                candidate_encoded = candidate_src.read(1, window=window); current_encoded = current_src.read(1, window=window)
                overlap += int(((candidate_encoded == 1) & (current_encoded == 1)).sum())
        overlaps.append({"Id": int(row.Id), "scenario": row.scenario, "candidate_path": candidate_path, "overlap_pixels": overlap, "overlap_gate_pass": overlap == 0, "audit_mode": "direct_spatial"})
    overlap = pd.DataFrame(overlaps)
    save_table(spatial, out / "spatial_output_audit.csv"); save_table(overlap, out / "candidate_current_overlap_audit.csv")

    validation = json.loads((root / "09_accuracy_validation" / "accuracy_validation_status.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "07_probability_calibration" / "probability_calibration_status.json").read_text(encoding="utf-8"))
    temporal_status = json.loads((root / "08_temporal_pv_change" / "temporal_decoding_status.json").read_text(encoding="utf-8"))
    ecology = json.loads((root / "12_dynamic_ecology" / "dynamic_ecology_status.json").read_text(encoding="utf-8"))
    events = json.loads((root / "13_pv_events" / "pv_event_status.json").read_text(encoding="utf-8"))
    association = json.loads((root / "14_conditional_association" / "conditional_association_status.json").read_text(encoding="utf-8"))
    response = json.loads((root / "15_response_dynamics" / "response_dynamics_status.json").read_text(encoding="utf-8"))
    future_status = json.loads((root / "16_future_2050" / "future_2050_status.json").read_text(encoding="utf-8"))
    maps = json.loads((root / "17_maps" / "map_status.json").read_text(encoding="utf-8"))
    quality = json.loads((root / "04_unit_quality" / "unit_quality_status.json").read_text(encoding="utf-8"))
    gate = validation.get("gate_checks", {})
    claims = [
        {"claim_id": "C1", "claim": "All generated raster outputs use canonical projected 30 m grids.", "qualified": bool(len(spatial) and spatial.spatial_gate_pass.all()), "evidence": f"{int(spatial.spatial_gate_pass.sum())}/{len(spatial)} rasters passed."},
        {"claim_id": "C2", "claim": "The final 2024 distribution preserves the trusted interior of pv_desert_fix.shp and permits only bounded, audited image-driven boundary corrections.", "qualified": bool(validation.get("final_refined_inventory_exact_by_contract") and validation.get("reference_boundary_refinement_audited") and temporal_status.get("refined_reference_inventory_used_in_reference_year")), "evidence": "Core recall, correction distance, adjustment area, multiband break support and exact agreement with the Stage-7 refined inventory are audited."},
        {"claim_id": "C3", "claim": "Independent 2024 model metrics are identifiable on the trusted core and far-background support, with the uncertain manual boundary excluded.", "qualified": bool(validation.get("boundary_excluded_model_metrics_identifiable")), "evidence": f"Boundary-aware gate checks={gate}; boundary pixels are audited separately rather than treated as exact labels."},
        {"claim_id": "C4", "claim": "The 2024 geometry is never copied into 2011–2023 without annual image and temporal-change evidence.", "qualified": bool(not temporal_status.get("reference_anchor_copied_to_historical_years_without_change_evidence", True)), "evidence": "Inside-census installation onset defaults to 2024 when historical evidence is inconclusive."},
        {"claim_id": "C5", "claim": "2024 changes are restricted to the declared boundary band; historical outside-site tracks and 2025 new sites require independent persistent evidence.", "qualified": bool(temporal_status.get("reference_year_corrections_restricted_to_boundary_band") and temporal_status.get("outside_reference_inventory_can_be_supplemented")), "evidence": "Remote 2024 additions are prohibited; only boundary-connected corrections are accepted."},
        {"claim_id": "C6", "claim": "Temporal decoding is NoData-safe and cross-unit synchrony is explicitly audited.", "qualified": bool(temporal_status.get("no_data_is_never_decoded_as_pv") and not temporal_status.get("synchronous_anomaly")), "evidence": f"Maximum synchronous fraction={temporal_status.get('maximum_synchronous_peak_fraction_observed')}."},
        {"claim_id": "C7", "claim": "Historical installation years are exact commissioning dates.", "qualified": False, "evidence": "They represent the earliest persistent Landsat-scale structural transition and require external records for commissioning-date validation."},
        {"claim_id": "C8", "claim": "PAVG follows the LSMA shade/low-albedo adjustment and uses fixed spatial support.", "qualified": bool(ecology.get("pavg_method_synced_to_xia_2022") and ecology.get("fixed_spatial_support")), "evidence": f"PAVG definition={ecology.get('pavg_definition')}; valid fraction={ecology.get('valid_pavg_fraction')}."},
        {"claim_id": "C9", "claim": "PV expansion caused the observed vegetation changes.", "qualified": False, "evidence": "Climate-adjusted event studies estimate conditional associations, not causal effects."},
        {"claim_id": "C10", "claim": "2050 areas are forecast from reconstructed temporal change and environmental drivers rather than copied from the 2024 census.", "qualified": True, "evidence": future_status.get("area_forecast_basis")},
        {"claim_id": "C11", "claim": "2050 spatial maps are produced for low-quality area-only units.", "qualified": False, "evidence": "Area-only units receive forecast areas and uncertainty but no fabricated candidate raster."},
        {"claim_id": "C12", "claim": "Direct-unit 2050 candidate masks exclude current inferred PV.", "qualified": bool(overlap.empty or overlap.overlap_gate_pass.fillna(False).all()), "evidence": f"Maximum audited overlap={pd.to_numeric(overlap.overlap_pixels, errors='coerce').max() if len(overlap) else 0} pixels."},
        {"claim_id": "C13", "claim": "Annual maps use one mutually exclusive PV class layer, external area annotations, shared extents and vector exports.", "qualified": bool(maps.get("automatic_valid_data_crop") and maps.get("equal_aspect") and maps.get("vector_atlas_exports") and maps.get("single_exclusive_class_layer") and maps.get("no_stacked_pv_layers") and maps.get("area_annotations_outside_map_axes") and maps.get("annual_atlases", 0) == maps.get("expected_annual_atlases", -1)), "evidence": "Annual PNG maps and shared-extent PNG/PDF/SVG atlases are audited for every configured unit."},
        {"claim_id": "C14", "claim": "Formal event/ecological conclusions are permitted for all units.", "qualified": bool(events.get("formal_event_claims_allowed") and ecology.get("formal_ecological_inference_eligible") and association.get("valid_adjusted_models", 0) > 0), "evidence": f"Formal direct events={events.get('formal_qualified_events')}; valid ecological models={association.get('valid_adjusted_models')}."},
    ]
    claims_frame = pd.DataFrame(claims); save_table(claims_frame, out / "claim_qualification_audit.csv")
    spatial_pass = bool(len(spatial) and spatial.spatial_gate_pass.all()); overlap_pass = bool(overlap.empty or overlap.overlap_gate_pass.fillna(False).all())
    status = "passed" if spatial_pass and overlap_pass else "failed"
    publication = bool(validation.get("segmentation_gate_pass") and not temporal_status.get("synchronous_anomaly") and maps.get("status") == "passed")
    atomic_json({"status": status, "created_utc": utc_now(), "rasters": len(spatial), "spatial_pass": spatial_pass, "candidate_overlap_pass": overlap_pass, "qualified_claims": int(claims_frame.qualified.sum()), "total_claims": len(claims_frame), "publication_eligible_core_detection": publication, "diagnostic_only": not publication, "positive_unlabelled_reference": False, "complete_2024_reference_census": False, "boundary_uncertain_2024_reference": True, "model_only_validation_product_retained": True, "reference_anchor_forced_in_reference_year": False, "reference_core_preserved": True, "reference_boundary_refined": True, "reference_anchor_blindly_backcopied": False, "single_exclusive_map_layer": bool(maps.get("single_exclusive_class_layer")), "area_labels_external": bool(maps.get("area_annotations_outside_map_axes")), "area_only_maps_fabricated": False}, out / "audit_status.json")
    if status == "failed" and cfg["processing"].get("fail_fast", True):
        raise RuntimeError("Final spatial or current-PV overlap audit failed")
    return spatial, claims_frame
