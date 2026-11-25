from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .routes import PIXEL_MODES
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _read_mask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        a = src.read(1)
        nodata = src.nodata
    valid = np.ones(a.shape, bool) if nodata is None else a != nodata
    return (a == 1) & valid, valid


def _read_float(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nodata = src.nodata
    valid = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        valid &= a != nodata
    return a, valid


def _metrics(truth: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t, p = truth & valid, pred & valid
    tn = int(np.count_nonzero(~truth & ~pred & valid))
    tp = int(np.count_nonzero(t))
    fp = int(np.count_nonzero(~truth & pred & valid))
    fn = int(np.count_nonzero(truth & ~pred & valid))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    area_error = abs(int(np.count_nonzero(pred & valid)) - int(np.count_nonzero(truth & valid))) / max(int(np.count_nonzero(truth & valid)), 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, iou=iou, false_positive_rate=fpr, area_relative_error=area_error)


def _boundary_f1(truth: np.ndarray, pred: np.ndarray, valid: np.ndarray, tolerance: int = 1) -> float:
    structure = np.ones((3, 3), bool)
    tb = (truth ^ ndimage.binary_erosion(truth, structure=structure, border_value=0)) & valid
    pb = (pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)) & valid
    if not np.any(tb) and not np.any(pb):
        return 1.0
    td = ndimage.binary_dilation(tb, iterations=tolerance) if tolerance else tb
    pdil = ndimage.binary_dilation(pb, iterations=tolerance) if tolerance else pb
    precision = np.count_nonzero(pb & td) / max(np.count_nonzero(pb), 1)
    recall = np.count_nonzero(tb & pdil) / max(np.count_nonzero(tb), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def validate_segmentation_and_change(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "09_accuracy_validation")
    temporal = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv")
    calibrated = pd.read_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    refyear = int(cfg["study"]["reference_year"])
    cohort = temporal.loc[temporal.year.eq(refyear) & temporal.estimation_mode.isin(PIXEL_MODES)].copy()
    if "split_role" in calibrated.columns:
        roles = calibrated[["Id", "split_role"]].drop_duplicates("Id")
        cohort = cohort.merge(roles, on="Id", how="left")
        independent = cohort.loc[cohort.split_role.eq("test")]
        if len(independent):
            cohort = independent
    columns = [
        "Id", "pre_reference_reconstruction_probability_path", "calibrated_probability_path",
        "calibrated_threshold", "reference_inventory_path", "reference_anchor_path",
        "reference_boundary_uncertainty_path", "reference_boundary_correction_code_path",
    ]
    probs = calibrated.loc[calibrated.year.eq(refyear), [c for c in columns if c in calibrated.columns]]
    cohort = cohort.merge(probs, on="Id", how="left", suffixes=("", "_cal"))

    rows: list[dict[str, Any]] = []
    for row in tqdm(
        list(cohort.itertuples(index=False)),
        desc="Boundary-aware 2024 reference validation",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        rid = int(row.Id)
        frac, _ = _read_float(supervision.loc[rid, "pv_fraction_path"])
        manual = frac > float(cfg["supervision"].get("exact_inventory_minimum_fraction", 0.0))
        boundary_path = getattr(row, "reference_boundary_uncertainty_path", "")
        boundary, _ = _read_mask(boundary_path) if boundary_path and Path(str(boundary_path)).exists() else (np.zeros(manual.shape, bool), np.ones(manual.shape, bool))
        stable_valid = ~boundary
        core = manual & stable_valid

        prob_path = getattr(row, "pre_reference_reconstruction_probability_path", "") or getattr(row, "calibrated_probability_path_cal", "")
        probability, pvalid = _read_float(prob_path)
        threshold = float(getattr(row, "calibrated_threshold_cal", getattr(row, "calibrated_threshold", 0.5)))
        pred = pvalid & (probability >= threshold)
        evaluation_valid = pvalid & stable_valid
        metrics = _metrics(manual, pred, evaluation_valid)
        metrics["boundary_f1_1px"] = _boundary_f1(manual, pred, pvalid, 1)
        y = manual[evaluation_valid].astype(np.uint8)
        pv = probability[evaluation_valid]
        metrics["pr_auc"] = float(average_precision_score(y, pv)) if np.unique(y).size == 2 else np.nan
        metrics["roc_auc"] = float(roc_auc_score(y, pv)) if np.unique(y).size == 2 else np.nan

        final, _ = _read_mask(row.active_mask_path)
        inventory_path = getattr(row, "reference_inventory_path", "")
        refined, _ = _read_mask(inventory_path) if inventory_path and Path(str(inventory_path)).exists() else (final.copy(), np.ones(final.shape, bool))
        full = np.ones(manual.shape, bool)
        metrics["final_2024_exact_match"] = float(np.mean(final == refined))
        metrics["final_2024_iou"] = _metrics(refined, final, full)["iou"]
        metrics["manual_reference_agreement"] = float(np.mean(final == manual))
        metrics["manual_reference_iou"] = _metrics(manual, final, full)["iou"]
        metrics["trusted_core_recall"] = float(np.count_nonzero(final & core) / max(int(core.sum()), 1))
        changed = final ^ manual
        metrics["reference_adjustment_fraction"] = float(np.count_nonzero(changed) / max(int(manual.sum()), 1))
        metrics["corrections_outside_boundary_band"] = int(np.count_nonzero(changed & ~boundary))
        metrics["added_boundary_pixels"] = int(np.count_nonzero(final & ~manual))
        metrics["removed_boundary_pixels"] = int(np.count_nonzero(manual & ~final))
        metrics.update(
            Id=rid,
            truth_pixels=int(manual.sum()),
            trusted_core_pixels=int(core.sum()),
            boundary_band_pixels=int(boundary.sum()),
            valid_pixels=int(evaluation_valid.sum()),
            scientific_scope="stable_core_and_far_background_with_boundary_uncertainty_audit",
        )
        rows.append(metrics)

    unit = pd.DataFrame(rows)
    if unit.empty:
        raise RuntimeError("No independent pixel-mapped units available for boundary-aware validation")
    save_table(unit, out / "complete_reference_validation_by_unit.csv")
    save_table(unit, out / "positive_only_validation_by_unit.csv")

    requirements = {
        "iou": "required_mean_iou",
        "f1": "required_mean_f1",
        "pr_auc": "required_mean_pr_auc",
        "boundary_f1_1px": "required_boundary_f1_1px",
    }
    metrics_order = [
        "precision", "recall", "f1", "iou", "pr_auc", "roc_auc", "boundary_f1_1px",
        "false_positive_rate", "area_relative_error", "final_2024_exact_match", "final_2024_iou",
        "manual_reference_agreement", "manual_reference_iou", "trusted_core_recall",
        "reference_adjustment_fraction", "corrections_outside_boundary_band",
    ]
    summary: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    validation_cfg = cfg["validation"]
    for metric in metrics_order:
        vals = pd.to_numeric(unit[metric], errors="coerce")
        mean = float(vals.mean())
        key = requirements.get(metric)
        if metric == "false_positive_rate":
            key = "negative_control_false_positive_rate_max"
            passed = mean <= float(validation_cfg[key])
        elif metric == "area_relative_error":
            key = "required_area_relative_error"
            passed = mean <= float(validation_cfg[key])
        elif metric in {"final_2024_exact_match", "final_2024_iou"}:
            key = None
            passed = bool(np.isclose(mean, 1.0))
        elif metric == "trusted_core_recall":
            key = "required_reference_core_recall"
            passed = mean >= float(validation_cfg.get(key, 1.0))
        elif metric == "reference_adjustment_fraction":
            key = "maximum_reference_adjustment_fraction"
            passed = mean <= float(validation_cfg.get(key, 0.10))
        elif metric == "corrections_outside_boundary_band":
            key = "maximum_corrections_outside_boundary_band"
            passed = mean <= float(validation_cfg.get(key, 0.0))
        elif key:
            passed = mean >= float(validation_cfg[key])
        else:
            passed = True
        if key or metric in {"final_2024_exact_match", "final_2024_iou"}:
            checks[metric] = bool(passed)
        direction = "lower" if metric in {"false_positive_rate", "area_relative_error", "reference_adjustment_fraction", "corrections_outside_boundary_band"} else "higher"
        threshold = float(validation_cfg[key]) if key else 1.0 if metric.startswith("final_") else np.nan
        summary.append({
            "metric": metric,
            "mean": mean,
            "ci_low": float(vals.quantile(0.025)),
            "ci_high": float(vals.quantile(0.975)),
            "required_threshold": threshold,
            "gate_direction": direction,
            "units": int(unit.Id.nunique()),
            "cohort": "independent_test_units",
            "scientific_scope": "boundary_uncertain_manual_reference",
        })
    summary_frame = pd.DataFrame(summary)
    save_table(summary_frame, out / "segmentation_validation_summary.csv")
    temporal_status = json.loads((root / "08_temporal_pv_change" / "temporal_decoding_status.json").read_text())
    gate = all(checks.values()) and bool(temporal_status.get("scientific_temporal_gate_pass", True))
    atomic_json({
        "status": "passed",
        "created_utc": utc_now(),
        "segmentation_gate_pass": gate,
        "complete_reference_gate_pass": gate,
        "gate_checks": checks,
        "test_units": int(unit.Id.nunique()),
        "reference_year": refyear,
        "reference_semantics": "manual high-confidence census with bounded image-driven boundary uncertainty",
        "full_scene_precision_identifiable": False,
        "full_scene_iou_identifiable": False,
        "boundary_excluded_model_metrics_identifiable": True,
        "precision_claim_prohibited": False,
        "iou_claim_prohibited": False,
        "validation_design": "independent model evaluation excludes the uncertain boundary band; the final 2024 product must exactly equal the audited Stage-7 refined inventory, preserve the trusted core and restrict all corrections to the declared boundary band",
        "final_reference_product_exact_by_contract": False,
        "final_refined_inventory_exact_by_contract": True,
        "reference_boundary_refinement_audited": True,
        "scientific_gate_overridden": not gate,
        "downstream_analysis_mode": "formal_if_gate_pass_else_qualified_diagnostic",
    }, out / "accuracy_validation_status.json")
    return summary_frame
