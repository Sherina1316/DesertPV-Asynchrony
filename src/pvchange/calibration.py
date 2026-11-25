from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import joblib
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy import ndimage
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .anchor_guided import (
    anchor_mask_for_window,
    buffered_support_mask,
    classify_anchor_guided,
    select_anchor_geometry,
)
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _windows(width: int, height: int, size: int = 512) -> Iterator[Window]:
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


@dataclass
class Reservoir:
    capacity: int
    rng: np.random.Generator
    seen: int = 0
    values: np.ndarray | None = None
    priorities: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.values = np.empty(0, dtype=np.float32)
        self.priorities = np.empty(0, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Uniform streaming sample using independent random priorities.

        This is vectorised: every observed pixel receives one independent key and
        the reservoir retains the ``capacity`` smallest keys.  It is exactly a
        uniform sample without the per-pixel Python loop used in legacy versions.
        """
        data = np.asarray(x, dtype=np.float32).reshape(-1)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return
        self.seen += int(data.size)
        assert self.values is not None and self.priorities is not None
        keys = self.rng.random(data.size)
        values = np.concatenate([self.values, data])
        priorities = np.concatenate([self.priorities, keys])
        if values.size > self.capacity:
            keep = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            values = values[keep]
            priorities = priorities[keep]
        self.values = values.astype(np.float32, copy=False)
        self.priorities = priorities.astype(np.float64, copy=False)

    def array(self) -> np.ndarray:
        assert self.values is not None
        return self.values.copy()


def _logit(p: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _design(method: str, p: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    if method == "platt":
        return _logit(x).reshape(-1, 1)
    if method == "beta":
        return np.column_stack([np.log(x), -np.log1p(-x)])
    raise ValueError(method)


def _fit_candidate(method: str, p: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> dict[str, Any]:
    if method == "identity":
        return {"method": method, "model": None}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y, sample_weight=w)
        return {"method": method, "model": model}
    if method in {"platt", "beta"}:
        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=500, random_state=seed
        )
        model.fit(_design(method, p), y, sample_weight=w)
        return {"method": method, "model": model}
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(bundle: dict[str, Any], probability: np.ndarray) -> np.ndarray:
    method = str(bundle["method"])
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    if method == "identity":
        out = p
    elif method == "isotonic":
        out = bundle["model"].predict(p.reshape(-1)).reshape(p.shape)
    else:
        out = bundle["model"].predict_proba(_design(method, p.reshape(-1)))[:, 1].reshape(p.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _weighted_ece(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 20) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(np.sum(w))
    if total <= 0:
        return float("nan")
    result = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        chosen = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not np.any(chosen):
            continue
        mass = float(np.sum(w[chosen]))
        observed = float(np.average(y[chosen], weights=w[chosen]))
        predicted = float(np.average(p[chosen], weights=w[chosen]))
        result += (mass / total) * abs(observed - predicted)
    return float(result)


def _weighted_confusion(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict[str, float]:
    truth = y.astype(bool)
    estimate = pred.astype(bool)
    return {
        "tp": float(np.sum(w[truth & estimate])),
        "tn": float(np.sum(w[~truth & ~estimate])),
        "fp": float(np.sum(w[~truth & estimate])),
        "fn": float(np.sum(w[truth & ~estimate])),
    }


def _threshold_row(y: np.ndarray, p: np.ndarray, w: np.ndarray, threshold: float) -> dict[str, float]:
    c = _weighted_confusion(y, p >= threshold, w)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1e-12)
    fpr = fp / max(fp + tn, 1e-12)
    area_error = abs((tp + fp) - (tp + fn)) / max(tp + fn, 1e-12)
    return {
        "threshold": float(threshold),
        "precision_against_hard_negatives": precision,
        "anchor_recall": recall,
        "anchor_hard_negative_f1": f1,
        "anchor_hard_negative_iou": iou,
        "hard_negative_fpr": fpr,
        # Kept only as a case-control diagnostic; it is not a full-area error
        # because the SHP is complete-reference and incomplete by design.
        "case_control_area_ratio_error": area_error,
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "negative_control_fpr": fpr, "area_relative_error": area_error,
        **c,
    }


def _choose_threshold(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, bool]:
    """Select a threshold using the complete 2024 inventory when available."""
    data = frame.copy()
    complete = bool(cfg.get("supervision", {}).get("reference_year_complete_inventory_assumed", False))
    if complete:
        min_precision = float(cfg["calibration"].get("minimum_precision", 0.70))
        min_recall = float(cfg["calibration"].get("minimum_anchor_recall", 0.85))
        max_fpr = float(cfg["calibration"].get("maximum_negative_fpr", 0.05))
        max_area = float(cfg["calibration"].get("maximum_area_relative_error", 0.20))
        data["feasible"] = (
            data["precision"].ge(min_precision)
            & data["recall"].ge(min_recall)
            & data["negative_control_fpr"].le(max_fpr)
            & data["area_relative_error"].le(max_area)
        )
        weights = cfg["calibration"].get("selection_weights", {})
        data["score"] = (
            float(weights.get("precision", 0.22)) * data["precision"]
            + float(weights.get("recall", 0.22)) * data["recall"]
            + float(weights.get("f1", 0.22)) * data["f1"]
            + float(weights.get("iou", 0.20)) * data["iou"]
            + float(weights.get("specificity", 0.08)) * (1.0 - data["negative_control_fpr"].clip(0, 1))
            + float(weights.get("area_accuracy", 0.06)) * (1.0 - data["area_relative_error"].clip(0, 1))
        )
        feasible = data.loc[data["feasible"]]
        if not feasible.empty:
            return feasible.sort_values(["score", "threshold"], ascending=[False, False]).iloc[0], True
        data["violation"] = (
            np.maximum(min_precision - data["precision"], 0) / max(min_precision, 1e-6)
            + np.maximum(min_recall - data["recall"], 0) / max(min_recall, 1e-6)
            + np.maximum(data["negative_control_fpr"] - max_fpr, 0) / max(max_fpr, 1e-6)
            + np.maximum(data["area_relative_error"] - max_area, 0) / max(max_area, 1e-6)
        )
        return data.sort_values(["violation", "score"], ascending=[True, False]).iloc[0], False

    minimum_recall = float(cfg["calibration"].get("minimum_anchor_recall", 0.90))
    maximum_fpr = float(cfg["calibration"].get("maximum_hard_negative_fpr", 0.05))
    data["feasible"] = data["anchor_recall"].ge(minimum_recall) & data["hard_negative_fpr"].le(maximum_fpr)
    data["score"] = 0.5 * data["anchor_recall"] + 0.3 * data["anchor_hard_negative_f1"] + 0.2 * (1 - data["hard_negative_fpr"].clip(0, 1))
    feasible = data.loc[data["feasible"]]
    if not feasible.empty:
        return feasible.sort_values(["score", "threshold"], ascending=[False, False]).iloc[0], True
    data["violation"] = np.maximum(minimum_recall-data["anchor_recall"],0)+np.maximum(data["hard_negative_fpr"]-maximum_fpr,0)
    return data.sort_values(["violation","score"],ascending=[True,False]).iloc[0], False

def _cleanup(mask: np.ndarray, minimum_component: int, maximum_hole: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if minimum_component > 1 and np.any(result):
        labels, count = ndimage.label(result)
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= minimum_component
        keep[0] = False
        result = keep[labels]
    if maximum_hole > 0 and np.any(result):
        holes = ~result
        labels, _ = ndimage.label(holes)
        sizes = np.bincount(labels.reshape(-1))
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        fill = (sizes <= maximum_hole)
        fill[border] = False
        result |= fill[labels]
    return result


def _collect_calibration_samples(cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(cfg["paths"]["output_root"])
    raw = pd.read_csv(root / "07_annual_pv_inference" / "annual_inference_registry.csv")
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv")[["Id", "direct_detection_eligible"]]

    # CSV round-trips can otherwise leave integer identifiers or booleans with
    # inconsistent dtypes, which makes a valid tiny selected-unit smoke test
    # appear to have no calibration/inference intersection.
    for frame_name, frame in (("annual inference", raw), ("supervision", supervision), ("unit quality", quality)):
        numeric_ids = pd.to_numeric(frame["Id"], errors="coerce")
        if numeric_ids.isna().any():
            raise RuntimeError(f"{frame_name} registry contains non-numeric unit identifiers")
        frame["Id"] = numeric_ids.astype(int)
    direct_raw = quality["direct_detection_eligible"]
    if not pd.api.types.is_bool_dtype(direct_raw):
        direct_raw = direct_raw.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    quality["direct_detection_eligible"] = direct_raw.fillna(False).astype(bool)
    supervision["split_role"] = supervision["split_role"].fillna("inference_only").astype(str).str.strip().str.lower()
    supervision = supervision.merge(quality, on="Id", how="left")

    reference_year = int(cfg["calibration"].get("reference_year", cfg["study"]["reference_year"]))
    raw = raw.loc[pd.to_numeric(raw["year"], errors="coerce").eq(reference_year)].copy()
    if "probability_path" in raw.columns:
        raw = raw.loc[raw["probability_path"].fillna("").astype(str).str.strip().ne("")]

    direct = supervision["direct_detection_eligible"].fillna(False).astype(bool)
    calibration_units = supervision.loc[supervision["split_role"].eq("calibration") & direct].copy()
    merged = calibration_units.merge(raw, on="Id", how="inner")
    calibration_role = "calibration"

    selected_ids = [int(value) for value in (cfg.get("study", {}).get("selected_unit_ids") or [])]
    test_mode = cfg.get("test_mode", {})
    allow_smoke_test_fallback = bool(selected_ids) and bool(test_mode.get("relax_formal_production_contracts", False))
    if merged.empty and allow_smoke_test_fallback:
        # Restricted to explicitly selected smoke-test units. Prefer formal
        # direct units, but permit another selected pixel-mapped unit when
        # quality-aware routing leaves no direct-calibration intersection. A
        # real reference-year probability raster and supervision label remain
        # mandatory. The formal 142-unit experiment never enters this branch.
        available_ids = set(raw["Id"].astype(int).tolist())
        selected_available = supervision.loc[
            supervision["Id"].astype(int).isin(selected_ids)
            & supervision["Id"].astype(int).isin(available_ids)
        ].copy()
        direct_selected = selected_available.loc[
            selected_available["direct_detection_eligible"].fillna(False).astype(bool)
        ].copy()
        fallback_units = direct_selected if not direct_selected.empty else selected_available
        merged = fallback_units.merge(raw, on="Id", how="inner")
        calibration_role = "selected_unit_smoke_test_fallback"
        if not merged.empty:
            chosen = sorted(merged["Id"].astype(int).unique().tolist())
            print(
                "[test-units] No formal direct calibration intersection; "
                f"using selected units {chosen} for smoke-test-only probability calibration."
            )

    if merged.empty:
        split_counts = supervision.loc[direct, "split_role"].value_counts().to_dict()
        inference_ids = sorted(raw["Id"].astype(int).unique().tolist())
        raise RuntimeError(
            "No calibration units intersect the reference-year inference registry; "
            f"direct split counts={split_counts}, reference-year inference units={inference_ids}, "
            f"selected units={selected_ids}"
        )

    audit: list[dict[str, Any]] = []
    sample_p: list[np.ndarray] = []
    sample_y: list[np.ndarray] = []
    sample_w: list[np.ndarray] = []
    seed = int(cfg["project"]["random_seed"])
    ignore = int(cfg["supervision"]["ignore_value"])
    max_pos = int(cfg["training"].get("calibration_max_positive_pixels_per_unit", 250000))
    max_neg = int(cfg["training"].get("calibration_max_negative_pixels_per_unit", 150000))

    for row in tqdm(list(merged.itertuples(index=False)), desc="Panoramic calibration sampling", unit="region", disable=not cfg["processing"].get("show_progress", True)):
        pos = Reservoir(max_pos, np.random.default_rng(seed + int(row.Id) * 11 + 1))
        neg = Reservoir(max_neg, np.random.default_rng(seed + int(row.Id) * 11 + 2))
        with rasterio.open(row.probability_path) as prob_src, rasterio.open(row.label_path) as label_src:
            if prob_src.shape != label_src.shape or prob_src.crs != label_src.crs or not prob_src.transform.almost_equals(label_src.transform):
                raise RuntimeError(f"Id {int(row.Id)}: calibration probability and label grids differ")
            for window in _windows(prob_src.width, prob_src.height):
                probability = prob_src.read(1, window=window).astype(np.float32)
                labels = label_src.read(1, window=window)
                valid = (probability > -9990) & (labels != ignore)
                pos.update(probability[valid & (labels == 1)])
                neg.update(probability[valid & (labels == 0)])
        p_pos, p_neg = pos.array(), neg.array()
        if pos.seen and len(p_pos):
            sample_p.append(p_pos); sample_y.append(np.ones(len(p_pos), dtype=np.uint8))
            sample_w.append(np.full(len(p_pos), pos.seen / len(p_pos), dtype=np.float64))
        if neg.seen and len(p_neg):
            sample_p.append(p_neg); sample_y.append(np.zeros(len(p_neg), dtype=np.uint8))
            sample_w.append(np.full(len(p_neg), neg.seen / len(p_neg), dtype=np.float64))
        audit.append({
            "Id": int(row.Id), "role": calibration_role, "positive_population": int(pos.seen),
            "negative_population": int(neg.seen), "positive_sample": int(len(p_pos)),
            "negative_sample": int(len(p_neg)), "natural_prevalence": pos.seen / max(pos.seen + neg.seen, 1),
        })
    if not sample_p:
        raise RuntimeError("Calibration sampling produced no reliable pixels")
    p = np.concatenate(sample_p)
    y = np.concatenate(sample_y)
    w = np.concatenate(sample_w)
    if np.unique(y).size < 2:
        raise RuntimeError("Calibration requires both positive and negative samples")
    max_total = int(cfg["training"].get("calibration_sample_max_pixels", 2000000))
    if len(y) > max_total:
        rng = np.random.default_rng(seed + 991)
        probability = w / np.sum(w)
        keep = rng.choice(len(y), max_total, replace=False, p=probability)
        p, y, w = p[keep], y[keep], w[keep]
    w = w / max(float(np.mean(w)), 1e-12)
    return pd.DataFrame(audit), p.astype(np.float32), y.astype(np.uint8), w.astype(np.float64)


def calibrate_annual_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "07_probability_calibration")
    model_root = ensure_dir(Path(cfg["paths"]["model_root"]) / "calibration")
    audit, raw_p, y, weights = _collect_calibration_samples(cfg)
    save_table(audit, out / "calibration_sampling_audit.csv")

    seed = int(cfg["project"]["random_seed"])
    candidate_rows: list[dict[str, Any]] = []
    candidate_bundles: dict[str, dict[str, Any]] = {}
    calibrated_values: dict[str, np.ndarray] = {}
    for method in cfg["calibration"]["candidate_methods"]:
        bundle = _fit_candidate(str(method), raw_p, y, weights, seed)
        calibrated = apply_calibrator(bundle, raw_p)
        candidate_bundles[str(method)] = bundle
        calibrated_values[str(method)] = calibrated
        brier = float(np.average((calibrated - y) ** 2, weights=weights))
        ece = _weighted_ece(y, calibrated, weights)
        pr_auc = float(average_precision_score(y, calibrated, sample_weight=weights))
        roc_auc = float(roc_auc_score(y, calibrated, sample_weight=weights))
        candidate_rows.append({"method": method, "weighted_brier": brier, "weighted_ece": ece, "weighted_pr_auc": pr_auc, "weighted_roc_auc": roc_auc})
    candidates = pd.DataFrame(candidate_rows)
    candidates["calibration_score"] = candidates["weighted_brier"] + candidates["weighted_ece"]
    chosen_method = str(candidates.sort_values(["calibration_score", "weighted_pr_auc"], ascending=[True, False]).iloc[0]["method"])
    bundle = candidate_bundles[chosen_method]
    calibrated = calibrated_values[chosen_method]
    save_table(candidates, out / "calibration_method_comparison.csv")

    grid = np.linspace(
        float(cfg["calibration"]["threshold_grid_start"]),
        float(cfg["calibration"]["threshold_grid_end"]),
        int(cfg["calibration"]["threshold_grid_count"]),
    )
    threshold_scan = pd.DataFrame([_threshold_row(y, calibrated, weights, float(t)) for t in grid])
    selected, feasible = _choose_threshold(threshold_scan, cfg)
    threshold_scan["selected"] = np.isclose(threshold_scan["threshold"], float(selected["threshold"]))
    save_table(threshold_scan, out / "panoramic_threshold_scan.csv")
    threshold = float(selected["threshold"])

    bundle_path = model_root / "probability_calibrator.joblib"
    joblib.dump(bundle, bundle_path)
    raw_registry = pd.read_csv(root / "07_annual_pv_inference" / "annual_inference_registry.csv").sort_values(["Id", "year"])
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    reference_shape = gpd.read_file(cfg["paths"]["reference_shp"])
    calibrated_root = ensure_dir(out / "calibrated_probability_30m")
    mask_root = ensure_dir(out / "calibrated_raw_masks_30m")
    candidate_class_root = ensure_dir(out / "calibrated_candidate_class_30m")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    unit_registry_root = ensure_dir(out / "unit_registries")
    rows: list[pd.DataFrame] = []
    mask_nodata = int(cfg["calibration"].get("mask_nodata", 255))

    for region_id, group in tqdm(list(raw_registry.groupby("Id")), desc="Applying complete-inventory probability calibration", unit="region", disable=not cfg["processing"].get("show_progress", True)):
        region_id = int(region_id)
        group = group.sort_values("year")
        if "inference_mode" in group and group["inference_mode"].astype(str).eq("area_only_fallback").all():
            fallback = group.copy()
            fallback["raw_probability_path"] = ""
            fallback["calibrated_probability_path"] = ""
            fallback["calibrated_mask_path"] = ""
            fallback["calibrated_candidate_class_path"] = ""
            fallback["calibration_method"] = "not_applicable_area_only"
            fallback["calibrated_threshold"] = threshold
            fallback["calibrated_predicted_pixels"] = 0
            fallback["calibrated_predicted_area_km2"] = np.nan
            rows.append(fallback)
            continue
        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
        if checkpoint.exists() and registry_path.exists() and not cfg["processing"].get("overwrite", False):
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing = pd.read_csv(registry_path)
                paths_ok = all(Path(path).exists() for path in existing["calibrated_probability_path"]) and all(Path(path).exists() for path in existing["calibrated_mask_path"])
                if payload.get("status") == "passed" and len(existing) == len(group) and paths_ok:
                    rows.append(existing); continue
            except Exception:
                pass
        unit_rows: list[dict[str, Any]] = []
        guided = str(group.iloc[0].get("inference_mode", "direct_pixel")) == "anchor_guided_candidate"
        fraction_src = None
        anchor_geometry = None
        if guided:
            first_probability = Path(str(group.iloc[0]["probability_path"]))
            with rasterio.open(first_probability) as preview:
                anchor_geometry = select_anchor_geometry(reference_shape, region_id, preview.crs, preview.bounds, cfg)
            fraction_src = rasterio.open(str(supervision.loc[region_id, "pv_fraction_path"]))
        for row in group.itertuples(index=False):
            probability_path = ensure_dir(calibrated_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{int(row.year)}_pv_probability_calibrated_30m.tif"
            mask_path = ensure_dir(mask_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{int(row.year)}_pv_mask_calibrated_raw_30m.tif"
            candidate_class_path = ensure_dir(candidate_class_root / f"unit_{region_id:03d}") / f"unit_{region_id:03d}_{int(row.year)}_candidate_class_calibrated_30m.tif" if guided else Path("")
            predicted_pixels = 0
            medium_pixels = 0
            with rasterio.open(row.probability_path) as src:
                float_profile = src.profile.copy(); float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"))
                mask_profile = src.profile.copy(); mask_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"))
                class_dst_context = rasterio.open(candidate_class_path, "w", **mask_profile) if guided else None
                similarity_src = rasterio.open(str(row.anchor_similarity_path)) if guided else None
                std_src = rasterio.open(str(row.epistemic_std_path)) if guided else None
                try:
                    with rasterio.open(probability_path, "w", **float_profile) as prob_dst, rasterio.open(mask_path, "w", **mask_profile) as mask_dst:
                        for _, window in src.block_windows(1):
                            raw = src.read(1, window=window).astype(np.float32)
                            valid = raw > -9990
                            calibrated_block = np.full(raw.shape, -9999.0, dtype=np.float32)
                            if np.any(valid):
                                calibrated_block[valid] = apply_calibrator(bundle, raw[valid])
                            if guided:
                                similarity = similarity_src.read(1, window=window).astype(np.float32)
                                uncertainty = std_src.read(1, window=window).astype(np.float32)
                                near_anchor = buffered_support_mask(anchor_geometry, window_transform(window, src.transform), raw.shape, cfg)
                                anchor_mask = anchor_mask_for_window(fraction_src, window, cfg)
                                prediction, medium, _ = classify_anchor_guided(
                                    np.nan_to_num(calibrated_block, nan=0.0, neginf=0.0),
                                    np.nan_to_num(similarity, nan=0.0, neginf=0.0),
                                    np.nan_to_num(uncertainty, nan=1.0, posinf=1.0),
                                    near_anchor, valid, anchor_mask,
                                    int(row.year) == int(cfg["study"]["reference_year"]), cfg,
                                )
                            else:
                                prediction = np.zeros(raw.shape, dtype=bool)
                                prediction[valid] = calibrated_block[valid] >= threshold
                                if bool(cfg["calibration"].get("apply_component_filter", True)):
                                    prediction = _cleanup(prediction, int(cfg["calibration"]["minimum_component_pixels"]), int(cfg["calibration"]["maximum_hole_pixels"])) & valid
                                medium = np.zeros(raw.shape, dtype=bool)
                            encoded = np.full(raw.shape, mask_nodata, dtype=np.uint8)
                            encoded[valid] = prediction[valid].astype(np.uint8)
                            prob_dst.write(calibrated_block, 1, window=window)
                            mask_dst.write(encoded, 1, window=window)
                            if guided and class_dst_context is not None:
                                classes = np.full(raw.shape, mask_nodata, dtype=np.uint8)
                                classes[valid] = 0
                                classes[medium & valid] = 3
                                classes[prediction & valid] = 2
                                if int(row.year) == int(cfg["study"]["reference_year"]):
                                    classes[anchor_mask & valid] = 1
                                class_dst_context.write(classes, 1, window=window)
                            predicted_pixels += int(np.count_nonzero(prediction))
                            medium_pixels += int(np.count_nonzero(medium))
                        prob_dst.update_tags(calibration_method=chosen_method, panoramic_threshold=threshold, calibration_role="independent_unit_holdout")
                        mask_dst.update_tags(calibration_method=chosen_method, panoramic_threshold=threshold, mask_nodata=mask_nodata)
                finally:
                    if similarity_src is not None: similarity_src.close()
                    if std_src is not None: std_src.close()
                    if class_dst_context is not None: class_dst_context.close()
            unit_rows.append({
                **row._asdict(), "raw_probability_path": row.probability_path,
                "calibrated_probability_path": str(probability_path), "calibrated_mask_path": str(mask_path),
                "calibration_method": chosen_method, "calibrated_threshold": threshold,
                "calibrated_predicted_pixels": predicted_pixels, "calibrated_predicted_area_km2": predicted_pixels * 0.0009,
                "calibrated_candidate_class_path": str(candidate_class_path) if guided else "",
                "calibrated_medium_candidate_pixels": int(medium_pixels) if guided else 0,
            })
        if fraction_src is not None:
            fraction_src.close()
        unit_frame = pd.DataFrame(unit_rows)
        save_table(unit_frame, registry_path)
        atomic_json({"status": "passed", "Id": region_id, "records": len(unit_frame), "registry_path": str(registry_path), "created_utc": utc_now()}, checkpoint)
        rows.append(unit_frame)

    registry = pd.concat(rows, ignore_index=True).sort_values(["Id", "year"])
    save_table(registry, out / "calibrated_inference_registry.csv")
    population_pos = int(audit["positive_population"].sum())
    population_neg = int(audit["negative_population"].sum())
    status_payload = {
        "status": "passed", "created_utc": utc_now(), "method": chosen_method,
        "calibrator_path": str(bundle_path), "panoramic_threshold": threshold,
        "threshold_constraints_feasible": bool(feasible),
        "selected_threshold_metrics": {key: float(selected[key]) for key in [
            "anchor_recall", "precision_against_hard_negatives", "anchor_hard_negative_f1",
            "anchor_hard_negative_iou", "hard_negative_fpr", "score"
        ]},
        "calibration_units": int(audit["Id"].nunique()), "positive_population": population_pos,
        "negative_population": population_neg, "natural_prevalence": population_pos / max(population_pos + population_neg, 1),
        "records": len(registry), "units": int(registry["Id"].nunique()), "mask_nodata": mask_nodata,
        "independent_role": "calibration", "test_units_used": False,
        "reference_semantics": "high-confidence 2024 manual census with an unlabelled boundary-uncertainty band",
        "outside_shp_is_not_negative": True,
        "boundary_band_is_unlabelled": True,
        "far_outside_is_negative": True,
        "full_scene_precision_identifiable": False,
        "direct_units": int(registry.groupby("Id")["inference_mode"].first().eq("direct_pixel").sum()),
        "anchor_guided_candidate_units": int(registry.groupby("Id")["inference_mode"].first().eq("anchor_guided_candidate").sum()),
        "area_only_units": int(registry.groupby("Id")["inference_mode"].first().eq("area_only_fallback").sum()),
    }
    atomic_json(status_payload, out / "probability_calibration_status.json")
    return registry
