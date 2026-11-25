from __future__ import annotations

from pvchange.study_area import test_mode_enabled
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import binary_confusion, calibration_error, metrics_from_confusion
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _make_model(name: str, cfg: dict[str, Any], seed: int):
    if name == "xgboost":
        from xgboost import XGBClassifier
        params = dict(cfg["training"]["xgboost"])
        params.update(random_state=seed, objective="binary:logistic", eval_metric="logloss")
        return XGBClassifier(**params)
    if name == "random_forest":
        params = dict(cfg["training"]["random_forest"])
        params.update(random_state=seed, class_weight="balanced_subsample")
        return RandomForestClassifier(**params)
    if name == "hist_gradient_boosting":
        params = dict(cfg["training"]["hist_gradient_boosting"])
        params.update(random_state=seed, class_weight="balanced")
        return HistGradientBoostingClassifier(**params)
    raise KeyError(name)


def _load_cached_features(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    root = Path(cfg["paths"]["output_root"])
    registry = pd.read_csv(root / "04_patch_dataset" / "training_feature_cache_registry.csv")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    feature_count: int | None = None
    for row in registry.itertuples(index=False):
        data = np.load(row.shard_path)
        x = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.uint8, copy=False)
        if feature_count is None:
            feature_count = x.shape[1]
        elif x.shape[1] != feature_count:
            raise RuntimeError(f"Feature-count mismatch in {row.shard_path}: {x.shape[1]} vs {feature_count}")
        xs.append(x); ys.append(y)
        groups.append(np.full(len(y), int(row.Id), dtype=np.int16))
        folds.append(np.full(len(y), int(row.fold), dtype=np.int8))
    if not xs:
        raise RuntimeError("Training feature cache is empty")
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    group = np.concatenate(groups, axis=0)
    fold = np.concatenate(folds, axis=0)
    return X, y, group, fold, registry


def _stratified_cap(indices: np.ndarray, y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    local_y = y[indices]
    pos = indices[local_y == 1]
    neg = indices[local_y == 0]
    pos_fraction = len(pos) / max(len(indices), 1)
    npos = min(len(pos), max(1, int(round(cap * pos_fraction)))) if len(pos) else 0
    nneg = min(len(neg), cap - npos)
    if npos + nneg < cap:
        extra_pos = min(len(pos) - npos, cap - npos - nneg)
        npos += max(extra_pos, 0)
    selected = np.concatenate([
        rng.choice(pos, npos, replace=False) if npos else np.array([], dtype=np.int64),
        rng.choice(neg, nneg, replace=False) if nneg else np.array([], dtype=np.int64),
    ])
    rng.shuffle(selected)
    return selected


def _safe_probability_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def _best_threshold(y: np.ndarray, p: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    pr_auc, roc_auc = _safe_probability_metrics(y, p)
    rows: list[dict[str, float]] = []
    for threshold in cfg["training"]["threshold_grid"]:
        pred = p >= float(threshold)
        confusion = binary_confusion(y, pred, np.ones_like(y, dtype=bool))
        metrics = metrics_from_confusion(confusion)
        true_positive_count = max(int((y == 1).sum()), 1)
        metrics.update({
            "threshold": float(threshold),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "ece": float(calibration_error(y, p, int(cfg["validation"]["calibration_bins"]))),
            "area_relative_error": abs(int(pred.sum()) - int((y == 1).sum())) / true_positive_count,
            "negative_control_fpr": float(pred[y == 0].mean()) if np.any(y == 0) else float("nan"),
        })
        rows.append(metrics)
    return max(rows, key=lambda z: (z["f1"], z["iou"], -z["area_relative_error"], -z["ece"]))


def _composite_score(row: pd.Series, cfg: dict[str, Any]) -> float:
    weights = cfg["training"]["selection_weights"]
    terms = {
        "iou": float(row.get("iou", np.nan)),
        "f1": float(row.get("f1", np.nan)),
        "pr_auc": float(row.get("pr_auc", np.nan)),
        "calibration": 1.0 - float(np.clip(row.get("ece", np.nan), 0, 1)),
        "area_accuracy": 1.0 - float(np.clip(row.get("area_relative_error", np.nan), 0, 1)),
        "negative_control": 1.0 - float(np.clip(row.get("negative_control_fpr", np.nan), 0, 1)),
    }
    score = 0.0
    used = 0.0
    for key, weight in weights.items():
        value = terms.get(key, np.nan)
        if np.isfinite(value):
            score += float(weight) * value
            used += float(weight)
    return score / used if used > 0 else float("nan")


def benchmark_models(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "05_model_benchmark")
    X, y, groups, fold_assignments, registry = _load_cached_features(cfg)
    seed = int(cfg["project"]["random_seed"])
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unique_folds = sorted(np.unique(fold_assignments).tolist())
    diagnostic_pixel_cv = False
    if test_mode_enabled(cfg) and len(unique_folds) < 2:
        # Diagnostic-only fallback for Units 1, 6 and 28. The full 142-unit
        # production run never uses this branch and retains spatial unit folds.
        rng = np.random.default_rng(seed)
        diagnostic = np.zeros(len(y), dtype=np.int8)
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            diagnostic[idx[1::2]] = 1
        fold_assignments = diagnostic
        unique_folds = [0, 1]
        diagnostic_pixel_cv = True
    min_pos = int(cfg["training"].get("minimum_positive_validation_pixels", 20))
    min_neg = int(cfg["training"].get("minimum_negative_validation_pixels", 20))

    for model_name in cfg["training"]["candidate_models"]:
        for fold in unique_folds:
            train_idx = np.flatnonzero(fold_assignments != fold)
            valid_idx = np.flatnonzero(fold_assignments == fold)
            train_idx = _stratified_cap(train_idx, y, int(cfg["training"]["maximum_training_pixels"]), seed + fold)
            valid_idx = _stratified_cap(valid_idx, y, int(cfg["training"]["maximum_validation_pixels"]), seed + 100 + fold)
            y_train = y[train_idx]; y_valid = y[valid_idx]
            pos_valid = int((y_valid == 1).sum()); neg_valid = int((y_valid == 0).sum())
            if np.unique(y_train).size < 2 or pos_valid < min_pos or neg_valid < min_neg:
                skipped.append({
                    "model": model_name, "fold": int(fold), "reason": "insufficient_class_support",
                    "train_positive": int((y_train == 1).sum()), "train_negative": int((y_train == 0).sum()),
                    "validation_positive": pos_valid, "validation_negative": neg_valid,
                })
                continue
            model = _make_model(model_name, cfg, seed + int(fold))
            model.fit(X[train_idx], y_train)
            probability = model.predict_proba(X[valid_idx])[:, 1].astype(np.float32)
            metrics = _best_threshold(y_valid, probability, cfg)
            model_path = ensure_dir(out / model_name / f"fold_{fold}") / "model.joblib"
            joblib.dump(model, model_path, compress=3)
            metrics.update({
                "model": model_name, "fold": int(fold), "train_pixels": len(train_idx), "validation_pixels": len(valid_idx),
                "train_regions": int(np.unique(groups[train_idx]).size), "validation_regions": int(np.unique(groups[valid_idx]).size),
                "validation_positive": pos_valid, "validation_negative": neg_valid, "model_path": str(model_path),
            })
            rows.append(metrics)

    folds = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    save_table(folds, out / "benchmark_fold_metrics.csv")
    save_table(skipped_df, out / "benchmark_skipped_folds.csv")
    if folds.empty:
        atomic_json({"status": "failed", "created_utc": utc_now(), "reason": "No valid spatial fold for any model"}, out / "model_benchmark_status.json")
        raise RuntimeError("No valid model benchmark folds")

    metric_columns = ["iou", "f1", "pr_auc", "roc_auc", "ece", "area_relative_error", "negative_control_fpr"]
    summary = folds.groupby("model", as_index=False)[metric_columns].mean(numeric_only=True)
    valid_counts = folds.groupby("model").size().rename("valid_folds").reset_index()
    summary = summary.merge(valid_counts, on="model", how="left")
    summary["composite_score"] = summary.apply(lambda row: _composite_score(row, cfg), axis=1)
    min_valid_folds = int(cfg["training"].get("minimum_valid_spatial_folds", 3))
    if test_mode_enabled(cfg):
        min_valid_folds = min(
            min_valid_folds,
            int(cfg.get("test_mode", {}).get("minimum_valid_spatial_folds", 1)),
        )
    summary["eligible"] = summary["valid_folds"] >= min_valid_folds
    summary = summary.sort_values(["eligible", "composite_score"], ascending=[False, False]).reset_index(drop=True)
    summary["selection_rank"] = np.arange(1, len(summary) + 1)
    save_table(summary, out / "benchmark_model_summary.csv")
    selected = summary.loc[summary.eligible].head(int(cfg["training"]["final_ensemble_top_models"]))["model"].tolist()
    if not selected:
        status = "failed"
        reason = f"No model reached minimum {min_valid_folds} valid spatial folds"
    else:
        status = "passed"
        reason = None
    atomic_json({
        "status": status, "created_utc": utc_now(), "selected_models": selected,
        "backend": "CPU tree ensemble with one-pass cached CCDC feature shards",
        "cached_samples": int(len(y)), "cached_regions": int(registry.Id.nunique()),
        "valid_fold_records": int(len(folds)), "skipped_fold_records": int(len(skipped_df)),
        "minimum_valid_spatial_folds": min_valid_folds, "reason": reason,
        "diagnostic_within_unit_blocked_pixel_cv": diagnostic_pixel_cv,
    }, out / "model_benchmark_status.json")
    if status == "failed":
        raise RuntimeError(reason)
    return summary


def train_final_ensemble(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "06_final_models")
    summary = pd.read_csv(root / "05_model_benchmark" / "benchmark_model_summary.csv")
    selected = summary.loc[summary.eligible.astype(bool)].head(int(cfg["training"]["final_ensemble_top_models"]))["model"].tolist()
    if not selected:
        raise RuntimeError("No eligible models were selected")
    X, y, groups, _, registry = _load_cached_features(cfg)
    all_idx = np.arange(len(y), dtype=np.int64)
    all_idx = _stratified_cap(all_idx, y, int(cfg["training"]["maximum_training_pixels"]), int(cfg["project"]["random_seed"]))
    if np.unique(y[all_idx]).size < 2:
        raise RuntimeError("Final training data do not contain both classes")
    selected_summary = summary.set_index("model").loc[selected]
    raw_weights = np.clip(selected_summary["composite_score"].to_numpy(dtype=float), 1e-6, None)
    weights = raw_weights / raw_weights.sum()
    rows: list[dict[str, Any]] = []
    for i, (name, weight) in enumerate(zip(selected, weights)):
        model = _make_model(name, cfg, int(cfg["project"]["random_seed"]) + i)
        model.fit(X[all_idx], y[all_idx])
        path = out / f"{name}_final.joblib"
        joblib.dump(model, path, compress=3)
        rows.append({
            "model": name, "model_path": str(path), "weight": float(weight),
            "feature_count": int(X.shape[1]), "training_pixels": int(len(all_idx)),
            "training_positive": int((y[all_idx] == 1).sum()), "training_negative": int((y[all_idx] == 0).sum()),
        })
    manifest = pd.DataFrame(rows)
    save_table(manifest, out / "final_ensemble_manifest.csv")
    fold_metrics = pd.read_csv(root / "05_model_benchmark" / "benchmark_fold_metrics.csv")
    selected_metrics = fold_metrics.loc[fold_metrics.model.isin(selected)].copy()
    threshold = float(selected_metrics["threshold"].median())
    feature_schema = json.loads((root / "04_patch_dataset" / "patch_dataset_status.json").read_text(encoding="utf-8"))["feature_channels"]
    atomic_json({
        "status": "passed", "created_utc": utc_now(), "models": selected, "threshold": threshold,
        "training_pixels": int(len(all_idx)), "training_regions": int(np.unique(groups[all_idx]).size),
        "feature_count": int(X.shape[1]), "feature_schema": feature_schema,
        "backend": "CPU XGBoost/RandomForest/HistGradientBoosting ensemble",
        "source_feature_cache_regions": int(registry.Id.nunique()),
    }, out / "final_model_status.json")
    return manifest
