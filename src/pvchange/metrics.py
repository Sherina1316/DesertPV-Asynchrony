from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray | None = None) -> dict[str, int]:
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)
    if valid is None:
        valid = np.ones_like(yt, dtype=bool)
    valid = np.asarray(valid).astype(bool)
    yt, yp = yt[valid], yp[valid]
    return {
        "tp": int(np.sum(yt & yp)),
        "tn": int(np.sum(~yt & ~yp)),
        "fp": int(np.sum(~yt & yp)),
        "fn": int(np.sum(yt & ~yp)),
    }


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced = 0.5 * (recall + specificity)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask ^ eroded


def boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance_pixels: int = 1, valid: np.ndarray | None = None) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 3:
        values = []
        for i in range(yt.shape[0]):
            vi = None if valid is None else np.asarray(valid)[i]
            values.append(boundary_f1(yt[i], yp[i], tolerance_pixels, vi))
        return float(np.mean(values)) if values else float("nan")
    if yt.ndim != 2:
        raise ValueError(f"boundary_f1 expects 2D or 3D arrays, got shape {yt.shape}")
    gt = boundary_mask(yt)
    pr = boundary_mask(yp)
    if valid is not None:
        vv = np.asarray(valid).astype(bool)
        gt &= vv
        pr &= vv
    if tolerance_pixels > 0:
        structure = ndimage.generate_binary_structure(2, 2)
        gt_d = ndimage.binary_dilation(gt, structure=structure, iterations=tolerance_pixels)
        pr_d = ndimage.binary_dilation(pr, structure=structure, iterations=tolerance_pixels)
    else:
        gt_d, pr_d = gt, pr
    precision = np.sum(pr & gt_d) / max(np.sum(pr), 1)
    recall = np.sum(gt & pr_d) / max(np.sum(gt), 1)
    if np.sum(gt) == 0 and np.sum(pr) == 0:
        return 1.0
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 15, valid: np.ndarray | None = None) -> float:
    y = np.asarray(y_true).astype(np.float32)
    p = np.asarray(prob).astype(np.float32)
    if valid is None:
        valid = np.ones_like(y, dtype=bool)
    y, p = y[valid], p[valid]
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if idx.any():
            ece += idx.mean() * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(ece)


def probability_metrics(y_true: np.ndarray, prob: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y_true).astype(np.uint8)
    p = np.asarray(prob).astype(np.float32)
    if valid is None:
        valid = np.ones_like(y, dtype=bool)
    y, p = y[valid], p[valid]
    if y.size == 0 or np.unique(y).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def object_metrics(y_true: np.ndarray, y_pred: np.ndarray, iou_threshold: float = 0.25) -> dict[str, float]:
    gt_lab, gt_n = ndimage.label(np.asarray(y_true).astype(bool))
    pr_lab, pr_n = ndimage.label(np.asarray(y_pred).astype(bool))
    matched_gt: set[int] = set()
    matched_pr: set[int] = set()
    for p_id in range(1, pr_n + 1):
        p_mask = pr_lab == p_id
        overlaps = np.unique(gt_lab[p_mask])
        for g_id in overlaps:
            if g_id == 0:
                continue
            g_mask = gt_lab == g_id
            iou = np.sum(p_mask & g_mask) / max(np.sum(p_mask | g_mask), 1)
            if iou >= iou_threshold:
                matched_gt.add(int(g_id))
                matched_pr.add(int(p_id))
                break
    precision = len(matched_pr) / max(pr_n, 1)
    recall = len(matched_gt) / max(gt_n, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"object_precision": precision, "object_recall": recall, "object_f1": f1, "gt_objects": gt_n, "pred_objects": pr_n}
