from __future__ import annotations

import numpy as np

EPS = 1e-6


def _safe_mean(y: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(y)
    count = valid.sum(axis=axis).astype(np.float32)
    total = np.nansum(y, axis=axis, dtype=np.float64).astype(np.float32)
    mean = np.divide(total, count, out=np.zeros_like(total, dtype=np.float32), where=count > 0)
    return mean, count


def _linear_fit(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized linear fit for ``y[T, N]`` with missing observations."""
    y = np.asarray(y, dtype=np.float32)
    t = np.arange(y.shape[0], dtype=np.float32)[:, None]
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype(np.float32)
    sy = np.nansum(y, axis=0, dtype=np.float64).astype(np.float32)
    st = np.sum(valid * t, axis=0, dtype=np.float64).astype(np.float32)
    stt = np.sum(valid * t * t, axis=0, dtype=np.float64).astype(np.float32)
    sty = np.nansum(y * t, axis=0, dtype=np.float64).astype(np.float32)
    denominator = n * stt - st * st
    numerator = n * sty - st * sy
    slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=np.abs(denominator) > EPS)
    intercept = np.divide(sy - slope * st, np.maximum(n, 1.0), out=np.zeros_like(sy), where=n > 0)
    prediction = intercept[None, :] + slope[None, :] * t
    residual = np.where(valid, y - prediction, np.nan).astype(np.float32)
    residual_count = np.isfinite(residual).sum(axis=0).astype(np.float32)
    residual_sse = np.nansum(residual * residual, axis=0, dtype=np.float64).astype(np.float32)
    rmse = np.sqrt(np.divide(residual_sse, residual_count, out=np.zeros_like(residual_sse), where=residual_count > 0))
    return slope.astype(np.float32), rmse.astype(np.float32), residual, n


def annual_ccdc_descriptors(series: np.ndarray, min_obs: int = 6, min_seg: int = 3) -> np.ndarray:
    """CCDC-inspired descriptors for annual composites.

    Parameters
    ----------
    series:
        Matrix shaped ``[T, N]``. Missing observations are represented by NaN.
    """
    y = np.asarray(series, dtype=np.float32)
    slope, rmse, residual, n = _linear_fit(y)
    time_count, pixel_count = y.shape
    best_sse = np.full(pixel_count, np.inf, dtype=np.float32)
    best_year = np.zeros(pixel_count, dtype=np.float32)
    best_magnitude = np.zeros(pixel_count, dtype=np.float32)
    for split in range(int(min_seg), time_count - int(min_seg) + 1):
        pre = y[:split]
        post = y[split:]
        pre_mean, pre_count = _safe_mean(pre)
        post_mean, post_count = _safe_mean(post)
        pre_sse = np.nansum((pre - pre_mean) ** 2, axis=0, dtype=np.float64)
        post_sse = np.nansum((post - post_mean) ** 2, axis=0, dtype=np.float64)
        sse = (pre_sse + post_sse).astype(np.float32)
        eligible = (pre_count >= min_seg) & (post_count >= min_seg)
        take = eligible & (sse < best_sse)
        best_sse[take] = sse[take]
        best_year[take] = split
        best_magnitude[take] = (post_mean - pre_mean)[take]
    break_z = np.abs(best_magnitude) / np.maximum(rmse, 1e-4)
    max_abs_residual = np.max(np.where(np.isfinite(residual), np.abs(residual), 0.0), axis=0)
    valid = n >= int(min_obs)
    descriptors = np.stack(
        [
            slope,
            rmse,
            best_magnitude,
            best_year / max(time_count - 1, 1),
            break_z,
            max_abs_residual,
            n / max(time_count, 1),
        ],
        axis=1,
    ).astype(np.float32)
    descriptors[~valid] = 0.0
    return descriptors


def causal_ccdc_features(series: np.ndarray, min_obs: int = 6, min_seg: int = 3) -> np.ndarray:
    """Return ``[T, N, 7]`` descriptors using observations only through each year."""
    series = np.asarray(series, dtype=np.float32)
    time_count, pixel_count = series.shape
    output = np.zeros((time_count, pixel_count, 7), dtype=np.float32)
    for index in range(time_count):
        observations = index + 1
        if observations >= int(min_obs):
            effective_segment = min(int(min_seg), max(1, observations // 2))
            output[index] = annual_ccdc_descriptors(series[:observations], min_obs=min_obs, min_seg=effective_segment)
    return output
