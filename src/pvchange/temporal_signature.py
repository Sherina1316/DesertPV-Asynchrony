from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy import ndimage

EPS = 1e-6


def _nanmedian(array: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(array, axis=axis)


def _nanmean(array: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(array, axis=axis)


def _nanmax(array: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(array, axis=axis)


def _nanmin(array: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(array, axis=axis)


@dataclass(frozen=True)
class SignatureResult:
    onset_index: np.ndarray
    transition_score: np.ndarray
    probability_shift: np.ndarray
    fraction_shift: np.ndarray
    break_z: np.ndarray
    post_presence: np.ndarray
    post_mad: np.ndarray
    confuser_penalty: np.ndarray
    baseline_preexisting: np.ndarray
    recent_right_censored: np.ndarray
    anchor_backcast: np.ndarray
    anchor_backcast_score: np.ndarray
    qualified: np.ndarray
    exclusion_code: np.ndarray
    valid_year_count: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def robust_probability_harmonization(
    probability: np.ndarray,
    valid: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Correct unit-wide annual probability shifts using stable background pixels."""
    settings = cfg.get("temporal_signature", {})
    ceiling = float(settings.get("stable_background_probability_ceiling", 0.22))
    max_range = float(settings.get("stable_background_maximum_range", 0.15))
    minimum = int(settings.get("harmonization_minimum_pixels", 500))
    p = np.asarray(probability, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    masked = np.where(v, p, np.nan)
    med = _nanmedian(masked, axis=0)
    rng = _nanmax(masked, axis=0) - _nanmin(masked, axis=0)
    stable = np.isfinite(med) & (med <= ceiling) & np.isfinite(rng) & (rng <= max_range)
    biases = np.zeros(p.shape[0], dtype=np.float32)
    if int(stable.sum()) < minimum:
        return p.copy(), biases
    logits = np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1))
    medians = []
    for index in range(p.shape[0]):
        values = logits[index][stable & v[index]]
        medians.append(float(_nanmedian(values)) if values.size else np.nan)
    medians = np.asarray(medians, dtype=np.float32)
    target = float(_nanmedian(medians)) if np.isfinite(medians).any() else 0.0
    biases = np.where(np.isfinite(medians), medians - target, 0.0).astype(np.float32)
    adjusted = _sigmoid(logits - biases[:, None, None]).astype(np.float32)
    adjusted[~v] = np.nan
    return adjusted, biases


def _presence_fraction(values: np.ndarray, threshold: float) -> np.ndarray:
    return _nanmean(
        np.where(np.isfinite(values), values >= float(threshold), np.nan),
        axis=0,
    )


def detect_temporal_signature(
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_evidence: np.ndarray,
    valid: np.ndarray,
    near_anchor: np.ndarray | None,
    cfg: dict[str, Any],
    anchor_core: np.ndarray | None = None,
) -> SignatureResult:
    """Detect PV installation using adaptive, anchor-aware temporal evidence.

    Four model-evidence routes are evaluated:
    1. pre-series facilities already present at the beginning of the record;
    2. standard within-series installation transitions;
    3. right-censored recent installations with fewer post-event observations;
    4. 2011-2015 retrospective backcasting inside confirmed 2024 anchors.

    This function returns model/LSMA/time-series evidence only. The separate
    worker-level endpoint constraint guarantees that confirmed 2024 anchor
    pixels are present in the final 2024 product while retaining this model-only
    result for unbiased sensitivity validation. Historical years are never
    filled from the reference anchor without detected change evidence.
    """
    s = cfg.get("temporal_signature", {})
    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    c = np.asarray(confuser_evidence, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    if p.shape != f.shape or p.shape != c.shape or p.shape != v.shape:
        raise ValueError("All temporal cubes must share [T,H,W]")

    t, h, w = p.shape
    n = h * w
    p2 = p.reshape(t, n)
    f2 = f.reshape(t, n)
    c2 = c.reshape(t, n)
    v2 = v.reshape(t, n)
    near = np.zeros(n, dtype=bool) if near_anchor is None else np.asarray(near_anchor, dtype=bool).reshape(n)
    anchor = np.zeros(n, dtype=bool) if anchor_core is None else np.asarray(anchor_core, dtype=bool).reshape(n)
    near |= anchor

    presence_thr = max(
        float(cfg.get("temporal_decoding", {}).get("change_year_probability_threshold", 0.70)),
        0.50,
    )
    reference_index = int(
        np.clip(
            int(cfg.get("study", {}).get("reference_year", 2024))
            - int(cfg.get("study", {}).get("start_year", 2011)),
            0,
            t - 1,
        )
    )
    reference_probability = p2[reference_index]
    reference_fraction = f2[reference_index]
    reference_valid = v2[reference_index]

    best_score = np.full(n, -np.inf, dtype=np.float32)
    best_idx = np.full(n, -1, dtype=np.int16)
    best_prob_shift = np.full(n, np.nan, dtype=np.float32)
    best_frac_shift = np.full(n, np.nan, dtype=np.float32)
    best_z = np.full(n, np.nan, dtype=np.float32)
    best_presence = np.full(n, np.nan, dtype=np.float32)
    best_mad = np.full(n, np.nan, dtype=np.float32)
    best_conf = np.full(n, np.nan, dtype=np.float32)
    baseline_preexisting = np.zeros(n, dtype=bool)
    recent_right_censored = np.zeros(n, dtype=bool)
    anchor_backcast = np.zeros(n, dtype=bool)
    anchor_backcast_score = np.zeros(n, dtype=np.float32)
    valid_year_count = np.sum(v2, axis=0).astype(np.int16)

    # ------------------------------------------------------------------
    # Route A: facilities present before or near the beginning of series.
    # Missing early observations are tolerated when later evidence remains
    # persistent and reference-year anchor evidence is strong.
    # ------------------------------------------------------------------
    if bool(s.get("baseline_preexisting_enabled", True)):
        early_n = min(t, int(s.get("baseline_early_year_count", 4)))
        early_p = np.where(v2[:early_n], p2[:early_n], np.nan)
        early_f = np.where(v2[:early_n], f2[:early_n], np.nan)
        all_p = np.where(v2, p2, np.nan)
        all_c = np.where(v2, c2, np.nan)
        early_count = np.sum(np.isfinite(early_p), axis=0)
        early_prob = _nanmedian(early_p, axis=0)
        early_fraction = _nanmedian(early_f, axis=0)
        all_median = _nanmedian(all_p, axis=0)
        all_mad = _nanmedian(np.abs(all_p - all_median[None, :]), axis=0)
        all_presence = _presence_fraction(all_p, presence_thr)
        all_conf = _nanmedian(all_c, axis=0)
        if t >= 2:
            pair_valid = np.isfinite(all_p[1:]) & np.isfinite(all_p[:-1])
            drops = np.where(pair_valid, np.maximum(all_p[:-1] - all_p[1:], 0.0), np.nan)
            all_drop = np.nan_to_num(_nanmax(drops, axis=0), nan=0.0)
        else:
            all_drop = np.zeros(n, dtype=np.float32)

        required_early = np.where(
            anchor,
            int(s.get("baseline_minimum_anchor_early_valid_years", 1)),
            np.where(
                near,
                int(s.get("baseline_minimum_near_early_valid_years", 2)),
                int(s.get("baseline_minimum_remote_early_valid_years", 3)),
            ),
        )
        probability_gate = np.where(
            anchor,
            float(s.get("baseline_minimum_anchor_probability", 0.62)),
            np.where(
                near,
                float(s.get("baseline_minimum_near_anchor_probability", 0.74)),
                float(s.get("baseline_minimum_remote_probability", 0.92)),
            ),
        )
        fraction_gate = np.where(
            anchor,
            float(s.get("baseline_minimum_anchor_fraction_evidence", 0.18)),
            np.where(
                near,
                float(s.get("baseline_minimum_near_anchor_fraction_evidence", 0.35)),
                float(s.get("baseline_minimum_remote_fraction_evidence", 0.62)),
            ),
        )
        presence_gate = np.where(
            anchor,
            float(s.get("baseline_minimum_anchor_presence_fraction", 0.58)),
            np.where(
                near,
                float(s.get("baseline_minimum_near_presence_fraction", 0.68)),
                float(s.get("baseline_minimum_remote_presence_fraction", 0.84)),
            ),
        )
        conf_gate = np.where(
            anchor,
            float(s.get("baseline_maximum_anchor_confuser_penalty", 0.48)),
            np.where(
                near,
                float(s.get("baseline_maximum_near_confuser_penalty", 0.38)),
                float(s.get("baseline_maximum_remote_confuser_penalty", 0.26)),
            ),
        )
        reference_support = (
            reference_valid
            & (reference_probability >= np.where(anchor, 0.55, np.where(near, 0.70, 0.90)))
            & (reference_fraction >= np.where(anchor, 0.10, np.where(near, 0.25, 0.55)))
        )
        baseline_preexisting = (
            (early_count >= required_early)
            & np.isfinite(early_prob)
            & np.isfinite(early_fraction)
            & (early_prob >= probability_gate)
            & (early_fraction >= fraction_gate)
            & (all_presence >= presence_gate)
            & (all_mad <= np.where(anchor, 0.22, np.where(near, 0.18, 0.14)))
            & (all_drop <= np.where(anchor, 0.40, np.where(near, 0.32, 0.22)))
            & (all_conf <= conf_gate)
            & (reference_support | ~anchor)
        )
        baseline_score = np.clip(
            0.30 * np.nan_to_num(early_prob)
            + 0.25 * np.nan_to_num(early_fraction)
            + 0.25 * np.nan_to_num(all_presence)
            + 0.10 * np.nan_to_num(reference_probability)
            + 0.10 * (1.0 - np.nan_to_num(all_conf, nan=1.0)),
            0.0,
            1.0,
        )
        best_idx[baseline_preexisting] = 0
        best_score[baseline_preexisting] = baseline_score[baseline_preexisting]
        best_prob_shift[baseline_preexisting] = 0.0
        best_frac_shift[baseline_preexisting] = 0.0
        best_z[baseline_preexisting] = 0.0
        best_presence[baseline_preexisting] = all_presence[baseline_preexisting]
        best_mad[baseline_preexisting] = all_mad[baseline_preexisting]
        best_conf[baseline_preexisting] = all_conf[baseline_preexisting]

    # ------------------------------------------------------------------
    # Routes B/C: adaptive standard and recent right-censored transitions.
    # ------------------------------------------------------------------
    min_pre = int(s.get("minimum_pre_years", 3))
    min_post = int(s.get("minimum_post_years", 2))
    recent_window = int(s.get("recent_installation_window_years", 3))
    min_p_shift = float(s.get("minimum_probability_level_shift", 0.18))
    min_f_shift = float(s.get("minimum_fraction_level_shift", 0.08))
    min_z = float(s.get("minimum_break_z", 2.0))
    min_post_presence = float(s.get("minimum_post_presence_fraction", 0.75))
    max_pre_presence = float(s.get("maximum_pre_presence_fraction", 0.30))
    max_post_mad = float(s.get("maximum_post_probability_mad", 0.18))
    max_drop = float(s.get("maximum_post_annual_drop", 0.30))
    conf_weight = float(s.get("confuser_penalty_weight", 0.25))
    near_relax = float(s.get("near_anchor_score_relaxation", 0.06))
    anchor_relax = float(s.get("anchor_core_score_relaxation", 0.12))
    remote_penalty = float(s.get("remote_score_penalty", 0.08))

    for onset in range(1, t):
        pre_p = np.where(v2[:onset], p2[:onset], np.nan)
        post_p = np.where(v2[onset:], p2[onset:], np.nan)
        pre_f = np.where(v2[:onset], f2[:onset], np.nan)
        post_f = np.where(v2[onset:], f2[onset:], np.nan)
        post_c = np.where(v2[onset:], c2[onset:], np.nan)

        pre_count = np.sum(np.isfinite(pre_p), axis=0)
        post_count = np.sum(np.isfinite(post_p), axis=0)
        recent = onset >= max(1, t - recent_window)

        pre_required = np.where(
            anchor,
            int(s.get("anchor_minimum_pre_observations", 1)),
            np.where(near, int(s.get("near_anchor_minimum_pre_observations", 2)), min_pre),
        )
        post_required = np.where(
            anchor & recent,
            int(s.get("recent_anchor_minimum_post_observations", 1)),
            np.where(
                near & recent,
                int(s.get("recent_near_minimum_post_observations", 1)),
                min_post,
            ),
        )
        enough = (pre_count >= pre_required) & (post_count >= post_required)

        pre_med = _nanmedian(pre_p, axis=0)
        post_med = _nanmedian(post_p, axis=0)
        pre_f_med = _nanmedian(pre_f, axis=0)
        post_f_med = _nanmedian(post_f, axis=0)
        p_shift = post_med - pre_med
        f_shift = post_f_med - pre_f_med
        pre_mad = _nanmedian(np.abs(pre_p - pre_med[None, :]), axis=0)
        post_mad = _nanmedian(np.abs(post_p - post_med[None, :]), axis=0)
        z = p_shift / np.maximum(1.4826 * pre_mad + 0.03, 0.03)
        post_presence = _presence_fraction(post_p, presence_thr)
        pre_presence = _presence_fraction(pre_p, presence_thr)
        conf = _nanmedian(post_c, axis=0)
        if post_p.shape[0] >= 2:
            pair_valid = np.isfinite(post_p[1:]) & np.isfinite(post_p[:-1])
            drops = np.where(pair_valid, np.maximum(post_p[:-1] - post_p[1:], 0.0), np.nan)
            annual_drop = np.nan_to_num(_nanmax(drops, axis=0), nan=0.0)
        else:
            annual_drop = np.zeros(n, dtype=np.float32)

        p_gate = np.where(anchor, min_p_shift * 0.45, np.where(near, min_p_shift * 0.72, min_p_shift))
        f_gate = np.where(anchor, min_f_shift * 0.35, np.where(near, min_f_shift * 0.65, min_f_shift))
        z_gate = np.where(anchor, min_z * 0.45, np.where(near, min_z * 0.70, min_z))
        presence_gate = np.where(anchor, 0.52, np.where(near, 0.64, min_post_presence))
        pre_presence_gate = np.where(anchor, 0.62, np.where(near, 0.45, max_pre_presence))
        mad_gate = np.where(anchor, 0.28, np.where(near, 0.23, max_post_mad))
        drop_gate = np.where(anchor, 0.48, np.where(near, 0.38, max_drop))
        conf_gate = np.where(anchor, 0.55, np.where(near, 0.45, 0.34))

        # Recent 2023-2025 installations are right-censored. They require very
        # strong endpoint evidence instead of an unavailable long post-period.
        recent_support = np.ones(n, dtype=bool)
        if recent:
            recent_support = (
                reference_valid
                & (reference_probability >= np.where(anchor, 0.58, np.where(near, 0.76, 0.92)))
                & (reference_fraction >= np.where(anchor, 0.12, np.where(near, 0.28, 0.60)))
            )

        score = (
            0.28 * _sigmoid((p_shift - p_gate) / 0.06)
            + 0.18 * _sigmoid((f_shift - f_gate) / 0.05)
            + 0.18 * _sigmoid((z - z_gate) / 0.75)
            + 0.20 * np.clip(post_presence, 0, 1)
            + 0.08 * np.clip(1.0 - pre_presence, 0, 1)
            + 0.08 * np.nan_to_num(reference_probability)
            - conf_weight * np.nan_to_num(conf, nan=1.0)
            - 0.08 * np.clip(post_mad / np.maximum(mad_gate, EPS), 0, 2)
            - 0.08 * np.clip(annual_drop / np.maximum(drop_gate, EPS), 0, 2)
        )
        score += np.where(anchor, anchor_relax, np.where(near, near_relax, -remote_penalty))

        hard = (
            enough
            & np.isfinite(score)
            & (p_shift >= p_gate)
            & (f_shift >= f_gate)
            & (z >= z_gate)
            & (post_presence >= presence_gate)
            & (pre_presence <= pre_presence_gate)
            & (post_mad <= mad_gate)
            & (annual_drop <= drop_gate)
            & (conf <= conf_gate)
            & recent_support
        )
        better = hard & (~baseline_preexisting) & (score > best_score)
        best_score[better] = score[better]
        best_idx[better] = onset
        best_prob_shift[better] = p_shift[better]
        best_frac_shift[better] = f_shift[better]
        best_z[better] = z[better]
        best_presence[better] = post_presence[better]
        best_mad[better] = post_mad[better]
        best_conf[better] = conf[better]
        if recent:
            recent_right_censored[better] = True

    # ------------------------------------------------------------------
    # Route D: reference-anchor-guided retrospective backcasting.
    #
    # The 2024 reference polygon is certain positive evidence at the
    # reference date, but it is never copied backwards without a detectable
    # temporal change.  For 2011-2015, the route allows one pre- and one
    # post-observation because the ordinary persistence rule can reject early
    # short-lived transitions or intermittent annual composites.  A candidate
    # still needs at least two independent probability/LSMA/break votes and
    # low confuser evidence.
    # ------------------------------------------------------------------
    back = cfg.get("anchor_backcasting", {})
    if bool(back.get("enabled", True)) and np.any(anchor):
        start_year = int(cfg.get("study", {}).get("start_year", 2011))
        target_start = max(start_year, int(back.get("target_start_year", start_year)))
        target_end = min(
            int(cfg.get("study", {}).get("reference_year", 2024)) - 1,
            int(back.get("target_end_year", 2015)),
        )
        onset_start = max(1, target_start - start_year)
        onset_stop = min(t - 1, target_end - start_year)
        pre_window = max(1, int(back.get("local_pre_window_years", 2)))
        post_window = max(1, int(back.get("local_post_window_years", 2)))
        min_pre_obs = max(1, int(back.get("minimum_pre_observations", 1)))
        min_post_obs = max(1, int(back.get("minimum_post_observations", 1)))
        p_gate_back = float(back.get("minimum_probability_shift", 0.08))
        f_gate_back = float(back.get("minimum_fraction_shift", 0.025))
        z_gate_back = float(back.get("minimum_break_z", 0.75))
        min_votes = int(back.get("minimum_evidence_votes", 2))
        later_presence_gate = float(back.get("minimum_later_presence_fraction", 0.25))
        later_presence_threshold = float(back.get("later_presence_probability_threshold", 0.55))
        conf_gate_back = float(back.get("maximum_confuser_penalty", 0.62))
        score_gate_back = float(back.get("minimum_score", 0.42))
        reference_probability_gate = float(back.get("reference_probability_support_threshold", 0.45))
        reference_fraction_gate = float(back.get("reference_fraction_support_threshold", 0.08))
        early_preference = float(back.get("earliest_credible_onset_preference", 0.03))

        reference_support = reference_valid & (
            (reference_probability >= reference_probability_gate)
            | (reference_fraction >= reference_fraction_gate)
        )

        if onset_stop >= onset_start:
            for onset in range(onset_start, onset_stop + 1):
                pre_slice = slice(max(0, onset - pre_window), onset)
                post_slice = slice(onset, min(t, onset + post_window))
                later_slice = slice(onset, reference_index + 1)

                pre_p = np.where(v2[pre_slice], p2[pre_slice], np.nan)
                post_p = np.where(v2[post_slice], p2[post_slice], np.nan)
                pre_f = np.where(v2[pre_slice], f2[pre_slice], np.nan)
                post_f = np.where(v2[post_slice], f2[post_slice], np.nan)
                local_c = np.where(v2[post_slice], c2[post_slice], np.nan)
                later_p = np.where(v2[later_slice], p2[later_slice], np.nan)

                pre_count = np.sum(np.isfinite(pre_p), axis=0)
                post_count = np.sum(np.isfinite(post_p), axis=0)
                pre_med = _nanmedian(pre_p, axis=0)
                post_med = _nanmedian(post_p, axis=0)
                pre_f_med = _nanmedian(pre_f, axis=0)
                post_f_med = _nanmedian(post_f, axis=0)
                p_shift = post_med - pre_med
                f_shift = post_f_med - pre_f_med
                pre_mad = _nanmedian(np.abs(pre_p - pre_med[None, :]), axis=0)
                post_mad = _nanmedian(np.abs(post_p - post_med[None, :]), axis=0)
                z = p_shift / np.maximum(1.4826 * pre_mad + 0.03, 0.03)
                local_presence = _presence_fraction(post_p, later_presence_threshold)
                later_presence = _presence_fraction(later_p, later_presence_threshold)
                conf = _nanmedian(local_c, axis=0)

                votes = (p_shift >= p_gate_back).astype(np.uint8)
                votes += (f_shift >= f_gate_back).astype(np.uint8)
                votes += (z >= z_gate_back).astype(np.uint8)
                votes += (local_presence >= 0.50).astype(np.uint8)
                votes += reference_support.astype(np.uint8)

                score = (
                    0.28 * _sigmoid((p_shift - p_gate_back) / 0.05)
                    + 0.20 * _sigmoid((f_shift - f_gate_back) / 0.035)
                    + 0.18 * _sigmoid((z - z_gate_back) / 0.60)
                    + 0.14 * np.clip(np.nan_to_num(local_presence), 0, 1)
                    + 0.10 * np.clip(np.nan_to_num(later_presence), 0, 1)
                    + 0.07 * np.nan_to_num(reference_probability)
                    + 0.03 * np.nan_to_num(reference_fraction)
                    - 0.18 * np.nan_to_num(conf, nan=1.0)
                    - 0.05 * np.clip(np.nan_to_num(post_mad) / 0.30, 0, 2)
                )
                if onset_stop > onset_start:
                    score += early_preference * (onset_stop - onset) / (onset_stop - onset_start)

                hard = (
                    anchor
                    & ~baseline_preexisting
                    & (pre_count >= min_pre_obs)
                    & (post_count >= min_post_obs)
                    & np.isfinite(score)
                    & (votes >= min_votes)
                    & (post_med > pre_med)
                    & (conf <= conf_gate_back)
                    & (reference_support | (later_presence >= later_presence_gate))
                    & (score >= score_gate_back)
                )
                # Prefer an earlier credible backcast over a later standard
                # transition, while never replacing a pre-series classification.
                # A credible 2011-2015 anchor backcast takes precedence over
                # a later endpoint/right-censored route. The retrospective
                # hard gate already requires independent probability, LSMA,
                # break and low-confuser evidence, so ranking it only by the
                # recent-route score would systematically bias onset years
                # toward 2023-2024.
                better = hard & (
                    (best_idx < 0)
                    | (best_idx > onset)
                    | (score > best_score + 0.08)
                )
                best_score[better] = score[better]
                best_idx[better] = onset
                best_prob_shift[better] = p_shift[better]
                best_frac_shift[better] = f_shift[better]
                best_z[better] = z[better]
                best_presence[better] = later_presence[better]
                best_mad[better] = post_mad[better]
                best_conf[better] = conf[better]
                anchor_backcast[better] = True
                anchor_backcast_score[better] = score[better]
                recent_right_censored[better] = False

    score_gate = np.where(
        anchor,
        float(s.get("minimum_anchor_transition_score", 0.46)),
        np.where(
            near,
            float(s.get("minimum_near_transition_score", 0.52)),
            float(s.get("minimum_transition_score", 0.58)),
        ),
    )
    # The retrospective route has its own multi-evidence hard gate and score
    # threshold. Do not reject a valid early backcast a second time with the
    # stricter generic transition-score gate.
    qualified = ((best_idx >= 0) & (best_score >= score_gate)) | anchor_backcast

    # Compact diagnostic code used only to explain why confirmed anchors were
    # lost. It does not alter the classification result.
    exclusion = np.full(n, 12, dtype=np.uint8)  # 12: no qualified temporal route
    exclusion[valid_year_count == 0] = 1       # no valid observations
    exclusion[(valid_year_count > 0) & (valid_year_count < 3)] = 2
    median_conf = _nanmedian(np.where(v2, c2, np.nan), axis=0)
    exclusion[np.isfinite(median_conf) & (median_conf > 0.55)] = 11
    exclusion[qualified] = 0

    return SignatureResult(
        onset_index=best_idx.reshape(h, w),
        transition_score=np.where(np.isfinite(best_score), best_score, 0).reshape(h, w),
        probability_shift=best_prob_shift.reshape(h, w),
        fraction_shift=best_frac_shift.reshape(h, w),
        break_z=best_z.reshape(h, w),
        post_presence=best_presence.reshape(h, w),
        post_mad=best_mad.reshape(h, w),
        confuser_penalty=best_conf.reshape(h, w),
        baseline_preexisting=baseline_preexisting.reshape(h, w),
        recent_right_censored=recent_right_censored.reshape(h, w),
        anchor_backcast=anchor_backcast.reshape(h, w),
        anchor_backcast_score=anchor_backcast_score.reshape(h, w),
        qualified=qualified.reshape(h, w),
        exclusion_code=exclusion.reshape(h, w),
        valid_year_count=valid_year_count.reshape(h, w),
    )


def propagate_anchor_backcast_within_objects(
    onset_index: np.ndarray,
    anchor_backcast: np.ndarray,
    anchor_reference: np.ndarray,
    transition_score: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a credible early change year within a confirmed anchor object.

    The operation is deliberately restricted to connected pixels inside the
    2024 confirmed-positive anchor.  It never expands outside the SHP and is
    applied only when a sufficient fraction of an anchor object independently
    supports a 2011-2015 backcast with a compact year distribution.  This fills
    internal roads, mixed pixels and short temporal gaps without copying the
    entire reference anchor backwards merely because it exists in 2024.
    """
    settings = cfg.get("anchor_backcasting", {})
    onset = np.asarray(onset_index, dtype=np.int16).copy()
    direct = np.asarray(anchor_backcast, dtype=bool).copy()
    anchor = np.asarray(anchor_reference, dtype=bool)
    score = np.asarray(transition_score, dtype=np.float32)
    filled = np.zeros_like(anchor, dtype=bool)
    if not bool(settings.get("object_consensus_enabled", True)) or not np.any(anchor):
        return onset, direct, filled

    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(anchor, structure=structure)
    minimum_fraction = float(settings.get("object_consensus_minimum_supported_fraction", 0.20))
    maximum_spread = int(settings.get("object_consensus_maximum_year_spread", 3))
    minimum_pixels = int(settings.get("object_consensus_minimum_supported_pixels", 1))
    target_end = int(settings.get("target_end_year", 2015))
    start_year = int(cfg.get("study", {}).get("start_year", 2011))
    maximum_index = max(0, target_end - start_year)

    for label_id in range(1, count + 1):
        component = labels == label_id
        size = int(np.count_nonzero(component))
        supported = component & direct & (onset >= 0) & (onset <= maximum_index)
        supported_count = int(np.count_nonzero(supported))
        if supported_count < minimum_pixels or supported_count / max(size, 1) < minimum_fraction:
            continue
        years = onset[supported].astype(int)
        if years.size == 0 or int(years.max() - years.min()) > maximum_spread:
            continue
        weights = np.clip(score[supported].astype(float), 0.01, None)
        order = np.argsort(years)
        ordered_years = years[order]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights)
        consensus = int(ordered_years[np.searchsorted(cumulative, cumulative[-1] * 0.5)])
        consensus = int(np.clip(consensus, 0, maximum_index))

        eligible = component & ((onset < 0) | (onset > consensus))
        onset[eligible] = consensus
        filled |= eligible
        direct |= eligible

    return onset, direct, filled



def _masked_local_mean(
    values: np.ndarray,
    support: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and support count without propagating NoData."""
    size = max(1, int(size))
    finite = np.isfinite(values) & np.asarray(support, dtype=bool)
    numerator = ndimage.uniform_filter(
        np.where(finite, values, 0.0).astype(np.float32),
        size=size,
        mode="nearest",
    )
    denominator = ndimage.uniform_filter(
        finite.astype(np.float32),
        size=size,
        mode="nearest",
    )
    mean = np.divide(
        numerator,
        denominator,
        out=np.full(values.shape, np.nan, dtype=np.float32),
        where=denominator > EPS,
    )
    count = denominator * float(size * size)
    return mean.astype(np.float32), count.astype(np.float32)


def compute_anchor_reference_spectral_similarity(
    reflectance: np.ndarray,
    anchor_reference: np.ndarray,
    valid: np.ndarray,
    reference_index: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Similarity to spatially matched 2024 PV spectra across the full series.

    A single tile-wide prototype is inadequate when one 2024 SHP object contains
    different panel types, illumination conditions and desert substrates.  The
    revised similarity blends three references: the same pixel in 2024, a local
    2024 anchor prototype, and a robust tile-wide anchor prototype.  The same-
    pixel/local terms preserve spatial heterogeneity and are essential for
    recovering partial installations visible in 2013--2015 inside a much larger
    2024 anchor object.
    """
    cube = np.asarray(reflectance, dtype=np.float32)
    anchor = np.asarray(anchor_reference, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    if cube.ndim != 4 or cube.shape[1] != 6:
        raise ValueError("reflectance must have shape [T,6,H,W]")
    if v.shape != (cube.shape[0], cube.shape[2], cube.shape[3]):
        raise ValueError("valid must have shape [T,H,W]")

    settings = cfg.get("anchor_full_series_decoder", {})
    reference_index = int(np.clip(reference_index, 0, cube.shape[0] - 1))
    reference = cube[reference_index]
    support = (
        anchor
        & v[reference_index]
        & np.all(np.isfinite(reference), axis=0)
    )
    output = np.full(v.shape, np.nan, dtype=np.float32)
    if int(np.count_nonzero(support)) < 3:
        return output

    rows = reference[:, support]
    global_prototype = np.nanmedian(rows, axis=1).astype(np.float32)
    q25 = np.nanquantile(rows, 0.25, axis=1)
    q75 = np.nanquantile(rows, 0.75, axis=1)
    floor = float(settings.get("reference_scale_floor", 0.015))
    scale = np.maximum((q75 - q25).astype(np.float32), floor)

    local_window = max(3, int(settings.get("local_reference_window_pixels", 9)))
    if local_window % 2 == 0:
        local_window += 1
    minimum_local_pixels = max(
        1,
        int(settings.get("minimum_local_reference_pixels", 5)),
    )

    spatial_reference = np.empty_like(reference, dtype=np.float32)
    for band in range(reference.shape[0]):
        local_mean, local_count = _masked_local_mean(
            reference[band],
            support,
            local_window,
        )
        local_reference = np.where(
            local_count >= minimum_local_pixels,
            local_mean,
            global_prototype[band],
        )
        same_pixel = support & np.isfinite(reference[band])
        spatial_reference[band] = np.where(
            same_pixel,
            reference[band],
            local_reference,
        ).astype(np.float32)

    temperature = float(settings.get("reference_similarity_temperature", 2.5))
    same_pixel_weight = float(settings.get("same_pixel_similarity_weight", 0.70))
    global_weight = float(settings.get("global_similarity_weight", 0.30))
    weight_sum = max(same_pixel_weight + global_weight, EPS)

    for index in range(cube.shape[0]):
        spatial_distance = _nanmedian(
            np.abs(
                (cube[index] - spatial_reference)
                / scale[:, None, None]
            ),
            axis=0,
        )
        global_distance = _nanmedian(
            np.abs(
                (cube[index] - global_prototype[:, None, None])
                / scale[:, None, None]
            ),
            axis=0,
        )
        spatial_similarity = np.exp(
            -spatial_distance / max(temperature, EPS)
        )
        global_similarity = np.exp(
            -global_distance / max(temperature, EPS)
        )
        similarity = (
            same_pixel_weight * spatial_similarity
            + global_weight * global_similarity
        ) / weight_sum
        similarity = np.maximum(
            similarity,
            float(settings.get("global_similarity_fallback_weight", 0.85))
            * global_similarity,
        )
        similarity[~v[index]] = np.nan
        output[index] = np.clip(similarity, 0.0, 1.0).astype(np.float32)
    return output


def _filter_anchor_annual_components(
    mask: np.ndarray,
    strong: np.ndarray,
    minimum_pixels: int,
    minimum_strong_pixels: int,
) -> np.ndarray:
    """Remove isolated annual detections while retaining coherent PV rows."""
    raw = np.asarray(mask, dtype=bool)
    if not np.any(raw):
        return np.zeros_like(raw, dtype=bool)
    labels, count = ndimage.label(
        raw,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    strong_counts = np.bincount(
        labels.ravel(),
        weights=np.asarray(strong, dtype=np.uint8).ravel(),
        minlength=count + 1,
    )
    keep = (
        (sizes >= max(1, int(minimum_pixels)))
        | (strong_counts >= max(1, int(minimum_strong_pixels)))
    )
    keep[0] = False
    return keep[labels]


def rescue_anchor_internal_onset_from_trajectory(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    anchor_reference: np.ndarray,
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_penalty: np.ndarray,
    valid: np.ndarray,
    cfg: dict[str, Any],
    spectral_similarity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode pixel/sub-object installation continuously from 2011 to 2024.

    Earlier versions used the median of an entire connected 2024 SHP object.
    When only one part of a large station was installed in 2013--2015, the many
    not-yet-built 2024 pixels dominated that median and forced the whole object
    to begin in 2016 or later.  This decoder works at annual pixel/sub-object
    scale, adds local-neighbour support, and traces coherent evidence backward
    from the certain 2024 endpoint.  It therefore preserves progressive build-
    out instead of assigning one onset year to the whole 2024 object.
    """
    settings = cfg.get(
        "anchor_full_series_decoder",
        cfg.get("anchor_internal_rescue", {}),
    )
    onset = np.asarray(onset_index, dtype=np.int16).copy()
    keep = np.asarray(qualified, dtype=bool).copy()
    anchor = np.asarray(anchor_reference, dtype=bool)
    rescued = np.zeros_like(anchor, dtype=bool)
    if not bool(settings.get("enabled", True)) or not np.any(anchor):
        return onset, keep, rescued

    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    c = np.asarray(confuser_penalty, dtype=np.float32)
    if c.ndim == 2:
        c = np.broadcast_to(c[None, :, :], p.shape)
    if c.shape != p.shape:
        raise ValueError("confuser_penalty must be [H,W] or [T,H,W]")

    if spectral_similarity is None:
        sim = np.zeros_like(p, dtype=np.float32)
        has_similarity = False
    else:
        sim = np.asarray(spectral_similarity, dtype=np.float32)
        if sim.shape != p.shape:
            raise ValueError("spectral_similarity must share [T,H,W]")
        has_similarity = True

    t = p.shape[0]
    start_year = int(cfg.get("study", {}).get("start_year", 2011))
    reference_year = int(cfg.get("study", {}).get("reference_year", 2024))
    reference_index = int(np.clip(reference_year - start_year, 0, t - 1))

    wp = float(settings.get("probability_weight", 0.28))
    wf = float(settings.get("fraction_weight", 0.22))
    ws = (
        float(settings.get("spectral_similarity_weight", 0.38))
        if has_similarity
        else 0.0
    )
    wc = float(settings.get("low_confuser_weight", 0.12))
    total = max(wp + wf + ws + wc, EPS)
    presence = np.clip(
        (
            wp * np.nan_to_num(p, nan=0.0)
            + wf * np.nan_to_num(f, nan=0.0)
            + ws * np.nan_to_num(sim, nan=0.0)
            + wc * (1.0 - np.nan_to_num(c, nan=1.0))
        ) / total,
        0.0,
        1.0,
    ).astype(np.float32)

    neighbourhood_pixels = max(
        1,
        int(settings.get("neighbourhood_radius_pixels", 1)),
    )
    neighbourhood_size = 2 * neighbourhood_pixels + 1
    neighbourhood = np.full_like(presence, np.nan, dtype=np.float32)
    for index in range(reference_index + 1):
        local_mean, _ = _masked_local_mean(
            presence[index],
            anchor & v[index],
            neighbourhood_size,
        )
        neighbourhood[index] = local_mean

    pixel_weight = float(settings.get("pixel_presence_weight", 0.68))
    neighbourhood_weight = float(
        settings.get("neighbourhood_presence_weight", 0.32)
    )
    local_total = max(pixel_weight + neighbourhood_weight, EPS)
    supported_score = np.clip(
        (
            pixel_weight * presence
            + neighbourhood_weight * np.nan_to_num(
                neighbourhood,
                nan=0.0,
            )
        ) / local_total,
        0.0,
        1.0,
    )

    annual_threshold = float(
        settings.get("annual_pixel_presence_threshold", 0.39)
    )
    strong_threshold = float(
        settings.get("annual_strong_presence_threshold", 0.52)
    )
    similarity_threshold = float(
        settings.get("pixel_spectral_similarity_threshold", 0.44)
    )
    probability_floor = float(
        settings.get("anchor_probability_floor", 0.16)
    )
    fraction_floor = float(
        settings.get("anchor_fraction_floor", 0.07)
    )
    maximum_confuser = float(
        settings.get("anchor_maximum_confuser", 0.68)
    )

    direct = (
        anchor[None, :, :]
        & v
        & (
            (supported_score >= annual_threshold)
            | (
                (sim >= similarity_threshold)
                & (
                    (p >= probability_floor)
                    | (f >= fraction_floor)
                )
                & (c <= maximum_confuser)
            )
        )
    )
    strong = (
        anchor[None, :, :]
        & v
        & (
            (supported_score >= strong_threshold)
            | (
                (sim >= float(settings.get("strong_similarity_threshold", 0.58)))
                & (f >= float(settings.get("strong_fraction_floor", 0.10)))
                & (c <= maximum_confuser)
            )
        )
    )

    support_window = max(
        1,
        int(settings.get("support_window_years", 3)),
    )
    minimum_supported = max(
        1,
        int(settings.get("minimum_supported_years_in_window", 2)),
    )
    maximum_gap = max(
        0,
        int(settings.get("maximum_internal_gap_years", 1)),
    )

    temporal_candidate = np.zeros_like(direct, dtype=bool)
    for index in range(reference_index + 1):
        stop = min(reference_index + 1, index + support_window)
        local_count = np.count_nonzero(direct[index:stop], axis=0)
        local_gap = (stop - index) - local_count
        temporal_candidate[index] = (
            direct[index]
            & (
                strong[index]
                | (
                    (local_count >= minimum_supported)
                    & (local_gap <= maximum_gap)
                )
            )
        )

    # Bridge at most one weak/missing internal year, but never invent a new
    # leading year. This preserves temporal continuity without moving onset
    # earlier than the first supported observation.
    if maximum_gap > 0 and reference_index >= 2:
        weak_threshold = float(
            settings.get("annual_weak_presence_threshold", 0.30)
        )
        weak = (
            anchor[None, :, :]
            & v
            & (supported_score >= weak_threshold)
            & (c <= maximum_confuser)
        )
        for index in range(1, reference_index):
            bridge = (
                weak[index]
                & temporal_candidate[index - 1]
                & temporal_candidate[index + 1]
            )
            temporal_candidate[index] |= bridge

    annual_min_pixels = max(
        1,
        int(settings.get("annual_minimum_component_pixels", 4)),
    )
    annual_min_strong = max(
        1,
        int(settings.get("annual_minimum_strong_pixels", 2)),
    )
    spatial_candidate = np.zeros_like(temporal_candidate, dtype=bool)
    for index in range(reference_index + 1):
        closed = ndimage.binary_closing(
            temporal_candidate[index],
            structure=np.ones((3, 3), dtype=bool),
            iterations=max(
                0,
                int(settings.get("annual_closing_iterations", 1)),
            ),
        )
        closed &= anchor
        spatial_candidate[index] = _filter_anchor_annual_components(
            closed,
            strong[index],
            annual_min_pixels,
            annual_min_strong,
        )

    # The 2024 endpoint confirms where PV exists by the reference year, but a
    # strict year-by-year backward chain would recreate the very 2015/2016
    # discontinuity this decoder is intended to remove: one cloudy or poorly
    # calibrated intermediate composite could sever a genuine earlier run.
    # Therefore local multi-year evidence is sufficient by default. Optional
    # backward endpoint continuity can still be enabled for sensitivity tests.
    accepted = spatial_candidate.copy()
    accepted[reference_index] = anchor
    if bool(settings.get("require_backward_endpoint_continuity", False)):
        constrained = np.zeros_like(spatial_candidate, dtype=bool)
        constrained[reference_index] = anchor
        continuity_radius = max(
            0,
            int(settings.get("backward_continuity_radius_pixels", 2)),
        )
        for index in range(reference_index - 1, -1, -1):
            future_stop = min(
                reference_index + 1,
                index + maximum_gap + 2,
            )
            future = np.any(constrained[index + 1:future_stop], axis=0)
            if continuity_radius > 0:
                future = ndimage.binary_dilation(
                    future,
                    structure=np.ones((3, 3), dtype=np.uint8),
                    iterations=continuity_radius,
                )
            constrained[index] = spatial_candidate[index] & future
        accepted = constrained

    inferred = np.full(anchor.shape, -1, dtype=np.int16)
    for index in range(reference_index + 1):
        first = accepted[index] & (inferred < 0)
        inferred[first] = np.int16(index)

    eligible = (
        anchor
        & (inferred >= 0)
        & (
            (~keep)
            | (onset < 0)
            | (onset > inferred)
        )
    )
    onset[eligible] = inferred[eligible]
    keep[eligible] = True
    rescued[eligible] = True
    return onset, keep, rescued

def enforce_spatiotemporal_topology(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    anchor_reference: np.ndarray,
    near_anchor: np.ndarray,
    transition_score: np.ndarray,
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_penalty: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suppress fragmented false detections and favor contiguous expansion.

    Remote new sites must be large, coherent and high-confidence. Small annual
    fronts are accepted only when they are spatially connected to an earlier
    accepted PV footprint. Historical candidates inside the confirmed 2024
    anchor follow a separate evidence route.
    """
    settings = cfg.get("spatial_topology", {})
    onset = np.asarray(onset_index, dtype=np.int16).copy()
    candidate = np.asarray(qualified, dtype=bool) & (onset >= 0)
    if not bool(settings.get("enabled", True)):
        return onset, candidate, np.zeros_like(candidate, dtype=bool)

    anchor = np.asarray(anchor_reference, dtype=bool)
    near = np.asarray(near_anchor, dtype=bool) | anchor
    score = np.asarray(transition_score, dtype=np.float32)
    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    c = np.asarray(confuser_penalty, dtype=np.float32)

    n_years = p.shape[0]
    start_year = int(cfg.get("study", {}).get("start_year", 2011))
    reference_year = int(cfg.get("study", {}).get("reference_year", 2024))
    reference_index = int(np.clip(reference_year - start_year, 0, n_years - 1))
    structure = np.ones((3, 3), dtype=np.uint8)
    growth_radius = max(1, int(settings.get("growth_connection_radius_pixels", 2)))
    anchor_radius = max(0, int(settings.get("anchor_connection_radius_pixels", 1)))

    accepted = np.zeros_like(candidate, dtype=bool)
    cleaned_onset = np.full_like(onset, -1, dtype=np.int16)
    site_labels, site_count = ndimage.label(candidate, structure=structure)
    site_sizes = np.bincount(site_labels.ravel(), minlength=site_count + 1)
    site_slices = ndimage.find_objects(site_labels)

    def finite_stat(values: np.ndarray, mode: str = "median", default: float = 0.0) -> float:
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float(default)
        if mode == "q75":
            return float(np.quantile(values, 0.75))
        return float(np.median(values))

    for year_index in range(n_years):
        current = candidate & (onset == year_index)
        if not np.any(current):
            continue
        labels, count = ndimage.label(current, structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        prior_zone = (
            ndimage.binary_dilation(accepted, structure=structure, iterations=growth_radius)
            if np.any(accepted) else np.zeros_like(accepted, dtype=bool)
        )
        anchor_zone = (
            ndimage.binary_dilation(anchor, structure=structure, iterations=anchor_radius)
            if anchor_radius > 0 and np.any(anchor) else anchor
        )

        for label_id, component_slice in enumerate(ndimage.find_objects(labels), start=1):
            if component_slice is None:
                continue
            local = labels[component_slice] == label_id
            component = np.zeros_like(candidate, dtype=bool)
            component[component_slice] = local
            component_size = int(sizes[label_id])

            site_ids = np.unique(site_labels[component])
            site_ids = site_ids[site_ids > 0]
            if site_ids.size:
                site_id = int(site_ids[np.argmax(site_sizes[site_ids])])
                site_size = int(site_sizes[site_id])
                site_slice = site_slices[site_id - 1]
                bbox_area = int(np.prod([axis.stop - axis.start for axis in site_slice])) if site_slice else site_size
                site_fill = float(site_size / max(bbox_area, 1))
            else:
                site_size = component_size
                site_fill = float(component_size / max(local.size, 1))

            touches_anchor = bool(np.any(component & anchor_zone))
            touches_prior = bool(np.any(component & prior_zone))
            near_fraction = float(np.count_nonzero(component & near) / max(component_size, 1))
            probability_median = finite_stat(p[year_index][component])
            fraction_median = finite_stat(f[year_index][component])
            confuser_median = finite_stat(c[component], default=1.0)
            score_median = finite_stat(score[component])
            score_q75 = finite_stat(score[component], mode="q75")
            high = (
                (p[year_index] >= float(settings.get("high_pixel_probability", 0.80)))
                & (f[year_index] >= float(settings.get("high_pixel_fraction", 0.30)))
            )
            high_fraction = float(np.count_nonzero(high & component) / max(component_size, 1))

            if touches_anchor and year_index <= reference_index:
                accept = (
                    component_size >= int(settings.get("historical_anchor_minimum_component_pixels", 6))
                    and score_q75 >= float(settings.get("historical_anchor_minimum_score", 0.52))
                    and probability_median >= float(settings.get("historical_anchor_minimum_probability", 0.60))
                    and confuser_median <= float(settings.get("historical_anchor_maximum_confuser", 0.48))
                )
            elif touches_prior:
                accept = (
                    component_size >= int(settings.get("connected_growth_minimum_component_pixels", 4))
                    and score_median >= float(settings.get("connected_growth_minimum_score", 0.56))
                    and probability_median >= float(settings.get("connected_growth_minimum_probability", 0.68))
                    and fraction_median >= float(settings.get("connected_growth_minimum_fraction", 0.18))
                    and confuser_median <= float(settings.get("connected_growth_maximum_confuser", 0.42))
                )
            elif near_fraction >= float(settings.get("near_anchor_component_fraction", 0.25)):
                accept = (
                    component_size >= int(settings.get("near_anchor_minimum_onset_pixels", 4))
                    and site_size >= int(settings.get("near_anchor_new_block_minimum_pixels", 12))
                    and site_fill >= float(settings.get("near_anchor_minimum_bbox_fill", 0.08))
                    and score_median >= float(settings.get("near_anchor_new_block_minimum_score", 0.62))
                    and probability_median >= float(settings.get("near_anchor_new_block_minimum_probability", 0.78))
                    and fraction_median >= float(settings.get("near_anchor_new_block_minimum_fraction", 0.32))
                    and confuser_median <= float(settings.get("near_anchor_new_block_maximum_confuser", 0.36))
                    and high_fraction >= float(settings.get("near_anchor_high_pixel_fraction", 0.40))
                )
            else:
                accept = (
                    component_size >= int(settings.get("remote_minimum_onset_pixels", 6))
                    and site_size >= int(settings.get("remote_new_site_minimum_pixels", 32))
                    and site_fill >= float(settings.get("remote_minimum_bbox_fill", 0.12))
                    and score_median >= float(settings.get("remote_new_site_minimum_score", 0.68))
                    and probability_median >= float(settings.get("remote_new_site_minimum_probability", 0.86))
                    and fraction_median >= float(settings.get("remote_new_site_minimum_fraction", 0.45))
                    and confuser_median <= float(settings.get("remote_new_site_maximum_confuser", 0.28))
                    and high_fraction >= float(settings.get("remote_high_pixel_fraction", 0.60))
                )

            if accept:
                accepted |= component
                cleaned_onset[component] = np.int16(year_index)

    removed = candidate & ~accepted
    return cleaned_onset, accepted, removed

def filter_transition_components(
    mask: np.ndarray,
    near_anchor: np.ndarray,
    cfg: dict[str, Any],
    *,
    transition_score: np.ndarray | None = None,
    reference_probability: np.ndarray | None = None,
    confuser_penalty: np.ndarray | None = None,
    anchor_core: np.ndarray | None = None,
) -> np.ndarray:
    """Object filter with anchor-safe edge recovery.

    Confirmed-anchor and anchor-connected components are not rejected merely
    because they are small, elongated, or touch tile edges. Remote candidates
    retain stricter area and compactness requirements. High-confidence edge
    pixels adjacent to accepted objects are recovered after filtering.
    """
    s = cfg.get("temporal_signature", {})
    raw = np.asarray(mask, dtype=bool)
    near = np.asarray(near_anchor, dtype=bool)
    anchor = np.zeros_like(raw) if anchor_core is None else np.asarray(anchor_core, dtype=bool)
    labels, count = ndimage.label(raw)
    if count == 0:
        return np.zeros_like(raw, dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.zeros(count + 1, dtype=bool)
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        component = labels[slc] == label_id
        size = int(sizes[label_id])
        fill = size / max(component.size, 1)
        touches_anchor = bool(np.any(anchor[slc] & component))
        touches_near = bool(np.any(near[slc] & component))
        if touches_anchor:
            minimum = int(s.get("anchor_core_minimum_component_pixels", 1))
            accepted = size >= minimum
        elif touches_near:
            minimum = int(s.get("near_anchor_minimum_component_pixels", 2))
            accepted = size >= minimum
        else:
            minimum = int(s.get("remote_minimum_component_pixels", 16))
            area_fraction = size / max(raw.size, 1)
            accepted = (
                size >= minimum
                and area_fraction <= float(s.get("maximum_component_area_fraction", 0.12))
                and fill >= float(s.get("minimum_bbox_fill", 0.06))
            )
        if accepted:
            keep[label_id] = True
    filtered = keep[labels]

    iterations = int(s.get("edge_recovery_iterations", 2))
    if iterations > 0 and filtered.any():
        neighborhood = ndimage.binary_dilation(filtered, iterations=iterations)
        recover = raw & neighborhood & ~filtered
        if transition_score is not None:
            recover &= np.asarray(transition_score) >= float(s.get("edge_recovery_minimum_score", 0.42))
        if reference_probability is not None:
            recover &= np.asarray(reference_probability) >= float(s.get("edge_recovery_minimum_reference_probability", 0.55))
        if confuser_penalty is not None:
            recover &= np.asarray(confuser_penalty) <= float(s.get("edge_recovery_maximum_confuser_penalty", 0.55))
        # Recovery is deliberately limited to anchor-supported neighborhoods.
        recover &= near | anchor
        filtered |= recover
    return filtered


def _safe_gradient_magnitude(values: np.ndarray) -> np.ndarray:
    """Gradient magnitude that is safe for tiny clipped windows."""
    x = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    if x.ndim != 2 or x.size == 0:
        return out
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    fill = float(_nanmedian(x[finite]))
    safe = np.where(finite, x, fill).astype(np.float32, copy=False)
    h, w = safe.shape
    if h >= 2 and w >= 2:
        gy, gx = np.gradient(safe)
        out = np.hypot(gx, gy).astype(np.float32)
    elif h >= 2 and w == 1:
        out = np.abs(np.gradient(safe[:, 0])).reshape(h, 1).astype(np.float32)
    elif h == 1 and w >= 2:
        out = np.abs(np.gradient(safe[0, :])).reshape(1, w).astype(np.float32)
    out[~finite] = 0.0
    return out


def _normalized_positive_shift(post: np.ndarray, pre: np.ndarray, scale: float) -> np.ndarray:
    shift = np.nan_to_num(post - pre, nan=0.0)
    return np.clip(shift / max(float(scale), EPS), 0.0, 1.0).astype(np.float32)


def _absolute_shift(post: np.ndarray, pre: np.ndarray, scale: float) -> np.ndarray:
    shift = np.nan_to_num(np.abs(post - pre), nan=0.0)
    return np.clip(shift / max(float(scale), EPS), 0.0, 1.0).astype(np.float32)


def compute_change_boundary_coherence_score(
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_penalty: np.ndarray,
    valid: np.ndarray,
    reflectance: np.ndarray,
    spectral_similarity: np.ndarray | None,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a full-series ET-PAVG/CBC score for PV installation evidence.

    The score is used only as supporting evidence for temporally coherent
    objects.  It intentionally differs from a simple spectral-similarity mask:
    annual PV evidence must be supported by local boundary change, greenness or
    brightness transition, low-confuser evidence and future persistence.
    """
    settings = cfg.get("change_cbc_sankey", {})
    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    c = np.asarray(confuser_penalty, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    refl = np.asarray(reflectance, dtype=np.float32)
    if c.ndim == 2:
        c = np.broadcast_to(c[None, :, :], p.shape)
    if refl.ndim != 4 or refl.shape[1] < 6:
        raise ValueError("reflectance must have shape [T,6,H,W]")
    sim = np.zeros_like(p, dtype=np.float32) if spectral_similarity is None else np.asarray(spectral_similarity, dtype=np.float32)

    blue = refl[:, 0]
    green = refl[:, 1]
    red = refl[:, 2]
    nir = refl[:, 3]
    swir1 = refl[:, 4]
    swir2 = refl[:, 5]
    ndvi = (nir - red) / (nir + red + EPS)
    ndmi = (nir - swir1) / (nir + swir1 + EPS)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)
    brightness = np.nanmean(refl[:, :6], axis=1)
    tc_greenness = (
        -0.2941 * blue - 0.2430 * green - 0.5424 * red
        + 0.7276 * nir + 0.0713 * swir1 - 0.1608 * swir2
    )

    t = p.shape[0]
    pre_window = max(1, int(settings.get("pre_window_years", 2)))
    post_window = max(1, int(settings.get("post_window_years", 2)))
    future_window = max(1, int(settings.get("future_support_window_years", 4)))
    future_min = max(1, int(settings.get("future_support_minimum_years", 2)))

    base_score = np.clip(
        float(settings.get("probability_weight", 0.26)) * np.nan_to_num(p, nan=0.0)
        + float(settings.get("fraction_weight", 0.22)) * np.nan_to_num(f, nan=0.0)
        + float(settings.get("similarity_weight", 0.24)) * np.nan_to_num(sim, nan=0.0)
        + float(settings.get("low_confuser_weight", 0.16)) * (1.0 - np.nan_to_num(c, nan=1.0)),
        0.0,
        1.0,
    ).astype(np.float32)

    boundary = np.zeros_like(p, dtype=np.float32)
    transition = np.zeros_like(p, dtype=np.float32)
    future_support = np.zeros_like(p, dtype=np.float32)

    base_presence_thr = float(settings.get("future_presence_threshold", 0.34))
    for i in range(t):
        pre_slice = slice(max(0, i - pre_window), i)
        post_slice = slice(i, min(t, i + post_window))
        if i == 0:
            pre_p = p[i]
            pre_f = f[i]
            pre_ndvi = ndvi[i]
            pre_green = tc_greenness[i]
            pre_bright = brightness[i]
            pre_bsi = bsi[i]
        else:
            pre_p = _nanmedian(p[pre_slice], axis=0)
            pre_f = _nanmedian(f[pre_slice], axis=0)
            pre_ndvi = _nanmedian(ndvi[pre_slice], axis=0)
            pre_green = _nanmedian(tc_greenness[pre_slice], axis=0)
            pre_bright = _nanmedian(brightness[pre_slice], axis=0)
            pre_bsi = _nanmedian(bsi[pre_slice], axis=0)
        # The installation jump is evaluated from the candidate year itself,
        # not from a future-window median.  Future years are used only for
        # persistence support.  This prevents assigning a pre-installation year
        # immediately before a real construction event.
        post_p = p[i]
        post_f = f[i]
        post_ndvi = ndvi[i]
        post_green = tc_greenness[i]
        post_bright = brightness[i]
        post_bsi = bsi[i]

        p_jump = _normalized_positive_shift(post_p, pre_p, float(settings.get("probability_jump_scale", 0.18)))
        f_jump = _normalized_positive_shift(post_f, pre_f, float(settings.get("fraction_jump_scale", 0.12)))
        green_jump = np.maximum(
            _absolute_shift(post_ndvi, pre_ndvi, float(settings.get("ndvi_jump_scale", 0.12))),
            _absolute_shift(post_green, pre_green, float(settings.get("tc_greenness_jump_scale", 0.09))),
        )
        bright_jump = _absolute_shift(post_bright, pre_bright, float(settings.get("brightness_jump_scale", 0.10)))
        bsi_jump = _absolute_shift(post_bsi, pre_bsi, float(settings.get("bsi_jump_scale", 0.10)))
        transition[i] = np.clip(
            0.30 * p_jump + 0.25 * f_jump + 0.20 * green_jump + 0.15 * bright_jump + 0.10 * bsi_jump,
            0.0,
            1.0,
        )

        edge_source = np.nan_to_num(base_score[i], nan=0.0) + 0.45 * np.nan_to_num(transition[i], nan=0.0)
        boundary[i] = np.clip(
            _safe_gradient_magnitude(edge_source) / max(float(settings.get("edge_gradient_scale", 0.12)), EPS),
            0.0,
            1.0,
        )
        stop = min(t, i + future_window)
        future_presence = (base_score[i:stop] >= base_presence_thr) & v[i:stop]
        future_support[i] = np.clip(
            np.count_nonzero(future_presence, axis=0) / float(max(future_min, 1)),
            0.0,
            1.0,
        ).astype(np.float32)

    cbc = np.clip(
        float(settings.get("base_score_weight", 0.40)) * base_score
        + float(settings.get("transition_weight", 0.22)) * transition
        + float(settings.get("boundary_weight", 0.18)) * boundary
        + float(settings.get("future_support_weight", 0.20)) * future_support,
        0.0,
        1.0,
    ).astype(np.float32)
    cbc[~v] = 0.0
    return cbc, transition.astype(np.float32), boundary.astype(np.float32), future_support.astype(np.float32)


def refine_anchor_onset_with_change_boundary_coherence(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    anchor_reference: np.ndarray,
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_penalty: np.ndarray,
    valid: np.ndarray,
    reflectance: np.ndarray,
    cfg: dict[str, Any],
    spectral_similarity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover coherent early PV objects inside the 2024 SHP using CBC-PAVG.

    This is the v24.13 retained correction for the 2011--2015 problem.  The same rule is
    applied to every year up to the reference year: first test whether a local
    connected object exists in the annual image, using CBC-PAVG, boundary jump,
    greenness/brightness transition and future support; only then assign pixels
    inside that accepted object.  This avoids the v26 one-size result and the
    v24.4 whole-object-median failure.
    """
    settings = cfg.get("change_cbc_sankey", {})
    if not bool(settings.get("enabled", True)):
        return onset_index, qualified, np.zeros_like(anchor_reference, dtype=bool), np.zeros_like(anchor_reference, dtype=np.float32)
    anchor = np.asarray(anchor_reference, dtype=bool)
    if not np.any(anchor):
        return onset_index, qualified, np.zeros_like(anchor, dtype=bool), np.zeros_like(anchor, dtype=np.float32)

    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    c = np.asarray(confuser_penalty, dtype=np.float32)
    if c.ndim == 2:
        c = np.broadcast_to(c[None, :, :], p.shape)
    v = np.asarray(valid, dtype=bool)
    sim = np.zeros_like(p, dtype=np.float32) if spectral_similarity is None else np.asarray(spectral_similarity, dtype=np.float32)
    cbc, transition, boundary, future_support = compute_change_boundary_coherence_score(
        p, f, c, v, reflectance, sim, cfg
    )

    onset = np.asarray(onset_index, dtype=np.int16).copy()
    keep = np.asarray(qualified, dtype=bool).copy()
    rescued = np.zeros_like(anchor, dtype=bool)
    diagnostic = np.zeros_like(anchor, dtype=np.float32)

    t = p.shape[0]
    start_year = int(cfg.get("study", {}).get("start_year", 2011))
    reference_year = int(cfg.get("study", {}).get("reference_year", 2024))
    reference_index = int(np.clip(reference_year - start_year, 0, t - 1))
    # The same logic is applied to all years through 2024.  The common visual
    # failure is before 2016, but using one rule avoids a hidden 2015/2016 seam.
    end_index = min(reference_index, int(settings.get("decode_until_year", reference_year)) - start_year)

    strong_thr = float(settings.get("object_seed_threshold", 0.48))
    weak_thr = float(settings.get("object_growth_threshold", 0.31))
    boundary_thr = float(settings.get("minimum_boundary_score", 0.22))
    transition_thr = float(settings.get("minimum_transition_score", 0.20))
    future_thr = float(settings.get("minimum_future_support", 0.50))
    max_conf = float(settings.get("maximum_anchor_confuser", 0.72))
    min_pixels = int(settings.get("minimum_object_pixels", 6))
    min_seed_pixels = int(settings.get("minimum_seed_pixels", 2))
    min_mean_score = float(settings.get("minimum_object_mean_score", 0.36))
    min_q80_score = float(settings.get("minimum_object_q80_score", 0.46))
    bridge_iters = int(settings.get("internal_gap_closing_iterations", 1))
    grow_iters = int(settings.get("weak_object_growth_iterations", 1))

    structure = np.ones((3, 3), dtype=np.uint8)
    for year_index in range(0, end_index + 1):
        current_observation = (
            (p[year_index] >= float(settings.get("minimum_current_probability", 0.16)))
            | (f[year_index] >= float(settings.get("minimum_current_fraction", 0.06)))
            | (sim[year_index] >= float(settings.get("minimum_current_similarity", 0.42)))
            | (
                (boundary[year_index] >= boundary_thr)
                & (transition[year_index] >= transition_thr)
            )
        )
        seed = (
            anchor
            & v[year_index]
            & current_observation
            & (c[year_index] <= max_conf)
            & (
                (cbc[year_index] >= strong_thr)
                | (
                    (cbc[year_index] >= weak_thr)
                    & (boundary[year_index] >= boundary_thr)
                    & (transition[year_index] >= transition_thr)
                    & (future_support[year_index] >= future_thr)
                )
                | (
                    (sim[year_index] >= float(settings.get("similarity_rescue_threshold", 0.52)))
                    & (future_support[year_index] >= future_thr)
                    & ((p[year_index] >= 0.18) | (f[year_index] >= 0.06))
                )
            )
        )
        if not np.any(seed):
            continue
        grown = seed.copy()
        if grow_iters > 0:
            weak = (
                anchor
                & v[year_index]
                & (c[year_index] <= max_conf)
                & (
                    (cbc[year_index] >= weak_thr)
                    | (
                        (sim[year_index] >= float(settings.get("weak_similarity_threshold", 0.42)))
                        & (future_support[year_index] >= future_thr)
                    )
                )
            )
            grown = ndimage.binary_dilation(seed, structure=structure, iterations=grow_iters) & weak
            grown |= seed
        if bridge_iters > 0:
            grown = ndimage.binary_closing(grown, structure=structure, iterations=bridge_iters) & anchor
            grown = ndimage.binary_fill_holes(grown) & anchor
        labels, count = ndimage.label(grown, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        seed_counts = np.bincount(
            labels.ravel(), weights=seed.astype(np.uint8).ravel(), minlength=count + 1
        )
        for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
            if slc is None:
                continue
            component = labels[slc] == label_id
            component_full = np.zeros_like(anchor, dtype=bool)
            component_full[slc] = component
            size = int(sizes[label_id])
            if size < min_pixels or int(seed_counts[label_id]) < min_seed_pixels:
                continue
            values = cbc[year_index][component_full]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean_score = float(np.mean(values))
            q80 = float(np.quantile(values, 0.80))
            bmean = float(np.mean(boundary[year_index][component_full]))
            tmean = float(np.mean(transition[year_index][component_full]))
            fsmean = float(np.mean(future_support[year_index][component_full]))
            if not (
                mean_score >= min_mean_score
                and q80 >= min_q80_score
                and fsmean >= future_thr
                and (bmean >= boundary_thr or tmean >= transition_thr)
            ):
                continue
            update = component_full & (
                (onset < 0) | (onset > year_index) | (~keep)
            )
            if not np.any(update):
                continue
            onset[update] = np.int16(year_index)
            keep[update] = True
            rescued[update] = True
            diagnostic[update] = np.maximum(diagnostic[update], cbc[year_index][update])
    return onset, keep, rescued, diagnostic

def annual_masks_from_onset(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    t = valid.shape[0]
    result = np.zeros_like(valid, dtype=bool)
    for index in range(t):
        result[index] = qualified & (onset_index >= 0) & (onset_index <= index) & valid[index]
    return result


def annual_latent_masks_from_onset(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    n_years: int,
) -> np.ndarray:
    """Cumulative installed footprint: installation is irreversible."""
    result = np.zeros((int(n_years), *qualified.shape), dtype=bool)
    for index in range(int(n_years)):
        result[index] = qualified & (onset_index >= 0) & (onset_index <= index)
    return result


def annual_active_masks_from_evidence(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    valid: np.ndarray,
    cfg: dict[str, Any],
    *,
    anchor_reference: np.ndarray | None = None,
    spectral_similarity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Persistent active state with reference-guided post-2024 observation.

    The 2024 anchor is an installed and active seed. In 2025 it is reassessed
    using actual probability, LSMA, confuser and spectral-similarity evidence,
    but one post-reference observation can never be labelled as retirement.
    Missing or ambiguous observations retain the previous state. Retirement
    requires multi-year consecutive strong negative evidence.
    """
    s = cfg.get("active_state", {})
    endpoint = cfg.get("reference_guided_endpoint", {})
    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    sim = np.zeros_like(p, dtype=np.float32) if spectral_similarity is None else np.asarray(spectral_similarity, dtype=np.float32)
    anchor = np.zeros(p.shape[1:], dtype=bool) if anchor_reference is None else np.asarray(anchor_reference, dtype=bool)
    t,h,w = p.shape
    cumulative = annual_latent_masks_from_onset(onset_index, qualified, t)
    active = np.zeros_like(cumulative, dtype=bool)
    retirement_index = np.full((h,w), -1, dtype=np.int16)
    start_year = int(cfg.get("study", {}).get("start_year", 2011))
    reference_year = int(cfg.get("study", {}).get("reference_year", 2024))
    reference_index = int(np.clip(reference_year-start_year,0,t-1))
    neg_p=float(endpoint.get("negative_probability_maximum", s.get("retirement_probability_maximum",0.22)))
    neg_f=float(endpoint.get("negative_fraction_maximum", s.get("retirement_fraction_maximum",0.18)))
    neg_sim=float(endpoint.get("negative_similarity_maximum",0.25))
    retirement_years=max(int(s.get("retirement_consecutive_valid_years",3)), int(endpoint.get("minimum_consecutive_negative_years",3)))
    pos_p=float(endpoint.get("positive_probability_minimum", s.get("reactivation_probability_minimum",0.72)))
    pos_f=float(endpoint.get("positive_fraction_minimum", s.get("reactivation_fraction_minimum",0.30)))
    pos_sim=float(endpoint.get("positive_similarity_minimum",0.48))
    reactivation_years=int(s.get("reactivation_consecutive_valid_years",2))
    state=np.zeros((h,w),dtype=bool); negative_run=np.zeros((h,w),dtype=np.uint8); positive_run=np.zeros((h,w),dtype=np.uint8)
    for index in range(t):
        installed=cumulative[index]
        newly=installed & ~state & (onset_index==index)
        state[newly]=True
        if index==reference_index:
            state[anchor & installed]=True
            negative_run[anchor]=0
        if bool(cfg.get("landsat_object_pavg", {}).get("post_reference_anchor_active_retention", True)) and index >= reference_index and index <= reference_index + 1:
            # The 2024 SHP is a confirmed installed endpoint.  One noisy
            # post-reference annual composite must not create internal holes
            # in the observed active footprint.
            state[anchor & installed] = True
            negative_run[anchor] = 0
        observed=v[index] & installed
        strong_positive=observed & (((p[index]>=pos_p)&(f[index]>=pos_f)) | (sim[index]>=pos_sim))
        strong_negative=observed & (p[index]<=neg_p) & (f[index]<=neg_f) & (sim[index]<=neg_sim)
        neutral=observed & ~strong_negative & ~strong_positive
        negative_run[strong_negative & state]=np.minimum(255,negative_run[strong_negative & state]+1)
        negative_run[strong_positive | neutral]=0
        # A single 2025 endpoint observation cannot create instant retirement.
        retire=state & (negative_run>=retirement_years)
        if bool(endpoint.get("single_post_reference_year_cannot_be_retired",True)) and index<=reference_index+1:
            retire &= ~anchor
        state[retire]=False
        retirement_index[retire & (retirement_index<0)]=index
        positive_run[strong_positive & ~state]=np.minimum(255,positive_run[strong_positive & ~state]+1)
        positive_run[~strong_positive]=0
        reactivate=installed & ~state & (positive_run>=reactivation_years)
        state[reactivate]=True; negative_run[reactivate]=0
        active[index]=installed & state
    observed_active=active & v
    if bool(cfg.get("landsat_object_pavg", {}).get("anchor_endpoint_hole_fill", True)):
        for index in range(reference_index, min(t, reference_index + 2)):
            observed_active[index] |= anchor & cumulative[index]
            active[index] |= anchor & cumulative[index]
    return active, observed_active, retirement_index



def _local_mean_valid(values: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    size = max(3, int(size) | 1)
    numerator = ndimage.uniform_filter(np.where(valid, values, 0.0).astype(np.float32), size=size, mode="nearest")
    denominator = ndimage.uniform_filter(valid.astype(np.float32), size=size, mode="nearest")
    return numerator / np.maximum(denominator, EPS)


def _local_std_valid(values: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    mean = _local_mean_valid(values, valid, size)
    second = _local_mean_valid(values * values, valid, size)
    return np.sqrt(np.maximum(second - mean * mean, 0.0)).astype(np.float32)


def _fill_small_component_holes(mask: np.ndarray, anchor: np.ndarray, max_pixels: int) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask) & anchor
    holes = filled & ~mask
    if not np.any(holes):
        return mask
    labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, count + 1):
        if int(sizes[label_id]) <= int(max_pixels):
            keep |= labels == label_id
    return mask | keep


def compute_landsat_object_pavg_score(
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_penalty: np.ndarray,
    valid: np.ndarray,
    reflectance: np.ndarray,
    cfg: dict[str, Any],
    spectral_similarity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute object-aware Landsat PV evidence for 30 m annual composites.

    The score follows the remote-sensing interpretation that utility-scale PV
    in Landsat imagery should be a coherent low-reflectance / high-texture
    object relative to its local desert background.  The score therefore uses
    local object contrast and row/edge texture in addition to CNN probability,
    LSMA PV fraction, spectral similarity and future persistence.  It is not a
    pixel-only spectral threshold.
    """
    s = cfg.get("landsat_object_pavg", {})
    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    c = np.asarray(confuser_penalty, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    if c.ndim == 2:
        c = np.broadcast_to(c[None, :, :], p.shape)
    refl = np.asarray(reflectance, dtype=np.float32)
    if refl.ndim != 4 or refl.shape[1] < 6:
        raise ValueError("reflectance must have shape [T,6,H,W]")
    sim = np.zeros_like(p, dtype=np.float32) if spectral_similarity is None else np.asarray(spectral_similarity, dtype=np.float32)

    win = int(s.get("local_window_pixels", 17))
    blue, green, red, nir, swir1, swir2 = [refl[:, i] for i in range(6)]
    # Bright desert background versus dark PV panel response. Use a broad-band
    # brightness average rather than a single band to reduce sensor noise.
    brightness = np.nanmean(refl[:, :6], axis=1).astype(np.float32)
    visible = np.nanmean(refl[:, :3], axis=1).astype(np.float32)
    swir_bright = np.nanmean(refl[:, 4:6], axis=1).astype(np.float32)
    ndvi = (nir - red) / (nir + red + EPS)
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)

    t = p.shape[0]
    dark = np.zeros_like(p, dtype=np.float32)
    texture = np.zeros_like(p, dtype=np.float32)
    object_edge = np.zeros_like(p, dtype=np.float32)
    future = np.zeros_like(p, dtype=np.float32)

    dark_scale = float(s.get("minimum_dark_contrast", 0.08)) * 2.2
    texture_scale = max(float(s.get("minimum_texture_score", 0.08)) * 1.8, EPS)
    for i in range(t):
        vv = v[i]
        local_bright = _local_mean_valid(brightness[i], vv, win)
        local_vis = _local_mean_valid(visible[i], vv, win)
        local_swir = _local_mean_valid(swir_bright[i], vv, win)
        dark_raw = np.maximum(local_bright - brightness[i], 0.0)
        dark_vis = np.maximum(local_vis - visible[i], 0.0)
        dark_swir = np.maximum(local_swir - swir_bright[i], 0.0)
        dark[i] = np.clip((0.55 * dark_raw + 0.25 * dark_vis + 0.20 * dark_swir) / max(dark_scale, EPS), 0.0, 1.0)
        # PV farms show sharp artificial row/block boundaries in otherwise
        # smooth desert; combine local std and gradient of brightness/BSI/NDVI.
        local_std = _local_std_valid(brightness[i], vv, max(5, win // 2))
        grad_bright = _safe_gradient_magnitude(np.nan_to_num(brightness[i], nan=0.0))
        grad_bsi = _safe_gradient_magnitude(np.nan_to_num(bsi[i], nan=0.0))
        grad_ndvi = _safe_gradient_magnitude(np.nan_to_num(ndvi[i], nan=0.0))
        tex = 0.45 * local_std + 0.35 * grad_bright + 0.12 * grad_bsi + 0.08 * grad_ndvi
        texture[i] = np.clip(tex / texture_scale, 0.0, 1.0)
        object_edge[i] = np.clip(_safe_gradient_magnitude(dark[i] + 0.35 * texture[i]) / 0.18, 0.0, 1.0)

    base = np.clip(
        float(s.get("probability_weight", 0.20)) * np.nan_to_num(p, nan=0.0)
        + float(s.get("fraction_weight", 0.18)) * np.nan_to_num(f, nan=0.0)
        + float(s.get("similarity_weight", 0.18)) * np.nan_to_num(sim, nan=0.0)
        + float(s.get("dark_contrast_weight", 0.18)) * dark
        + float(s.get("texture_weight", 0.12)) * texture
        + float(s.get("boundary_weight", 0.08)) * object_edge
        + 0.06 * (1.0 - np.nan_to_num(c, nan=1.0)),
        0.0,
        1.0,
    ).astype(np.float32)

    window = max(1, int(s.get("future_support_window_years", 4)))
    minimum = max(1, int(s.get("future_support_minimum_years", 2)))
    presence_thr = float(s.get("grow_score_threshold", 0.36))
    for i in range(t):
        stop = min(t, i + window)
        future_presence = (base[i:stop] >= presence_thr) & v[i:stop]
        future[i] = np.clip(np.count_nonzero(future_presence, axis=0) / float(minimum), 0.0, 1.0)

    score = np.clip(base + float(s.get("future_weight", 0.06)) * future, 0.0, 1.0).astype(np.float32)
    score[~v] = 0.0
    return score, dark.astype(np.float32), texture.astype(np.float32), future.astype(np.float32)


def refine_anchor_onset_with_landsat_object_pavg(
    onset_index: np.ndarray,
    qualified: np.ndarray,
    anchor_reference: np.ndarray,
    probability: np.ndarray,
    fraction_evidence: np.ndarray,
    confuser_penalty: np.ndarray,
    valid: np.ndarray,
    reflectance: np.ndarray,
    cfg: dict[str, Any],
    spectral_similarity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover early and internally complete PV objects inside 2024 anchors.

    This v24.8 refinement is deliberately anchored to the v24 line: it keeps
    the 2024 SHP endpoint rule but removes the early-year seam by applying the
    same annual object evidence to 2011--2024.  A year is accepted only if the
    current annual composite contains a coherent PV-like object.  Future support
    proves persistence, not pre-construction presence.
    """
    s = cfg.get("landsat_object_pavg", {})
    if not bool(s.get("enabled", True)) or not bool(s.get("anchor_internal_enabled", True)):
        return onset_index, qualified, np.zeros_like(anchor_reference, dtype=bool), np.zeros_like(anchor_reference, dtype=np.float32)
    anchor = np.asarray(anchor_reference, dtype=bool)
    if not np.any(anchor):
        return onset_index, qualified, np.zeros_like(anchor, dtype=bool), np.zeros_like(anchor, dtype=np.float32)

    p = np.asarray(probability, dtype=np.float32)
    f = np.asarray(fraction_evidence, dtype=np.float32)
    c = np.asarray(confuser_penalty, dtype=np.float32)
    if c.ndim == 2:
        c = np.broadcast_to(c[None, :, :], p.shape)
    v = np.asarray(valid, dtype=bool)
    sim = np.zeros_like(p, dtype=np.float32) if spectral_similarity is None else np.asarray(spectral_similarity, dtype=np.float32)

    score, dark, texture, future = compute_landsat_object_pavg_score(
        p, f, c, v, reflectance, cfg, spectral_similarity=sim
    )

    onset = np.asarray(onset_index, dtype=np.int16).copy()
    keep = np.asarray(qualified, dtype=bool).copy()
    rescued = np.zeros_like(anchor, dtype=bool)
    diagnostic = np.zeros_like(anchor, dtype=np.float32)

    start_year = int(cfg.get("study", {}).get("start_year", 2011))
    reference_year = int(cfg.get("study", {}).get("reference_year", 2024))
    reference_index = int(np.clip(reference_year - start_year, 0, p.shape[0] - 1))
    end_index = min(reference_index, int(s.get("decode_until_year", reference_year)) - start_year)
    seed_thr = float(s.get("seed_score_threshold", 0.54))
    grow_thr = float(s.get("grow_score_threshold", 0.36))
    min_obj = int(s.get("minimum_object_pixels", 5))
    min_seed = int(s.get("minimum_seed_pixels", 2))
    min_mean = float(s.get("minimum_object_mean_score", 0.42))
    min_q75 = float(s.get("minimum_object_q75_score", 0.50))
    min_dark = float(s.get("minimum_dark_contrast", 0.08))
    min_tex = float(s.get("minimum_texture_score", 0.08))
    future_thr = float(s.get("minimum_future_support", 0.45))
    max_conf = float(s.get("maximum_anchor_confuser", 0.76))
    grow_iter = int(s.get("grow_iterations", 2))
    close_iter = int(s.get("gap_close_iterations", 1))
    hole_max = int(s.get("hole_fill_max_pixels", 48))
    structure = np.ones((3, 3), dtype=np.uint8)

    for year_index in range(0, end_index + 1):
        # Current-year evidence is mandatory.  Similarity is allowed only when
        # accompanied by a dark/texture object response or model/LSMA support.
        current = (
            (p[year_index] >= 0.12)
            | (f[year_index] >= 0.045)
            | ((sim[year_index] >= 0.38) & ((dark[year_index] >= min_dark) | (texture[year_index] >= min_tex)))
            | ((dark[year_index] >= min_dark) & (texture[year_index] >= min_tex))
        )
        seed = (
            anchor
            & v[year_index]
            & current
            & (c[year_index] <= max_conf)
            & (future[year_index] >= future_thr)
            & (
                (score[year_index] >= seed_thr)
                | (
                    (score[year_index] >= grow_thr)
                    & (dark[year_index] >= min_dark)
                    & (texture[year_index] >= min_tex)
                )
            )
        )
        if not np.any(seed):
            continue
        weak = (
            anchor
            & v[year_index]
            & (c[year_index] <= max_conf)
            & (score[year_index] >= grow_thr)
            & current
        )
        grown = seed.copy()
        if grow_iter > 0:
            grown = (ndimage.binary_dilation(seed, structure=structure, iterations=grow_iter) & weak) | seed
        if close_iter > 0:
            grown = ndimage.binary_closing(grown, structure=structure, iterations=close_iter) & anchor
            grown = _fill_small_component_holes(grown, anchor, hole_max)
        labels, count = ndimage.label(grown, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        seed_counts = np.bincount(labels.ravel(), weights=seed.astype(np.uint8).ravel(), minlength=count + 1)
        objects = ndimage.find_objects(labels)
        for label_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            component = labels[slc] == label_id
            comp = np.zeros_like(anchor, dtype=bool)
            comp[slc] = component
            if int(sizes[label_id]) < min_obj or int(seed_counts[label_id]) < min_seed:
                continue
            vals = score[year_index][comp]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean = float(np.mean(vals))
            q75 = float(np.quantile(vals, 0.75))
            dmean = float(np.mean(dark[year_index][comp]))
            tmean = float(np.mean(texture[year_index][comp]))
            fsmean = float(np.mean(future[year_index][comp]))
            if not (mean >= min_mean and q75 >= min_q75 and fsmean >= future_thr and (dmean >= min_dark or tmean >= min_tex)):
                continue
            update = comp & ((onset < 0) | (onset > year_index) | (~keep))
            if not np.any(update):
                continue
            onset[update] = np.int16(year_index)
            keep[update] = True
            rescued[update] = True
            diagnostic[update] = np.maximum(diagnostic[update], score[year_index][update])
    return onset, keep, rescued, diagnostic
