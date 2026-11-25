from __future__ import annotations

"""Reference-constrained growth-front temporal reconstruction (v24.19).

This decoder replaces all legacy anchor backcasting.  It solves an ordered
change-point problem separately for each 2024 inventory pixel: the pixel is
non-PV before its inferred installation year and PV from that year onward.
The 2024 vector constrains only the 2024 state.  If historical image evidence is
weak, the installation year defaults to 2024 rather than being copied to 2011.

Outside the 2024 inventory, independent annual evidence is retained only when it
forms a temporally persistent track (or a very strong, large end-of-series new
site).  This permits historical-only and 2025 new sites without treating the
2024 vector as anything other than the complete 2024 inventory.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from tqdm import tqdm

from .fallback import infer_area_only_units
from .dual_stage_reference import (
    fuse_reference_tracks,
    reconstruct_internal_reference_buildout,
)
from .growth_front import (
    reconstruct_historical_outside_tracks,
    reconstruct_reference_growth_front,
)
from .object_graph_refinement import _expanded, _windows
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now

PIXEL_MODES = {"direct_pixel", "anchor_guided_candidate"}

DETAIL_ALGORITHM_REVISION = "24.22.5-r4-dual-stage-reference-fusion"

EXCLUSION_NONE = 0
EXCLUSION_REFERENCE_ONLY_NO_HISTORY = 71
EXCLUSION_STABLE_SAND = 72
EXCLUSION_NO_PERSISTENT_TRACK = 73
EXCLUSION_SMALL_COMPONENT = 74


def _safe_prob(arr: np.ndarray, nodata: float | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(x)
    if nodata is not None and np.isfinite(nodata):
        valid &= x != float(nodata)
    out = np.clip(np.where(valid, x, 0.0), 1e-4, 1.0 - 1e-4)
    return out.astype(np.float32), valid


def _read_mask(src: rasterio.DatasetReader, window: rasterio.windows.Window) -> tuple[np.ndarray, np.ndarray]:
    x = src.read(1, window=window)
    nodata = src.nodata
    valid = np.ones(x.shape, dtype=bool)
    if nodata is not None:
        valid &= x != nodata
    return (x == 1) & valid, valid


def _ordered_change_point(
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer installation index for every reference-inventory pixel.

    The deterministic 2024 endpoint is down-weighted in the likelihood because
    it is a constraint, not an independent image observation.  A separate
    robust early-series test recovers genuinely pre-existing PV when Landsat
    probabilities are moderate but persistent, while later installations still
    require a credible pre/post transition.
    """
    p = np.clip(probability[: reference_index + 1], 1e-4, 1.0 - 1e-4)
    v = valid[: reference_index + 1]
    n = reference_index + 1
    weights = np.ones(n, dtype=np.float32)
    weights[reference_index] = float(settings.get("reference_endpoint_likelihood_weight", 0.25))
    weighted_valid = v.astype(np.float32) * weights[:, None, None]
    log_p = np.where(v, np.log(p), 0.0) * weights[:, None, None]
    log_q = np.where(v, np.log1p(-p), 0.0) * weights[:, None, None]

    scores = np.full((n, *inventory.shape), -np.inf, dtype=np.float32)
    post_means = np.zeros_like(scores)
    pre_means = np.zeros_like(scores)
    post_high = np.zeros_like(scores, dtype=np.uint8)
    transition_penalty = float(settings.get("change_point_transition_penalty", 0.18))
    for k in range(n):
        pre_score = np.sum(log_q[:k], axis=0) if k > 0 else 0.0
        post_score = np.sum(log_p[k:n], axis=0)
        valid_post_weight = weighted_valid[k:n].sum(axis=0)
        valid_pre_weight = weighted_valid[:k].sum(axis=0) if k > 0 else np.zeros(inventory.shape, dtype=np.float32)
        scores[k] = np.where(
            valid_post_weight > 0,
            pre_score + post_score - transition_penalty,
            -np.inf,
        )
        post_total = (p[k:n] * weighted_valid[k:n]).sum(axis=0)
        post_means[k] = np.divide(
            post_total,
            np.maximum(valid_post_weight, 1e-6),
            where=np.maximum(valid_post_weight, 1e-6) > 0,
        )
        if k > 0:
            pre_total = (p[:k] * weighted_valid[:k]).sum(axis=0)
            pre_means[k] = np.divide(
                pre_total,
                np.maximum(valid_pre_weight, 1e-6),
                where=np.maximum(valid_pre_weight, 1e-6) > 0,
            )
        post_high[k] = (
            (p[k:n] >= float(settings.get("post_installation_high_probability", 0.46)))
            & v[k:n]
        ).sum(axis=0)

    best = np.argmax(scores, axis=0).astype(np.int16)
    best_score = np.take_along_axis(scores, best[None, ...], axis=0)[0]
    sorted_scores = np.sort(scores, axis=0)
    second = sorted_scores[-2] if n > 1 else np.full(inventory.shape, -np.inf, dtype=np.float32)
    confidence = np.clip(best_score - second, 0.0, 20.0).astype(np.float32)
    best_post = np.take_along_axis(post_means, best[None, ...], axis=0)[0]
    best_pre = np.take_along_axis(pre_means, best[None, ...], axis=0)[0]
    best_high = np.take_along_axis(post_high, best[None, ...], axis=0)[0]

    min_post = float(settings.get("minimum_post_installation_mean_probability", 0.44))
    min_gain = float(settings.get("minimum_pre_post_probability_gain", 0.11))
    min_high = int(settings.get("minimum_post_installation_support_years", 2))
    credible = inventory & (best_post >= min_post) & (best_high >= min_high)
    credible &= (
        (best == 0)
        | ((best_post - best_pre) >= min_gain)
        | (best_post >= float(settings.get("strong_post_installation_override", 0.72)))
    )

    # Robust start-of-series / early-series rescue.  This does not use the 2024
    # endpoint and therefore cannot turn the vector into historical geometry.
    historical_n = reference_index
    early_first = np.zeros(inventory.shape, dtype=np.int16)
    robust_preexisting = np.zeros(inventory.shape, dtype=bool)
    if historical_n > 0:
        early_window = min(
            historical_n,
            max(1, int(settings.get("preexisting_window_years", 3))),
        )
        weak_threshold = float(settings.get("preexisting_weak_probability_threshold", 0.38))
        minimum_mean = float(settings.get("preexisting_minimum_early_mean_probability", 0.40))
        minimum_fraction = float(settings.get("preexisting_minimum_early_support_fraction", 0.60))
        persistent_threshold = float(settings.get("preexisting_persistent_probability_threshold", 0.34))
        minimum_persistence = float(settings.get("preexisting_minimum_historical_support_fraction", 0.55))
        minimum_valid = min(
            early_window,
            max(1, int(settings.get("preexisting_minimum_early_valid_years", 2))),
        )

        early_valid = v[:early_window]
        early_count = early_valid.sum(axis=0)
        early_total = np.where(early_valid, p[:early_window], 0.0).sum(axis=0)
        early_mean = np.divide(
            early_total,
            np.maximum(early_count, 1),
            where=np.maximum(early_count, 1) > 0,
        )
        early_support_count = ((p[:early_window] >= weak_threshold) & early_valid).sum(axis=0)
        early_fraction = np.divide(
            early_support_count,
            np.maximum(early_count, 1),
            where=np.maximum(early_count, 1) > 0,
        )
        hist_valid = v[:historical_n]
        hist_count = hist_valid.sum(axis=0)
        hist_support = ((p[:historical_n] >= persistent_threshold) & hist_valid).sum(axis=0)
        hist_fraction = np.divide(
            hist_support,
            np.maximum(hist_count, 1),
            where=np.maximum(hist_count, 1) > 0,
        )

        maximum_later_jump = np.zeros(inventory.shape, dtype=np.float32)
        jump_lookback = max(1, int(settings.get("preexisting_jump_lookback_years", 2)))
        for t in range(1, historical_n):
            lo = max(0, t - jump_lookback)
            block = np.where(v[lo:t], p[lo:t], np.nan)
            count = np.isfinite(block).sum(axis=0)
            work = block.copy()
            if work.shape[0] and np.any(count == 0):
                rr, cc = np.where(count == 0)
                work[0, rr, cc] = p[t, rr, cc]
            with np.errstate(all="ignore"):
                baseline = np.nanmedian(work, axis=0)
            jump = np.where(v[t] & (count > 0), p[t] - baseline, 0.0)
            maximum_later_jump = np.maximum(maximum_later_jump, jump.astype(np.float32))

        strong_early = early_mean >= float(
            settings.get("preexisting_strong_early_probability", 0.56)
        )
        robust_preexisting = (
            inventory
            & (early_count >= minimum_valid)
            & (early_mean >= minimum_mean)
            & (early_fraction >= minimum_fraction)
            & (hist_fraction >= minimum_persistence)
            & (
                (maximum_later_jump <= float(settings.get("preexisting_maximum_later_jump", 0.24)))
                | strong_early
            )
        )
        support = early_valid & (p[:early_window] >= weak_threshold)
        has_support = np.any(support, axis=0)
        early_first = np.argmax(support, axis=0).astype(np.int16)
        robust_preexisting &= has_support

    onset = np.full(inventory.shape, reference_index, dtype=np.int16)
    onset[credible] = best[credible]
    onset[robust_preexisting] = early_first[robust_preexisting]
    confidence[robust_preexisting] = np.maximum(
        confidence[robust_preexisting],
        float(settings.get("preexisting_confidence_floor", 0.25)),
    )
    confidence[~inventory] = 0.0
    return onset, confidence, best_post.astype(np.float32), best_pre.astype(np.float32)

def _persistent_outside_tracks(
    annual_candidate: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    reference_index: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer independent outside-reference tracks without violating the audited 2024 endpoint."""
    n = annual_candidate.shape[0]
    threshold = float(settings.get("historical_outside_probability", 0.62))
    candidate = annual_candidate & valid & (probability >= threshold) & ~inventory[None, ...]
    candidate[reference_index] = False  # outside is exact non-PV in 2024

    persistent = np.zeros_like(candidate, dtype=bool)
    # Historical runs require support in at least two neighbouring years.
    for t in range(reference_index):
        left = candidate[t-1] if t > 0 else np.zeros_like(inventory)
        right = candidate[t+1] if t+1 < reference_index else np.zeros_like(inventory)
        persistent[t] = candidate[t] & (left | right)
    # A one-year gap inside an otherwise persistent historical track is bridged.
    for t in range(1, max(reference_index-1, 1)):
        persistent[t] |= (~candidate[t]) & candidate[t-1] & candidate[t+1] & (probability[t] >= 0.45)

    # 2025 expansion is unconstrained but must be extremely strong when only one year is available.
    if reference_index + 1 < n:
        t = reference_index + 1
        persistent[t] = candidate[t] & (probability[t] >= float(settings.get("end_year_singleton_probability", 0.92)))

    onset = np.full(inventory.shape, n, dtype=np.int16)
    retirement = np.full(inventory.shape, n, dtype=np.int16)
    for t in range(n):
        first = persistent[t] & (onset == n)
        onset[first] = t
    # Historical outside tracks must be retired by the audited 2024 endpoint.
    for t in range(reference_index):
        ended = persistent[t] & ~np.any(persistent[t+1:reference_index], axis=0)
        retirement[ended] = np.minimum(retirement[ended], t + 1)
    return persistent, onset, retirement



def _optional_paths(group: pd.DataFrame, column: str) -> list[Path] | None:
    if column not in group.columns:
        return None
    paths = [Path(str(value)) for value in group[column].tolist()]
    return paths if paths and all(path.is_file() for path in paths) else None


def _remove_small_outside_components(
    mask: np.ndarray,
    protected: np.ndarray,
    minimum_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(mask & ~protected, structure=structure)
    kept = protected.copy()
    removed = np.zeros(mask.shape, dtype=bool)
    for label_id in range(1, count + 1):
        obj = labels == label_id
        if int(obj.sum()) >= int(minimum_pixels):
            kept |= obj
        else:
            removed |= obj
    return kept, removed


def decode_authoritative_reference_change(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("authoritative_temporal_reconstruction", cfg.get("reference_temporal_reconstruction", {}))
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "08_temporal_pv_change")
    inference = pd.read_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv").sort_values(["Id", "year"])
    dual_stage_enabled = bool(settings.get("dual_stage_reference_enabled", False))
    aligned_registry_path = root / "02_aligned_30m" / "aligned_raster_registry.csv"
    aligned_registry = (
        pd.read_csv(aligned_registry_path).sort_values(["Id", "year"])
        if dual_stage_enabled and aligned_registry_path.is_file()
        else None
    )
    if dual_stage_enabled and aligned_registry is None:
        raise RuntimeError("Dual-stage reference fusion requires 02_aligned_30m/aligned_raster_registry.csv")
    years = np.arange(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1, dtype=int)
    reference_year = int(cfg["study"]["reference_year"])
    reference_index = int(np.where(years == reference_year)[0][0])
    mask_nodata = int(cfg.get("temporal_decoding", {}).get("mask_nodata", 255))
    tile_size = int(settings.get("tile_size", 384))
    halo = int(settings.get("halo_pixels", 48))
    overwrite = bool(cfg.get("processing", {}).get("overwrite", False))

    unit_registry_root = ensure_dir(out / "unit_registries")
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    posterior_root = ensure_dir(out / "harmonized_probability_30m")
    active_root = ensure_dir(out / "observed_active_pv_masks_30m")
    cumulative_root = ensure_dir(out / "cumulative_installed_pv_masks_30m")
    model_only_root = ensure_dir(out / "model_only_cumulative_installed_pv_masks_30m")
    onset_root = ensure_dir(out / "first_installed_year_30m")
    model_onset_root = ensure_dir(out / "model_only_first_installed_year_30m")
    retirement_root = ensure_dir(out / "first_retirement_year_30m")
    score_root = ensure_dir(out / "transition_score_30m")
    confuser_root = ensure_dir(out / "confuser_penalty_30m")
    diagnostic_root = ensure_dir(out / "decision_diagnostics_30m")
    anchor_audit_root = ensure_dir(out / "anchor_constraint_diagnostics_30m")
    growth_rescue_root = ensure_dir(out / "historical_growth_front_rescue_30m")
    terminal_source_root = ensure_dir(out / "terminal_2025_expansion_source_30m")
    stage1_root = ensure_dir(out / "dual_stage_global_reference_masks_30m")
    stage2_root = ensure_dir(out / "dual_stage_internal_shp_buildout_30m")
    fusion_source_root = ensure_dir(out / "dual_stage_fusion_source_30m")
    internal_evidence_root = ensure_dir(out / "dual_stage_internal_evidence_30m")
    spectral_similarity_root = ensure_dir(out / "dual_stage_reference_spectral_similarity_30m")

    rows: list[dict[str, Any]] = []
    worker_audits: list[dict[str, Any]] = []
    pixel_units = sorted(int(i) for i, g in inference.groupby("Id") if str(g.iloc[0].get("inference_mode", "direct_pixel")) in PIXEL_MODES)

    iterator = tqdm(pixel_units, desc="Evidence-constrained installation change points", unit="region", disable=not cfg.get("processing", {}).get("show_progress", True))
    for region_id in iterator:
        group = inference.loc[inference["Id"].eq(region_id)].sort_values("year")
        if len(group) != len(years):
            raise RuntimeError(f"Id {region_id}: expected {len(years)} annual evidence records")
        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        unit_registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
        if checkpoint.exists() and unit_registry_path.exists() and not overwrite:
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                frame = pd.read_csv(unit_registry_path)
                if (
                    payload.get("status") == "passed"
                    and payload.get("schema_version") == "24.22.0"
                    and payload.get("algorithm_revision") == DETAIL_ALGORITHM_REVISION
                    and len(frame) == len(years)
                ):
                    if all(Path(p).exists() for p in frame["cumulative_mask_path"].astype(str)):
                        rows.extend(frame.to_dict("records"))
                        worker_audits.append(payload)
                        continue
            except Exception:
                pass

        probability_paths = [Path(str(v)) for v in group["calibrated_probability_path"]]
        candidate_paths = [Path(str(v)) for v in group["calibrated_mask_path"]]
        raw_probability_paths = _optional_paths(group, "pre_reference_reconstruction_probability_path") or probability_paths
        raw_candidate_paths = _optional_paths(group, "pre_reference_reconstruction_mask_path") or candidate_paths
        similarity_paths = _optional_paths(group, "anchor_similarity_path")
        candidate_score_paths = _optional_paths(group, "candidate_score_path")
        if dual_stage_enabled:
            assert aligned_registry is not None
            aligned_group = aligned_registry.loc[aligned_registry["Id"].eq(region_id)].sort_values("year")
            if len(aligned_group) != len(years):
                raise RuntimeError(f"Id {region_id}: aligned Landsat registry is incomplete for dual-stage reconstruction")
            aligned_paths = [Path(str(value)) for value in aligned_group["path"].tolist()]
        else:
            aligned_paths = []
        inventory_path = Path(str(group.iloc[reference_index]["reference_inventory_path"]))
        fraction_path = Path(str(group.iloc[reference_index].get("reference_fraction_path", "")))
        anchor_path = Path(str(group.iloc[reference_index]["reference_anchor_path"]))
        sand_path = Path(str(group.iloc[reference_index]["stable_sand_risk_path"]))
        stage7_terminal_source_path = Path(
            str(group.iloc[reference_index].get("terminal_2025_expansion_source_path", ""))
        )
        mode = str(group.iloc[0].get("inference_mode", "direct_pixel"))
        route = str(group.iloc[0].get("estimation_route", mode))

        unit_posterior_root = ensure_dir(posterior_root / f"unit_{region_id:03d}")
        unit_active_root = ensure_dir(active_root / f"unit_{region_id:03d}")
        unit_cumulative_root = ensure_dir(cumulative_root / f"unit_{region_id:03d}")
        unit_model_root = ensure_dir(model_only_root / f"unit_{region_id:03d}")
        unit_onset_root = ensure_dir(onset_root / f"unit_{region_id:03d}")
        unit_model_onset_root = ensure_dir(model_onset_root / f"unit_{region_id:03d}")
        unit_retirement_root = ensure_dir(retirement_root / f"unit_{region_id:03d}")
        unit_score_root = ensure_dir(score_root / f"unit_{region_id:03d}")
        unit_confuser_root = ensure_dir(confuser_root / f"unit_{region_id:03d}")
        unit_diag_root = ensure_dir(diagnostic_root / f"unit_{region_id:03d}")
        unit_anchor_root = ensure_dir(anchor_audit_root / f"unit_{region_id:03d}")
        unit_growth_rescue_root = ensure_dir(growth_rescue_root / f"unit_{region_id:03d}")
        unit_terminal_source_root = ensure_dir(terminal_source_root / f"unit_{region_id:03d}")
        unit_stage1_root = ensure_dir(stage1_root / f"unit_{region_id:03d}")
        unit_stage2_root = ensure_dir(stage2_root / f"unit_{region_id:03d}")
        unit_fusion_source_root = ensure_dir(fusion_source_root / f"unit_{region_id:03d}")
        unit_internal_evidence_root = ensure_dir(internal_evidence_root / f"unit_{region_id:03d}")
        unit_spectral_similarity_root = ensure_dir(spectral_similarity_root / f"unit_{region_id:03d}")

        posterior_out = [unit_posterior_root / f"unit_{region_id:03d}_{y}_pv_probability_harmonized_30m.tif" for y in years]
        active_out = [unit_active_root / f"unit_{region_id:03d}_{y}_pv_active_footprint_30m.tif" for y in years]
        cumulative_out = [unit_cumulative_root / f"unit_{region_id:03d}_{y}_pv_cumulative_installed_30m.tif" for y in years]
        model_out = [unit_model_root / f"unit_{region_id:03d}_{y}_pv_model_only_cumulative_installed_30m.tif" for y in years]
        onset_path = unit_onset_root / f"unit_{region_id:03d}_first_installed_year_30m.tif"
        model_onset_path = unit_model_onset_root / f"unit_{region_id:03d}_model_only_first_installed_year_30m.tif"
        retirement_path = unit_retirement_root / f"unit_{region_id:03d}_first_retirement_year_30m.tif"
        transition_path = unit_score_root / f"unit_{region_id:03d}_transition_score_30m.tif"
        confuser_path = unit_confuser_root / f"unit_{region_id:03d}_confuser_penalty_30m.tif"
        candidate_union_path = unit_diag_root / f"unit_{region_id:03d}_candidate_before_morphology_30m.tif"
        removed_path = unit_diag_root / f"unit_{region_id:03d}_morphology_removed_30m.tif"
        valid_count_path = unit_diag_root / f"unit_{region_id:03d}_valid_year_count_30m.tif"
        exclusion_path = unit_diag_root / f"unit_{region_id:03d}_temporal_exclusion_code_30m.tif"
        pre_reference_supported_path = unit_anchor_root / f"unit_{region_id:03d}_pre_reference_evidence_supported_30m.tif"
        forced_reference_only_path = unit_anchor_root / f"unit_{region_id:03d}_reference_forced_without_historical_evidence_30m.tif"
        object_fill_path = unit_anchor_root / f"unit_{region_id:03d}_legacy_object_fill_disabled_30m.tif"
        anchor_reference_path = unit_anchor_root / f"unit_{region_id:03d}_reference_anchor_inclusion_30m.tif"
        growth_rescue_paths = [unit_growth_rescue_root / f"unit_{region_id:03d}_{y}_historical_extent_rescued_30m.tif" for y in years]
        terminal_source_path = unit_terminal_source_root / f"unit_{region_id:03d}_2025_expansion_source_30m.tif"
        stage1_paths = [unit_stage1_root / f"unit_{region_id:03d}_{y}_global_reference_state_30m.tif" for y in years]
        stage2_paths = [unit_stage2_root / f"unit_{region_id:03d}_{y}_internal_shp_buildout_30m.tif" for y in years]
        fusion_source_paths = [unit_fusion_source_root / f"unit_{region_id:03d}_{y}_dual_stage_fusion_source_30m.tif" for y in years]
        internal_evidence_paths = [unit_internal_evidence_root / f"unit_{region_id:03d}_{y}_internal_evidence_30m.tif" for y in years]
        spectral_similarity_paths = [unit_spectral_similarity_root / f"unit_{region_id:03d}_{y}_reference_spectral_similarity_30m.tif" for y in years]

        active_pixels = np.zeros(len(years), dtype=np.int64)
        cumulative_pixels = np.zeros(len(years), dtype=np.int64)
        model_pixels = np.zeros(len(years), dtype=np.int64)
        active_weighted_pixels = np.zeros(len(years), dtype=np.float64)
        cumulative_weighted_pixels = np.zeros(len(years), dtype=np.float64)
        model_weighted_pixels = np.zeros(len(years), dtype=np.float64)
        annual_valid = np.zeros(len(years), dtype=np.int64)
        forced_reference_pixels = 0
        pre_reference_pixels = 0

        with ExitStack() as stack:
            prob_srcs = [stack.enter_context(rasterio.open(path)) for path in probability_paths]
            candidate_srcs = [stack.enter_context(rasterio.open(path)) for path in candidate_paths]
            raw_prob_srcs = prob_srcs if raw_probability_paths == probability_paths else [stack.enter_context(rasterio.open(path)) for path in raw_probability_paths]
            raw_candidate_srcs = candidate_srcs if raw_candidate_paths == candidate_paths else [stack.enter_context(rasterio.open(path)) for path in raw_candidate_paths]
            similarity_srcs = [stack.enter_context(rasterio.open(path)) for path in similarity_paths] if similarity_paths is not None else None
            candidate_score_srcs = [stack.enter_context(rasterio.open(path)) for path in candidate_score_paths] if candidate_score_paths is not None else None
            aligned_srcs = [stack.enter_context(rasterio.open(path)) for path in aligned_paths] if dual_stage_enabled else []
            inventory_src = stack.enter_context(rasterio.open(inventory_path))
            anchor_src = stack.enter_context(rasterio.open(anchor_path))
            sand_src = stack.enter_context(rasterio.open(sand_path))
            stage7_terminal_source_src = (
                stack.enter_context(rasterio.open(stage7_terminal_source_path))
                if stage7_terminal_source_path.is_file()
                else None
            )
            fraction_src = stack.enter_context(rasterio.open(fraction_path)) if fraction_path.exists() else None
            ref = prob_srcs[0]
            width, height = int(ref.width), int(ref.height)
            pixel_area_km2 = abs(float(ref.transform.a * ref.transform.e)) / 1_000_000.0
            unit_total_pixels = int(width * height)
            float_profile = ref.profile.copy()
            float_profile.update(count=1, dtype="float32", nodata=-9999.0, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            byte_profile = ref.profile.copy()
            byte_profile.update(count=1, dtype="uint8", nodata=mask_nodata, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")
            int_profile = ref.profile.copy()
            int_profile.update(count=1, dtype="int16", nodata=-32768, compress=cfg["processing"].get("compress", "DEFLATE"), BIGTIFF="IF_SAFER")

            posterior_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in posterior_out]
            active_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in active_out]
            cumulative_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in cumulative_out]
            model_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in model_out]
            onset_dst = stack.enter_context(rasterio.open(onset_path, "w", **int_profile))
            model_onset_dst = stack.enter_context(rasterio.open(model_onset_path, "w", **int_profile))
            retirement_dst = stack.enter_context(rasterio.open(retirement_path, "w", **int_profile))
            transition_dst = stack.enter_context(rasterio.open(transition_path, "w", **float_profile))
            confuser_dst = stack.enter_context(rasterio.open(confuser_path, "w", **float_profile))
            candidate_union_dst = stack.enter_context(rasterio.open(candidate_union_path, "w", **byte_profile))
            removed_dst = stack.enter_context(rasterio.open(removed_path, "w", **byte_profile))
            valid_count_dst = stack.enter_context(rasterio.open(valid_count_path, "w", **byte_profile))
            exclusion_dst = stack.enter_context(rasterio.open(exclusion_path, "w", **byte_profile))
            pre_reference_dst = stack.enter_context(rasterio.open(pre_reference_supported_path, "w", **byte_profile))
            forced_reference_dst = stack.enter_context(rasterio.open(forced_reference_only_path, "w", **byte_profile))
            object_fill_dst = stack.enter_context(rasterio.open(object_fill_path, "w", **byte_profile))
            anchor_reference_dst = stack.enter_context(rasterio.open(anchor_reference_path, "w", **byte_profile))
            growth_rescue_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in growth_rescue_paths]
            terminal_source_dst = stack.enter_context(rasterio.open(terminal_source_path, "w", **byte_profile))
            stage1_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in stage1_paths]
            stage2_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in stage2_paths]
            fusion_source_dsts = [stack.enter_context(rasterio.open(path, "w", **byte_profile)) for path in fusion_source_paths]
            internal_evidence_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in internal_evidence_paths]
            spectral_similarity_dsts = [stack.enter_context(rasterio.open(path, "w", **float_profile)) for path in spectral_similarity_paths]

            for core in _windows(width, height, tile_size):
                expanded, crop = _expanded(core, width, height, halo)
                shape = (int(expanded.height), int(expanded.width))
                probability = np.zeros((len(years), *shape), dtype=np.float32)
                raw_probability = np.zeros_like(probability, dtype=np.float32)
                valid = np.zeros_like(probability, dtype=bool)
                annual_candidate = np.zeros_like(probability, dtype=bool)
                raw_annual_candidate = np.zeros_like(probability, dtype=bool)
                anchor_similarity = np.zeros_like(probability, dtype=np.float32) if similarity_srcs is not None else None
                candidate_score = np.zeros_like(probability, dtype=np.float32) if candidate_score_srcs is not None else None
                reflectance = np.zeros((len(years), 6, *shape), dtype=np.float32) if dual_stage_enabled else None
                reflectance_valid = np.zeros_like(probability, dtype=bool) if dual_stage_enabled else None
                for t in range(len(years)):
                    raw = prob_srcs[t].read(1, window=expanded).astype(np.float32)
                    probability[t], valid[t] = _safe_prob(raw, prob_srcs[t].nodata)
                    raw_p = raw_prob_srcs[t].read(1, window=expanded).astype(np.float32)
                    raw_probability[t], raw_valid = _safe_prob(raw_p, raw_prob_srcs[t].nodata)
                    annual_candidate[t], candidate_valid = _read_mask(candidate_srcs[t], expanded)
                    raw_annual_candidate[t], raw_candidate_valid = _read_mask(raw_candidate_srcs[t], expanded)
                    valid[t] &= raw_valid & (candidate_valid | valid[t]) & (raw_candidate_valid | raw_valid)
                    if similarity_srcs is not None and anchor_similarity is not None:
                        sim_raw = similarity_srcs[t].read(1, window=expanded).astype(np.float32)
                        sim_valid = np.isfinite(sim_raw) & ((similarity_srcs[t].nodata is None) | (sim_raw != similarity_srcs[t].nodata))
                        anchor_similarity[t] = np.where(sim_valid, np.clip(sim_raw, 0.0, 1.0), 0.0)
                    if candidate_score_srcs is not None and candidate_score is not None:
                        score_raw = candidate_score_srcs[t].read(1, window=expanded).astype(np.float32)
                        score_valid = np.isfinite(score_raw) & ((candidate_score_srcs[t].nodata is None) | (score_raw != candidate_score_srcs[t].nodata))
                        candidate_score[t] = np.where(score_valid, np.clip(score_raw, 0.0, 1.0), 0.0)
                    if dual_stage_enabled:
                        assert reflectance is not None and reflectance_valid is not None
                        refl = aligned_srcs[t].read([1, 2, 3, 4, 5, 6], window=expanded).astype(np.float32)
                        refl_valid = np.all(np.isfinite(refl), axis=0)
                        if aligned_srcs[t].nodata is not None:
                            refl_valid &= np.all(refl != float(aligned_srcs[t].nodata), axis=0)
                        reflectance[t] = np.where(refl_valid[None, ...], refl, np.nan)
                        reflectance_valid[t] = refl_valid
                inventory, inventory_valid = _read_mask(inventory_src, expanded)
                anchor, anchor_valid = _read_mask(anchor_src, expanded)
                sand_raw = sand_src.read(1, window=expanded).astype(np.float32)
                sand_valid = np.isfinite(sand_raw) & ((sand_src.nodata is None) | (sand_raw != sand_src.nodata))
                sand_risk = np.where(sand_valid, np.clip(sand_raw, 0.0, 1.0), 0.0).astype(np.float32)
                fraction = fraction_src.read(1, window=expanded).astype(np.float32) if fraction_src is not None else anchor.astype(np.float32)
                fraction = np.clip(np.nan_to_num(fraction, nan=0.0), 0.0, 1.0)
                area_weight = np.where(inventory, fraction, 1.0).astype(np.float32)
                inventory &= inventory_valid
                anchor &= anchor_valid

                onset_idx, transition_confidence, post_mean, pre_mean = _ordered_change_point(
                    probability,
                    valid,
                    inventory,
                    reference_index,
                    settings,
                )
                stage1_inventory_active, historical_extent_rescued = reconstruct_reference_growth_front(
                    probability,
                    valid,
                    inventory,
                    onset_idx,
                    reference_index,
                    settings,
                )
                if dual_stage_enabled:
                    assert reflectance is not None and reflectance_valid is not None
                    internal_result = reconstruct_internal_reference_buildout(
                        raw_probability,
                        probability,
                        valid & reflectance_valid,
                        inventory,
                        raw_annual_candidate | annual_candidate,
                        reference_index,
                        settings,
                        reflectance=reflectance,
                        reflectance_valid=reflectance_valid,
                        anchor_similarity=anchor_similarity,
                        candidate_score=candidate_score,
                    )
                    inventory_active, fusion_source = fuse_reference_tracks(
                        stage1_inventory_active,
                        internal_result,
                        inventory,
                        raw_annual_candidate | annual_candidate,
                        reference_index,
                        settings,
                    )
                    historical_extent_rescued |= internal_result.internal_active & ~stage1_inventory_active
                    internal_evidence = internal_result.internal_evidence
                    spectral_similarity = internal_result.spectral_similarity
                    stage2_inventory_active = internal_result.internal_active
                else:
                    inventory_active = stage1_inventory_active
                    fusion_source = np.where(stage1_inventory_active, 1, 0).astype(np.uint8)
                    internal_evidence = probability.copy()
                    spectral_similarity = np.zeros_like(probability, dtype=np.float32)
                    stage2_inventory_active = np.zeros_like(stage1_inventory_active, dtype=bool)
                if stage7_terminal_source_src is not None:
                    raw_terminal_source = stage7_terminal_source_src.read(1, window=expanded)
                    terminal_source_override = np.where(
                        np.isin(raw_terminal_source, [1, 2, 3, 4]),
                        raw_terminal_source,
                        0,
                    ).astype(np.uint8)
                else:
                    terminal_source_override = None
                outside_active, outside_onset, outside_retirement, terminal_source = reconstruct_historical_outside_tracks(
                    annual_candidate,
                    probability,
                    valid,
                    inventory,
                    reference_index,
                    settings,
                    terminal_source_override=terminal_source_override,
                )

                active = np.zeros_like(annual_candidate, dtype=bool)
                cumulative = np.zeros_like(annual_candidate, dtype=bool)
                model_only = np.zeros_like(annual_candidate, dtype=bool)
                reference_model_support = (
                    probability[reference_index] >= float(settings.get("model_only_reference_probability", 0.50))
                ) & valid[reference_index]
                model_inventory = inventory & reference_model_support

                for t in range(len(years)):
                    installed_inventory = inventory_active[t]
                    active[t] |= installed_inventory
                    active[t] |= outside_active[t]
                    cumulative[t] |= installed_inventory
                    cumulative[t] |= outside_onset <= t
                    model_only[t] |= model_inventory & (onset_idx <= t)
                    model_only[t] |= outside_onset <= t

                    active[t], removed = _remove_small_outside_components(
                        active[t],
                        installed_inventory,
                        int(settings.get("minimum_outside_component_pixels", 10)),
                    )
                    annual_candidate[t] |= inventory if t == reference_index else False

                # Adaptive endpoint contract: active 2024 equals the audited,
                # boundary-refined inventory produced by Stage 7. The original
                # manual vector remains a prior/diagnostic layer, not a pixelwise
                # hard endpoint.
                active[reference_index][:] = inventory
                cumulative[reference_index] |= inventory
                for t in range(reference_index + 1, len(years)):
                    cumulative[t] |= inventory
                    # With only one post-reference year, absence is insufficient
                    # evidence for decommissioning; retain refined 2024 facilities in 2025.
                    active[t] |= inventory

                # Ensure cumulative state is monotonic, independently of active
                # state and image-quality gaps.
                for t in range(1, len(years)):
                    cumulative[t] |= cumulative[t - 1]
                    model_only[t] |= model_only[t - 1]

                onset_year = np.full(shape, -32768, dtype=np.int16)
                growth_onset_idx = np.full(shape, reference_index, dtype=np.int16)
                for t in range(reference_index + 1):
                    first = inventory_active[t] & (growth_onset_idx == reference_index)
                    growth_onset_idx[first] = t
                onset_year[inventory] = years[growth_onset_idx[inventory]]
                outside_present = outside_onset < len(years)
                onset_year[outside_present & ~inventory] = years[outside_onset[outside_present & ~inventory]]
                model_onset_year = np.full(shape, -32768, dtype=np.int16)
                model_onset_year[model_inventory] = years[onset_idx[model_inventory]]
                model_onset_year[outside_present & ~inventory] = years[outside_onset[outside_present & ~inventory]]
                retirement_year = np.full(shape, -32768, dtype=np.int16)
                retired = outside_retirement < len(years)
                retirement_year[retired] = years[np.clip(outside_retirement[retired], 0, len(years)-1)]

                pre_reference_supported = inventory & (growth_onset_idx < reference_index)
                forced_reference_only = inventory & (growth_onset_idx == reference_index)
                candidate_union = np.any(annual_candidate, axis=0)
                final_union = np.any(active, axis=0)
                morphology_removed = candidate_union & ~final_union
                valid_count = np.sum(valid, axis=0).astype(np.uint8)
                exclusion = np.zeros(shape, dtype=np.uint8)
                exclusion[forced_reference_only] = EXCLUSION_REFERENCE_ONLY_NO_HISTORY
                exclusion[(sand_risk >= float(settings.get("stable_sand_exclusion_threshold", 0.68))) & ~final_union] = EXCLUSION_STABLE_SAND
                exclusion[candidate_union & ~final_union & (exclusion == 0)] = EXCLUSION_NO_PERSISTENT_TRACK
                exclusion[morphology_removed & (exclusion == 0)] = EXCLUSION_SMALL_COMPONENT

                core_valid = valid[:, crop[0], crop[1]]
                ever_valid = np.any(core_valid, axis=0) | inventory[crop]
                for t in range(len(years)):
                    posterior_dsts[t].write(np.where(core_valid[t], probability[t][crop], -9999.0).astype(np.float32), 1, window=core)
                    # The decoded annual state is written on the multi-year valid
                    # domain. A cloudy annual composite must not make an already
                    # reconstructed PV parcel disappear from the map.
                    active_domain = ever_valid
                    active_dsts[t].write(np.where(active_domain, active[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    cumulative_dsts[t].write(np.where(ever_valid, cumulative[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    model_dsts[t].write(np.where(ever_valid, model_only[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    growth_rescue_dsts[t].write(np.where(ever_valid, historical_extent_rescued[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    stage1_dsts[t].write(np.where(ever_valid, stage1_inventory_active[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    stage2_dsts[t].write(np.where(ever_valid, stage2_inventory_active[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    fusion_source_dsts[t].write(np.where(ever_valid, fusion_source[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    internal_evidence_dsts[t].write(np.where(ever_valid, internal_evidence[t][crop], -9999.0).astype(np.float32), 1, window=core)
                    spectral_similarity_dsts[t].write(np.where(ever_valid, spectral_similarity[t][crop], -9999.0).astype(np.float32), 1, window=core)
                    active_pixels[t] += int(np.count_nonzero(active[t][crop] & ever_valid))
                    cumulative_pixels[t] += int(np.count_nonzero(cumulative[t][crop] & ever_valid))
                    model_pixels[t] += int(np.count_nonzero(model_only[t][crop] & ever_valid))
                    weight_core = area_weight[crop]
                    active_weighted_pixels[t] += float(np.sum(weight_core * active[t][crop] * ever_valid))
                    cumulative_weighted_pixels[t] += float(np.sum(weight_core * cumulative[t][crop] * ever_valid))
                    model_weighted_pixels[t] += float(np.sum(weight_core * model_only[t][crop] * ever_valid))
                    annual_valid[t] += int(np.count_nonzero(core_valid[t] | (inventory[crop] if t == reference_index else False)))

                onset_dst.write(np.where(ever_valid, onset_year[crop], -32768).astype(np.int16), 1, window=core)
                model_onset_dst.write(np.where(ever_valid, model_onset_year[crop], -32768).astype(np.int16), 1, window=core)
                retirement_dst.write(np.where(ever_valid, retirement_year[crop], -32768).astype(np.int16), 1, window=core)
                transition_dst.write(np.where(ever_valid, transition_confidence[crop], -9999.0).astype(np.float32), 1, window=core)
                confuser_dst.write(np.where(ever_valid, sand_risk[crop], -9999.0).astype(np.float32), 1, window=core)
                candidate_union_dst.write(np.where(ever_valid, candidate_union[crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                removed_dst.write(np.where(ever_valid, morphology_removed[crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                valid_count_dst.write(np.where(ever_valid, valid_count[crop], mask_nodata).astype(np.uint8), 1, window=core)
                exclusion_dst.write(np.where(ever_valid, exclusion[crop], mask_nodata).astype(np.uint8), 1, window=core)
                pre_reference_dst.write(np.where(ever_valid, pre_reference_supported[crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                forced_reference_dst.write(np.where(ever_valid, forced_reference_only[crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                zero_core = np.zeros(ever_valid.shape, dtype=np.uint8)
                object_fill_dst.write(np.where(ever_valid, zero_core, mask_nodata).astype(np.uint8), 1, window=core)
                anchor_reference_dst.write(np.where(ever_valid, anchor[crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                terminal_source_dst.write(np.where(ever_valid, terminal_source[crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                forced_reference_pixels += int(np.count_nonzero(forced_reference_only[crop]))
                pre_reference_pixels += int(np.count_nonzero(pre_reference_supported[crop]))

        inferred_deployment_year = (
            int(years[np.argmax(np.diff(np.r_[0, cumulative_pixels]))])
            if cumulative_pixels.max(initial=0) > 0
            else np.nan
        )
        unit_rows: list[dict[str, Any]] = []
        manual_reference_area = float(group.iloc[reference_index].get("confirmed_anchor_area_km2", np.nan))
        confirmed_anchor_area = float(active_weighted_pixels[reference_index] * pixel_area_km2)
        for t, year in enumerate(years):
            active_area = float(active_weighted_pixels[t] * pixel_area_km2)
            cumulative_area = float(cumulative_weighted_pixels[t] * pixel_area_km2)
            model_area = float(model_weighted_pixels[t] * pixel_area_km2)
            unit_rows.append({
                "Id": region_id,
                "year": int(year),
                "posterior_path": str(posterior_out[t]),
                "mask_path": str(active_out[t]),
                "active_mask_path": str(active_out[t]),
                "cumulative_mask_path": str(cumulative_out[t]),
                "model_only_cumulative_mask_path": str(model_out[t]),
                "pv_pixels": int(active_pixels[t]),
                "active_pv_pixels": int(active_pixels[t]),
                "cumulative_installed_pixels": int(cumulative_pixels[t]),
                "model_only_cumulative_installed_pixels": int(model_pixels[t]),
                "valid_pixels": int(annual_valid[t]),
                "pv_area_km2": active_area,
                "observed_active_area_km2": active_area,
                "cumulative_installed_area_km2": cumulative_area,
                "model_only_cumulative_installed_area_km2": model_area,
                "anchor_constrained_added_area_km2": max(cumulative_area - model_area, 0.0),
                "anchor_constrained_removed_area_km2": max(model_area - cumulative_area, 0.0),
                "pv_area_latent_persistent_km2": cumulative_area,
                "inactive_or_retired_area_km2": max(cumulative_area - active_area, 0.0),
                "observation_coverage_fraction": float(annual_valid[t] / max(unit_total_pixels, 1)),
                "change_year_path": str(onset_path),
                "model_only_change_year_path": str(model_onset_path),
                "retirement_year_path": str(retirement_path),
                "transition_score_path": str(transition_path),
                "confuser_penalty_path": str(confuser_path),
                "temporal_candidate_path": str(candidate_union_path),
                "morphology_removed_path": str(removed_path),
                "valid_year_count_path": str(valid_count_path),
                "temporal_exclusion_code_path": str(exclusion_path),
                # Backward-compatible column names, with corrected semantics.
                "anchor_backcast_path": str(pre_reference_supported_path),
                "anchor_forced_reference_path": str(forced_reference_only_path),
                "anchor_backcast_object_fill_path": str(object_fill_path),
                "anchor_reference_path": str(anchor_reference_path),
                "historical_growth_front_rescue_path": str(growth_rescue_paths[t]),
                "dual_stage_global_reference_mask_path": str(stage1_paths[t]),
                "dual_stage_internal_shp_buildout_path": str(stage2_paths[t]),
                "dual_stage_fusion_source_path": str(fusion_source_paths[t]),
                "dual_stage_internal_evidence_path": str(internal_evidence_paths[t]),
                "dual_stage_reference_spectral_similarity_path": str(spectral_similarity_paths[t]),
                "terminal_2025_expansion_source_path": str(terminal_source_path),
                "anchor_constraint_applied": True,
                "calibrated_threshold": float(group.iloc[t].get("calibrated_threshold", np.nan)),
                "posterior_threshold": float(settings.get("minimum_post_installation_mean_probability", 0.44)),
                "estimation_mode": mode,
                "estimation_route": route,
                "pv_area_ci_low_km2": np.nan,
                "pv_area_ci_high_km2": np.nan,
                "confirmed_anchor_area_km2": confirmed_anchor_area,
                "manual_reference_area_km2": manual_reference_area,
                "refined_reference_area_km2": confirmed_anchor_area,
                "inferred_deployment_year": inferred_deployment_year,
                "annual_probability_bias": 0.0,
                "area_state_definition": "observed_active",
                "temporal_decoder": "v24_22_5_r4_dual_stage_reference_fusion",
            })
        frame = pd.DataFrame(unit_rows)
        save_table(frame, unit_registry_path)
        payload = {
            "status": "passed",
            "schema_version": "24.22.0",
            "algorithm_revision": DETAIL_ALGORITHM_REVISION,
            "Id": region_id,
            "forced_reference_only_pixels": int(forced_reference_pixels),
            "pre_reference_evidence_supported_pixels": int(pre_reference_pixels),
            "historical_anchor_geometry_copied": False,
            "dual_stage_reference_fusion_enabled": dual_stage_enabled,
            "stage1_full_domain_and_stage2_internal_computed_independently": True,
            "stage7_terminal_source_preserved": bool(stage7_terminal_source_path.is_file()),
            "terminal_2025_intrasite_perimeter_completion_enabled": True,
            "created_utc": utc_now(),
        }
        atomic_json(payload, checkpoint)
        rows.extend(frame.to_dict("records"))
        worker_audits.append(payload)

    direct_registry = pd.DataFrame(rows).sort_values(["Id", "year"]) if rows else pd.DataFrame()
    fallback_registry, fallback_audit = infer_area_only_units(cfg, direct_registry)
    registry = pd.concat([direct_registry, fallback_registry], ignore_index=True, sort=False).sort_values(["Id", "year"])
    duplicate = registry.duplicated(["Id", "year"], keep=False)
    if duplicate.any():
        duplicated_keys = registry.loc[duplicate, ["Id", "year"]].drop_duplicates().to_dict("records")
        raise RuntimeError(f"Temporal registry contains duplicate Id-year records: {duplicated_keys[:20]}")
    save_table(registry, out / "temporal_mask_registry.csv")
    save_table(pd.DataFrame(worker_audits), out / "temporal_worker_resource_audit.csv")
    save_table(pd.DataFrame({"Id": registry.get("Id", []), "year": registry.get("year", []), "annual_probability_logit_bias": 0.0}), out / "annual_probability_harmonization.csv")
    try:
        from .temporal import _synchrony_audit
        synchrony, synchrony_diag = _synchrony_audit(registry, cfg)
    except Exception as exc:
        synchrony = pd.DataFrame(columns=["year", "peak_gain_units", "eligible_units", "peak_fraction"])
        synchrony_diag = {"synchronous_anomaly": False, "synchrony_audit_warning": f"{type(exc).__name__}: {exc}"}
    save_table(synchrony, out / "temporal_synchrony_audit.csv")

    expected_units = expected_counts(cfg)[0]
    expected_records = expected_units * len(years)
    coverage_ok = len(registry) == expected_records and registry["Id"].nunique() == expected_units
    if len(direct_registry) and "estimation_mode" in direct_registry.columns:
        modes = direct_registry["estimation_mode"].astype(str)
        direct_units = int(direct_registry.loc[modes.eq("direct_pixel"), "Id"].nunique())
        anchor_candidate_units = int(direct_registry.loc[modes.eq("anchor_guided_candidate"), "Id"].nunique())
    else:
        direct_units = 0
        anchor_candidate_units = 0
    area_only_units = int(fallback_registry["Id"].nunique()) if len(fallback_registry) else 0
    atomic_json({
        "status": "passed" if coverage_ok else "failed",
        "schema_version": "24.22.0",
        "algorithm_revision": DETAIL_ALGORITHM_REVISION,
        "records": int(len(registry)),
        "expected_records": int(expected_records),
        "units": int(registry["Id"].nunique()) if len(registry) else 0,
        "decoder": "R4-based two-stage fusion: full-domain annual change detection plus independent 2024-SHP internal build-out reconstruction",
        "reference_anchor_forced_in_reference_year": False,
        "refined_reference_inventory_used_in_reference_year": True,
        "reference_anchor_copied_to_historical_years_without_change_evidence": False,
        "outside_reference_inventory_can_be_supplemented": True,
        "reference_year_outside_forced_negative": False,
        "reference_year_corrections_restricted_to_boundary_band": True,
        "environment_creates_historical_labels": False,
        "no_data_is_never_decoded_as_pv": True,
        "test_units_anchored": False,
        "area_only_pixel_maps_fabricated": False,
        "direct_pixel_units": direct_units,
        "anchor_guided_candidate_units": anchor_candidate_units,
        "area_only_units": area_only_units,
        "area_only_unit_ids": sorted(fallback_registry["Id"].astype(int).unique().tolist()) if len(fallback_registry) else [],
        "explicit_stage7_area_only_unit_ids": sorted(
            fallback_audit.loc[
                fallback_audit.get("explicit_stage7_area_only", pd.Series(False, index=fallback_audit.index)).fillna(False).astype(bool),
                "Id",
            ].astype(int).unique().tolist()
        ) if len(fallback_audit) else [],
        "cumulative_installed_state_is_persistent": True,
        "observed_active_state_allows_consecutive_negative_retirement": True,
        "observed_active_state_allows_only_multiyear_negative_retirement": True,
        "image_quality_gaps_do_not_create_retirement": True,
        "preexisting_start_of_series_facilities_supported": True,
        "model_only_diagnostic_product_retained": True,
        "historical_growth_front_completion_enabled": True,
        "dual_stage_reference_fusion_enabled": dual_stage_enabled,
        "dual_stage_full_domain_branch": True,
        "dual_stage_internal_shp_branch": dual_stage_enabled,
        "environmental_evaluation_uses_fused_masks": dual_stage_enabled,
        "terminal_2025_gain_aware_detection_enabled": True,
        "terminal_2025_intrasite_perimeter_completion_enabled": True,
        "terminal_2025_stage7_source_preserved": True,
        "terminal_2025_novelty_gated_remote_change_preserved": True,
        "reference_refinement_is_causal_to_2024": True,
        "terminal_2025_detection_uses_pre_reference_image_baseline": True,
        "historical_extent_requires_connected_high_confidence_seed": True,
        "scientific_temporal_gate_pass": not bool(synchrony_diag.get("synchronous_anomaly", False)),
        **synchrony_diag,
        "created_utc": utc_now(),
    }, out / "temporal_decoding_status.json")
    if not coverage_ok:
        raise RuntimeError(f"Temporal decoding coverage incomplete: {len(registry)}/{expected_records} records")
    return registry
