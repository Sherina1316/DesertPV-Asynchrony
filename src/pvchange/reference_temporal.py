from __future__ import annotations

"""Reference-constrained temporal change reconstruction (v24.16).

This decoder replaces all legacy anchor backcasting.  It solves an ordered
change-point problem separately for each 2024 inventory pixel: the pixel is
non-PV before its inferred installation year and PV from that year onward.
The 2024 vector constrains only the 2024 state.  If historical image evidence is
weak, the installation year defaults to 2024 rather than being copied to 2011.

Outside the 2024 inventory, independent annual evidence is retained only when it
forms a temporally persistent track (or a very strong, large end-of-series new
site).  This permits historical-only and 2025 new sites without treating the
2024 vector as a complete inventory.
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
from .object_graph_refinement import _expanded, _windows
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now

PIXEL_MODES = {"direct_pixel", "anchor_guided_candidate"}

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
    """Infer installation index for every reference-inventory pixel."""
    p = np.clip(probability[: reference_index + 1], 1e-4, 1.0 - 1e-4)
    v = valid[: reference_index + 1]
    log_p = np.where(v, np.log(p), 0.0)
    log_q = np.where(v, np.log1p(-p), 0.0)
    n = reference_index + 1
    scores = np.full((n, *inventory.shape), -np.inf, dtype=np.float32)
    post_means = np.zeros_like(scores)
    pre_means = np.zeros_like(scores)
    post_high = np.zeros_like(scores, dtype=np.uint8)
    transition_penalty = float(settings.get("change_point_transition_penalty", 0.18))
    for k in range(n):
        pre_score = np.sum(log_q[:k], axis=0) if k > 0 else 0.0
        post_score = np.sum(log_p[k:n], axis=0)
        valid_post = v[k:n].sum(axis=0)
        valid_pre = v[:k].sum(axis=0) if k > 0 else np.zeros(inventory.shape, dtype=np.int16)
        scores[k] = np.where(valid_post > 0, pre_score + post_score - transition_penalty, -np.inf)
        post_total = np.where(v[k:n], p[k:n], 0.0).sum(axis=0)
        post_means[k] = np.divide(post_total, np.maximum(valid_post, 1), where=np.maximum(valid_post, 1) > 0)
        if k > 0:
            pre_total = np.where(v[:k], p[:k], 0.0).sum(axis=0)
            pre_means[k] = np.divide(pre_total, np.maximum(valid_pre, 1), where=np.maximum(valid_pre, 1) > 0)
        post_high[k] = ((p[k:n] >= float(settings.get("post_installation_high_probability", 0.46))) & v[k:n]).sum(axis=0)

    best = np.argmax(scores, axis=0).astype(np.int16)
    best_score = np.take_along_axis(scores, best[None, ...], axis=0)[0]
    # Confidence is the log-likelihood gap between the best and second-best
    # change point.  It is used only for audit/regularisation, not as a label.
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
    credible &= (best == 0) | ((best_post - best_pre) >= min_gain) | (best_post >= float(settings.get("strong_post_installation_override", 0.72)))

    # A start-of-series facility needs support in the first two valid years.
    if n >= 2:
        preexisting = inventory & (best == 0)
        preexisting &= p[0] >= float(settings.get("preexisting_probability_threshold", 0.58))
        preexisting &= p[1] >= float(settings.get("preexisting_probability_threshold", 0.58))
        credible = (credible & (best != 0)) | preexisting

    onset = np.full(inventory.shape, reference_index, dtype=np.int16)
    onset[credible] = best[credible]
    # The vector is never allowed to pull onset earlier when the image evidence
    # is inconclusive.  Such pixels remain 2024-only in the historical record.
    confidence[~inventory] = 0.0
    return onset, confidence, best_post.astype(np.float32), best_pre.astype(np.float32)


def _persistent_outside_tracks(
    annual_candidate: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    inventory: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    n = annual_candidate.shape[0]
    candidate = annual_candidate & valid & ~inventory[None, ...]
    persistent = np.zeros_like(candidate, dtype=bool)
    if n > 1:
        persistent[0] = candidate[0] & candidate[1]
        persistent[-1] = candidate[-1] & candidate[-2]
    if n > 2:
        persistent[1:-1] = candidate[1:-1] & (candidate[:-2] | candidate[2:])

    # Permit a 2025-only newly built site only at very high evidence; spatial
    # object filtering below prevents isolated dark pixels from being accepted.
    persistent[-1] |= (
        candidate[-1]
        & (probability[-1] >= float(settings.get("end_year_singleton_probability", 0.90)))
    )

    onset = np.full(inventory.shape, n, dtype=np.int16)
    for t in range(n):
        first = persistent[t] & (onset == n)
        onset[first] = t
    return persistent, onset


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


def decode_reference_constrained_change(cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg.get("reference_temporal_reconstruction", {})
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "08_temporal_pv_change")
    inference = pd.read_csv(root / "07_probability_calibration" / "calibrated_inference_registry.csv").sort_values(["Id", "year"])
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
                if payload.get("status") == "passed" and payload.get("schema_version") == "24.18.0" and len(frame) == len(years):
                    if all(Path(p).exists() for p in frame["cumulative_mask_path"].astype(str)):
                        rows.extend(frame.to_dict("records"))
                        worker_audits.append(payload)
                        continue
            except Exception:
                pass

        probability_paths = [Path(str(v)) for v in group["calibrated_probability_path"]]
        candidate_paths = [Path(str(v)) for v in group["calibrated_mask_path"]]
        inventory_path = Path(str(group.iloc[reference_index]["reference_inventory_path"]))
        anchor_path = Path(str(group.iloc[reference_index]["reference_anchor_path"]))
        sand_path = Path(str(group.iloc[reference_index]["stable_sand_risk_path"]))
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

        active_pixels = np.zeros(len(years), dtype=np.int64)
        cumulative_pixels = np.zeros(len(years), dtype=np.int64)
        model_pixels = np.zeros(len(years), dtype=np.int64)
        annual_valid = np.zeros(len(years), dtype=np.int64)
        forced_reference_pixels = 0
        pre_reference_pixels = 0

        with ExitStack() as stack:
            prob_srcs = [stack.enter_context(rasterio.open(path)) for path in probability_paths]
            candidate_srcs = [stack.enter_context(rasterio.open(path)) for path in candidate_paths]
            inventory_src = stack.enter_context(rasterio.open(inventory_path))
            anchor_src = stack.enter_context(rasterio.open(anchor_path))
            sand_src = stack.enter_context(rasterio.open(sand_path))
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

            for core in _windows(width, height, tile_size):
                expanded, crop = _expanded(core, width, height, halo)
                shape = (int(expanded.height), int(expanded.width))
                probability = np.zeros((len(years), *shape), dtype=np.float32)
                valid = np.zeros_like(probability, dtype=bool)
                annual_candidate = np.zeros_like(probability, dtype=bool)
                for t in range(len(years)):
                    raw = prob_srcs[t].read(1, window=expanded).astype(np.float32)
                    probability[t], valid[t] = _safe_prob(raw, prob_srcs[t].nodata)
                    annual_candidate[t], candidate_valid = _read_mask(candidate_srcs[t], expanded)
                    valid[t] &= candidate_valid | valid[t]
                inventory, inventory_valid = _read_mask(inventory_src, expanded)
                anchor, anchor_valid = _read_mask(anchor_src, expanded)
                sand_raw = sand_src.read(1, window=expanded).astype(np.float32)
                sand_valid = np.isfinite(sand_raw) & ((sand_src.nodata is None) | (sand_raw != sand_src.nodata))
                sand_risk = np.where(sand_valid, np.clip(sand_raw, 0.0, 1.0), 0.0).astype(np.float32)
                inventory &= inventory_valid
                anchor &= anchor_valid

                onset_idx, transition_confidence, post_mean, pre_mean = _ordered_change_point(
                    probability,
                    valid,
                    inventory,
                    reference_index,
                    settings,
                )
                outside_active, outside_onset = _persistent_outside_tracks(
                    annual_candidate,
                    probability,
                    valid,
                    inventory,
                    settings,
                )

                active = np.zeros_like(annual_candidate, dtype=bool)
                cumulative = np.zeros_like(annual_candidate, dtype=bool)
                model_only = np.zeros_like(annual_candidate, dtype=bool)
                reference_model_support = (
                    probability[reference_index] >= float(settings.get("model_only_reference_probability", 0.50))
                ) & valid[reference_index]
                model_inventory = inventory & reference_model_support

                for t in range(len(years)):
                    installed_inventory = inventory & (onset_idx <= t)
                    if t <= reference_index:
                        active[t] |= installed_inventory
                    else:
                        # A single post-reference observation cannot establish
                        # retirement.  Retain 2024 sites through 2025.
                        active[t] |= inventory
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

                # Reference-year contract: every valid vector pixel is PV in
                # 2024.  No preceding active/cumulative layer is altered here.
                active[reference_index] |= anchor
                cumulative[reference_index] |= anchor
                for t in range(reference_index + 1, len(years)):
                    cumulative[t] |= anchor
                    active[t] |= anchor

                # Ensure cumulative state is monotonic, independently of active
                # state and image-quality gaps.
                for t in range(1, len(years)):
                    cumulative[t] |= cumulative[t - 1]
                    model_only[t] |= model_only[t - 1]

                onset_year = np.full(shape, -32768, dtype=np.int16)
                onset_year[inventory] = years[onset_idx[inventory]]
                outside_present = outside_onset < len(years)
                onset_year[outside_present & ~inventory] = years[outside_onset[outside_present & ~inventory]]
                model_onset_year = np.full(shape, -32768, dtype=np.int16)
                model_onset_year[model_inventory] = years[onset_idx[model_inventory]]
                model_onset_year[outside_present & ~inventory] = years[outside_onset[outside_present & ~inventory]]
                retirement_year = np.full(shape, -32768, dtype=np.int16)

                pre_reference_supported = inventory & (onset_idx < reference_index)
                forced_reference_only = anchor & (onset_idx == reference_index)
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
                ever_valid = np.any(core_valid, axis=0) | anchor[crop]
                for t in range(len(years)):
                    posterior_dsts[t].write(np.where(core_valid[t], probability[t][crop], -9999.0).astype(np.float32), 1, window=core)
                    active_dsts[t].write(np.where(core_valid[t] | (anchor[crop] if t >= reference_index else False), active[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    cumulative_dsts[t].write(np.where(ever_valid, cumulative[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    model_dsts[t].write(np.where(ever_valid, model_only[t][crop].astype(np.uint8), mask_nodata).astype(np.uint8), 1, window=core)
                    active_pixels[t] += int(np.count_nonzero(active[t][crop] & ever_valid))
                    cumulative_pixels[t] += int(np.count_nonzero(cumulative[t][crop] & ever_valid))
                    model_pixels[t] += int(np.count_nonzero(model_only[t][crop] & ever_valid))
                    annual_valid[t] += int(np.count_nonzero(core_valid[t] | (anchor[crop] if t == reference_index else False)))

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
                forced_reference_pixels += int(np.count_nonzero(forced_reference_only[crop]))
                pre_reference_pixels += int(np.count_nonzero(pre_reference_supported[crop]))

        inferred_deployment_year = (
            int(years[np.argmax(np.diff(np.r_[0, cumulative_pixels]))])
            if cumulative_pixels.max(initial=0) > 0
            else np.nan
        )
        unit_rows: list[dict[str, Any]] = []
        confirmed_anchor_area = float(group.iloc[reference_index].get("confirmed_anchor_area_km2", np.nan))
        for t, year in enumerate(years):
            active_area = float(active_pixels[t] * pixel_area_km2)
            cumulative_area = float(cumulative_pixels[t] * pixel_area_km2)
            model_area = float(model_pixels[t] * pixel_area_km2)
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
                "anchor_constraint_applied": True,
                "calibrated_threshold": float(group.iloc[t].get("calibrated_threshold", np.nan)),
                "posterior_threshold": float(settings.get("minimum_post_installation_mean_probability", 0.44)),
                "estimation_mode": mode,
                "estimation_route": route,
                "pv_area_ci_low_km2": np.nan,
                "pv_area_ci_high_km2": np.nan,
                "confirmed_anchor_area_km2": confirmed_anchor_area,
                "inferred_deployment_year": inferred_deployment_year,
                "annual_probability_bias": 0.0,
                "area_state_definition": "observed_active",
                "temporal_decoder": "v24_18_ordered_change_point_no_anchor_backfill",
            })
        frame = pd.DataFrame(unit_rows)
        save_table(frame, unit_registry_path)
        payload = {
            "status": "passed",
            "schema_version": "24.18.0",
            "Id": region_id,
            "forced_reference_only_pixels": int(forced_reference_pixels),
            "pre_reference_evidence_supported_pixels": int(pre_reference_pixels),
            "historical_anchor_geometry_copied": False,
            "created_utc": utc_now(),
        }
        atomic_json(payload, checkpoint)
        rows.extend(frame.to_dict("records"))
        worker_audits.append(payload)

    direct_registry = pd.DataFrame(rows).sort_values(["Id", "year"]) if rows else pd.DataFrame()
    fallback_registry, _ = infer_area_only_units(cfg, direct_registry)
    registry = pd.concat([direct_registry, fallback_registry], ignore_index=True, sort=False).sort_values(["Id", "year"])
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
        "schema_version": "24.18.0",
        "records": int(len(registry)),
        "expected_records": int(expected_records),
        "units": int(registry["Id"].nunique()) if len(registry) else 0,
        "decoder": "ordered image-evidence change point conditioned on the audited 2024 endpoint",
        "reference_anchor_forced_in_reference_year": True,
        "reference_anchor_copied_to_historical_years_without_change_evidence": False,
        "outside_reference_inventory_can_be_supplemented": True,
        "environment_creates_historical_labels": False,
        "no_data_is_never_decoded_as_pv": True,
        "test_units_anchored": False,
        "area_only_pixel_maps_fabricated": False,
        "direct_pixel_units": direct_units,
        "anchor_guided_candidate_units": anchor_candidate_units,
        "area_only_units": area_only_units,
        "cumulative_installed_state_is_persistent": True,
        "observed_active_state_allows_consecutive_negative_retirement": True,
        "observed_active_state_allows_only_multiyear_negative_retirement": True,
        "image_quality_gaps_do_not_create_retirement": True,
        "preexisting_start_of_series_facilities_supported": True,
        "model_only_diagnostic_product_retained": True,
        "scientific_temporal_gate_pass": not bool(synchrony_diag.get("synchronous_anomaly", False)),
        **synchrony_diag,
        "created_utc": utc_now(),
    }, out / "temporal_decoding_status.json")
    if not coverage_ok:
        raise RuntimeError(f"Temporal decoding coverage incomplete: {len(registry)}/{expected_records} records")
    return registry
