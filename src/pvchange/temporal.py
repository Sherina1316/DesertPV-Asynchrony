from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from tqdm import tqdm

from .fallback import infer_area_only_units
from .features import compute_features
from .routes import PIXEL_MODES
from .spectral_mixture import load_endmember_matrix
from .temporal_signature import (
    annual_latent_masks_from_onset,
    annual_masks_from_onset,
    detect_temporal_signature,
    filter_transition_components,
    robust_probability_harmonization,
)
from .study_area import expected_counts
from .utils import atomic_json, ensure_dir, save_table, utc_now


def forward_backward(prob: np.ndarray, birth: float, retirement: float, initial_pv: float) -> np.ndarray:
    """Backward-compatible two-state smoother used by regression tests."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-5, 1 - 1e-5)
    t, n = p.shape
    trans = np.array([[1 - birth, birth], [retirement, 1 - retirement]], dtype=np.float64)
    emission = np.stack([1 - p, p], axis=-1)
    alpha = np.zeros((t, n, 2), dtype=np.float64)
    alpha[0, :, 0] = (1 - initial_pv) * emission[0, :, 0]
    alpha[0, :, 1] = initial_pv * emission[0, :, 1]
    alpha[0] /= np.maximum(alpha[0].sum(-1, keepdims=True), 1e-12)
    for i in range(1, t):
        alpha[i] = (alpha[i - 1] @ trans) * emission[i]
        alpha[i] /= np.maximum(alpha[i].sum(-1, keepdims=True), 1e-12)
    beta = np.ones((t, n, 2), dtype=np.float64)
    for i in range(t - 2, -1, -1):
        beta[i] = (beta[i + 1] * emission[i + 1]) @ trans.T
        beta[i] /= np.maximum(beta[i].sum(-1, keepdims=True), 1e-12)
    posterior = alpha * beta
    posterior /= np.maximum(posterior.sum(-1, keepdims=True), 1e-12)
    return posterior[:, :, 1].astype(np.float32)


def first_persistent_year(mask: np.ndarray, valid: np.ndarray, years: np.ndarray, persistence: int) -> np.ndarray:
    t, n = mask.shape
    result = np.zeros(n, dtype=np.int16)
    for i in range(0, t - persistence + 1):
        sustained = np.all(mask[i:i + persistence] & valid[i:i + persistence], axis=0) & (result == 0)
        result[sustained] = int(years[i])
    return result


def build_endpoint_aware_synchrony_audit(
    registry: pd.DataFrame,
    cfg: dict[str, Any],
    years: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit cross-unit synchrony using the full independent-test denominator.

    A pipeline-wide synchrony artefact means that a large fraction of *all*
    independent, unanchored test units exhibits its largest substantive annual
    gain in the same interior year. Units without a substantive gain remain in
    the denominator because excluding them can turn two coincident active units
    out of three into an apparent 0.667 system-wide anomaly, even when the test
    cohort contains many unaffected units.

    The active-only concentration is retained as a diagnostic, but it is never
    used alone as a hard gate. The hard gate additionally requires an absolute
    count and a Bonferroni-adjusted exact binomial tail probability.
    """
    temporal_cfg = cfg["temporal_decoding"]
    reference_year = int(cfg["study"].get("reference_year", int(years[-1])))
    start_year = int(cfg["study"].get("start_year", int(years[0])))
    role_by_unit = (
        registry[["Id", "split_role"]]
        .drop_duplicates("Id")
        .set_index("Id")["split_role"]
        .astype(str)
    )
    area = registry.pivot(index="Id", columns="year", values="pv_area_km2").sort_index(axis=1)
    gains = area.diff(axis=1)
    minimum_gain = float(temporal_cfg.get("synchrony_minimum_peak_gain_km2", 0.01))
    allowed = float(temporal_cfg["maximum_synchronous_peak_fraction"])
    gate_roles = set(map(str, temporal_cfg.get("synchrony_gate_roles", ["test"])))
    exclude_reference = bool(temporal_cfg.get("exclude_reference_year_from_synchrony_gate", True))
    exclude_start = bool(temporal_cfg.get("exclude_start_year_from_synchrony_gate", True))
    minimum_units = int(temporal_cfg.get("minimum_synchrony_gate_units", 10))
    minimum_peak_units = int(temporal_cfg.get("minimum_synchronous_peak_units", 5))
    familywise_alpha = float(temporal_cfg.get("synchrony_familywise_alpha", 0.05))

    audit_rows: list[dict[str, Any]] = []

    def summarize(cohort: str, unit_ids: list[int], candidate_years: list[int], gate: bool) -> dict[str, Any]:
        subset = gains.reindex(index=unit_ids, columns=candidate_years)
        cohort_units = int(len(subset.index))
        if subset.empty or not candidate_years:
            active_peak = pd.Series(dtype=float)
        else:
            peak_gain = subset.max(axis=1, skipna=True)
            active_ids = peak_gain[peak_gain >= minimum_gain].index
            active_peak = (
                subset.loc[active_ids].idxmax(axis=1)
                if len(active_ids)
                else pd.Series(dtype=float)
            )
        active_units = int(len(active_peak))
        counts = active_peak.value_counts() if active_units else pd.Series(dtype=int)
        null_probability = 1.0 / max(len(candidate_years), 1)
        per_year_p: dict[int, float] = {}
        for year in candidate_years:
            count = int(counts.get(int(year), 0))
            if cohort_units > 0:
                raw_p = float(
                    binomtest(
                        count,
                        n=cohort_units,
                        p=null_probability,
                        alternative="greater",
                    ).pvalue
                )
                adjusted_p = min(1.0, raw_p * max(len(candidate_years), 1))
            else:
                adjusted_p = 1.0
            per_year_p[int(year)] = adjusted_p
        for year in years:
            count = int(counts.get(int(year), 0))
            all_fraction = float(count / max(cohort_units, 1))
            active_fraction = float(count / max(active_units, 1))
            audit_rows.append({
                "cohort": cohort,
                "year": int(year),
                "peak_gain_units": count,
                "cohort_units": cohort_units,
                "active_gain_units": active_units,
                # Backward-compatible field: denominator is now all cohort units.
                "eligible_units": cohort_units,
                "peak_fraction": all_fraction,
                "peak_fraction_all_cohort_units": all_fraction,
                "peak_fraction_active_units": active_fraction,
                "bonferroni_adjusted_binomial_p": float(per_year_p.get(int(year), 1.0)),
                "minimum_peak_gain_km2": minimum_gain,
                "minimum_synchronous_peak_units": minimum_peak_units,
                "is_gate_cohort": bool(gate),
                "year_in_gate_window": int(year) in candidate_years,
                "reference_year": reference_year,
            })
        if counts.empty:
            maximum_count = 0
            anomalous_year = None
            maximum_fraction = 0.0
            maximum_active_fraction = 0.0
            adjusted_p = 1.0
        else:
            anomalous_year = int(counts.idxmax())
            maximum_count = int(counts.max())
            maximum_fraction = float(maximum_count / max(cohort_units, 1))
            maximum_active_fraction = float(maximum_count / max(active_units, 1))
            adjusted_p = float(per_year_p.get(anomalous_year, 1.0))
        evaluable = cohort_units >= minimum_units and len(candidate_years) >= 2
        anomaly = bool(
            gate
            and evaluable
            and maximum_count >= minimum_peak_units
            and maximum_fraction > allowed
            and adjusted_p < familywise_alpha
        )
        return {
            "cohort": cohort,
            "cohort_units": cohort_units,
            "eligible_units": cohort_units,
            "active_gain_units": active_units,
            "maximum_count": maximum_count,
            "maximum_fraction": maximum_fraction,
            "maximum_active_fraction": maximum_active_fraction,
            "anomalous_year": anomalous_year,
            "adjusted_p_value": adjusted_p,
            "evaluable": evaluable,
            "anomaly": anomaly,
        }

    all_units = [int(value) for value in area.index]
    raw_years = [int(y) for y in years if int(y) != start_year]
    summarize("all_units_raw_endpoint_inclusive", all_units, raw_years, False)

    for role in sorted(role_by_unit.unique()):
        ids = [int(value) for value in role_by_unit[role_by_unit.eq(role)].index]
        summarize(f"split_{role}_raw_endpoint_inclusive", ids, raw_years, False)

    gate_years = [int(y) for y in years]
    if exclude_start:
        gate_years = [y for y in gate_years if y != start_year]
    if exclude_reference:
        gate_years = [y for y in gate_years if y != reference_year]
    gate_ids = [int(value) for value in role_by_unit[role_by_unit.isin(gate_roles)].index]
    gate_summary = summarize("independent_unanchored_internal_year_gate", gate_ids, gate_years, True)

    raw_audit = pd.DataFrame(audit_rows)
    endpoint_row = raw_audit.loc[
        (raw_audit["cohort"] == "all_units_raw_endpoint_inclusive")
        & (raw_audit["year"] == reference_year)
    ]
    endpoint_fraction = (
        float(endpoint_row["peak_fraction_all_cohort_units"].iloc[0])
        if not endpoint_row.empty
        else 0.0
    )
    diagnostics = {
        "synchrony_gate_definition": (
            "independent unanchored test units; interior years; denominator is all test units; "
            "active-only concentration is diagnostic; hard failure requires fraction, absolute-count, "
            "and Bonferroni-adjusted exact-binomial criteria"
        ),
        "synchrony_gate_roles": sorted(gate_roles),
        "synchrony_gate_years": gate_years,
        "synchrony_gate_eligible_units": int(gate_summary["cohort_units"]),
        "synchrony_gate_active_gain_units": int(gate_summary["active_gain_units"]),
        "synchrony_gate_minimum_units": minimum_units,
        "minimum_synchronous_peak_units": minimum_peak_units,
        "synchrony_familywise_alpha": familywise_alpha,
        "synchrony_gate_evaluable": bool(gate_summary["evaluable"]),
        "synchrony_gate_conclusive": bool(gate_summary["evaluable"]),
        "maximum_synchronous_peak_unit_count": int(gate_summary["maximum_count"]),
        "maximum_synchronous_peak_fraction_observed": float(gate_summary["maximum_fraction"]),
        "maximum_active_only_peak_fraction_diagnostic": float(gate_summary["maximum_active_fraction"]),
        "synchrony_adjusted_binomial_p_value": float(gate_summary["adjusted_p_value"]),
        "synchronous_anomaly_year": gate_summary["anomalous_year"],
        "synchronous_anomaly": bool(gate_summary["anomaly"]),
        "maximum_synchronous_peak_fraction_allowed": allowed,
        "reference_year_peak_fraction_all_units_raw": endpoint_fraction,
        "reference_year_endpoint_concentration_warning": bool(endpoint_fraction > allowed),
        "reference_year_excluded_from_hard_gate": exclude_reference,
        "start_year_excluded_from_hard_gate": exclude_start,
        "minimum_peak_gain_km2": minimum_gain,
    }
    return raw_audit, diagnostics



def _synchrony_audit(registry: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    direct = registry.loc[registry["estimation_mode"].isin(PIXEL_MODES)].copy()
    years = sorted(direct.year.unique())
    area = direct.pivot(index="Id", columns="year", values="pv_area_km2").reindex(columns=years)
    gains = area.diff(axis=1)
    minimum = float(cfg["temporal_decoding"].get("synchrony_minimum_peak_gain_km2", 0.01))
    peak_gain = gains.max(axis=1, skipna=True)
    active = peak_gain[peak_gain >= minimum].index
    peak_year = gains.loc[active].idxmax(axis=1) if len(active) else pd.Series(dtype=float)
    counts = peak_year.value_counts()
    denominator = max(len(area), 1)
    rows = []
    for year in years:
        count = int(counts.get(year, 0))
        rows.append({
            "year": int(year), "peak_gain_units": count, "all_pixel_mapped_units": len(area),
            "active_gain_units": len(active), "peak_fraction_all_units": count / denominator,
            "peak_fraction_active_units": count / max(len(active), 1),
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        max_fraction, anomaly_year, max_count = 0.0, None, 0
    else:
        top = audit.sort_values("peak_fraction_all_units", ascending=False).iloc[0]
        max_fraction, anomaly_year, max_count = float(top.peak_fraction_all_units), int(top.year), int(top.peak_gain_units)
    allowed = float(cfg["temporal_decoding"].get("maximum_synchronous_peak_fraction", 0.3))
    anomaly = bool(max_fraction > allowed and max_count >= int(cfg["temporal_decoding"].get("minimum_synchronous_peak_units", 5)))
    return audit, {
        "maximum_synchronous_peak_fraction_observed": max_fraction,
        "synchronous_anomaly_year": anomaly_year,
        "maximum_synchronous_peak_unit_count": max_count,
        "synchronous_anomaly": anomaly,
        "synchrony_gate_eligible_units": int(len(area)),
        "synchrony_gate_active_gain_units": int(len(active)),
    }



def _valid_worker_checkpoint(checkpoint: Path, registry_path: Path, expected_years: int) -> bool:
    if not checkpoint.exists() or not registry_path.exists():
        return False
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        frame = pd.read_csv(registry_path)
        required_columns = {
            "mask_path",
            "cumulative_mask_path",
            "model_only_cumulative_mask_path",
            "anchor_backcast_path",
            "anchor_forced_reference_path",
            "anchor_backcast_object_fill_path",
            "anchor_reference_path",
        }
        return (
            payload.get("status") == "passed"
            and payload.get("schema_version") == "24.15.0"
            and len(frame) == expected_years
            and required_columns.issubset(frame.columns)
            and all(Path(value).exists() for value in frame["mask_path"].astype(str))
            and all(Path(value).exists() for value in frame["cumulative_mask_path"].astype(str))
            and all(Path(value).exists() for value in frame["model_only_cumulative_mask_path"].astype(str))
            and all(Path(value).exists() for value in frame["anchor_backcast_path"].astype(str))
            and all(Path(value).exists() for value in frame["anchor_forced_reference_path"].astype(str))
            and all(Path(value).exists() for value in frame["anchor_backcast_object_fill_path"].astype(str))
            and all(Path(value).exists() for value in frame["anchor_reference_path"].astype(str))
        )
    except Exception:
        return False


def _worker_environment(cfg: dict[str, Any]) -> dict[str, str]:
    code_root = Path(__file__).resolve().parents[2]
    settings = cfg.get("resource_control", {})
    return {
        **os.environ,
        "PYTHONPATH": str(code_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
        "GDAL_CACHEMAX": str(int(settings.get("gdal_cache_mb", 32))),
        "VSI_CACHE": "FALSE",
        "CPL_VSIL_CURL_CACHE_SIZE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "1",
        "MALLOC_TRIM_THRESHOLD_": "65536",
        "PYTHONMALLOC": "malloc",
    }


def temporal_decode(cfg: dict[str, Any]) -> pd.DataFrame:
    import signal
    import subprocess
    import sys

    from .resource_control import memory_snapshot, wait_for_resource_headroom

    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "08_temporal_pv_change")
    inference = pd.read_csv(
        root / "07_probability_calibration" / "calibrated_inference_registry.csv"
    ).sort_values(["Id", "year"])
    years = np.arange(
        int(cfg["study"]["start_year"]),
        int(cfg["study"]["end_year"]) + 1,
        dtype=int,
    )
    mask_nodata = int(cfg.get("temporal_decoding", {}).get("mask_nodata", 255))
    checkpoint_root = ensure_dir(out / "unit_checkpoints")
    unit_registry_root = ensure_dir(out / "unit_registries")
    diagnostics_root = ensure_dir(out / "resource_diagnostics")

    pixel_units: list[int] = []
    for region_id, group in inference.groupby("Id"):
        mode = str(group.iloc[0].get("inference_mode", "direct_pixel"))
        if mode in PIXEL_MODES:
            pixel_units.append(int(region_id))
    pixel_units.sort()

    rows: list[dict[str, Any]] = []
    harmonization_rows: list[dict[str, Any]] = []
    worker_audit_rows: list[dict[str, Any]] = []
    overwrite = bool(cfg.get("processing", {}).get("overwrite", False))

    iterator = tqdm(
        pixel_units,
        desc="Persistent transition decoding",
        unit="region",
        disable=not cfg.get("processing", {}).get("show_progress", True),
    )
    for region_id in iterator:
        iterator.set_postfix_str(f"unit {region_id:03d}")
        checkpoint = checkpoint_root / f"unit_{region_id:03d}.json"
        registry_path = unit_registry_root / f"unit_{region_id:03d}.csv"
        if not overwrite and _valid_worker_checkpoint(checkpoint, registry_path, len(years)):
            existing = pd.read_csv(registry_path)
            rows.extend(existing.to_dict("records"))
            for record in existing.itertuples(index=False):
                harmonization_rows.append({
                    "Id": int(record.Id),
                    "year": int(record.year),
                    "annual_probability_logit_bias": float(record.annual_probability_bias),
                })
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                worker_audit_rows.append(payload)
            except Exception:
                pass
            continue

        before = wait_for_resource_headroom(
            cfg,
            stage_label=f"Stage 8 unit {region_id:03d}",
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "pvchange.temporal_worker",
            "--config",
            str(root / "effective_analysis_config.yaml"),
            "--unit-id",
            str(region_id),
        ]
        if overwrite:
            command.append("--overwrite")
        print(
            f"Starting isolated temporal worker for unit {region_id:03d}; "
            f"MemAvailable={before.available_gb:.1f} GiB, "
            f"SwapFree={before.swap_free_gb:.2f} GiB",
            flush=True,
        )
        completed = subprocess.run(
            command,
            env=_worker_environment(cfg),
            check=False,
        )
        if completed.returncode != 0:
            after = memory_snapshot(
                cfg["paths"]["project_root"],
                float(cfg.get("resource_control", {}).get("minimum_competing_process_rss_mb", 512.0)),
            )
            signal_name = None
            if completed.returncode < 0:
                try:
                    signal_name = signal.Signals(-completed.returncode).name
                except Exception:
                    signal_name = f"signal {-completed.returncode}"
            payload = {
                "status": "failed",
                "Id": int(region_id),
                "return_code": int(completed.returncode),
                "signal": signal_name,
                "memory_before": before.to_dict(),
                "memory_after": after.to_dict(),
                "created_utc": utc_now(),
            }
            atomic_json(
                payload,
                diagnostics_root / f"unit_{region_id:03d}_failure.json",
            )
            if signal_name:
                raise RuntimeError(
                    f"Stage 8 isolated worker for unit {region_id:03d} was terminated by {signal_name}. "
                    f"MemAvailable after termination was {after.available_gb:.1f} GiB and "
                    f"SwapFree was {after.swap_free_gb:.2f} GiB. "
                    f"No partial unit result was accepted; see {diagnostics_root}."
                )
            raise RuntimeError(
                f"Stage 8 isolated worker for unit {region_id:03d} failed with return code "
                f"{completed.returncode}; see {diagnostics_root}."
            )

        if not _valid_worker_checkpoint(checkpoint, registry_path, len(years)):
            raise RuntimeError(
                f"Stage 8 worker for unit {region_id:03d} exited successfully but its output contract is incomplete"
            )
        frame = pd.read_csv(registry_path)
        rows.extend(frame.to_dict("records"))
        for record in frame.itertuples(index=False):
            harmonization_rows.append({
                "Id": int(record.Id),
                "year": int(record.year),
                "annual_probability_logit_bias": float(record.annual_probability_bias),
            })
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        worker_audit_rows.append(payload)

    direct_registry = pd.DataFrame(rows).sort_values(["Id", "year"])
    fallback_registry, _ = infer_area_only_units(cfg, direct_registry)
    registry = pd.concat(
        [direct_registry, fallback_registry],
        ignore_index=True,
        sort=False,
    ).sort_values(["Id", "year"])
    save_table(registry, out / "temporal_mask_registry.csv")
    save_table(pd.DataFrame(harmonization_rows), out / "annual_probability_harmonization.csv")
    save_table(pd.DataFrame(worker_audit_rows), out / "temporal_worker_resource_audit.csv")
    synchrony, synchrony_diag = _synchrony_audit(registry, cfg)
    save_table(synchrony, out / "temporal_synchrony_audit.csv")

    expected_units = expected_counts(cfg)[0]
    expected_records = expected_units * len(years)
    coverage_ok = len(registry) == expected_records and registry.Id.nunique() == expected_units
    peak_worker_rss = (
        float(pd.to_numeric(pd.DataFrame(worker_audit_rows).get("peak_worker_rss_mb"), errors="coerce").max())
        if worker_audit_rows
        else float("nan")
    )
    payload = {
        "status": "passed" if coverage_ok else "failed",
        "created_utc": utc_now(),
        "records": len(registry),
        "expected_records": expected_records,
        "units": int(registry.Id.nunique()),
        "mask_nodata": mask_nodata,
        "decoder": "isolated-unit streaming temporal signature + 2024 anchor endpoint + full-series 2011-2025 anchor spectral-prototype decoding + reference-guided 2025 endpoint + contiguous-growth topology + dual cumulative/active states",
        "worker_isolation": True,
        "reference_shp_loaded_in_stage8": False,
        "reference_anchor_source": "pre-rasterized 2024 complete-reference anchor support",
        "reference_anchor_forced_in_reference_year": True,
        "reference_anchor_copied_to_historical_years_without_change_evidence": False,
        "reference_shp_drawn_as_overlay": False,
        "reference_shp_role": "certain 2024 endpoint constraint; historical dates require model/LSMA temporal evidence or anchor-internal spectral-trajectory support",
        "anchor_internal_spectral_rescue_enabled": True,
        "spatiotemporal_topology_filter_enabled": True,
        "remote_isolated_small_components_suppressed": True,
        "model_only_diagnostic_product_retained": True,
        "anchor_full_series_2011_2025_decoder_enabled": True,
        "reference_guided_2025_endpoint_enabled": True,
        "instant_2025_retirement_from_one_observation": False,
        "no_data_is_never_decoded_as_pv": True,
        "cumulative_installed_state_is_persistent": True,
        "observed_active_state_allows_consecutive_negative_retirement": True,
        "observed_active_state_allows_only_multiyear_negative_retirement": True,
        "image_quality_gaps_do_not_create_retirement": True,
        "preexisting_start_of_series_facilities_supported": True,
        "area_accounting": "anchor-constrained cumulative installed footprint and observed active footprint; model-only cumulative footprint retained for validation",
        "direct_pixel_units": int(direct_registry.loc[direct_registry.estimation_mode.eq("direct_pixel"), "Id"].nunique()),
        "anchor_guided_candidate_units": int(direct_registry.loc[direct_registry.estimation_mode.eq("anchor_guided_candidate"), "Id"].nunique()),
        "pixel_mapping_units": int(direct_registry.Id.nunique()),
        "area_only_units": int(fallback_registry.Id.nunique()) if len(fallback_registry) else 0,
        "area_only_pixel_maps_fabricated": False,
        "maximum_worker_peak_rss_mb": peak_worker_rss,
        "scientific_temporal_gate_pass": not bool(synchrony_diag["synchronous_anomaly"]),
        **synchrony_diag,
    }
    atomic_json(payload, out / "temporal_decoding_status.json")
    if not coverage_ok:
        raise RuntimeError(
            f"Temporal decoding coverage incomplete: {registry.Id.nunique()}/{expected_units} units and "
            f"{len(registry)}/{expected_records} unit-years"
        )
    return registry
