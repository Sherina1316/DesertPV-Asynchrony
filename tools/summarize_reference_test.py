from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read1(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        a = src.read([3, 2, 1]).astype(np.float32)
    rgb = np.moveaxis(a, 0, -1)
    valid = np.all(np.isfinite(rgb), axis=2)
    out = np.zeros_like(rgb)
    for b in range(3):
        vals = rgb[..., b][valid]
        if vals.size:
            lo, hi = np.nanquantile(vals, [0.02, 0.98])
            out[..., b] = np.clip((rgb[..., b] - lo) / max(float(hi - lo), 1e-6), 0, 1)
    return out


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    out = rgb.copy()
    m = mask == 1
    out[m] = 0.38 * out[m] + 0.62 * np.asarray(color, dtype=np.float32)
    return out


def _status_rows(root: Path) -> list[dict]:
    paths = {
        "authoritative_reference": root / "07_authoritative_reference/authoritative_reference_status.json",
        "temporal_change_point": root / "08_temporal_pv_change/temporal_decoding_status.json",
        "validation": root / "09_accuracy_validation/accuracy_validation_status.json",
        "maps": root / "17_maps/map_status.json",
        "figures": root / "18_figures/figure_status.json",
        "audit": root / "19_audit/audit_status.json",
        "forecast": root / "21_strict_noleak_forecast/noleak_forecast_status.json",
    }
    rows = []
    for name, path in paths.items():
        payload = _json(path)
        status = payload.get("status", "missing")
        rows.append({"check": name, "status": status, "passed": status in {"passed", "completed", "completed_diagnostic", "completed_with_scientific_gate_failure"}, "path": str(path)})
    completed = (root / ".step_markers/step_21_completed.json").exists()
    rows.append({"check": "complete_21_stage_loop", "status": "passed" if completed else "incomplete", "passed": completed, "path": str(root / ".step_markers/step_21_completed.json")})
    return rows


def render_unit_audit(root: Path, outdir: Path, unit_id: int) -> list[dict]:
    aligned = _csv(root / "02_aligned_30m/aligned_raster_registry.csv")
    stage7 = _csv(root / "07_authoritative_reference/authoritative_reference_registry.csv")
    temporal = _csv(root / "08_temporal_pv_change/temporal_mask_registry.csv")
    rows = []
    required = {
        "aligned": {"Id", "year", "path"},
        "stage7": {"Id", "year", "calibrated_mask_path", "reference_inventory_path", "reference_anchor_path", "multiband_break_support_path", "reference_boundary_correction_code_path", "stable_sand_risk_path"},
        "temporal": {"Id", "year", "active_mask_path", "historical_growth_front_rescue_path", "terminal_2025_expansion_source_path"},
    }
    if (
        not required["aligned"].issubset(aligned.columns)
        or not required["stage7"].issubset(stage7.columns)
        or not required["temporal"].issubset(temporal.columns)
    ):
        # A failed/incomplete pipeline may not have written Stage 7 or Stage 8
        # registries yet.  The diagnostic summarizer must report incomplete
        # status rather than masking the original pipeline error with KeyError.
        return rows
    inspect_years = [2011, 2020, 2024, 2025]
    fig, axes = plt.subplots(len(inspect_years), 6, figsize=(19, 3.1 * len(inspect_years)))
    for r, year in enumerate(inspect_years):
        land = aligned.loc[(aligned["Id"] == unit_id) & (aligned["year"] == year)]
        s7 = stage7.loc[(stage7["Id"] == unit_id) & (stage7["year"] == year)]
        s8 = temporal.loc[(temporal["Id"] == unit_id) & (temporal["year"] == year)]
        if land.empty or s7.empty or s8.empty:
            for ax in axes[r]:
                ax.axis("off")
            continue
        land = land.iloc[0]
        s7 = s7.iloc[0]
        s8 = s8.iloc[0]
        rgb = _rgb(Path(str(land["path"])))
        evidence = _read1(Path(str(s7["calibrated_mask_path"])))
        active = _read1(Path(str(s8["active_mask_path"])))
        inventory = _read1(Path(str(s7["reference_inventory_path"])))
        manual = _read1(Path(str(s7["reference_anchor_path"])))
        change = _read1(Path(str(s7["multiband_break_support_path"])))
        correction = _read1(Path(str(s7["reference_boundary_correction_code_path"])))
        sand = _read1(Path(str(s7["stable_sand_risk_path"])))
        rescue = _read1(Path(str(s8["historical_growth_front_rescue_path"])))
        terminal_source = _read1(Path(str(s8["terminal_2025_expansion_source_path"])))
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{year} Landsat")
        axes[r, 1].imshow(_overlay(rgb, manual, (0.40, 0.40, 0.40)))
        axes[r, 1].set_title("Manual 2024 prior")
        axes[r, 2].imshow(_overlay(rgb, inventory, (0.00, 0.45, 0.78)))
        axes[r, 2].set_title("Refined 2024 inventory")
        axes[r, 3].imshow(_overlay(rgb, active, (0.00, 0.45, 0.78)))
        axes[r, 3].set_title("Final active PV")
        axes[r, 4].imshow(np.where(change > -1000, change, np.nan), vmin=0, vmax=1)
        axes[r, 4].set_title("Multiband break support")
        axes[r, 5].imshow(correction, vmin=0, vmax=4)
        axes[r, 5].set_title("Boundary correction code")
        for ax in axes[r]:
            ax.axis("off")
        rows.append({
            "Id": unit_id,
            "year": year,
            "annual_evidence_pixels": int(np.count_nonzero(evidence == 1)),
            "final_active_pixels": int(np.count_nonzero(active == 1)),
            "reference_inventory_pixels": int(np.count_nonzero(inventory == 1)),
            "manual_reference_pixels": int(np.count_nonzero(manual == 1)),
            "reference_added_pixels": int(np.count_nonzero(correction == 4)),
            "reference_removed_pixels": int(np.count_nonzero(correction == 3)),
            "mean_multiband_break_support": float(np.nanmean(np.where(change > -1000, change, np.nan))),
            "mean_stable_sand_risk": float(np.nanmean(np.where(sand > -1000, sand, np.nan))),
            "historical_extent_rescued_pixels": int(np.count_nonzero(rescue == 1)),
            "terminal_2025_adjacent_pixels": int(np.count_nonzero(terminal_source == 1)),
            "terminal_2025_remote_pixels": int(np.count_nonzero(terminal_source == 2)),
            "terminal_2025_perimeter_extension_pixels": int(np.count_nonzero(terminal_source == 3)),
            "terminal_2025_novelty_gated_remote_change_pixels": int(np.count_nonzero(terminal_source == 4)),
        })
    fig.suptitle(f"Unit {unit_id:03d}: causal 2024 boundary, growth-front and 2025 expansion audit")
    fig.tight_layout()
    fig.savefig(outdir / f"unit_{unit_id:03d}_adaptive_reference_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"unit_{unit_id:03d}_adaptive_reference_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows



def render_terminal_extension_audit(root: Path, outdir: Path, unit_id: int) -> dict:
    aligned = _csv(root / "02_aligned_30m/aligned_raster_registry.csv")
    stage7 = _csv(root / "07_authoritative_reference/authoritative_reference_registry.csv")
    required = {
        "Id", "year", "reference_inventory_path",
        "terminal_2025_expansion_source_path",
        "terminal_2025_site_likelihood_path",
        "terminal_2025_component_similarity_path",
        "terminal_2025_multicue_change_path",
        "terminal_2025_front_proximity_path",
        "terminal_2025_remote_census_likelihood_path",
        "terminal_2025_remote_change_support_path",
        "terminal_2025_remote_candidate_density_path",
        "terminal_2025_absolute_darkness_path",
        "terminal_2025_novelty_score_path",
        "terminal_2025_direct_change_support_path",
        "terminal_2025_preexisting_state_risk_path",
        "terminal_2025_persistent_nonpv_risk_path",
    }
    if not required.issubset(stage7.columns) or not {"Id", "year", "path"}.issubset(aligned.columns):
        return {}
    ref = stage7.loc[(stage7["Id"] == unit_id) & (stage7["year"] == 2024)]
    y24 = aligned.loc[(aligned["Id"] == unit_id) & (aligned["year"] == 2024)]
    y25 = aligned.loc[(aligned["Id"] == unit_id) & (aligned["year"] == 2025)]
    if ref.empty or y24.empty or y25.empty:
        return {}
    row = ref.iloc[0]
    rgb24 = _rgb(Path(str(y24.iloc[0]["path"])))
    rgb25 = _rgb(Path(str(y25.iloc[0]["path"])))
    inventory = _read1(Path(str(row["reference_inventory_path"])))
    source = _read1(Path(str(row["terminal_2025_expansion_source_path"])))
    site = _read1(Path(str(row["terminal_2025_site_likelihood_path"])))
    similarity = _read1(Path(str(row["terminal_2025_component_similarity_path"])))
    change = _read1(Path(str(row["terminal_2025_multicue_change_path"])))
    proximity = _read1(Path(str(row["terminal_2025_front_proximity_path"])))
    remote_census = _read1(Path(str(row["terminal_2025_remote_census_likelihood_path"])))
    remote_change = _read1(Path(str(row["terminal_2025_remote_change_support_path"])))
    remote_density = _read1(Path(str(row["terminal_2025_remote_candidate_density_path"])))
    absolute_darkness = _read1(Path(str(row["terminal_2025_absolute_darkness_path"])))
    novelty = _read1(Path(str(row["terminal_2025_novelty_score_path"])))
    direct_change = _read1(Path(str(row["terminal_2025_direct_change_support_path"])))
    preexisting_risk = _read1(Path(str(row["terminal_2025_preexisting_state_risk_path"])))
    persistent_risk = _read1(Path(str(row["terminal_2025_persistent_nonpv_risk_path"])))
    site = np.where(site > -1000, site, np.nan)
    similarity = np.where(similarity > -1000, similarity, np.nan)
    change = np.where(change > -1000, change, np.nan)
    proximity = np.where(proximity > -1000, proximity, np.nan)
    remote_census = np.where(remote_census > -1000, remote_census, np.nan)
    remote_change = np.where(remote_change > -1000, remote_change, np.nan)
    remote_density = np.where(remote_density > -1000, remote_density, np.nan)
    absolute_darkness = np.where(absolute_darkness > -1000, absolute_darkness, np.nan)
    novelty = np.where(novelty > -1000, novelty, np.nan)
    direct_change = np.where(direct_change > -1000, direct_change, np.nan)
    preexisting_risk = np.where(preexisting_risk > -1000, preexisting_risk, np.nan)
    persistent_risk = np.where(persistent_risk > -1000, persistent_risk, np.nan)

    fig, axes = plt.subplots(4, 4, figsize=(15.5, 13.0))
    axes[0, 0].imshow(rgb24); axes[0, 0].set_title("2024 Landsat")
    axes[0, 1].imshow(_overlay(rgb24, inventory, (0.95, 0.80, 0.05))); axes[0, 1].set_title("Frozen 2024 inventory")
    axes[0, 2].imshow(rgb25); axes[0, 2].set_title("2025 Landsat")
    axes[0, 3].imshow(source, vmin=0, vmax=4); axes[0, 3].set_title("Source: 1 adjacent / 2 strict remote / 3 perimeter / 4 census")
    axes[1, 0].imshow(site, vmin=0, vmax=1); axes[1, 0].set_title("2025 site likelihood")
    axes[1, 1].imshow(similarity, vmin=0, vmax=1); axes[1, 1].set_title("Component similarity")
    axes[1, 2].imshow(change, vmin=0, vmax=1); axes[1, 2].set_title("Multicue change")
    axes[1, 3].imshow(proximity, vmin=0, vmax=1); axes[1, 3].set_title("Front proximity")
    axes[2, 0].imshow(remote_census, vmin=0, vmax=1); axes[2, 0].set_title("Full-domain remote likelihood")
    axes[2, 1].imshow(remote_change, vmin=0, vmax=1); axes[2, 1].set_title("Novelty-gated change support")
    axes[2, 2].imshow(remote_density, vmin=0, vmax=1); axes[2, 2].set_title("Remote candidate density")
    axes[2, 3].imshow(absolute_darkness, vmin=0, vmax=1); axes[2, 3].set_title("2025 absolute darkness")
    axes[3, 0].imshow(novelty, vmin=0, vmax=1); axes[3, 0].set_title("2025 novelty score")
    axes[3, 1].imshow(direct_change, vmin=0, vmax=1); axes[3, 1].set_title("Direct 2024→2025 change")
    axes[3, 2].imshow(preexisting_risk, vmin=0, vmax=1); axes[3, 2].set_title("Pre-existing-state risk")
    axes[3, 3].imshow(persistent_risk, vmin=0, vmax=1); axes[3, 3].set_title("Persistent non-PV risk")
    for ax in axes.ravel(): ax.axis("off")
    fig.suptitle(f"Unit {unit_id:03d}: causal 2024 freeze and 2025 terminal-extension audit")
    fig.tight_layout()
    fig.savefig(outdir / f"unit_{unit_id:03d}_terminal_2025_extension_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"unit_{unit_id:03d}_terminal_2025_extension_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "Id": unit_id,
        "terminal_gain_adjacent_pixels": int(np.count_nonzero(source == 1)),
        "terminal_remote_pixels": int(np.count_nonzero(source == 2)),
        "terminal_perimeter_extension_pixels": int(np.count_nonzero(source == 3)),
        "terminal_novelty_gated_remote_change_pixels": int(np.count_nonzero(source == 4)),
        "terminal_total_pixels": int(np.count_nonzero(np.isin(source, [1, 2, 3, 4]))),
    }

def render_area(temporal: pd.DataFrame, outdir: Path, units: list[int]) -> None:
    required = {"Id", "year", "observed_active_area_km2", "cumulative_installed_area_km2"}
    if temporal.empty or not required.issubset(temporal.columns):
        return
    data = temporal.loc[temporal["Id"].isin(units)].copy()
    data.to_csv(outdir / "annual_area_units_6_28.csv", index=False)
    fig, axes = plt.subplots(len(units), 1, figsize=(9, 3.2 * len(units)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, uid in zip(axes, units):
        sub = data.loc[data["Id"] == uid].sort_values("year")
        ax.plot(sub["year"], sub["observed_active_area_km2"], marker="o", label="Observed active")
        ax.plot(sub["year"], sub["cumulative_installed_area_km2"], marker="o", label="Cumulative installed")
        ax.axvline(2024, linewidth=0.8, linestyle="--")
        ax.set_title(f"Unit {uid:03d}")
        ax.set_ylabel("Area (km²)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Causal-reference annual PV trajectories with terminal extensions")
    fig.tight_layout()
    fig.savefig(outdir / "annual_area_units_6_28.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "annual_area_units_6_28.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--unit-ids", default="6,28")
    args = ap.parse_args()
    units = [int(x) for x in args.unit_ids.split(",") if x.strip()]
    outdir = args.output_root / "algorithm_effect_test"
    outdir.mkdir(parents=True, exist_ok=True)

    temporal = _csv(args.output_root / "08_temporal_pv_change/temporal_mask_registry.csv")
    scorecard = pd.DataFrame(_status_rows(args.output_root))
    audit_rows: list[dict] = []
    terminal_rows: list[dict] = []
    for uid in units:
        audit_rows.extend(render_unit_audit(args.output_root, outdir, uid))
        terminal_row = render_terminal_extension_audit(args.output_root, outdir, uid)
        if terminal_row:
            terminal_rows.append(terminal_row)
    audit = pd.DataFrame(audit_rows)
    terminal_audit = pd.DataFrame(terminal_rows)
    render_area(temporal, outdir, units)
    scorecard.to_csv(outdir / "algorithm_effect_scorecard_units_6_28.csv", index=False)
    audit.to_csv(outdir / "adaptive_reference_audit_units_6_28.csv", index=False)
    terminal_audit.to_csv(outdir / "terminal_2025_extension_audit_units_6_28.csv", index=False)

    unit_rows = []
    temporal_ready = {"Id", "year", "observed_active_area_km2"}.issubset(temporal.columns)
    for uid in units:
        sub = (
            temporal.loc[temporal["Id"] == uid].sort_values("year")
            if temporal_ready
            else pd.DataFrame()
        )
        row = {"Id": uid, "records": int(len(sub))}
        for year in [2011, 2020, 2024, 2025]:
            hit = sub.loc[sub["year"] == year] if "year" in sub.columns else pd.DataFrame()
            row[f"active_area_{year}_km2"] = (
                float(hit.iloc[0]["observed_active_area_km2"])
                if not hit.empty and "observed_active_area_km2" in hit.columns
                else np.nan
            )
        unit_rows.append(row)
    unit_summary = pd.DataFrame(unit_rows)
    unit_summary.to_csv(outdir / "algorithm_effect_unit_summary_units_6_28.csv", index=False)

    complete = bool(scorecard.loc[scorecard["check"] == "complete_21_stage_loop", "passed"].all())
    status = "passed" if complete else "incomplete"
    payload = {
        "status": status,
        "unit_ids": units,
        "complete_21_stage_loop": complete,
        "scorecard": str(outdir / "algorithm_effect_scorecard_units_6_28.csv"),
        "unit_summary": str(outdir / "algorithm_effect_unit_summary_units_6_28.csv"),
        "audit": str(outdir / "adaptive_reference_audit_units_6_28.csv"),
        "terminal_audit": str(outdir / "terminal_2025_extension_audit_units_6_28.csv"),
    }
    (outdir / "algorithm_effect_test_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# v24.22 Unit 6/28 full-domain remote-census test",
        "",
        f"Overall status: **{status}**",
        "",
        "The manually interpreted 2024 vector is treated as a high-confidence prior. The 2024 boundary is refined causally using only 2011–2024 observations and is frozen before 2025 is evaluated. Historical extents use connected growth fronts. Terminal expansion uses gain-supported adjacency, strict remote-site detection and component-conditioned perimeter completion; Stage 8 preserves the exact Stage-7 source raster.",
        "",
        "## Unit summary",
        "",
        unit_summary.to_markdown(index=False),
        "",
        "## Stage scorecard",
        "",
        scorecard.to_markdown(index=False),
    ]
    text = "\n".join(report)
    (outdir / "algorithm_effect_test_report_units_6_28.md").write_text(text, encoding="utf-8")
    (outdir / "algorithm_effect_test_report_units_6_28.html").write_text("<html><meta charset='utf-8'><body>" + text.replace("\n", "<br>\n") + "</body></html>", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
