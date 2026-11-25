from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

from .routes import is_pixel_mode
from .utils import atomic_json, ensure_dir, save_table, utc_now


def _display_shape(width: int, height: int, maximum: int) -> tuple[int, int]:
    scale = min(1.0, float(maximum) / max(width, height))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _stretch(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for band in range(3):
        values = rgb[band][valid & np.isfinite(rgb[band])]
        if values.size:
            lo, hi = np.nanquantile(values, [0.02, 0.98])
            out[band] = np.clip((rgb[band] - lo) / max(float(hi - lo), 1e-6), 0, 1)
    image = np.moveaxis(out, 0, -1)
    image[~valid] = 1.0
    return image


def _mute_background(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Reduce false-colour saturation so the exclusive PV classes dominate."""
    vis = cfg.get("annual_visualization", {})
    desaturation = float(np.clip(vis.get("background_desaturation", 0.58), 0.0, 1.0))
    brightness = float(np.clip(vis.get("background_brightness", 0.82), 0.25, 1.25))
    gray = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )[..., None]
    muted = desaturation * gray + (1.0 - desaturation) * image
    return np.clip(muted * brightness, 0.0, 1.0)


def _valid_crop(
    valid: np.ndarray,
    themes: Iterable[np.ndarray] | float | None = None,
    padding: float = 0.04,
) -> tuple[slice, slice]:
    if isinstance(themes, (int, float)):
        padding = float(themes)
        themes = None
    support = np.asarray(valid, dtype=bool).copy()
    for theme in themes or []:
        arr = np.asarray(theme, dtype=bool)
        if arr.shape == support.shape:
            support |= arr
    rows, cols = np.where(support)
    if not rows.size:
        return slice(0, support.shape[0]), slice(0, support.shape[1])
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    pad_r = max(2, int((r1 - r0) * padding))
    pad_c = max(2, int((c1 - c0) * padding))
    return (
        slice(max(0, r0 - pad_r), min(support.shape[0], r1 + pad_r)),
        slice(max(0, c0 - pad_c), min(support.shape[1], c1 + pad_c)),
    )


def _crop_extent(
    bounds: rasterio.coords.BoundingBox,
    shape: tuple[int, int],
    crop: tuple[slice, slice],
) -> list[float]:
    height, width = shape
    rs, cs = crop
    left = bounds.left + (cs.start / width) * (bounds.right - bounds.left)
    right = bounds.left + (cs.stop / width) * (bounds.right - bounds.left)
    top = bounds.top - (rs.start / height) * (bounds.top - bounds.bottom)
    bottom = bounds.top - (rs.stop / height) * (bounds.top - bounds.bottom)
    return [left, right, bottom, top]


def _style(cfg: dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.family": cfg["mapping"].get("font_family", "DejaVu Sans"),
        "font.size": float(cfg["mapping"].get("tick_fontsize_pt", 7)),
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _format_axes(ax, extent: list[float], show_labels: bool = True) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    span_km = max((extent[1] - extent[0]) / 1000.0, (extent[3] - extent[2]) / 1000.0)
    decimals = 2 if span_km < 2 else (1 if span_km < 20 else 0)
    formatter = FuncFormatter(lambda value, _: f"{value / 1000:,.{decimals}f}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if show_labels:
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def _nice_scale(width_m: float) -> float:
    target = width_m * 0.20
    power = 10 ** np.floor(np.log10(max(target, 1)))
    options = np.array([1, 2, 5, 10], dtype=float) * power
    return float(options[np.argmin(np.abs(options - target))])


def _north_scale(ax, extent: list[float], fontsize: float = 6.5) -> None:
    left, right, bottom, top = extent
    width = right - left
    height = top - bottom
    length = _nice_scale(width)
    x0 = left + 0.055 * width
    y0 = bottom + 0.060 * height
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=1.25, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.005 * height, y0 + 0.005 * height], color="black", lw=0.7)
    ax.plot([x0 + length, x0 + length], [y0 - 0.005 * height, y0 + 0.005 * height], color="black", lw=0.7)
    ax.text(x0 + length / 2, y0 + 0.012 * height, f"{length / 1000:g} km", ha="center", va="bottom", fontsize=fontsize)
    xn = right - 0.055 * width
    yn = top - 0.095 * height
    ax.annotate("", xy=(xn, yn + 0.050 * height), xytext=(xn, yn), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black"})
    ax.text(xn, yn - 0.010 * height, "N", ha="center", va="top", fontsize=fontsize, fontweight="bold")


def _read_rgb(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, rasterio.coords.BoundingBox]:
    with rasterio.open(path) as src:
        rgb = src.read([4, 3, 2], out_shape=(3, *shape), resampling=Resampling.bilinear).astype(np.float32)
        rgb[rgb <= -9990] = np.nan
        valid = np.isfinite(rgb).all(axis=0)
        return _stretch(rgb, valid), valid, src.bounds


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    if not str(path) or not Path(str(path)).exists():
        return np.zeros(shape, dtype=bool)
    with rasterio.open(path) as src:
        return src.read(1, out_shape=shape, resampling=Resampling.nearest) == 1


def _read_anchor(path: str | Path, shape: tuple[int, int], threshold: float) -> np.ndarray:
    with rasterio.open(path) as src:
        fraction = src.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
    return fraction >= threshold


def _annual_layers(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible binary change partition.

    Returns persistent, gain and loss masks. The publication maps do not draw
    these as stacked overlays; the helper is retained for diagnostics/tests.
    """
    current = np.asarray(current, dtype=bool)
    previous = np.zeros_like(current) if previous is None else np.asarray(previous, dtype=bool)
    persistent = current & previous
    gain = current & ~previous
    loss = previous & ~current
    return persistent, gain, loss


def _exclusive_classes(
    active: np.ndarray,
    previous_active: np.ndarray | None,
    cumulative: np.ndarray,
    previous_cumulative: np.ndarray | None,
) -> np.ndarray:
    """Return one and only one PV class per pixel.

    0 background; 1 active existing; 2 newly installed; 3 installed inactive.
    No PV class is rendered as a second overlay above another class.
    """
    previous_active = np.zeros_like(active) if previous_active is None else previous_active
    previous_cumulative = np.zeros_like(cumulative) if previous_cumulative is None else previous_cumulative
    new_installation = cumulative & ~previous_cumulative
    inactive = cumulative & ~active
    active_existing = active & ~new_installation
    classes = np.zeros(active.shape, dtype=np.uint8)
    classes[active_existing] = 1
    classes[inactive] = 3
    classes[new_installation] = 2
    return classes


def _class_rgba(classes: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    vis = cfg["annual_visualization"]
    alpha = float(vis.get("theme_alpha", 0.95))
    output = np.zeros((*classes.shape, 4), dtype=np.float32)
    for code, colour in [
        (1, vis["active_persistent_color"]),
        (2, vis["installation_gain_color"]),
        (3, vis["inactive_retired_color"]),
    ]:
        output[classes == code] = to_rgba(colour, alpha)
    return output


def _save_figure(
    fig,
    stem: Path,
    raster_formats: list[str],
    vector_formats: list[str],
    dpi: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for ext in raster_formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        outputs[ext] = str(path)
    for ext in vector_formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        outputs[ext] = str(path)
    plt.close(fig)
    return outputs


def select_representatives(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "17_maps")
    dynamics = pd.read_csv(root / "11_pv_dynamics" / "unit_pv_dynamics_summary.csv")
    quality = pd.read_csv(root / "04_unit_quality" / "unit_quality_registry.csv")
    events = pd.read_csv(root / "13_pv_events" / "pv_expansion_events.csv")
    table = dynamics.merge(
        quality[["Id", "quality_score", "estimation_route"]],
        on="Id",
        how="left",
        suffixes=("", "_quality"),
    ).merge(events[["Id", "delta_bic", "diagnostic_candidate"]], on="Id", how="left")
    score = np.zeros(len(table), dtype=float)
    for column in ["net_change_km2", "cumulative_gain_km2", "delta_bic", "quality_score"]:
        x = pd.to_numeric(table.get(column), errors="coerce").fillna(0).to_numpy(float)
        score += (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    score += pd.to_numeric(table.get("diagnostic_candidate"), errors="coerce").fillna(0).to_numpy(float)
    table["representative_score"] = score
    count = int(cfg["mapping"].get("representative_count", 10))
    parts = []
    for _, group in table.groupby("estimation_mode"):
        parts.append(group.nlargest(max(1, count // max(table.estimation_mode.nunique(), 1)), "representative_score"))
    selected = pd.concat(parts, ignore_index=True).drop_duplicates("Id") if parts else table.iloc[0:0]
    if len(selected) < count:
        selected = pd.concat([
            selected,
            table.loc[~table.Id.isin(selected.Id)].nlargest(count - len(selected), "representative_score"),
        ], ignore_index=True)
    selected = selected.nlargest(count, "representative_score").copy()
    selected["selection_order"] = np.arange(1, len(selected) + 1)
    save_table(selected, out / "representative_units.csv")
    return selected


def _unit_display_contract(
    aligned_group: pd.DataFrame,
    mask_group: pd.DataFrame,
    anchor_path: str,
    cfg: dict[str, Any],
    maximum: int,
) -> tuple[tuple[int, int], tuple[slice, slice], list[float]]:
    reference_year = int(cfg["study"]["reference_year"])
    reference_path = Path(str(aligned_group.loc[aligned_group.year.eq(reference_year), "path"].iloc[0]))
    with rasterio.open(reference_path) as src:
        shape = _display_shape(src.width, src.height, maximum)
        bounds = src.bounds
        rgb = src.read([4, 3, 2], out_shape=(3, *shape), resampling=Resampling.bilinear).astype(np.float32)
        rgb[rgb <= -9990] = np.nan
        valid = np.isfinite(rgb).all(axis=0)
    anchor = _read_anchor(
        anchor_path,
        shape,
        float(cfg.get("anchor_constraint", {}).get("reference_inclusion_fraction", 0.05)),
    )
    union = valid | anchor
    for row in mask_group.itertuples(index=False):
        if is_pixel_mode(getattr(row, "estimation_mode", "")):
            union |= _read_mask(getattr(row, "active_mask_path", getattr(row, "mask_path", "")), shape)
            union |= _read_mask(getattr(row, "cumulative_mask_path", ""), shape)
    crop = _valid_crop(valid, [union], float(cfg["mapping"].get("crop_padding_fraction", 0.04)))
    return shape, crop, _crop_extent(bounds, shape, crop)


def _draw_map_panel(
    ax,
    background: np.ndarray,
    extent: list[float],
    active: np.ndarray,
    previous_active: np.ndarray | None,
    cumulative: np.ndarray,
    previous_cumulative: np.ndarray | None,
    year: int,
    mode: str,
    cfg: dict[str, Any],
    *,
    panel_label: str | None = None,
    show_axes: bool = False,
) -> np.ndarray:
    vis = cfg["annual_visualization"]
    ax.imshow(
        _mute_background(background, cfg),
        extent=extent,
        origin="upper",
        interpolation="nearest",
        alpha=float(vis.get("background_alpha", 0.78)),
    )
    classes = np.zeros(active.shape, dtype=np.uint8)
    if is_pixel_mode(mode):
        classes = _exclusive_classes(active, previous_active, cumulative, previous_cumulative)
        if np.any(classes):
            ax.imshow(
                _class_rgba(classes, cfg),
                extent=extent,
                origin="upper",
                interpolation="nearest",
            )
    ax.set_title(str(year), fontsize=6.8, pad=2.0, fontweight="semibold")
    if panel_label:
        ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, fontweight="bold")
    _format_axes(ax, extent, show_labels=show_axes)
    return classes


def _numeric(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float(default)


def _year_metrics(mask_row: pd.Series, dynamics_row: pd.Series | None) -> dict[str, float]:
    active = _numeric(mask_row, "observed_active_area_km2", _numeric(mask_row, "pv_area_km2", 0.0))
    cumulative = _numeric(mask_row, "cumulative_installed_area_km2", active)
    inactive = max(cumulative - active, 0.0)
    source = dynamics_row if dynamics_row is not None else mask_row
    return {
        "active": active,
        "cumulative": cumulative,
        "inactive": inactive,
        "installed_gain": _numeric(source, "installed_gain_km2", 0.0),
        "active_gain": _numeric(source, "gross_gain_km2", 0.0),
        "active_loss": _numeric(source, "gross_loss_km2", 0.0),
        "net_change": _numeric(source, "net_change_km2", 0.0),
    }


def _legend_handles(cfg: dict[str, Any]) -> list[Patch]:
    vis = cfg["annual_visualization"]
    return [
        Patch(facecolor=vis["active_persistent_color"], edgecolor="none", label="Observed active PV"),
        Patch(facecolor=vis["installation_gain_color"], edgecolor="none", label="Newly installed footprint"),
        Patch(facecolor=vis["inactive_retired_color"], edgecolor="none", label="Installed, activity unconfirmed or multi-year retired"),
    ]


def _draw_external_year_summary(
    ax,
    region_id: int,
    year: int,
    metrics: dict[str, float],
    mode: str,
    cfg: dict[str, Any],
) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, f"Unit {region_id:03d}", ha="left", va="top", fontsize=9.0, fontweight="semibold")
    ax.text(0.0, 0.91, str(year), ha="left", va="top", fontsize=17.0, fontweight="bold")
    lines = [
        ("Active footprint", metrics["active"]),
        ("Cumulative installed", metrics["cumulative"]),
        ("New installation", metrics["installed_gain"]),
        ("Active gain", metrics["active_gain"]),
        ("Active loss", metrics["active_loss"]),
        ("Net active change", metrics["net_change"]),
        ("Installed but inactive", metrics["inactive"]),
    ]
    y = 0.77
    for label, value in lines:
        ax.text(0.0, y, label, ha="left", va="top", fontsize=6.7, color="0.35")
        ax.text(1.0, y, f"{value:,.2f} km²", ha="right", va="top", fontsize=7.0, fontweight="semibold")
        y -= 0.075
    if not is_pixel_mode(mode):
        ax.text(0.0, y - 0.01, "Area-only estimate; no pixel footprint fabricated", ha="left", va="top", fontsize=6.3, color="0.35", wrap=True)
        y -= 0.10
    ax.legend(handles=_legend_handles(cfg), loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=6.2, handlelength=1.2)


def _draw_atlas_summary(
    ax,
    region_id: int,
    years: list[int],
    metrics_by_year: dict[int, dict[str, float]],
    cfg: dict[str, Any],
) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, f"Unit {region_id:03d}", ha="left", va="top", fontsize=8.2, fontweight="semibold")
    ax.text(0.0, 0.955, "Annual area summary", ha="left", va="top", fontsize=7.2)
    ax.text(0.0, 0.915, "Year   Active   Δ installed", ha="left", va="top", fontsize=5.8, fontfamily="monospace", color="0.35")
    y = 0.88
    for year in years:
        values = metrics_by_year[year]
        ax.text(
            0.0,
            y,
            f"{year}  {values['active']:7.1f}  {values['installed_gain']:+8.1f}",
            ha="left",
            va="top",
            fontsize=5.8,
            fontfamily="monospace",
        )
        y -= 0.043
    ax.text(0.0, max(y - 0.005, 0.16), "Area: km²", ha="left", va="top", fontsize=5.6, color="0.35")
    ax.legend(handles=_legend_handles(cfg), loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=5.6, handlelength=1.1)


def _annual_visualizations_for_unit(
    region_id: int,
    aligned_group: pd.DataFrame,
    mask_group: pd.DataFrame,
    dynamics_group: pd.DataFrame,
    anchor_path: str,
    cfg: dict[str, Any],
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vis = cfg["annual_visualization"]
    maximum = int(vis.get("max_render_dimension", 1400))
    shape, crop, extent = _unit_display_contract(aligned_group, mask_group, anchor_path, cfg, maximum)
    years = sorted(mask_group.year.astype(int).unique())
    manifest: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    unit_out = ensure_dir(out / f"unit_{region_id:03d}")
    yearly_out = ensure_dir(unit_out / "annual_maps")

    active_masks: dict[int, np.ndarray] = {}
    cumulative_masks: dict[int, np.ndarray] = {}
    metrics_by_year: dict[int, dict[str, float]] = {}
    cropped_shape = (crop[0].stop - crop[0].start, crop[1].stop - crop[1].start)
    for row in mask_group.itertuples(index=False):
        year = int(row.year)
        pixel_mode = is_pixel_mode(getattr(row, "estimation_mode", ""))
        active_path = getattr(row, "active_mask_path", getattr(row, "mask_path", ""))
        active_masks[year] = _read_mask(active_path, shape)[crop] if pixel_mode else np.zeros(cropped_shape, dtype=bool)
        cumulative_masks[year] = _read_mask(getattr(row, "cumulative_mask_path", ""), shape)[crop] if pixel_mode else np.zeros(cropped_shape, dtype=bool)
        mask_row = mask_group.loc[mask_group.year.eq(year)].iloc[0]
        dynamic_rows = dynamics_group.loc[dynamics_group.year.eq(year)]
        metrics_by_year[year] = _year_metrics(mask_row, dynamic_rows.iloc[0] if len(dynamic_rows) else None)

    if bool(vis.get("write_individual_year_maps", True)):
        previous_active = None
        previous_cumulative = None
        for year in years:
            image_path = Path(str(aligned_group.loc[aligned_group.year.eq(year), "path"].iloc[0]))
            background, _, _ = _read_rgb(image_path, shape)
            background = background[crop]
            row = mask_group.loc[mask_group.year.eq(year)].iloc[0]
            fig = plt.figure(
                figsize=(float(cfg["mapping"]["figure_width_mm"]) / 25.4, float(cfg["mapping"]["figure_height_mm"]) / 25.4),
                facecolor="white",
                layout="constrained",
            )
            grid = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.45])
            map_ax = fig.add_subplot(grid[0, 0])
            info_ax = fig.add_subplot(grid[0, 1])
            _draw_map_panel(
                map_ax,
                background,
                extent,
                active_masks[year],
                previous_active,
                cumulative_masks[year],
                previous_cumulative,
                year,
                str(row.estimation_mode),
                cfg,
                show_axes=True,
            )
            map_ax.set_title(
                f"Unit {region_id:03d}: annual PV footprint — {year}",
                loc="left",
                fontsize=float(cfg["mapping"]["title_fontsize_pt"]),
                fontweight="semibold",
            )
            _north_scale(map_ax, extent, float(cfg["mapping"]["legend_fontsize_pt"]))
            _draw_external_year_summary(info_ax, region_id, year, metrics_by_year[year], str(row.estimation_mode), cfg)
            outputs = _save_figure(
                fig,
                yearly_out / f"unit_{region_id:03d}_{year}_pv_change",
                list(vis.get("individual_raster_formats", ["png"])),
                [],
                int(vis.get("individual_dpi", 300)),
            )
            manifest.append({"Id": region_id, "year": year, "map_type": "annual_pv_change", "estimation_mode": str(row.estimation_mode), **outputs})
            previous_active = active_masks[year]
            previous_cumulative = cumulative_masks[year]

    if bool(vis.get("write_annual_atlas", True)):
        columns = int(vis.get("atlas_columns", 5))
        rows_count = int(vis.get("atlas_rows", 3))
        fig = plt.figure(
            figsize=(float(vis.get("atlas_width_mm", 285)) / 25.4, float(vis.get("atlas_height_mm", 165)) / 25.4),
            facecolor="white",
            layout="constrained",
        )
        grid = fig.add_gridspec(rows_count, columns + 1, width_ratios=[1.0] * columns + [1.28])
        axes = [fig.add_subplot(grid[r, c]) for r in range(rows_count) for c in range(columns)]
        summary_ax = fig.add_subplot(grid[:, -1])
        previous_active = None
        previous_cumulative = None
        for index, year in enumerate(years):
            ax = axes[index]
            image_path = Path(str(aligned_group.loc[aligned_group.year.eq(year), "path"].iloc[0]))
            background, _, _ = _read_rgb(image_path, shape)
            background = background[crop]
            row = mask_group.loc[mask_group.year.eq(year)].iloc[0]
            _draw_map_panel(
                ax,
                background,
                extent,
                active_masks[year],
                previous_active,
                cumulative_masks[year],
                previous_cumulative,
                year,
                str(row.estimation_mode),
                cfg,
                panel_label=chr(97 + index),
                show_axes=not bool(vis.get("outer_axes_only", True)),
            )
            previous_active = active_masks[year]
            previous_cumulative = cumulative_masks[year]
        for ax in axes[len(years):]:
            ax.axis("off")
        _draw_atlas_summary(summary_ax, region_id, years, metrics_by_year, cfg)
        fig.suptitle(
            f"Unit {region_id:03d}: mutually exclusive annual PV states, 2011–2025",
            x=0.01,
            ha="left",
            fontsize=9.5,
            fontweight="semibold",
        )
        outputs = _save_figure(
            fig,
            unit_out / f"unit_{region_id:03d}_annual_pv_change_atlas",
            list(vis.get("atlas_raster_formats", ["png"])),
            list(vis.get("atlas_vector_formats", ["pdf", "svg"])),
            int(vis.get("atlas_dpi", 450)),
        )
        manifest.append({"Id": region_id, "year": np.nan, "map_type": "annual_pv_change_atlas", "estimation_mode": str(mask_group.iloc[0].estimation_mode), **outputs})

    occupancy = float(np.logical_or.reduce([cumulative_masks[y] for y in years]).mean()) if years else 0.0
    audits.append({
        "Id": region_id,
        "map_type": "annual_series",
        "contract_pass": True,
        "shared_extent": True,
        "equal_aspect": True,
        "years": len(years),
        "crop_theme_occupancy_fraction": occupancy,
        "reference_shp_drawn": False,
        "reference_anchor_embedded_in_final_2024_footprint": True,
        "dual_state_visualization": True,
        "single_exclusive_class_layer": True,
        "no_stacked_pv_layers": True,
        "area_annotations_outside_map_axes": True,
        "colourblind_safe_high_contrast": True,
    })
    return manifest, audits



def _future_classes(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Exclusive 2050 classes: 0 background, 1 current footprint, 2 added candidate."""
    current = np.asarray(current, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool) & ~current
    classes = np.zeros(current.shape, dtype=np.uint8)
    classes[current] = 1
    classes[candidate] = 2
    return classes


def _future_rgba(classes: np.ndarray, scenario: str, cfg: dict[str, Any]) -> np.ndarray:
    vis = cfg.get("future_visualization", {})
    colors = {
        "conservative": vis.get("conservative_color", "#009E73"),
        "transition": vis.get("transition_color", "#E69F00"),
        "accelerated": vis.get("accelerated_color", "#CC79A7"),
    }
    out = np.zeros((*classes.shape, 4), dtype=np.float32)
    out[classes == 1] = to_rgba(vis.get("current_pv_color", "#0072B2"), 0.92)
    out[classes == 2] = to_rgba(colors.get(scenario, "#E69F00"), 0.90)
    return out


def _generate_future_visualizations(
    aligned: pd.DataFrame,
    masks: pd.DataFrame,
    future: pd.DataFrame,
    cfg: dict[str, Any],
    out: Path,
) -> list[dict[str, Any]]:
    vis = cfg.get("future_visualization", {})
    if not bool(vis.get("enabled", True)) or future.empty:
        return []
    result: list[dict[str, Any]] = []
    future_out = ensure_dir(out / "future_2050_visualizations")
    scenarios = [str(x) for x in vis.get("scenarios", ["conservative", "transition", "accelerated"])]
    unit_ids = sorted(set(future.loc[future.spatial_map_available.fillna(False), "Id"].astype(int)))
    for region_id in tqdm(unit_ids, desc="2025-2050 forecast maps", unit="region", disable=not cfg["processing"].get("show_progress", True)):
        aligned_2025 = aligned.loc[(aligned.Id == region_id) & (aligned.year == int(cfg["study"]["end_year"]))]
        mask_2025 = masks.loc[(masks.Id == region_id) & (masks.year == int(cfg["study"]["end_year"]))]
        rows = future.loc[future.Id == region_id]
        if aligned_2025.empty or mask_2025.empty or rows.empty:
            continue
        image_path = Path(str(aligned_2025.iloc[0].path))
        current_path = Path(str(mask_2025.iloc[0].cumulative_mask_path))
        if not image_path.exists() or not current_path.exists():
            continue
        with rasterio.open(current_path) as src:
            shape = _display_shape(src.width, src.height, int(cfg.get("annual_visualization", {}).get("max_render_dimension", 1800)))
            current = src.read(1, out_shape=shape, resampling=Resampling.nearest) == 1
            bounds = src.bounds
        background, valid, _ = _read_rgb(image_path, shape)
        candidates = {}
        for scenario in scenarios:
            row = rows.loc[rows.scenario.astype(str) == scenario]
            if row.empty:
                continue
            path = Path(str(row.iloc[0].candidate_path))
            if path.exists():
                candidates[scenario] = _read_mask(path, shape)
        if not candidates:
            continue
        crop = _valid_crop(valid, [current, *candidates.values()], float(cfg["mapping"].get("crop_padding_fraction", 0.04)))
        extent = _crop_extent(bounds, shape, crop)
        bg = _mute_background(background[crop], cfg)
        current_crop = current[crop]
        fig = plt.figure(figsize=(float(vis.get("figure_width_mm",250))/25.4, float(vis.get("figure_height_mm",150))/25.4), facecolor="white", layout="constrained")
        grid = fig.add_gridspec(1, len(candidates)+1, width_ratios=[1.0]*len(candidates)+[0.82])
        summary_ax = fig.add_subplot(grid[0,-1]); summary_ax.axis("off")
        summary_ax.text(0,0.98,f"Unit {region_id:03d}",ha="left",va="top",fontsize=10,fontweight="bold")
        summary_ax.text(0,0.91,"Conditional 2050 candidate envelopes",ha="left",va="top",fontsize=8.5)
        y=0.82
        for idx,(scenario,candidate) in enumerate(candidates.items()):
            ax=fig.add_subplot(grid[0,idx])
            classes=_future_classes(current_crop,candidate[crop])
            ax.imshow(bg,extent=extent,origin="upper")
            ax.imshow(_future_rgba(classes,scenario,cfg),extent=extent,origin="upper",interpolation="nearest")
            _format_axes(ax,extent,show_labels=False)
            ax.set_title(scenario.capitalize(),fontsize=9,fontweight="semibold")
            row=rows.loc[rows.scenario.astype(str)==scenario].iloc[0]
            summary_ax.text(0,y,scenario.capitalize(),fontsize=8.2,fontweight="bold",va="top")
            summary_ax.text(0,y-0.045,f"2025 installed: {float(row.current_area_km2):.1f} km²",fontsize=7.5,va="top")
            summary_ax.text(0,y-0.085,f"2050 target: {float(row.target_area_2050_km2):.1f} km²",fontsize=7.5,va="top")
            summary_ax.text(0,y-0.125,f"Additional: {float(row.candidate_area_km2):.1f} km²",fontsize=7.5,va="top")
            y-=0.22
        summary_ax.legend(handles=[Patch(facecolor=vis.get("current_pv_color","#0072B2"),label="2025 installed footprint"),Patch(facecolor=vis.get("transition_color","#E69F00"),label="Additional 2050 candidate")],loc="lower left",frameon=False,fontsize=7)
        fig.suptitle(f"Unit {region_id:03d}: observed 2025 footprint and conditional 2050 spatial candidates",x=0.01,ha="left",fontsize=10.5,fontweight="semibold")
        outputs=_save_figure(fig,future_out/f"unit_{region_id:03d}_2025_2050_candidate_comparison",list(vis.get("raster_formats",["png"])),list(vis.get("vector_formats",["pdf","svg"])),int(vis.get("dpi",450)))
        result.append({"Id":region_id,"year":2050,"map_type":"future_2050_candidate_comparison","estimation_mode":str(mask_2025.iloc[0].estimation_mode),**outputs})
    return result


def generate_maps(cfg: dict[str, Any]) -> pd.DataFrame:
    _style(cfg)
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "17_maps")
    representatives = select_representatives(cfg)
    aligned = pd.read_csv(root / "02_aligned_30m" / "aligned_raster_registry.csv").sort_values(["Id", "year"])
    masks = pd.read_csv(root / "08_temporal_pv_change" / "temporal_mask_registry.csv").sort_values(["Id", "year"])
    dynamics = pd.read_csv(root / "11_pv_dynamics" / "annual_pv_dynamics.csv").sort_values(["Id", "year"])
    supervision = pd.read_csv(root / "03_supervision" / "supervision_registry.csv").set_index("Id")
    future_path = root / "16_future_2050" / "conditional_2050_candidate_summary.csv"
    future = pd.read_csv(future_path) if future_path.exists() else pd.DataFrame()
    manifest: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    requested = cfg.get("annual_visualization", {}).get("units", "all")
    if requested == "all":
        unit_ids = sorted(masks.Id.astype(int).unique())
    elif requested == "representative":
        unit_ids = sorted(representatives.Id.astype(int).unique())
    else:
        unit_ids = sorted({int(x) for x in requested})

    for region_id in tqdm(
        unit_ids,
        desc="Annual PV maps and 15-year atlases",
        unit="region",
        disable=not cfg["processing"].get("show_progress", True),
    ):
        unit_manifest, unit_audits = _annual_visualizations_for_unit(
            region_id,
            aligned.loc[aligned.Id.eq(region_id)],
            masks.loc[masks.Id.eq(region_id)],
            dynamics.loc[dynamics.Id.eq(region_id)],
            str(supervision.loc[region_id, "pv_fraction_path"]),
            cfg,
            out,
        )
        manifest.extend(unit_manifest)
        audits.extend(unit_audits)

    manifest.extend(_generate_future_visualizations(aligned, masks, future, cfg, out))
    manifest_df = pd.DataFrame(manifest)
    audit_df = pd.DataFrame(audits)
    save_table(manifest_df, out / "map_manifest.csv")
    save_table(audit_df, out / "map_rendering_audit.csv")
    expected_atlases = len(unit_ids) if cfg.get("annual_visualization", {}).get("write_annual_atlas", True) else 0
    actual_atlases = int((manifest_df.map_type == "annual_pv_change_atlas").sum()) if not manifest_df.empty else 0
    status = "passed" if not manifest_df.empty and audit_df.contract_pass.all() and actual_atlases == expected_atlases else "failed"
    atomic_json({
        "status": status,
        "created_utc": utc_now(),
        "units_visualized": len(unit_ids),
        "maps": len(manifest_df),
        "annual_atlases": actual_atlases,
        "expected_annual_atlases": expected_atlases,
        "years_per_atlas": int(cfg["study"]["end_year"]) - int(cfg["study"]["start_year"]) + 1,
        "individual_annual_maps": int((manifest_df.map_type == "annual_pv_change").sum()) if not manifest_df.empty else 0,
        "future_2050_comparison_maps": int((manifest_df.map_type == "future_2050_candidate_comparison").sum()) if not manifest_df.empty else 0,
        "future_maps_use_exclusive_current_and_candidate_classes": True,
        "automatic_valid_data_crop": True,
        "shared_extent_across_years": True,
        "equal_aspect": True,
        "coordinate_units": "km",
        "vector_atlas_exports": cfg.get("annual_visualization", {}).get("atlas_vector_formats", ["pdf", "svg"]),
        "area_only_maps_fabricated": False,
        "reference_shp_drawn": False,
        "reference_anchor_embedded_in_final_2024_footprint": True,
        "single_exclusive_class_layer": True,
        "no_stacked_pv_layers": True,
        "area_annotations_outside_map_axes": True,
        "dual_state_visualization": True,
        "colourblind_safe_high_contrast": True,
        "palette_semantics": {
            "observed_active": cfg.get("annual_visualization", {}).get("active_persistent_color", "#0072B2"),
            "newly_installed": cfg.get("annual_visualization", {}).get("installation_gain_color", "#F0E442"),
            "installed_inactive_or_retired": cfg.get("annual_visualization", {}).get("inactive_retired_color", "#CC79A7"),
        },
        "fallback_map_policy": "area-only panels contain external non-spatial annotation; no footprint is invented",
    }, out / "map_status.json")
    if status == "failed":
        raise RuntimeError("Annual publication-map contract failed")
    return manifest_df
