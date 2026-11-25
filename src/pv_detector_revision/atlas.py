from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio

from .raster_io import existing_path, read_bool, read_float, require_same_grid


def _stretch(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(channel.shape, dtype=np.float32)
    values = channel[valid & np.isfinite(channel)]
    if values.size == 0:
        return result
    low, high = np.nanquantile(values, [0.02, 0.98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    return np.clip((channel - float(low)) / float(high - low), 0.0, 1.0).astype(np.float32)


def _read_rgb(path: Path, template_profile: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        require_same_grid(template_profile, profile, f"atlas background {path}")
        nodata = src.nodata
        if src.count >= 5:
            bands = src.read([5, 4, 3]).astype(np.float32)
        elif src.count >= 4:
            bands = src.read([4, 3, 2]).astype(np.float32)
        elif src.count >= 3:
            bands = src.read([1, 2, 3]).astype(np.float32)
        else:
            one = src.read(1).astype(np.float32)
            bands = np.stack([one, one, one])
    valid = np.isfinite(bands).all(axis=0)
    if nodata is not None and np.isfinite(nodata):
        valid &= np.all(bands != nodata, axis=0)
    rgb = np.stack([_stretch(bands[i], valid) for i in range(3)], axis=-1)
    rgb[~valid] = 0.92
    return rgb, valid


def _discover_background(
    output_root: Path,
    unit_id: int,
    year: int,
    template_profile: Mapping[str, object],
    score_path: Path,
) -> Path:
    keywords = ("annual", "composite", "aligned", "landsat", "reflectance", "stack", "image")
    candidates: list[tuple[int, Path]] = []
    for csv_path in sorted(output_root.glob("0[1-7]_*/**/*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if not {"Id", "year"}.issubset(frame.columns):
            continue
        ids = pd.to_numeric(frame["Id"], errors="coerce")
        years = pd.to_numeric(frame["year"], errors="coerce")
        rows = frame.loc[ids.eq(unit_id) & years.eq(year)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for column in frame.columns:
            name = str(column).lower()
            if "path" not in name or not any(keyword in name for keyword in keywords):
                continue
            value = row.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            path = Path(str(value))
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            try:
                with rasterio.open(path) as src:
                    profile = src.profile.copy()
                    if src.count < 3:
                        continue
                require_same_grid(template_profile, profile, f"background candidate {path}")
            except Exception:
                continue
            score = 0
            if "aligned" in name:
                score += 4
            if "composite" in name:
                score += 3
            if "landsat" in name:
                score += 2
            if "annual" in name:
                score += 1
            candidates.append((score, path))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], str(item[1])))
        return candidates[0][1]
    return score_path


def generate_atlas(output_root: str | Path, unit_id: int, config: Mapping[str, Any]) -> Path:
    root = Path(output_root).resolve()
    years = [int(value) for value in config["study"]["years"]]
    audit_root = root / str(config["outputs"]["audit_subdir"])
    registry_path = audit_root / "temporal_mask_registry_detector_revision_v111.csv"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Revised registry is missing: {registry_path}")
    registry = pd.read_csv(registry_path)
    registry["Id"] = pd.to_numeric(registry["Id"], errors="coerce")
    registry["year"] = pd.to_numeric(registry["year"], errors="coerce")
    rows = registry.loc[registry["Id"].eq(unit_id) & registry["year"].isin(years)].sort_values("year")
    if len(rows) != len(years):
        raise RuntimeError(f"Unit {unit_id}: revised registry does not contain all annual rows")
    rows = rows.set_index("year", drop=False)

    first_row = rows.loc[years[0]]
    score_path = Path(str(first_row["detector_revision_score_path"]))
    _, template_profile, _ = read_float(score_path)

    fig = plt.figure(figsize=(18, 11.3), constrained_layout=False)
    grid = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.78], hspace=0.16, wspace=0.08)
    cmap = ListedColormap([
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.42, 0.75, 0.82),
        (1.0, 0.72, 0.0, 0.92),
        (0.73, 0.22, 0.52, 0.80),
    ])
    summary: list[tuple[int, float, float]] = []

    for index, year in enumerate(years):
        row = rows.loc[year]
        state_path = Path(str(row["annual_state_mask_path"]))
        score_path = Path(str(row["detector_revision_score_path"]))
        background_path = _discover_background(root, unit_id, year, template_profile, score_path)
        if background_path == score_path:
            score, profile, valid = read_float(score_path)
            require_same_grid(template_profile, profile, f"score background {score_path}")
            stretched = _stretch(np.nan_to_num(score, nan=0.0), valid)
            rgb = np.stack([stretched, stretched, stretched], axis=-1)
        else:
            rgb, valid = _read_rgb(background_path, template_profile)

        state, profile, _ = read_float(state_path)
        require_same_grid(template_profile, profile, f"state mask {state_path}")
        state = np.nan_to_num(state, nan=0.0).astype(np.uint8)
        ax = fig.add_subplot(grid[index // 5, index % 5])
        ax.imshow(rgb, interpolation="nearest")
        masked = np.ma.masked_where((state == 0) | ~valid, state)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(str(year), fontsize=10, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        summary.append(
            (
                year,
                float(pd.to_numeric(pd.Series([row["observed_active_area_km2"]]), errors="coerce").iloc[0]),
                float(pd.to_numeric(pd.Series([row["newly_installed_area_km2"]]), errors="coerce").iloc[0]),
            )
        )

    side = fig.add_subplot(grid[:, 5])
    side.axis("off")
    side.text(0.0, 0.99, f"Unit {unit_id:03d}", fontsize=12, fontweight="bold", va="top")
    side.text(0.0, 0.95, "Detector revision v1.1.1", fontsize=9.5, va="top")
    side.text(0.0, 0.90, "Year   Active km²   New km²", fontsize=8, family="monospace", va="top")
    y = 0.86
    for year, active_area, new_area in summary:
        side.text(0.0, y, f"{year}   {active_area:8.3f}   {new_area:7.3f}", fontsize=7.4, family="monospace", va="top")
        y -= 0.043
    side.text(0.0, 0.10, "Blue: active PV", fontsize=8)
    side.text(0.0, 0.075, "Orange: newly installed", fontsize=8)
    side.text(0.0, 0.05, "Purple: installed, inactive/unconfirmed", fontsize=8)
    side.text(0.0, 0.015, "All rasters retain the original v24.22 grid.", fontsize=7.2)
    fig.suptitle(
        f"Unit {unit_id:03d}: annual PV states using one detector for 2011–2025",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path = root / str(config["outputs"]["figure_subdir"]) / f"unit_{unit_id:03d}_annual_pv_change_atlas_v111.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
