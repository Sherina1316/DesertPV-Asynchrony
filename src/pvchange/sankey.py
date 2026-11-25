from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from .utils import atomic_json, ensure_dir, save_table, utc_now

PALETTE = {
    "blue": "#2B6CB0",
    "cyan": "#2C7FB8",
    "green": "#2F855A",
    "olive": "#6B8E23",
    "yellow": "#D69E2E",
    "orange": "#DD6B20",
    "red": "#C53030",
    "purple": "#6B46C1",
    "gray": "#718096",
    "light": "#EDF2F7",
    "dark": "#1A202C",
}

LAYER_COLORS = [
    "#2B6CB0", "#2F855A", "#D69E2E", "#DD6B20", "#6B46C1", "#718096"
]


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _quantile_class(series: pd.Series, labels: list[str]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series([labels[0]] * len(x), index=series.index, dtype=object)
    finite = x[np.isfinite(x)]
    if finite.empty:
        return pd.Series(["Not estimable"] * len(x), index=series.index, dtype=object)
    # Duplicate quantiles are common when many units have zero expansion; rank is safer.
    ranks = x.rank(method="average", pct=True)
    bins = np.linspace(0, 1, len(labels) + 1)
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = ranks <= hi
        elif i == len(labels) - 1:
            mask = ranks > lo
        else:
            mask = (ranks > lo) & (ranks <= hi)
        out.loc[mask.fillna(False)] = label
    out.loc[~np.isfinite(x)] = "Not estimable"
    return out


def _lag_class(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    result = pd.Series("No clear response", index=values.index, dtype=object)
    result.loc[x <= 1] = "Immediate (0-1 yr)"
    result.loc[(x > 1) & (x <= 3)] = "Short lag (2-3 yr)"
    result.loc[(x > 3) & (x <= 5)] = "Medium lag (4-5 yr)"
    result.loc[x > 5] = "Long lag (>5 yr)"
    return result


def _response_class(direction: pd.Series, amplitude: pd.Series | None = None) -> pd.Series:
    d = direction.astype(str).str.lower() if len(direction) else pd.Series(dtype=object)
    result = pd.Series("Weak / mixed response", index=direction.index, dtype=object)
    result.loc[d.isin(["increase", "greening", "positive"])] = "Greening response"
    result.loc[d.isin(["decline", "decrease", "browning", "negative"])] = "Browning response"
    result.loc[d.isin(["none", "insufficient", "nan", "missing"])] = "No clear response"
    if amplitude is not None and len(amplitude):
        amp = pd.to_numeric(amplitude, errors="coerce").abs()
        result.loc[amp.fillna(0) < 1e-6] = "Weak / mixed response"
    return result


def _future_pressure(future: pd.DataFrame) -> pd.DataFrame:
    if future.empty or "Id" not in future:
        return pd.DataFrame(columns=["Id", "future_additional_area_2050_km2"])
    area_col = "additional_area_2050_km2" if "additional_area_2050_km2" in future else None
    if area_col is None:
        numeric_cols = [c for c in future.columns if c.endswith("_km2")]
        area_col = numeric_cols[0] if numeric_cols else None
    if area_col is None:
        return pd.DataFrame({"Id": future["Id"].astype(int).drop_duplicates(), "future_additional_area_2050_km2": np.nan})
    g = future.copy()
    if "scenario" in g:
        transition = g.loc[g["scenario"].astype(str).str.lower().eq("transition")]
        if not transition.empty:
            g = transition
    return g.groupby("Id", as_index=False)[area_col].sum().rename(columns={area_col: "future_additional_area_2050_km2"})


def build_ecological_response_sankey_tables(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create node/link tables for the expansion--ecology--future Sankey.

    The table is deliberately unit-level so the figure remains interpretable:
    each desert unit contributes its 2024 cumulative installed area where
    available, otherwise one equal-weight unit.  The resulting flow shows how
    PV expansion intensity maps into ecological response classes, response
    lags and 2050 pressure categories.
    """
    root = Path(cfg["paths"]["output_root"])
    fig_dir = root / "18_figures"
    speed = _read_csv(fig_dir / "expansion_ecological_speed_by_unit.csv")
    response = _read_csv(fig_dir / "model_consistent_response_timing.csv")
    future = _read_csv(root / "16_future_2050" / "conditional_2050_candidate_summary.csv")
    future_unit = _future_pressure(future)

    if speed.empty:
        speed = pd.DataFrame({"Id": [0], "pv_expansion_speed_normalized": [np.nan], "reference_cumulative_installed_area_km2": [1.0]})
    base = speed.copy()
    if "Id" not in base:
        base["Id"] = np.arange(len(base))
    base["Id"] = pd.to_numeric(base["Id"], errors="coerce").fillna(-1).astype(int)
    if "reference_cumulative_installed_area_km2" not in base:
        base["reference_cumulative_installed_area_km2"] = 1.0
    if "pv_expansion_speed_normalized" not in base:
        base["pv_expansion_speed_normalized"] = np.nan

    cols = ["Id"]
    for c in ["response_direction", "response_lag_years", "response_amplitude", "stabilization_observed"]:
        if c in response:
            cols.append(c)
    if not response.empty and "Id" in response:
        base = base.merge(response[cols].drop_duplicates("Id"), on="Id", how="left")
    else:
        base["response_direction"] = "insufficient"
        base["response_lag_years"] = np.nan
        base["response_amplitude"] = np.nan
        base["stabilization_observed"] = False
    base = base.merge(future_unit, on="Id", how="left")

    base["Expansion intensity"] = _quantile_class(base["pv_expansion_speed_normalized"], ["Low expansion", "Moderate expansion", "High expansion", "Very high expansion"])
    base["Ecological response"] = _response_class(base.get("response_direction", pd.Series("insufficient", index=base.index)), base.get("response_amplitude"))
    base["Response lag"] = _lag_class(base.get("response_lag_years", pd.Series(np.nan, index=base.index)))
    base["2050 pressure"] = _quantile_class(base.get("future_additional_area_2050_km2", pd.Series(np.nan, index=base.index)), ["Low future pressure", "Moderate future pressure", "High future pressure", "Very high future pressure"])
    weight = pd.to_numeric(base["reference_cumulative_installed_area_km2"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    base["flow_weight"] = weight.fillna(1.0).clip(lower=0.01)
    base["sankey_type"] = "ecological_response"

    layers = ["Expansion intensity", "Ecological response", "Response lag", "2050 pressure"]
    return _tables_from_classification(base, layers, "flow_weight")


def build_multimetric_sankey_tables(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a second Sankey focused on individual ecological indicators."""
    root = Path(cfg["paths"]["output_root"])
    fig_dir = root / "18_figures"
    speed = _read_csv(fig_dir / "expansion_ecological_speed_by_unit.csv")
    anomaly = _read_csv(root / "12_dynamic_ecology" / "external_ecology_anomaly_panel.csv")
    response = _read_csv(fig_dir / "model_consistent_response_timing.csv")

    if speed.empty:
        speed = pd.DataFrame({"Id": [0], "reference_cumulative_installed_area_km2": [1.0], "pv_expansion_speed_normalized": [np.nan]})
    base = speed[[c for c in ["Id", "reference_cumulative_installed_area_km2", "pv_expansion_speed_normalized"] if c in speed]].copy()
    if "Id" not in base:
        base["Id"] = np.arange(len(base))
    base["Id"] = pd.to_numeric(base["Id"], errors="coerce").fillna(-1).astype(int)
    base["PV stage"] = _quantile_class(base.get("pv_expansion_speed_normalized", pd.Series(np.nan, index=base.index)), ["Slow buildout", "Moderate buildout", "Fast buildout", "Very fast buildout"])

    # Metric-specific classes are computed from median post-event anomalies when available.
    metric_classes: dict[str, pd.Series] = {}
    for metric, output_name, pos, neg in [
        ("NDVI", "Greenness class", "Greening", "Browning"),
        ("TC_Greenness", "Texture greenness", "TC greenness increase", "TC greenness decline"),
        ("NDMI", "Moisture class", "Moistening", "Drying"),
        ("TC_Wetness", "Wetness class", "Wetness increase", "Wetness decline"),
    ]:
        if not anomaly.empty and {"Id", "metric", "external_anomaly"}.issubset(anomaly.columns):
            g = anomaly.loc[anomaly["metric"].astype(str).eq(metric)].groupby("Id", as_index=False)["external_anomaly"].median()
            tmp = base[["Id"]].merge(g, on="Id", how="left")
            val = pd.to_numeric(tmp["external_anomaly"], errors="coerce")
            cls = pd.Series("Weak / mixed", index=base.index, dtype=object)
            cls.loc[val > 0.05] = pos
            cls.loc[val < -0.05] = neg
            cls.loc[~np.isfinite(val)] = "Not estimable"
        else:
            cls = pd.Series("Not estimable", index=base.index, dtype=object)
        metric_classes[output_name] = cls

    for name, cls in metric_classes.items():
        base[name] = cls
    if not response.empty and {"Id", "stabilization_observed"}.issubset(response.columns):
        tmp = base[["Id"]].merge(response[["Id", "stabilization_observed"]].drop_duplicates("Id"), on="Id", how="left")
        stable = tmp["stabilization_observed"].fillna(False).astype(bool)
        base["Integrated outcome"] = np.where(stable, "Stabilized response", "Right-censored or unstable")
    else:
        base["Integrated outcome"] = "Not estimable"
    base["flow_weight"] = pd.to_numeric(base.get("reference_cumulative_installed_area_km2", pd.Series(1.0, index=base.index)), errors="coerce").fillna(1.0).clip(lower=0.01)
    base["sankey_type"] = "multimetric_ecology"
    layers = ["PV stage", "Greenness class", "Moisture class", "Wetness class", "Integrated outcome"]
    return _tables_from_classification(base, layers, "flow_weight")


def _tables_from_classification(base: pd.DataFrame, layers: list[str], weight_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for layer_index, layer in enumerate(layers):
        values = base[layer].fillna("Not estimable").astype(str)
        for label, g in base.assign(_label=values).groupby("_label", dropna=False):
            rows.append({"node_id": f"L{layer_index}:{label}", "layer_index": layer_index, "layer": layer, "label": label, "value": float(pd.to_numeric(g[weight_col], errors="coerce").fillna(0).sum())})
    nodes = pd.DataFrame(rows)
    link_rows = []
    for layer_index in range(len(layers) - 1):
        src_col, tgt_col = layers[layer_index], layers[layer_index + 1]
        g = base.groupby([src_col, tgt_col], dropna=False)[weight_col].sum().reset_index()
        for row in g.itertuples(index=False):
            src = str(getattr(row, src_col.replace(" ", "_"), row[0])) if False else str(row[0])
            tgt = str(row[1])
            value = float(row[2])
            if value <= 0:
                continue
            link_rows.append({
                "source": f"L{layer_index}:{src}",
                "target": f"L{layer_index + 1}:{tgt}",
                "source_layer": src_col,
                "target_layer": tgt_col,
                "source_label": src,
                "target_label": tgt,
                "value": value,
            })
    links = pd.DataFrame(link_rows)
    return nodes, links, base


@dataclass
class NodeGeom:
    x: float
    y0: float
    y1: float
    color: str
    value: float
    layer: str
    label: str


def _layout_nodes(nodes: pd.DataFrame, layers: list[str]) -> dict[str, NodeGeom]:
    geoms: dict[str, NodeGeom] = {}
    x_positions = np.linspace(0.08, 0.92, len(layers))
    for layer_index, layer in enumerate(layers):
        sub = nodes.loc[nodes["layer"].eq(layer)].copy().sort_values("value", ascending=False)
        total = float(sub["value"].sum())
        if total <= 0:
            total = 1.0
        gap = 0.018
        usable = max(0.70, 0.90 - gap * max(len(sub) - 1, 0))
        y_top = 0.92
        for j, row in enumerate(sub.itertuples(index=False)):
            h = max(0.018, float(row.value) / total * usable)
            y1 = y_top
            y0 = max(0.04, y1 - h)
            color = LAYER_COLORS[layer_index % len(LAYER_COLORS)]
            geoms[str(row.node_id)] = NodeGeom(x=float(x_positions[layer_index]), y0=float(y0), y1=float(y1), color=color, value=float(row.value), layer=layer, label=str(row.label))
            y_top = y0 - gap
    return geoms


def _draw_sankey(nodes: pd.DataFrame, links: pd.DataFrame, layers: list[str], title: str, output_stem: Path, cfg: dict[str, Any]) -> None:
    dpi = int(cfg.get("mapping", {}).get("map_dpi", 600))
    fig = plt.figure(figsize=(13.2, 7.6), dpi=160, facecolor="white")
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.99, title, ha="center", va="top", fontsize=16, fontweight="bold", color=PALETTE["dark"])
    geoms = _layout_nodes(nodes, layers)
    max_value = float(links["value"].max()) if len(links) else 1.0
    # Draw links first.
    offsets_out = {node: geom.y1 for node, geom in geoms.items()}
    offsets_in = {node: geom.y1 for node, geom in geoms.items()}
    for row in links.sort_values("value", ascending=False).itertuples(index=False):
        if row.source not in geoms or row.target not in geoms:
            continue
        src = geoms[row.source]
        tgt = geoms[row.target]
        v = float(row.value)
        h_src = max(0.004, (src.y1 - src.y0) * v / max(src.value, 1e-9))
        h_tgt = max(0.004, (tgt.y1 - tgt.y0) * v / max(tgt.value, 1e-9))
        y0 = offsets_out[row.source] - h_src / 2
        y1 = offsets_in[row.target] - h_tgt / 2
        offsets_out[row.source] -= h_src
        offsets_in[row.target] -= h_tgt
        x0 = src.x + 0.018
        x1 = tgt.x - 0.018
        xm0 = x0 + (x1 - x0) * 0.45
        xm1 = x0 + (x1 - x0) * 0.55
        path = MplPath([(x0, y0), (xm0, y0), (xm1, y1), (x1, y1)], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        lw = 0.8 + 12.0 * np.sqrt(v / max(max_value, 1e-9))
        patch = PathPatch(path, facecolor="none", edgecolor=src.color, lw=lw, alpha=0.22, capstyle="round", joinstyle="round")
        ax.add_patch(patch)
    # Draw nodes and labels.
    for geom in geoms.values():
        ax.add_patch(Rectangle((geom.x - 0.016, geom.y0), 0.032, geom.y1 - geom.y0, facecolor=geom.color, edgecolor="white", lw=0.8, alpha=0.95))
        label = f"{geom.label}\n{geom.value:.1f}"
        ha = "right" if geom.x > 0.72 else "left"
        tx = geom.x - 0.024 if ha == "right" else geom.x + 0.024
        ax.text(tx, (geom.y0 + geom.y1) / 2, label, ha=ha, va="center", fontsize=8.2, color=PALETTE["dark"], linespacing=1.15)
    for i, layer in enumerate(layers):
        x = np.linspace(0.08, 0.92, len(layers))[i]
        ax.text(x, 0.947, layer, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#333333")
    for ext in cfg.get("mapping", {}).get("raster_formats", ["png"]):
        fig.savefig(output_stem.with_suffix(f".{ext}"), dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    for ext in cfg.get("mapping", {}).get("vector_formats", ["pdf", "svg"]):
        fig.savefig(output_stem.with_suffix(f".{ext}"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def generate_ecology_sankey_figures(cfg: dict[str, Any]) -> pd.DataFrame:
    """Generate both Sankey figures and all supporting CSV tables."""
    root = Path(cfg["paths"]["output_root"])
    out = ensure_dir(root / "18_figures")
    manifests: list[dict[str, Any]] = []

    nodes, links, units = build_ecological_response_sankey_tables(cfg)
    save_table(nodes, out / "ecology_response_sankey_nodes.csv")
    save_table(links, out / "ecology_response_sankey_links.csv")
    save_table(units, out / "ecology_response_sankey_unit_classification.csv")
    _draw_sankey(nodes, links, ["Expansion intensity", "Ecological response", "Response lag", "2050 pressure"], "PV expansion, ecological response and 2050 pressure", out / "figure_10_ecological_response_sankey", cfg)
    manifests.append({"figure": 10, "name": "ecological_response_sankey", "png": str(out / "figure_10_ecological_response_sankey.png"), "pdf": str(out / "figure_10_ecological_response_sankey.pdf"), "svg": str(out / "figure_10_ecological_response_sankey.svg")})

    nodes2, links2, units2 = build_multimetric_sankey_tables(cfg)
    save_table(nodes2, out / "multimetric_ecology_sankey_nodes.csv")
    save_table(links2, out / "multimetric_ecology_sankey_links.csv")
    save_table(units2, out / "multimetric_ecology_sankey_unit_classification.csv")
    _draw_sankey(nodes2, links2, ["PV stage", "Greenness class", "Moisture class", "Wetness class", "Integrated outcome"], "Multi-indicator ecological pathways after PV installation", out / "figure_11_multimetric_ecology_sankey", cfg)
    manifests.append({"figure": 11, "name": "multimetric_ecology_sankey", "png": str(out / "figure_11_multimetric_ecology_sankey.png"), "pdf": str(out / "figure_11_multimetric_ecology_sankey.pdf"), "svg": str(out / "figure_11_multimetric_ecology_sankey.svg")})

    manifest = pd.DataFrame(manifests)
    atomic_json({"status": "passed", "created_utc": utc_now(), "figures": len(manifest), "ecology_response_nodes": len(nodes), "ecology_response_links": len(links), "multimetric_nodes": len(nodes2), "multimetric_links": len(links2)}, out / "sankey_status.json")
    return manifest
