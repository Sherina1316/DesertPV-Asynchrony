from __future__ import annotations

from pvchange.study_area import study_ids, expected_counts, expected_id_set, selected_mode, test_mode_enabled, selected_unit_count
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_table

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls"}

_ID_ALIASES = {
    "id", "regionid", "region_id", "unitid", "unit_id", "desertid", "desert_id",
    "srcid", "src_id", "fid", "objectid", "zoneid", "zone_id",
}
_YEAR_ALIASES = {"year", "yr", "dateyear", "date_year", "timeyear", "time_year"}
_VARIABLE_ALIASES = {"variable", "var", "indicator", "metric", "covariate", "feature", "name"}
_VALUE_ALIASES = {"value", "val", "mean", "average", "avg", "estimate"}

_COVARIATE_ALIASES = {
    "precipitation": {
        "precipitation", "precip", "rainfall", "rain", "ppt", "tp", "annual_precipitation",
        "total_precipitation", "prcp", "pcp", "gs_p_mm",
    },
    "temperature": {
        "temperature", "temp", "t2m", "air_temperature", "mean_temperature", "annual_temperature",
        "tmean", "tas", "gs_tmean_c",
    },
    "potential_evapotranspiration": {
        "potential_evapotranspiration", "potential_et", "pet", "eto", "reference_evapotranspiration",
        "gs_pet_mm",
    },
    "evapotranspiration": {
        "evapotranspiration", "evapo", "aet", "actual_evapotranspiration", "actual_et", "et",
    },
    "soil_moisture": {
        "soil_moisture", "soilwater", "soil_water", "swvl", "sm", "volumetric_soil_water",
        "soil_water_content", "swc", "vwc", "gs_sm_m3m3",
    },
    "radiation": {
        "radiation", "solar_radiation", "shortwave_radiation", "ssrd", "surface_solar_radiation",
        "surface_solar_radiation_downwards", "solar", "shortwave", "srad", "ssr", "gs_ssr_mjm2",
    },
    "wind_speed": {
        "wind_speed", "windspeed", "wind", "ws", "wind_velocity", "10m_wind_speed", "gs_wind_ms",
    },
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")



def _canonical_covariate(value: str, configured: Iterable[str], source_map: dict[str, str] | None = None) -> str | None:
    key = _norm(value)
    compact = key.replace("_", "")
    if source_map:
        mapped = source_map.get(key)
        if mapped is not None:
            mapped_norm = _norm(mapped)
            if mapped_norm in {_norm(c) for c in configured}:
                return mapped_norm
    configured_norm = [_norm(c) for c in configured]
    for cov in configured_norm:
        if key == cov or compact == cov.replace("_", ""):
            return cov
    for cov in configured_norm:
        aliases = _COVARIATE_ALIASES.get(cov, {cov})
        for alias in aliases:
            a = _norm(alias)
            ac = a.replace("_", "")
            if key == a or compact == ac:
                return cov
            # Long descriptive column names and file names are common. Avoid
            # matching ambiguous two-character aliases inside arbitrary words.
            if len(ac) >= 4 and (a in key or ac in compact):
                return cov
    return None


def _infer_year_from_name(path: Path) -> int | None:
    matches = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", path.stem)
    return int(matches[-1]) if matches else None


def _coerce_year_columns(df: pd.DataFrame, path: Path, configured: Iterable[str], source_map: dict[str, str] | None = None) -> pd.DataFrame | None:
    id_col = _find_column(df.columns, _ID_ALIASES)
    if id_col is None:
        return None
    year_cols = [c for c in df.columns if re.fullmatch(r"(?:19|20)\d{2}", str(c).strip())]
    cov = _canonical_covariate(str(path), configured, source_map)
    if not year_cols or cov is None:
        return None
    x = df[[id_col, *year_cols]].melt(id_vars=[id_col], var_name="year", value_name=cov)
    x = x.rename(columns={id_col: "Id"})
    x["Id"] = pd.to_numeric(x["Id"], errors="coerce")
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    x[cov] = pd.to_numeric(x[cov], errors="coerce")
    return x.dropna(subset=["Id", "year"])


def _coerce_single_year_table(df: pd.DataFrame, path: Path, configured: Iterable[str], source_map: dict[str, str] | None = None) -> pd.DataFrame | None:
    id_col = _find_column(df.columns, _ID_ALIASES)
    year = _infer_year_from_name(path)
    if id_col is None or year is None:
        return None
    candidates: list[tuple[str, str]] = []
    for col in df.columns:
        if col == id_col:
            continue
        cov = _canonical_covariate(col, configured, source_map)
        if cov is not None and pd.to_numeric(df[col], errors="coerce").notna().any():
            candidates.append((col, cov))
    if not candidates:
        cov = _canonical_covariate(str(path), configured, source_map)
        numeric = [c for c in df.columns if c != id_col and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if cov is not None and len(numeric) == 1:
            candidates = [(numeric[0], cov)]
    if not candidates:
        return None
    x = pd.DataFrame({"Id": pd.to_numeric(df[id_col], errors="coerce"), "year": year})
    for col, cov in candidates:
        values = pd.to_numeric(df[col], errors="coerce")
        x[cov] = values if cov not in x else pd.concat([x[cov], values], axis=1).mean(axis=1)
    return x.dropna(subset=["Id", "year"])

def _find_column(columns: Iterable[str], aliases: set[str]) -> str | None:
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
        compact = alias.replace("_", "")
        for key, original in lookup.items():
            if key.replace("_", "") == compact:
                return original
    return None


def _safe_extract(zip_path: Path, target: Path) -> None:
    ensure_dir(target)
    root = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            destination = (target / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        zf.extractall(target)


def materialize_table_root(cfg: dict[str, Any]) -> Path:
    configured = Path(cfg["paths"].get("input_table_dir", ""))
    if configured.exists() and any(p.suffix.lower() in SUPPORTED_SUFFIXES for p in configured.rglob("*")):
        return configured
    zip_path = Path(cfg["paths"].get("input_table_zip", ""))
    if not zip_path.exists():
        return configured
    target = ensure_dir(Path(cfg["paths"]["cache_root"]) / "environmental_tables")
    marker = target / ".source_zip.txt"
    signature = f"{zip_path.resolve()}|{zip_path.stat().st_size}|{zip_path.stat().st_mtime_ns}"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == signature:
        return target
    if target.exists():
        shutil.rmtree(target)
    ensure_dir(target)
    _safe_extract(zip_path, target)
    marker.write_text(signature, encoding="utf-8")
    return target


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(path)


def _coerce_long_table(df: pd.DataFrame, configured: Iterable[str], source_map: dict[str, str] | None = None) -> pd.DataFrame | None:
    id_col = _find_column(df.columns, _ID_ALIASES)
    year_col = _find_column(df.columns, _YEAR_ALIASES)
    var_col = _find_column(df.columns, _VARIABLE_ALIASES)
    value_col = _find_column(df.columns, _VALUE_ALIASES)
    if not all([id_col, year_col, var_col, value_col]):
        return None
    x = df[[id_col, year_col, var_col, value_col]].copy()
    x.columns = ["Id", "year", "variable", "value"]
    x["Id"] = pd.to_numeric(x["Id"], errors="coerce")
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    x["variable"] = x["variable"].map(lambda v: _canonical_covariate(v, configured, source_map))
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna(subset=["Id", "year", "variable"])
    if x.empty:
        return None
    wide = x.pivot_table(index=["Id", "year"], columns="variable", values="value", aggfunc="mean").reset_index()
    wide.columns.name = None
    return wide


def _coerce_wide_table(df: pd.DataFrame, configured: Iterable[str], source_map: dict[str, str] | None = None) -> pd.DataFrame | None:
    id_col = _find_column(df.columns, _ID_ALIASES)
    year_col = _find_column(df.columns, _YEAR_ALIASES)
    if id_col is None or year_col is None:
        return None
    x = df.copy()
    x = x.rename(columns={id_col: "Id", year_col: "year"})
    x["Id"] = pd.to_numeric(x["Id"], errors="coerce")
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    keep = ["Id", "year"]
    rename: dict[str, str] = {}
    for col in x.columns:
        if col in keep:
            continue
        numeric = pd.to_numeric(x[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        canonical = _canonical_covariate(col, configured, source_map)
        if canonical is None:
            continue
        x[col] = numeric
        keep.append(col)
        rename[col] = canonical
    if len(keep) <= 2:
        return None
    x = x[keep].rename(columns=rename).dropna(subset=["Id", "year"])
    # Multiple source columns may map to one canonical covariate.
    base = x[["Id", "year"]].copy()
    for cov in dict.fromkeys(rename.values()):
        block = x.loc[:, x.columns == cov]
        base[cov] = block.mean(axis=1, numeric_only=True) if block.shape[1] > 1 else pd.to_numeric(block.iloc[:, 0], errors="coerce")
    return base


def normalize_environmental_panel(cfg: dict[str, Any], output_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = materialize_table_root(cfg)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES) if root.exists() else []
    environment_cfg = cfg.get("environment", {})
    preferred_names = {_norm(name) for name in environment_cfg.get("preferred_table_names", [])}
    if preferred_names:
        preferred = [p for p in files if _norm(p.name) in preferred_names]
        if preferred:
            files = preferred
    covariates = [_norm(c) for c in cfg.get("ecology", {}).get("climate_covariates", [])]
    source_map = {
        _norm(source): _norm(target)
        for source, target in environment_cfg.get("source_column_map", {}).items()
    }
    pieces: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    for path in files:
        try:
            raw = _read_table(path)
            normalized = _coerce_long_table(raw, covariates, source_map)
            table_type = "long"
            if normalized is None:
                normalized = _coerce_wide_table(raw, covariates, source_map)
                table_type = "wide"
            if normalized is None:
                normalized = _coerce_year_columns(raw, path, covariates, source_map)
                table_type = "year_columns"
            if normalized is None:
                normalized = _coerce_single_year_table(raw, path, covariates, source_map)
                table_type = "single_year"
            if normalized is None:
                inventory_rows.append({"path": str(path), "status": "ignored", "reason": "no Id/year numeric panel"})
                continue
            available = [c for c in normalized.columns if c not in {"Id", "year"}]
            selected = [c for c in available if c in covariates]
            if not selected:
                inventory_rows.append({"path": str(path), "status": "ignored", "reason": "no configured covariates", "columns": ";".join(available)})
                continue
            part = normalized[["Id", "year", *selected]].copy()
            pieces.append(part)
            inventory_rows.append({
                "path": str(path), "status": "used", "table_type": table_type,
                "rows": len(part), "covariates": ";".join(selected),
            })
        except Exception as exc:
            inventory_rows.append({"path": str(path), "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    inventory = pd.DataFrame(inventory_rows)
    if not pieces:
        panel = pd.DataFrame(columns=["Id", "year", *covariates])
    else:
        stacked = pd.concat(pieces, ignore_index=True, sort=False)
        panel = stacked.groupby(["Id", "year"], as_index=False).mean(numeric_only=True)
        for cov in covariates:
            if cov not in panel:
                panel[cov] = np.nan
        panel = panel[["Id", "year", *covariates]]
    if not panel.empty:
        panel["Id"] = panel["Id"].round().astype(int)
        panel["year"] = panel["year"].round().astype(int)
        panel = panel.sort_values(["Id", "year"]).reset_index(drop=True)
    if output_path is not None:
        save_table(panel, output_path)
    return panel, inventory


def environmental_coverage(panel: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    ids = study_ids(cfg)
    years = range(int(cfg["study"]["start_year"]), int(cfg["study"]["end_year"]) + 1)
    expected = pd.MultiIndex.from_product([ids, years], names=["Id", "year"]).to_frame(index=False)
    covariates = [_norm(c) for c in cfg.get("ecology", {}).get("climate_covariates", [])]
    merged = expected.merge(panel, on=["Id", "year"], how="left")
    by_cov = {cov: float(pd.to_numeric(merged.get(cov), errors="coerce").notna().mean()) if cov in merged else 0.0 for cov in covariates}
    complete = float(merged[covariates].notna().all(axis=1).mean()) if covariates else 1.0
    return {
        "expected_records": len(expected),
        "panel_records": int(len(panel)),
        "complete_case_fraction": complete,
        "coverage_by_covariate": by_cov,
        "units_present": int(panel["Id"].nunique()) if not panel.empty else 0,
        "years_present": int(panel["year"].nunique()) if not panel.empty else 0,
    }
