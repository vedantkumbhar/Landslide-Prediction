"""
Shared GIS, Metadata Validation, Freshness, Backup, and Spatial Cell Utilities
for the Landslide Prediction System Standard Refresh Pipeline.

Provides unified, zero-fallback helpers across all three target regions:
- Himalayas (5 km UTM regular grid across 29 DEM tiles)
- Northeast India (5 km UTM regular grid across 33 DEM tiles)
- Western Ghats (1,582 validated event/background locations across 14 DEM tiles)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon
import shapely.geometry


MAX_DATA_LATENCY_HOURS = 12.0
EXPECTED_GRANULES_24H = 48

CATEGORY_COLORS: Dict[str, str] = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Very High": "#c0392b",
    "UNASSESSED_MISSING_RAINFALL": "#95a5a6",
    "STALE_DATA_UNASSESSED": "#95a5a6",
    "UNASSESSED": "#95a5a6",
}


def parse_iso8601_utc(ts_str: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO-8601 UTC timestamp string strictly requiring explicit 'Z' or '+00:00'.
    Rejects naive, missing, or non-UTC timestamps to prevent timezone ambiguity.
    """
    if not ts_str or not isinstance(ts_str, str):
        return None
    cleaned = ts_str.strip()
    if not (cleaned.endswith("Z") or cleaned.endswith("+00:00")):
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
            return None
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def validate_dynamic_metadata_thresholds(metadata: Dict[str, Any]) -> Dict[str, float]:
    """
    Strictly validate relative category cutoffs parsed dynamically from Stage metadata.

    Dynamic risk pipelines recalculate P50/P80/P95 empirical percentiles on each fresh
    rainfall snapshot. Map and GeoJSON exporters must validate those cutoffs dynamically
    with zero silent fallbacks and strict ordering: 0 < P50 < P80 < P95 < 100.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    risk_spec = metadata.get("risk_formula_specification")
    if not isinstance(risk_spec, dict):
        raise ValueError("Metadata missing 'risk_formula_specification' dictionary.")

    cutoffs = risk_spec.get("relative_category_cutoffs")
    if not isinstance(cutoffs, dict):
        raise ValueError("Metadata missing 'relative_category_cutoffs' dictionary.")

    required_keys = [
        "P50_low_to_moderate",
        "P80_moderate_to_high",
        "P95_high_to_very_high",
    ]

    validated: Dict[str, float] = {}
    for key in required_keys:
        if key not in cutoffs:
            raise ValueError(f"Missing required threshold key in metadata: '{key}'")
        raw_val = cutoffs[key]
        if (
            raw_val is None
            or isinstance(raw_val, bool)
            or not isinstance(raw_val, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"Invalid or non-numeric threshold value for '{key}': {raw_val!r}")
        val = float(raw_val)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite threshold value for '{key}': {val}")
        validated[key] = val

    p50 = validated["P50_low_to_moderate"]
    p80 = validated["P80_moderate_to_high"]
    p95 = validated["P95_high_to_very_high"]
    if not (0.0 < p50 < p80 < p95 < 100.0):
        raise ValueError(
            f"Metadata relative thresholds must satisfy 0 < P50 < P80 < P95 < 100 "
            f"(got P50={p50}, P80={p80}, P95={p95})."
        )

    return validated


def evaluate_freshness_condition(
    obs_end_utc_str: Optional[str],
    is_24h_complete: Optional[bool] = True,
    available_granules: Optional[int] = EXPECTED_GRANULES_24H,
    missing_intervals: int = 0,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Evaluate Standard Refresh freshness conditions cleanly:
    - Data age <= 12h (and complete)          -> FRESH
    - Data age > 12h                          -> STALE / UNASSESSED (STALE_DATA_UNASSESSED)
    - Missing or invalid timestamp            -> UNASSESSED
    - Future timestamp                        -> UNASSESSED
    - Incomplete or invalid rainfall data     -> UNASSESSED
    """
    ref_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    else:
        ref_now = ref_now.astimezone(timezone.utc)

    if not obs_end_utc_str or not isinstance(obs_end_utc_str, str) or not obs_end_utc_str.strip():
        return {
            "freshness_label": "UNASSESSED",
            "status_code": "UNASSESSED_MISSING_TIMESTAMP",
            "is_fresh": False,
            "is_stale": True,
            "latency_hours": None,
            "reason": "Missing observation end timestamp",
        }

    obs_end_dt = parse_iso8601_utc(obs_end_utc_str)
    if obs_end_dt is None:
        return {
            "freshness_label": "UNASSESSED",
            "status_code": "UNASSESSED_INVALID_TIMESTAMP",
            "is_fresh": False,
            "is_stale": True,
            "latency_hours": None,
            "reason": f"Invalid or non-UTC timestamp: {obs_end_utc_str!r}",
        }

    latency_hours = (ref_now - obs_end_dt).total_seconds() / 3600.0
    if latency_hours < 0.0:
        return {
            "freshness_label": "UNASSESSED",
            "status_code": "UNASSESSED_FUTURE_TIMESTAMP",
            "is_fresh": False,
            "is_stale": True,
            "latency_hours": round(latency_hours, 4),
            "reason": f"Observation end {obs_end_utc_str} is ahead of reference time",
        }

    if (
        not isinstance(is_24h_complete, bool)
        or is_24h_complete is not True
        or (available_granules is not None and available_granules < EXPECTED_GRANULES_24H)
        or missing_intervals > 0
    ):
        return {
            "freshness_label": "UNASSESSED",
            "status_code": "INCOMPLETE_DATA",
            "is_fresh": False,
            "is_stale": True,
            "latency_hours": round(latency_hours, 4),
            "reason": (
                f"Incomplete 24h GPM window (complete={is_24h_complete!r}, "
                f"granules={available_granules}/{EXPECTED_GRANULES_24H}, missing={missing_intervals})"
            ),
        }

    if latency_hours > max_latency_hours:
        return {
            "freshness_label": "STALE / UNASSESSED",
            "status_code": "STALE_DATA_UNASSESSED",
            "is_fresh": False,
            "is_stale": True,
            "latency_hours": round(latency_hours, 4),
            "reason": f"Latency {latency_hours:.2f}h exceeds {max_latency_hours:.1f}h limit",
        }

    return {
        "freshness_label": "FRESH",
        "status_code": "FRESH",
        "is_fresh": True,
        "is_stale": False,
        "latency_hours": round(latency_hours, 4),
        "reason": f"Fresh complete observation ({latency_hours:.2f}h <= {max_latency_hours:.1f}h)",
    }


def create_timestamped_backup(
    paths: Sequence[Path],
    backup_dir: Path,
    run_ts: Optional[str] = None,
) -> List[Path]:
    """
    Create timestamped backup copies of existing files before replacing them.
    Never removes existing valid files automatically.
    """
    ts = run_ts or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    for p in paths:
        if p and p.exists() and p.is_file():
            dest = backup_dir / f"{p.stem}_backup_{ts}{p.suffix}"
            shutil.copy2(p, dest)
            created.append(dest)
    return created


def atomic_write_text(target: Path, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write text file via staged temp file and os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(target: Path, payload: Any, indent: int = 2) -> Path:
    """Atomically write JSON document via staged temp file and os.replace."""
    return atomic_write_text(target, json.dumps(payload, indent=indent), encoding="utf-8")


def atomic_write_csv(df: pd.DataFrame, target: Path) -> Path:
    """Atomically write DataFrame CSV via staged temp file and os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_copy_file(src: Path, dst: Path) -> Path:
    """Atomically copy src to dst via staged temp file and os.replace."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return dst


def utm_epsg_for_tile(tile: str) -> int:
    """Compute Northern Hemisphere WGS84 UTM EPSG code from a SRTM tile label (e.g. N27E088)."""
    lon_deg = int(str(tile)[4:7])
    zone = (lon_deg // 6) + 31
    return 32600 + zone


def build_5km_spatial_cells_utm(
    df: pd.DataFrame,
    expected_count: int,
    spacing_m: float = 5000.0,
) -> gpd.GeoDataFrame:
    """
    Construct exact 5 km square spatial cells in each tile's native projected UTM CRS
    and transform vertices to RFC 7946 WGS84 polygons.
    """
    half = spacing_m / 2.0
    fwd_transformers: Dict[int, Transformer] = {}
    inv_transformers: Dict[int, Transformer] = {}

    geoms: List[Polygon] = []
    for _, row in df.iterrows():
        epsg = utm_epsg_for_tile(str(row["tile"]))
        if epsg not in fwd_transformers:
            crs_str = f"EPSG:{epsg}"
            fwd_transformers[epsg] = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
            inv_transformers[epsg] = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)

        t_to_utm = fwd_transformers[epsg]
        t_to_wgs = inv_transformers[epsg]
        x, y = t_to_utm.transform(float(row["longitude"]), float(row["latitude"]))
        xg = round(x / spacing_m) * spacing_m
        yg = round(y / spacing_m) * spacing_m

        corners_utm = [
            (xg - half, yg - half),
            (xg + half, yg - half),
            (xg + half, yg + half),
            (xg - half, yg + half),
            (xg - half, yg - half),
        ]
        geoms.append(Polygon([t_to_wgs.transform(cx, cy) for cx, cy in corners_utm]))

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    if not gdf.geometry.is_valid.all():
        raise ValueError("Generated 5 km cell polygons contain invalid geometries.")
    if len(gdf) != expected_count:
        raise ValueError(f"Expected {expected_count} cells, generated {len(gdf)}")
    return gdf


def dissolve_hazard_layer_into_regions(
    gdf_cells: gpd.GeoDataFrame,
    group_col: str,
    layer_name: str,
    color_map: Dict[str, str],
    obs_start_utc: str,
    obs_end_utc: str,
    region_prefix: str,
    disclaimer: str,
    is_stale: bool = False,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Dissolve adjacent 5 km cells sharing identical screening categories into continuous
    geographical hazard polygons without bridging disconnected footprints.
    """
    region_records: List[Dict[str, Any]] = []
    total_matched_cells = 0
    all_matched_grid_ids: List[str] = []

    unique_cats = sorted(gdf_cells[group_col].dropna().unique())
    reg_counter = 1

    if layer_name == "dynamic_hazard_screening":
        pfx = f"{region_prefix}_HAZARD_REG"
    elif layer_name == "terrain_susceptibility":
        pfx = f"{region_prefix}_TERRAIN_REG"
    else:
        pfx = f"{region_prefix}_RAIN_REG"

    for cat in unique_cats:
        cat_cells = gdf_cells[gdf_cells[group_col] == cat].copy()
        if len(cat_cells) == 0:
            continue

        dissolved = cat_cells.dissolve()
        exploded = dissolved.explode(index_parts=False).reset_index(drop=True)

        rep_pts = cat_cells.geometry.representative_point()
        tree = exploded.sindex
        _, right_idx = tree.query(rep_pts, predicate="within")
        cat_cells["_reg_idx"] = right_idx

        for r_i, poly in enumerate(exploded.geometry):
            sub_cells = cat_cells[cat_cells["_reg_idx"] == r_i]
            c_count = len(sub_cells)
            if c_count == 0:
                continue

            total_matched_cells += c_count
            gids = sub_cells["grid_id"].astype(str).tolist()
            all_matched_grid_ids.extend(gids)

            reg_id = f"{pfx}_{reg_counter:04d}"
            reg_counter += 1

            def _stat(col: str, fn: str, digits: int) -> Optional[float]:
                if col not in sub_cells or not sub_cells[col].notna().any():
                    return None
                val = getattr(sub_cells[col], fn)()
                return round(float(val), digits)

            region_records.append(
                {
                    "geometry": poly,
                    "region_id": reg_id,
                    "layer": layer_name,
                    group_col: str(cat),
                    "color": color_map.get(str(cat), "#95a5a6"),
                    "cell_count": c_count,
                    "area_sqkm": round(c_count * 25.0, 1),
                    "grid_ids": gids,
                    "dynamic_risk_min": _stat("dynamic_risk_index", "min", 2),
                    "dynamic_risk_mean": _stat("dynamic_risk_index", "mean", 2),
                    "dynamic_risk_max": _stat("dynamic_risk_index", "max", 2),
                    "terrain_contrib_mean": _stat("terrain_contribution", "mean", 2),
                    "rainfall_contrib_mean": _stat("rainfall_contribution", "mean", 2),
                    "terrain_rf_prob_mean": _stat("terrain_rf_prob_balanced", "mean", 3),
                    **{
                        f"gpm_{w}_accumulation_mean_mm": _stat(f"gpm_{w}_accumulation_mm", "mean", 2)
                        for w in ("1h", "3h", "6h", "12h", "24h")
                    },
                    "rainfall_intensity_index_mean": _stat("rainfall_intensity_index", "mean", 3),
                    "tiles_represented": sorted(sub_cells["tile"].astype(str).unique().tolist()),
                    "observation_start_utc": obs_start_utc,
                    "observation_end_utc": obs_end_utc,
                    "freshness_status": "STALE_DATA_UNASSESSED" if is_stale else "FRESH",
                    "disclaimer": disclaimer,
                }
            )

    gdf_regions = gpd.GeoDataFrame(region_records, crs="EPSG:4326")
    if total_matched_cells != len(gdf_cells):
        raise ValueError(
            f"Layer '{layer_name}' cell count mismatch: {total_matched_cells} != {len(gdf_cells)}"
        )
    if len(set(all_matched_grid_ids)) != len(gdf_cells):
        raise ValueError(
            f"Layer '{layer_name}' has duplicate/missing grid IDs: "
            f"{len(set(all_matched_grid_ids))} of {len(gdf_cells)}"
        )
    if not gdf_regions.geometry.is_valid.all():
        raise ValueError(f"Layer '{layer_name}' contains invalid polygon geometries.")

    meta = {
        "layer_name": layer_name,
        "group_column": group_col,
        "total_regions": len(gdf_regions),
        "total_cells_covered": total_matched_cells,
        "all_geometries_valid": True,
        "category_region_counts": {
            str(k): int(v) for k, v in gdf_regions[group_col].value_counts().items()
        },
    }
    return gdf_regions, meta


def export_regions_geojson(
    gdf_regions: gpd.GeoDataFrame,
    metadata: Dict[str, Any],
    output_path: Path,
    title: str,
    region: str,
    disclaimer: str,
) -> Path:
    """Export dissolved regional hazard polygons as RFC 7946 GeoJSON."""
    features: List[Dict[str, Any]] = []
    for _, r in gdf_regions.iterrows():
        props = {k: v for k, v in r.items() if k != "geometry"}
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(r.geometry),
                "properties": props,
            }
        )

    obs_meta = metadata.get("gpm_observation_metadata", {})
    doc = {
        "type": "FeatureCollection",
        "metadata": {
            "title": title,
            "region": region,
            "grid_spacing_km": 5.0,
            "total_regions": len(gdf_regions),
            "total_cells_covered": int(gdf_regions["cell_count"].sum()),
            "observation_start_utc": obs_meta.get("observation_start_utc"),
            "observation_end_utc": obs_meta.get("observation_end_utc"),
            "crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
            "disclaimer": disclaimer,
        },
        "features": features,
    }
    return atomic_write_json(output_path, doc)


def export_cells_geojson(
    gdf_cells: gpd.GeoDataFrame,
    output_path: Path,
    title: str,
    region: str,
    disclaimer: str,
) -> Path:
    """Export individual 5 km spatial cell polygons as RFC 7946 GeoJSON."""
    cell_cols = [
        "grid_id",
        "tile",
        "latitude",
        "longitude",
        "elevation_m",
        "slope_degrees",
        "primary_screening_category",
        "dynamic_risk_index",
        "terrain_contribution",
        "rainfall_contribution",
        "terrain_rf_prob_balanced",
        "terrain_baseline_screening_tier",
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_24h_accumulation_mm",
        "gpm_observation_end_utc",
    ]
    features: List[Dict[str, Any]] = []
    for _, r in gdf_cells.iterrows():
        props: Dict[str, Any] = {}
        for c in cell_cols:
            if c in r:
                v = r[c]
                if pd.notna(v):
                    if isinstance(v, (float, np.floating)):
                        props[c] = float(v)
                    elif isinstance(v, (int, np.integer)):
                        props[c] = int(v)
                    else:
                        props[c] = str(v)
                else:
                    props[c] = None
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(r.geometry),
                "properties": props,
            }
        )

    doc = {
        "type": "FeatureCollection",
        "metadata": {
            "title": title,
            "region": region,
            "total_cells": len(features),
            "grid_spacing_km": 5.0,
            "crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
            "disclaimer": disclaimer,
        },
        "features": features,
    }
    return atomic_write_json(output_path, doc)


def build_compute_freshness_js(max_latency_hours: float = MAX_DATA_LATENCY_HOURS) -> str:
    """
    Return the canonical client-side JS computeFreshness function (literal JS braces).
    Evaluates obsEndUtc against browser Date.now() without hardcoded stale status.
    """
    return f"""
    function computeFreshness() {{
        if (!obsEndUtc || typeof obsEndUtc !== 'string' || obsEndUtc.trim() === '') {{
            return {{ isStale: true, latencyHours: NaN, statusStr: 'UNASSESSED_MISSING_TIMESTAMP' }};
        }}
        var trimmed = obsEndUtc.trim();
        if (!(trimmed.endsWith('Z') || trimmed.endsWith('+00:00'))) {{
            return {{ isStale: true, latencyHours: NaN, statusStr: 'UNASSESSED_INVALID_TIMESTAMP' }};
        }}
        var obsTimeMs = Date.parse(trimmed);
        if (isNaN(obsTimeMs)) {{
            return {{ isStale: true, latencyHours: NaN, statusStr: 'UNASSESSED_INVALID_TIMESTAMP' }};
        }}
        var nowMs = Date.now();
        var elapsedMs = nowMs - obsTimeMs;
        var elapsedHours = elapsedMs / 3600000.0;

        if (elapsedHours < 0) {{
            return {{ isStale: true, latencyHours: elapsedHours, statusStr: 'UNASSESSED_FUTURE_TIMESTAMP' }};
        }}
        if (elapsedHours > {float(max_latency_hours):.6f}) {{
            return {{ isStale: true, latencyHours: elapsedHours, statusStr: 'STALE_DATA_UNASSESSED' }};
        }}
        return {{ isStale: false, latencyHours: elapsedHours, statusStr: 'FRESH' }};
    }}
    """.strip()


def build_gpm_rainfall_rows_js() -> str:
    """
    Return shared client-side JS helpers to render all five GPM IMERG rainfall
    accumulation windows (1h, 3h, 6h, 12h, 24h) in mm from polygon properties
    or legacy 25-element compat arrays.
    """
    windows = [
        ("1h", 14, "gpm_1h_accumulation_mean_mm", False),
        ("3h", 15, "gpm_3h_accumulation_mean_mm", False),
        ("6h", 16, "gpm_6h_accumulation_mean_mm", False),
        ("12h", 17, "gpm_12h_accumulation_mean_mm", False),
        ("24h", 18, "gpm_24h_accumulation_mean_mm", True),
    ]
    rows_js = []
    for label, idx, prop, bold in windows:
        border = "" if bold else ' style="border-bottom:1px solid #e1edf5;"'
        lbl_style = 'color:#546e7a; font-weight:bold;' if bold else 'color:#546e7a;'
        val_style = 'text-align:right; font-weight:bold; color:#1b4f72;' if bold else 'text-align:right; font-weight:bold;'
        rows_js.append(
            f"'<tr{border}><td style=\"{lbl_style}\">Mean {label} Accumulation:</td>"
            f"<td style=\"{val_style}\">' + formatVal(isArray ? propsOrData[{idx}] : propsOrData.{prop}, 'mm', 2) + '</td></tr>'"
        )
    joined_rows = " +\n            ".join(rows_js)
    return f"""
    function renderGpmRainfallRowsHtml(propsOrData) {{
        var isArray = Array.isArray(propsOrData);
        return '<table style="width:100%; border-collapse:collapse; font-size:11px; margin-bottom:4px;">' +
            {joined_rows} +
        '</table>';
    }}
    """.strip()


def resolve_gpm_window_with_telemetry(
    archive_dir: Path,
    raw_dirs: Optional[List[Path]] = None,
    target_end_time: Optional[datetime] = None,
    allow_download: bool = True,
    workers: int = 2,
    now_utc: Optional[datetime] = None,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[datetime], str, float, List[datetime], Dict[str, Any]]:
    """
    Resolve and retrieve the contiguous 24-hour (48 half-hourly granules) NASA GPM IMERG window,
    returning both the canonical 5-tuple values AND detailed discovery/download/validation telemetry.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from backend.extract_western_ghats_gpm_rainfall import parse_granule_times
    from backend.refresh_western_ghats_rainfall import (
        EXISTING_RAW_DIRS,
        GPMGranuleDownloader,
        LocalGranuleIndex,
        is_valid_hdf5_granule,
    )

    ref_now = now_utc or datetime.now(timezone.utc)
    raw_directories = raw_dirs or EXISTING_RAW_DIRS
    local_index = LocalGranuleIndex(raw_directories)
    downloader = GPMGranuleDownloader(archive_dir=archive_dir)

    target_granules: Optional[List[Dict[str, Any]]] = None
    target_end: Optional[datetime] = None
    missing_starts: List[datetime] = []
    discovered_online_count = 0
    downloaded_count = 0
    cached_count = 0
    window_source = "LOCAL_ARCHIVE"

    if target_end_time is None and allow_download and downloader.token:
        print("\nQuerying NASA CMR for latest available contiguous 24h GPM IMERG window...")
        try:
            candidate_end, online_granules, missing = downloader.discover_latest_contiguous_window_online()
            discovered_online_count = getattr(downloader, "last_discovered_cmr_count", len(online_granules or []))
            if candidate_end is not None and online_granules and not missing:
                print(f"Discovered contiguous CMR window ending at: {candidate_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                target_end = candidate_end
                target_granules = online_granules
                window_source = "NASA_CMR_ONLINE"
            elif candidate_end is not None and missing:
                print(
                    f"Notice: Latest CMR window ending at {candidate_end.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"has {len(missing)} missing intervals."
                )
                target_end = candidate_end
                missing_starts = list(missing)
                window_source = "NASA_CMR_PARTIAL"
        except Exception as exc:
            print(f"Notice: Online CMR query encountered an issue ({exc}). Checking local archive.")
    elif target_end_time is not None:
        target_end = target_end_time
        window_source = "TARGET_END_SPECIFIED"

    if target_end is not None:
        expected_starts = [
            target_end - timedelta(minutes=30 * (EXPECTED_GRANULES_24H - i))
            for i in range(EXPECTED_GRANULES_24H)
        ]
        local_index.build_index()
        cached_items: Dict[datetime, Dict[str, Any]] = {}
        to_download: List[Dict[str, Any]] = []
        pre_discovered_map = {g["start"]: g for g in (target_granules or [])}

        for s_dt in expected_starts:
            cached = local_index.get_granule(s_dt)
            if cached and is_valid_hdf5_granule(cached["path"]):
                cached_items[s_dt] = cached
                continue

            item = pre_discovered_map.get(s_dt)
            if not item and allow_download and downloader.token:
                w_end = s_dt + timedelta(minutes=30)
                disc = downloader.search_cmr_granules(s_dt, w_end)
                if disc:
                    item = disc[0]
                    discovered_online_count += 1

            if item and allow_download:
                to_download.append(item)
            else:
                if s_dt not in missing_starts:
                    missing_starts.append(s_dt)

        cached_count = len(cached_items)
        if to_download and allow_download:
            print(f"Downloading {len(to_download)} missing granules into archive (workers={workers})...")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(downloader.download_single_granule, it): it for it in to_download}
                for f in as_completed(futures):
                    try:
                        p = f.result()
                        if is_valid_hdf5_granule(p):
                            s_dt, e_dt = parse_granule_times(p.name)
                            cached_items[s_dt] = {"path": p, "filename": p.name, "start": s_dt, "end": e_dt}
                            downloaded_count += 1
                            if downloaded_count % 5 == 0 or downloaded_count == len(to_download):
                                print(f"   [{downloaded_count}/{len(to_download)}] Granules downloaded and verified.")
                        else:
                            it = futures[f]
                            missing_starts.append(it["start"])
                    except Exception as dl_err:
                        it = futures[f]
                        print(f"Download failed for {it.get('filename')}: {dl_err}")
                        missing_starts.append(it["start"])

        resolved_granules = []
        missing_starts_recalc = []
        for s_dt in expected_starts:
            if s_dt in cached_items:
                resolved_granules.append(cached_items[s_dt])
            else:
                missing_starts_recalc.append(s_dt)
        missing_starts = missing_starts_recalc

        if len(resolved_granules) == EXPECTED_GRANULES_24H and not missing_starts:
            target_granules = resolved_granules

    if target_granules is None or len(target_granules) < EXPECTED_GRANULES_24H:
        print("Scanning local archive directories for latest contiguous 48-granule block...")
        local_index.build_index()
        all_local = local_index.all_granules_sorted()
        if all_local:
            granule_map = {g["start"]: g for g in all_local}
            available_starts = sorted(granule_map.keys())
            for latest_start in reversed(available_starts):
                c_end = latest_start + timedelta(minutes=30)
                exp_starts = [
                    c_end - timedelta(minutes=30 * (EXPECTED_GRANULES_24H - i))
                    for i in range(EXPECTED_GRANULES_24H)
                ]
                miss = [s for s in exp_starts if s not in granule_map]
                if not miss:
                    valid_seq = [
                        granule_map[s] for s in exp_starts if is_valid_hdf5_granule(granule_map[s]["path"])
                    ]
                    if len(valid_seq) == EXPECTED_GRANULES_24H:
                        target_granules = valid_seq
                        target_end = c_end
                        missing_starts = []
                        cached_count = len(valid_seq)
                        window_source = "LOCAL_ARCHIVE_FALLBACK"
                        break

    validated_count = len(target_granules) if target_granules else 0
    telemetry: Dict[str, Any] = {
        "window_discovery_source": window_source,
        "granules_discovered_cmr": discovered_online_count,
        "granules_reused_cached": cached_count,
        "granules_downloaded": downloaded_count,
        "granules_validated": validated_count,
        "expected_granules_24h": EXPECTED_GRANULES_24H,
        "missing_intervals_count": len(missing_starts),
        "source_granules": [g["filename"] for g in target_granules] if target_granules else [],
    }

    if target_end is None or target_granules is None or len(target_granules) < EXPECTED_GRANULES_24H:
        return target_granules, target_end, "INCOMPLETE_DATA", 999.0, missing_starts, telemetry

    latency_seconds = (ref_now - target_end).total_seconds()
    if latency_seconds < 0.0:
        raise ValueError(
            f"Future observation timestamp detected: {target_end.isoformat()} is ahead of reference now {ref_now.isoformat()}"
        )

    latency_hours = latency_seconds / 3600.0
    status = "SUCCESS" if latency_hours <= max_latency_hours else "STALE_DATA"
    return target_granules, target_end, status, latency_hours, missing_starts, telemetry
