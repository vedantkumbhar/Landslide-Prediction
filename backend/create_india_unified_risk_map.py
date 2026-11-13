"""
Unified India Dynamic Landslide Risk Map Generator (Stage 1).

Combines synchronized, fresh operational outputs across all three validated Indian regions
from the Standard Refresh pipeline without modifying or overwriting any regional output:
  1. Indian Himalayas (29 DEM tiles, 12,052 5 km cells, 556 events, 429 background samples)
  2. Northeast India (33 DEM tiles, 14,550 5 km cells, 441 events, 477 background samples)
  3. Western Ghats — Limited Pilot Only (14 validated DEM tiles, 4,819 5 km cells,
     182 events, 1,400 background samples)

Combined Totals:
  - Dynamic 5 km risk grid cells: 31,421
  - Historical landslide events:   1,179
  - Provisional background samples: 2,306

Strict Safety & Scientific Separation Guarantees:
  - Regional relative thresholds (P50 / P80 / P95) and classifications remain strictly
    separate per region; numerical risk indices are never compared on one universal scale.
  - All three regions must share an identical, valid UTC GPM observation window that passes
    the 12-hour freshness guard before generation proceeds; mismatched or stale inputs fail closed.
  - Browser HTML includes a 60-second live freshness guard that marks rainfall-driven risk
    UNASSESSED if elapsed latency exceeds 12.0 hours.
  - Uses compact array serialization and lazy toggleable layer instantiation on Leaflet
    prefer_canvas=True (polygons) + L.svg({pane: 'markerPane'}) (point markers).
  - Writes outputs/india_unified_dynamic_risk_map.html atomically via atomic_save_map_html().
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.gis_refresh_utils import (
    CATEGORY_COLORS,
    MAX_DATA_LATENCY_HOURS,
    atomic_save_map_html,
    build_compute_freshness_js,
    build_gpm_rainfall_rows_js,
    create_timestamped_backup,
    evaluate_freshness_condition,
    parse_iso8601_utc,
    validate_dynamic_metadata_thresholds,
)

# Canonical Input Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_HIM_CELLS_GEOJSON = OUTPUTS_DIR / "himalayas_5km_spatial_cells.geojson"
DEFAULT_HIM_POINTS_GEOJSON = OUTPUTS_DIR / "himalayas_dynamic_risk_map.geojson"
DEFAULT_HIM_MAP_META = OUTPUTS_DIR / "himalayas_dynamic_risk_map_metadata.json"
DEFAULT_HIM_RISK_META = (
    DATA_PROCESSED_DIR / "himalayas_dynamic_risk" / "himalayas_dynamic_risk_metadata.json"
)

DEFAULT_NE_CELLS_GEOJSON = OUTPUTS_DIR / "northeast_india_5km_spatial_cells.geojson"
DEFAULT_TRAINING_CSV = DATA_PROCESSED_DIR / "regional_training_dataset_30oct.csv"
DEFAULT_NE_MAP_META = OUTPUTS_DIR / "northeast_dynamic_risk_map_metadata.json"
DEFAULT_NE_RISK_META = (
    DATA_PROCESSED_DIR / "northeast_dynamic_risk" / "northeast_dynamic_risk_metadata.json"
)

DEFAULT_WG_CELLS_GEOJSON = OUTPUTS_DIR / "western_ghats_5km_spatial_cells.geojson"
DEFAULT_WG_EVENTS_CSV = DATA_PROCESSED_DIR / "western_ghats_regional_event_features.csv"
DEFAULT_WG_BACKGROUND_CSV = (
    DATA_PROCESSED_DIR / "western_ghats_regional_background_samples_v2.csv"
)
DEFAULT_WG_MAP_META = OUTPUTS_DIR / "western_ghats_dynamic_risk_map_metadata.json"
DEFAULT_WG_RISK_META = (
    DATA_PROCESSED_DIR
    / "western_ghats_grid_dynamic_risk"
    / "western_ghats_dynamic_risk_metadata.json"
)

DEFAULT_OUTPUT_HTML = OUTPUTS_DIR / "india_unified_dynamic_risk_map.html"

# Contracted Scope Counts
EXPECTED_COUNTS: Dict[str, Dict[str, int]] = {
    "himalayas": {"cells": 12052, "tiles": 29, "events": 556, "background": 429},
    "northeast": {"cells": 14550, "tiles": 33, "events": 441, "background": 477},
    "western_ghats": {"cells": 4819, "tiles": 14, "events": 182, "background": 1400},
}
EXPECTED_TOTAL_CELLS = 31421
EXPECTED_TOTAL_EVENTS = 1179
EXPECTED_TOTAL_BACKGROUND = 2306

# Basemaps (avoid OpenStreetMap tile.openstreetmap.org due to HTTP 403)
ESRI_TOPO_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_TOPO_ATTR = "Esri, HERE, Garmin, USGS, EPA, NPS"
ESRI_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_IMAGERY_ATTR = "Esri World Imagery"
CARTO_LIGHT_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
CARTO_LIGHT_ATTR = "&copy; OpenStreetMap &copy; CARTO"

EXPERIMENTAL_DISCLAIMER = (
    "EXPERIMENTAL PROVISIONAL SCREENING ONLY — NOT AN OFFICIAL WARNING OR CALIBRATED PROBABILITY. "
    "Dynamic categories (Low / Moderate / High / Very High) are relative empirical percentiles "
    "(P50 / P80 / P95) calculated independently per region; numerical indices are not comparable across regions."
)

WG_LIMITED_PILOT_DISCLOSURE = (
    "Western Ghats — Limited Pilot Only: covers strictly 14 validated DEM tiles "
    "(4,819 5 km grid cells, 182 historical landslide events, 1,400 background samples) "
    "and does not imply complete Western Ghats coverage."
)

REGION_DISPLAY_CONFIG: Dict[str, Dict[str, Any]] = {
    "himalayas": {
        "display_name": "Indian Himalayas",
        "short_name": "Himalayas",
        "badge_color": "#1b4f72",
        "regional_map_href": "himalayas_dynamic_risk_map.html",
        "center": [31.54, 77.41],
        "zoom": 7,
        "scope_note": "29 validated DEM tiles · 12,052 cells (5 km)",
        "is_limited_pilot": False,
    },
    "northeast": {
        "display_name": "Northeast India",
        "short_name": "Northeast India",
        "badge_color": "#117864",
        "regional_map_href": "northeast_india_dynamic_risk_map.html",
        "center": [25.72, 92.61],
        "zoom": 7,
        "scope_note": "33 validated DEM tiles · 14,550 cells (5 km)",
        "is_limited_pilot": False,
    },
    "western_ghats": {
        "display_name": "Western Ghats (Limited Pilot)",
        "short_name": "Western Ghats Pilot",
        "badge_color": "#7d3c98",
        "regional_map_href": "western_ghats_dynamic_risk_map.html",
        "center": [14.51, 75.32],
        "zoom": 6,
        "scope_note": "14 validated DEM tiles ONLY · 4,819 cells (5 km)",
        "is_limited_pilot": True,
    },
}


def safe_html(val: Any) -> str:
    """Escape value safely for HTML embedding."""
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "N/A"
    return html.escape(str(val), quote=True)


def _clean_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    if isinstance(val, float) and not math.isfinite(val):
        return default
    s = str(val).strip()
    if s.lower() in ("nan", "none", "nat"):
        return default
    return s


def _clean_float(val: Any, digits: int, default: float = 0.0) -> float:
    try:
        f = float(val)
        if not math.isfinite(f):
            return round(default, digits)
        return round(f, digits)
    except (TypeError, ValueError):
        return round(default, digits)


def _load_json_checked(path: Path, description: str) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing required {description}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {description}: {path}")
    return data


def _extract_map_obs_block(map_meta: Dict[str, Any], region_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Normalize (obs_start_utc, obs_end_utc, status) across regional map metadata variants."""
    if "observation_metadata" in map_meta and isinstance(map_meta["observation_metadata"], dict):
        blk = map_meta["observation_metadata"]
        status = blk.get("status")
        if not status and blk.get("is_fresh") is True:
            status = "SUCCESS"
        return blk.get("observation_start_utc"), blk.get("observation_end_utc"), status
    if "gpm_observation_metadata" in map_meta and isinstance(map_meta["gpm_observation_metadata"], dict):
        blk = map_meta["gpm_observation_metadata"]
        status = blk.get("freshness_status") or blk.get("status")
        return blk.get("observation_start_utc"), blk.get("observation_end_utc"), status
    raise ValueError(f"Map metadata for '{region_key}' missing observation metadata block.")


def _extract_map_cutoffs(map_meta: Dict[str, Any], region_key: str) -> Optional[Dict[str, float]]:
    raw = map_meta.get("approved_relative_thresholds") or map_meta.get("relative_category_cutoffs")
    if not isinstance(raw, dict):
        return None
    keys = ("P50_low_to_moderate", "P80_moderate_to_high", "P95_high_to_very_high")
    if all(k in raw for k in keys):
        return {k: float(raw[k]) for k in keys}
    return None


def _validate_region_metadata(
    region_key: str,
    map_meta_path: Path,
    risk_meta_path: Path,
) -> Dict[str, Any]:
    """Load and internally cross-check one region's map metadata and dynamic risk metadata."""
    map_meta = _load_json_checked(map_meta_path, f"{region_key} map metadata JSON")
    risk_meta = _load_json_checked(risk_meta_path, f"{region_key} dynamic risk metadata JSON")

    cutoffs = validate_dynamic_metadata_thresholds(risk_meta)
    map_cutoffs = _extract_map_cutoffs(map_meta, region_key)
    if map_cutoffs is not None:
        for k, v in cutoffs.items():
            if abs(float(map_cutoffs[k]) - float(v)) > 1e-4:
                raise ValueError(
                    f"Threshold mismatch in {region_key} metadata ({k}): "
                    f"risk_meta={v} vs map_meta={map_cutoffs[k]}"
                )

    gpm_meta = risk_meta.get("gpm_observation_metadata")
    if not isinstance(gpm_meta, dict):
        raise ValueError(f"{region_key} risk metadata missing 'gpm_observation_metadata'.")

    r_start = gpm_meta.get("observation_start_utc")
    r_end = gpm_meta.get("observation_end_utc")
    r_status = gpm_meta.get("freshness_status")
    if r_status != "SUCCESS":
        raise ValueError(
            f"{region_key} dynamic risk freshness_status is {r_status!r} (expected 'SUCCESS')."
        )

    m_start, m_end, m_status = _extract_map_obs_block(map_meta, region_key)
    if m_status not in ("SUCCESS", "FRESH", "PASSED"):
        raise ValueError(f"{region_key} map metadata freshness status is {m_status!r}.")

    r_start_dt = parse_iso8601_utc(r_start)
    r_end_dt = parse_iso8601_utc(r_end)
    m_start_dt = parse_iso8601_utc(m_start)
    m_end_dt = parse_iso8601_utc(m_end)

    if not all((r_start_dt, r_end_dt, m_start_dt, m_end_dt)):
        raise ValueError(
            f"Invalid or missing explicit UTC timestamp in {region_key} metadata: "
            f"risk=({r_start!r}, {r_end!r}), map=({m_start!r}, {m_end!r})."
        )
    if r_start_dt != m_start_dt or r_end_dt != m_end_dt:
        raise ValueError(
            f"Internal timestamp mismatch for {region_key}: "
            f"risk_meta=({r_start}->{r_end}) vs map_meta=({m_start}->{m_end})."
        )
    if r_end_dt <= r_start_dt:
        raise ValueError(
            f"Invalid observation window ordering for {region_key}: {r_start} -> {r_end}."
        )

    is_24h_complete = map_meta.get("gpm_observation_metadata", {}).get("is_24h_complete", True)
    if is_24h_complete is not True:
        raise ValueError(f"{region_key} map metadata indicates incomplete 24h GPM window.")

    return {
        "region_key": region_key,
        "obs_start_utc": r_start,
        "obs_end_utc": r_end,
        "obs_start_dt": r_start_dt,
        "obs_end_dt": r_end_dt,
        "cutoffs": cutoffs,
        "map_generation_timestamp_utc": map_meta.get("generation_timestamp_utc"),
    }


def _load_and_serialize_cells(
    region_key: str,
    cells_geojson_path: Path,
    expected_obs_end_dt: datetime,
    expected_cells: int,
    expected_tiles: int,
) -> Tuple[List[List[Any]], Dict[str, int]]:
    """
    Validate and serialize 5 km grid cells from GeoJSON into compact rows:
    [corners4_latlon, grid_id, tile, cat, risk, terrain_c, rain_c,
     r1h, r3h, r6h, r12h, r24h, elev, slope, t_prob, t_tier, lat, lon]
    """
    gj = _load_json_checked(cells_geojson_path, f"{region_key} 5 km cells GeoJSON")
    features = gj.get("features")
    if not isinstance(features, list):
        raise ValueError(f"{region_key} cells GeoJSON missing 'features' list.")
    if len(features) != expected_cells:
        raise ValueError(
            f"{region_key} cells count mismatch in {cells_geojson_path.name}: "
            f"expected {expected_cells}, got {len(features)}."
        )

    rows: List[List[Any]] = []
    tiles_seen: set = set()
    category_counts: Dict[str, int] = {
        "Low": 0,
        "Moderate": 0,
        "High": 0,
        "Very High": 0,
        "Unassessed": 0,
    }

    for idx, feat in enumerate(features):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            raise ValueError(
                f"{region_key} cell #{idx} has non-Polygon geometry {geom.get('type')!r}."
            )
        rings = geom.get("coordinates") or []
        if not rings or len(rings[0]) < 4:
            raise ValueError(f"{region_key} cell #{idx} has malformed exterior polygon ring.")

        ext_ring = rings[0]
        # Drop closing duplicate vertex if present (5 vertices -> 4 unique corners)
        if len(ext_ring) >= 5 and ext_ring[0] == ext_ring[-1]:
            ring_pts = ext_ring[:-1]
        else:
            ring_pts = ext_ring
        corners_latlon = [
            [_clean_float(pt[1], 4), _clean_float(pt[0], 4)]
            for pt in ring_pts
        ]

        props = feat.get("properties") or {}
        cell_obs_end_str = props.get("gpm_observation_end_utc")
        cell_obs_end_dt = parse_iso8601_utc(cell_obs_end_str)
        if cell_obs_end_dt is None or cell_obs_end_dt != expected_obs_end_dt:
            raise ValueError(
                f"{region_key} cell {props.get('grid_id', idx)!r} has mismatched or invalid "
                f"gpm_observation_end_utc={cell_obs_end_str!r} "
                f"(expected {expected_obs_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')!r})."
            )

        grid_id = _clean_str(props.get("grid_id"), f"{region_key.upper()}_{idx}")
        tile = _clean_str(props.get("tile"))
        if tile:
            tiles_seen.add(tile)

        cat = _clean_str(props.get("primary_screening_category"), "UNASSESSED")
        if cat in category_counts:
            category_counts[cat] += 1
        else:
            category_counts["Unassessed"] += 1

        rows.append([
            corners_latlon,
            grid_id,
            tile,
            cat,
            _clean_float(props.get("dynamic_risk_index"), 2),
            _clean_float(props.get("terrain_contribution"), 2),
            _clean_float(props.get("rainfall_contribution"), 2),
            _clean_float(props.get("gpm_1h_accumulation_mm"), 2),
            _clean_float(props.get("gpm_3h_accumulation_mm"), 2),
            _clean_float(props.get("gpm_6h_accumulation_mm"), 2),
            _clean_float(props.get("gpm_12h_accumulation_mm"), 2),
            _clean_float(props.get("gpm_24h_accumulation_mm"), 2),
            _clean_float(props.get("elevation_m"), 1),
            _clean_float(props.get("slope_degrees"), 1),
            _clean_float(props.get("terrain_rf_prob_balanced"), 3),
            _clean_str(props.get("terrain_baseline_screening_tier"), "Unknown"),
            _clean_float(props.get("latitude"), 4),
            _clean_float(props.get("longitude"), 4),
        ])

    if len(tiles_seen) != expected_tiles:
        raise ValueError(
            f"{region_key} tile count mismatch in {cells_geojson_path.name}: "
            f"expected {expected_tiles}, got {len(tiles_seen)}."
        )

    return rows, category_counts


def _load_himalayas_points(
    points_geojson_path: Path,
    expected_obs_start_dt: datetime,
    expected_obs_end_dt: datetime,
    expected_events: int,
    expected_bg: int,
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Load Himalayas historical events and provisional background samples from point GeoJSON."""
    gj = _load_json_checked(points_geojson_path, "Himalayas point GeoJSON")
    meta = gj.get("metadata") or {}
    if meta.get("observation_start_utc") or meta.get("observation_end_utc"):
        ps_dt = parse_iso8601_utc(meta.get("observation_start_utc"))
        pe_dt = parse_iso8601_utc(meta.get("observation_end_utc"))
        if ps_dt != expected_obs_start_dt or pe_dt != expected_obs_end_dt:
            raise ValueError(
                f"Himalayas point GeoJSON observation window mismatch: "
                f"{meta.get('observation_start_utc')} -> {meta.get('observation_end_utc')}."
            )

    features = gj.get("features")
    if not isinstance(features, list):
        raise ValueError("Himalayas point GeoJSON missing 'features' list.")

    events_rows: List[List[Any]] = []
    bg_rows: List[List[Any]] = []

    for feat in features:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        ftype = _clean_str(props.get("feature_type"))
        stype = _clean_str(props.get("sample_type"))

        if ftype == "historical_landslide_event" or stype == "historical_landslide":
            if len(coords) < 2:
                raise ValueError(f"Malformed Himalayas historical event geometry: {props}")
            events_rows.append([
                _clean_float(coords[1], 5),
                _clean_float(coords[0], 5),
                _clean_str(props.get("record_id"), "event"),
                _clean_str(props.get("tile")),
                _clean_float(props.get("elevation_m"), 1),
                _clean_float(props.get("slope_degrees"), 1),
                "",
                "",
                "",
                "Himalayas",
            ])
        elif ftype == "provisional_background_sample" or stype == "provisional_background":
            if len(coords) < 2:
                raise ValueError(f"Malformed Himalayas background sample geometry: {props}")
            bg_rows.append([
                _clean_float(coords[1], 5),
                _clean_float(coords[0], 5),
                _clean_str(props.get("record_id"), "bg"),
                _clean_str(props.get("tile")),
                _clean_float(props.get("elevation_m"), 1),
                _clean_float(props.get("slope_degrees"), 1),
                "Himalayas",
                "",
                "provisional_background",
            ])

    if len(events_rows) != expected_events:
        raise ValueError(
            f"Himalayas historical events count mismatch: expected {expected_events}, got {len(events_rows)}."
        )
    if len(bg_rows) != expected_bg:
        raise ValueError(
            f"Himalayas background samples count mismatch: expected {expected_bg}, got {len(bg_rows)}."
        )
    return events_rows, bg_rows


def _load_northeast_points(
    training_csv_path: Path,
    expected_events: int,
    expected_bg: int,
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Load Northeast India historical events and background samples using the exact
    filtering logic from backend/create_northeast_risk_map.py.
    """
    if not training_csv_path.exists() or not training_csv_path.is_file():
        raise FileNotFoundError(f"Missing Northeast training dataset CSV: {training_csv_path}")

    df_train = pd.read_csv(training_csv_path, low_memory=False)
    for col in ("dem_region", "sample_type", "latitude", "longitude", "tile"):
        if col not in df_train.columns:
            raise ValueError(f"Missing required column {col!r} in {training_csv_path.name}.")

    ne_mask = df_train["dem_region"].str.lower().str.contains("northeast", na=False)
    ne_train = df_train[ne_mask]
    df_events = ne_train[ne_train["sample_type"] == "historical_landslide"].copy()
    df_bg = ne_train[ne_train["sample_type"] == "provisional_background"].copy()

    if len(df_events) != expected_events:
        raise ValueError(
            f"Northeast historical events count mismatch: expected {expected_events}, got {len(df_events)}."
        )
    if len(df_bg) != expected_bg:
        raise ValueError(
            f"Northeast background samples count mismatch: expected {expected_bg}, got {len(df_bg)}."
        )

    events_rows: List[List[Any]] = []
    for _, r in df_events.iterrows():
        rec_id = _clean_str(r.get("record_id")) or _clean_str(r.get("event_id"), "event")
        events_rows.append([
            _clean_float(r["latitude"], 5),
            _clean_float(r["longitude"], 5),
            rec_id,
            _clean_str(r.get("tile")),
            _clean_float(r.get("elevation_m"), 1),
            _clean_float(r.get("slope_degrees"), 1),
            "",
            "",
            "",
            "Northeast India",
        ])

    bg_rows: List[List[Any]] = []
    for _, r in df_bg.iterrows():
        rec_id = _clean_str(r.get("record_id")) or _clean_str(r.get("sample_id"), "bg")
        dist_km = r.get("nearest_recorded_event_km")
        dist_str = (
            f"{_clean_float(float(dist_km) * 1000.0, 0):.0f} m"
            if dist_km is not None and pd.notna(dist_km)
            else ""
        )
        bg_rows.append([
            _clean_float(r["latitude"], 5),
            _clean_float(r["longitude"], 5),
            rec_id,
            _clean_str(r.get("tile")),
            _clean_float(r.get("elevation_m"), 1),
            _clean_float(r.get("slope_degrees"), 1),
            "Northeast India",
            dist_str,
            _clean_str(r.get("sampling_method"), "provisional_background"),
        ])

    return events_rows, bg_rows


def _load_western_ghats_points(
    events_csv_path: Path,
    background_csv_path: Path,
    expected_events: int,
    expected_bg: int,
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Load Western Ghats Limited Pilot validated events and background CSVs."""
    if not events_csv_path.exists() or not events_csv_path.is_file():
        raise FileNotFoundError(f"Missing Western Ghats events CSV: {events_csv_path}")
    if not background_csv_path.exists() or not background_csv_path.is_file():
        raise FileNotFoundError(f"Missing Western Ghats background CSV: {background_csv_path}")

    df_events = pd.read_csv(events_csv_path, low_memory=False)
    df_bg = pd.read_csv(background_csv_path, low_memory=False)

    if len(df_events) != expected_events:
        raise ValueError(
            f"Western Ghats historical events count mismatch: expected {expected_events}, got {len(df_events)}."
        )
    if len(df_bg) != expected_bg:
        raise ValueError(
            f"Western Ghats background samples count mismatch: expected {expected_bg}, got {len(df_bg)}."
        )

    events_rows: List[List[Any]] = []
    for _, r in df_events.iterrows():
        ev_id = _clean_str(r.get("event_id"), "event")
        tile = _clean_str(r.get("dem_tile")) or _clean_str(r.get("tile"))
        ev_date = _clean_str(r.get("event_date_parsed")) or _clean_str(r.get("event_date"))
        if len(ev_date) > 10 and " " in ev_date:
            ev_date = ev_date.split(" ")[0]
        events_rows.append([
            _clean_float(r["latitude"], 5),
            _clean_float(r["longitude"], 5),
            ev_id,
            tile,
            _clean_float(r.get("elevation_m"), 1),
            _clean_float(r.get("slope_degrees"), 1),
            _clean_str(r.get("event_title")),
            ev_date,
            _clean_str(r.get("landslide_trigger")),
            _clean_str(r.get("subregion"), "Western Ghats Pilot"),
        ])

    bg_rows: List[List[Any]] = []
    for _, r in df_bg.iterrows():
        dist_m = r.get("nearest_event_distance_m")
        dist_str = (
            f"{_clean_float(dist_m, 0):.0f} m"
            if dist_m is not None and pd.notna(dist_m)
            else ""
        )
        bg_rows.append([
            _clean_float(r["latitude"], 5),
            _clean_float(r["longitude"], 5),
            _clean_str(r.get("sample_id"), "bg"),
            _clean_str(r.get("tile")),
            _clean_float(r.get("elevation_m"), 1),
            _clean_float(r.get("slope_degrees"), 1),
            _clean_str(r.get("subregion"), "Western Ghats Pilot"),
            dist_str,
            _clean_str(r.get("sampling_method"), "random_valid_land_cells"),
        ])

    return events_rows, bg_rows


def _build_dashboard_legend_html(
    region_cutoffs: Dict[str, Dict[str, float]],
    category_counts_by_region: Dict[str, Dict[str, int]],
    obs_start_utc: str,
    obs_end_utc: str,
    latency_hours: float,
    max_latency_hours: float,
    gen_ts_utc: str,
    map_name: str,
    layer_names: Dict[str, Dict[str, str]],
) -> str:
    """Build the premium geospatial command-center layout: Top Telemetry Bar, Left Command Sidebar & Right Analytics Drawer."""
    region_cards_html: List[str] = []
    accent_colors = {
        "himalayas": "#38bdf8",      # Cyan
        "northeast": "#10b981",      # Emerald
        "western_ghats": "#c084fc",  # Purple
    }
    for rkey in ("himalayas", "northeast", "western_ghats"):
        cfg = REGION_DISPLAY_CONFIG[rkey]
        cuts = region_cutoffs[rkey]
        counts = category_counts_by_region[rkey]
        p50 = cuts["P50_low_to_moderate"]
        p80 = cuts["P80_moderate_to_high"]
        p95 = cuts["P95_high_to_very_high"]
        accent = accent_colors.get(rkey, cfg["badge_color"])
        pilot_tag = (
            '<span style="background:rgba(192,132,252,0.25); color:#e9d5ff; border:1px solid rgba(192,132,252,0.45); padding:1px 6px; border-radius:3px; '
            'font-size:8.5px; font-weight:700; margin-left:5px; text-transform:uppercase;">14 TILES PILOT</span>'
            if cfg["is_limited_pilot"]
            else ""
        )
        region_cards_html.append(
            f"""
            <div style="border:1px solid rgba(56, 189, 248, 0.18); border-left:4px solid {accent};
                        border-radius:6px; padding:7px 9px; margin-bottom:7px; background:rgba(30, 41, 59, 0.70);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                    <span style="font-weight:700; font-size:11.8px; color:#f8fafc;">
                        {safe_html(cfg['display_name'])}{pilot_tag}
                    </span>
                    <span style="display:flex; gap:4px; align-items:center;">
                        <button type="button" onclick="window.zoomToUnifiedRegion('{rkey}')"
                                style="font-size:9.5px; color:#38bdf8; font-weight:600; cursor:pointer;
                                       border:1px solid rgba(56, 189, 248, 0.4); padding:1px 6px; border-radius:3px; background:rgba(56, 189, 248, 0.12);">
                           Focus
                        </button>
                        <a href="{safe_html(cfg['regional_map_href'])}" target="_blank" rel="noopener noreferrer"
                           style="font-size:9.5px; color:#34d399; text-decoration:none; font-weight:600;
                                  border:1px solid rgba(16, 185, 129, 0.4); padding:1px 6px; border-radius:3px; background:rgba(16, 185, 129, 0.12);">
                           Regional Map &nearr;
                        </a>
                    </span>
                </div>
                <div style="font-size:9.5px; color:#94a3b8; margin-bottom:4px;">
                    {safe_html(cfg['scope_note'])} · Events: {EXPECTED_COUNTS[rkey]['events']:,} · BG: {EXPECTED_COUNTS[rkey]['background']:,}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:2px 6px; font-size:9.5px; margin-bottom:5px; background:rgba(15, 23, 42, 0.6); padding:4px 6px; border-radius:4px;">
                    <div><span style="display:inline-block;width:8px;height:8px;background:{CATEGORY_COLORS['Low']};border-radius:2px;margin-right:4px;"></span>Low (&lt;{p50:.2f}): <b style="color:#f8fafc;">{counts['Low']:,}</b></div>
                    <div><span style="display:inline-block;width:8px;height:8px;background:{CATEGORY_COLORS['Moderate']};border-radius:2px;margin-right:4px;"></span>Mod ({p50:.2f}–{p80:.2f}): <b style="color:#f8fafc;">{counts['Moderate']:,}</b></div>
                    <div><span style="display:inline-block;width:8px;height:8px;background:{CATEGORY_COLORS['High']};border-radius:2px;margin-right:4px;"></span>High ({p80:.2f}–{p95:.2f}): <b style="color:#f8fafc;">{counts['High']:,}</b></div>
                    <div><span style="display:inline-block;width:8px;height:8px;background:{CATEGORY_COLORS['Very High']};border-radius:2px;margin-right:4px;"></span>Very High (&ge;{p95:.2f}): <b style="color:#f8fafc;">{counts['Very High']:,}</b></div>
                </div>
                <div style="display:flex; gap:10px; font-size:9.5px; border-top:1px solid rgba(255,255,255,0.08); padding-top:4px; color:#cbd5e1;">
                    <label style="cursor:pointer; display:flex; align-items:center; gap:4px;"><input type="checkbox" id="chk_{rkey}_cells" checked onchange="window.toggleUnifiedLayer('{rkey}','cells',this.checked)"> Cells</label>
                    <label style="cursor:pointer; display:flex; align-items:center; gap:4px;"><input type="checkbox" id="chk_{rkey}_events" checked onchange="window.toggleUnifiedLayer('{rkey}','events',this.checked)"> Events</label>
                    <label style="cursor:pointer; display:flex; align-items:center; gap:4px;"><input type="checkbox" id="chk_{rkey}_bg" onchange="window.toggleUnifiedLayer('{rkey}','bg',this.checked)"> Background</label>
                </div>
            </div>
            """.strip()
        )

    cards_joined = "\n".join(region_cards_html)
    _ = (map_name, layer_names)

    return f"""
    <style>
        :root {{
            --cc-bg-nav: rgba(11, 19, 43, 0.94);
            --cc-bg-panel: rgba(15, 23, 42, 0.92);
            --cc-bg-card: rgba(30, 41, 59, 0.70);
            --cc-border: rgba(56, 189, 248, 0.22);
            --cc-cyan: #38bdf8;
            --cc-emerald: #10b981;
            --cc-purple: #c084fc;
            --cc-text-light: #f8fafc;
            --cc-text-muted: #94a3b8;
        }}

        #unified-top-bar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 52px;
            background: var(--cc-bg-nav);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--cc-border);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--cc-text-light);
            box-sizing: border-box;
        }}

        .top-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .radar-dot {{
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #38bdf8;
            box-shadow: 0 0 10px #38bdf8;
            animation: pulse-dot 2.2s infinite;
        }}

        @keyframes pulse-dot {{
            0% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.8); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 7px rgba(56, 189, 248, 0); }}
            100% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }}
        }}

        .top-title {{
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 1.2px;
            color: #ffffff;
            text-transform: uppercase;
        }}

        .top-sub {{
            font-size: 9px;
            color: var(--cc-text-muted);
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }}

        .status-pill {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.45);
            color: #34d399;
            font-size: 9.5px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            letter-spacing: 0.4px;
        }}

        .top-center {{
            display: flex;
            flex-direction: column;
            align-items: center;
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid var(--cc-border);
            padding: 3px 12px;
            border-radius: 6px;
        }}

        .top-center-label {{
            font-size: 8.5px;
            font-weight: 700;
            color: var(--cc-cyan);
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }}

        .top-center-val {{
            font-size: 11px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-weight: 600;
            color: #f1f5f9;
        }}

        .top-right {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 10.5px;
        }}

        #live-utc-clock {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 10.5px;
            color: var(--cc-cyan);
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.28);
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: 600;
        }}

        /* Left Sidebar & Right Drawer */
        #india-unified-panel, #unified-right-panel {{
            background: var(--cc-bg-panel);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--cc-border);
            border-radius: 8px;
            box-shadow: 0 12px 36px rgba(0,0,0,0.65);
            color: var(--cc-text-light);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 11px;
            line-height: 1.4;
            box-sizing: border-box;
        }}

        #india-unified-panel {{
            position: fixed;
            top: 64px;
            left: 16px;
            width: 345px;
            max-height: calc(100vh - 80px);
            overflow-y: auto;
            z-index: 9998;
            padding: 10px 12px;
        }}

        #unified-right-panel {{
            position: fixed;
            top: 64px;
            right: 16px;
            width: 325px;
            max-height: calc(100vh - 80px);
            overflow-y: auto;
            z-index: 9998;
            padding: 10px 12px;
        }}

        /* Scrollbars */
        #india-unified-panel::-webkit-scrollbar,
        #unified-right-panel::-webkit-scrollbar {{
            width: 5px;
        }}
        #india-unified-panel::-webkit-scrollbar-track,
        #unified-right-panel::-webkit-scrollbar-track {{
            background: rgba(15, 23, 42, 0.6);
        }}
        #india-unified-panel::-webkit-scrollbar-thumb,
        #unified-right-panel::-webkit-scrollbar-thumb {{
            background: rgba(56, 189, 248, 0.35);
            border-radius: 3px;
        }}

        /* Leaflet Controls Dark Theme */
        .leaflet-control-layers {{
            background: rgba(15, 23, 42, 0.92) !important;
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
            color: #e2e8f0 !important;
            border-radius: 6px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.55) !important;
            backdrop-filter: blur(10px) !important;
        }}
        .leaflet-control-layers-expanded {{
            padding: 8px 12px !important;
            font-size: 11px !important;
        }}
        .leaflet-control-zoom a {{
            background: rgba(15, 23, 42, 0.92) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
        }}
        .leaflet-control-zoom a:hover {{
            background: rgba(56, 189, 248, 0.25) !important;
            color: #ffffff !important;
        }}
        .leaflet-container {{
            background: #080d1a !important;
        }}
        .leaflet-popup-content-wrapper {{
            background: #0f172a !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(56, 189, 248, 0.35) !important;
            border-radius: 8px !important;
            box-shadow: 0 12px 32px rgba(0,0,0,0.7) !important;
        }}
        .leaflet-popup-tip {{
            background: #0f172a !important;
        }}
    </style>

    <header id="unified-top-bar">
        <div class="top-brand">
            <span class="radar-dot"></span>
            <div>
                <div class="top-title">LANDSLIDE INDIA</div>
                <div class="top-sub">Geospatial Intelligence Platform</div>
            </div>
            <span class="status-pill"><span style="width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;"></span>OPERATIONAL</span>
        </div>

        <div class="top-center">
            <span class="top-center-label">NASA GPM IMERG V07C (24H WINDOW)</span>
            <span class="top-center-val">{safe_html(obs_start_utc)} &rarr; {safe_html(obs_end_utc)}</span>
        </div>

        <div class="top-right">
            <div>
                <span id="dashboard-latency-text" style="color:#cbd5e1; font-size:10.5px;">{latency_hours:.1f} hours (FRESH, limit: {max_latency_hours:.1f}h)</span>
            </div>
            <span id="dashboard-freshness-badge" style="background:#27ae60; color:#ffffff; padding:2px 8px; border-radius:4px; font-size:9.5px; font-weight:700; letter-spacing:0.4px; text-transform:uppercase;">FRESH (OPERATIONAL)</span>
            <span id="live-utc-clock">--:--:-- UTC</span>
            <span style="font-size:9px; color:#64748b;" title="Refresh Run Generation UTC">Gen: {safe_html(gen_ts_utc)}</span>
        </div>
    </header>

    <div id="india-unified-panel">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid rgba(56, 189, 248, 0.22); padding-bottom:5px; margin-bottom:7px;">
            <div>
                <div style="font-weight:700; font-size:12px; color:#f8fafc; text-transform:uppercase; letter-spacing:0.8px;">
                    Regional Command & Layers
                </div>
                <div style="font-size:9.5px; color:#38bdf8; font-weight:600;">
                    {EXPECTED_TOTAL_CELLS:,} Grid Cells (5 km) · {EXPECTED_TOTAL_EVENTS:,} Events · {EXPECTED_TOTAL_BACKGROUND:,} Background Samples
                </div>
            </div>
            <button type="button" onclick="var b=document.getElementById('unified-panel-body'); b.style.display=(b.style.display==='none'?'block':'none'); this.textContent=(b.style.display==='none'?'Expand':'Collapse');"
                    style="font-size:9.5px; padding:2px 7px; border:1px solid rgba(56, 189, 248, 0.3); border-radius:4px; background:rgba(30, 41, 59, 0.8); color:#38bdf8; cursor:pointer;">
                Collapse
            </button>
        </div>

        <div id="unified-panel-body">
            <div style="background:rgba(234, 179, 8, 0.12); border:1px solid rgba(234, 179, 8, 0.35); border-radius:5px; padding:6px 8px; margin-bottom:7px; font-size:9.5px; color:#fef08a; line-height:1.35;">
                <b>Separate Regional Thresholds:</b> Categories use each region's own P50 / P80 / P95 quantiles. Do not compare numerical scores across regions as a universal national scale.
            </div>

            <div style="display:flex; gap:5px; flex-wrap:wrap; margin-bottom:7px;">
                <button type="button" onclick="window.zoomToUnifiedRegion('india')"
                        style="font-size:10px; font-weight:600; padding:3px 9px; border:1px solid #38bdf8; border-radius:4px; background:rgba(56, 189, 248, 0.2); color:#38bdf8; cursor:pointer;">
                    India View
                </button>
                <button type="button" onclick="window.setAllCategoryLayers('cells', true)"
                        style="font-size:10px; padding:3px 7px; border:1px solid rgba(255,255,255,0.2); border-radius:4px; background:rgba(30, 41, 59, 0.8); color:#f8fafc; cursor:pointer;">
                    All Cells
                </button>
                <button type="button" onclick="window.setAllCategoryLayers('events', true)"
                        style="font-size:10px; padding:3px 7px; border:1px solid rgba(192, 132, 252, 0.4); border-radius:4px; background:rgba(192, 132, 252, 0.15); color:#e9d5ff; cursor:pointer;">
                    All Events
                </button>
                <button type="button" onclick="window.setAllCategoryLayers('bg', true)"
                        style="font-size:10px; padding:3px 7px; border:1px solid rgba(16, 185, 129, 0.4); border-radius:4px; background:rgba(16, 185, 129, 0.15); color:#a7f3d0; cursor:pointer;">
                    Show BG
                </button>
                <button type="button" onclick="window.setAllCategoryLayers('bg', false)"
                        style="font-size:10px; padding:3px 7px; border:1px solid rgba(255,255,255,0.15); border-radius:4px; background:rgba(30, 41, 59, 0.6); color:#94a3b8; cursor:pointer;">
                    Hide BG
                </button>
            </div>

            {cards_joined}

            <div style="display:flex; gap:10px; align-items:center; font-size:9.5px; margin:5px 0 6px 0; color:#cbd5e1; flex-wrap:wrap;">
                <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#8e44ad;border:1px solid #fff;box-shadow:0 0 0 1px #8e44ad;margin-right:4px;"></span>Historical Events ({EXPECTED_TOTAL_EVENTS:,})</span>
                <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#16a085;border:1px solid #0e6251;margin-right:4px;"></span>Background Samples ({EXPECTED_TOTAL_BACKGROUND:,})</span>
                <span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:{CATEGORY_COLORS['UNASSESSED_MISSING_RAINFALL']};margin-right:4px;"></span>Unassessed (&gt;12h)</span>
            </div>
        </div>
    </div>

    <div id="unified-right-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(56, 189, 248, 0.22); padding-bottom:5px; margin-bottom:7px;">
            <div>
                <div style="font-weight:700; font-size:12px; color:#f8fafc; text-transform:uppercase; letter-spacing:0.8px;">
                    National Telemetry & Analytics
                </div>
                <div style="font-size:9.5px; color:#94a3b8;">
                    GPM IMERG V07C Satellite Screening
                </div>
            </div>
            <button type="button" onclick="var b=document.getElementById('unified-right-body'); b.style.display=(b.style.display==='none'?'block':'none'); this.textContent=(b.style.display==='none'?'Expand':'Collapse');"
                    style="font-size:9.5px; padding:2px 7px; border:1px solid rgba(56, 189, 248, 0.3); border-radius:4px; background:rgba(30, 41, 59, 0.8); color:#38bdf8; cursor:pointer;">
                Collapse
            </button>
        </div>

        <div id="unified-right-body">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-bottom:8px;">
                <div style="background:rgba(30, 41, 59, 0.8); border:1px solid rgba(56, 189, 248, 0.2); border-radius:5px; padding:6px 4px; text-align:center;">
                    <div style="font-size:14px; font-weight:800; color:#38bdf8;">{EXPECTED_TOTAL_CELLS:,}</div>
                    <div style="font-size:8.5px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.3px;">5 km Cells</div>
                </div>
                <div style="background:rgba(30, 41, 59, 0.8); border:1px solid rgba(192, 132, 252, 0.25); border-radius:5px; padding:6px 4px; text-align:center;">
                    <div style="font-size:14px; font-weight:800; color:#c084fc;">{EXPECTED_TOTAL_EVENTS:,}</div>
                    <div style="font-size:8.5px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.3px;">Events</div>
                </div>
                <div style="background:rgba(30, 41, 59, 0.8); border:1px solid rgba(16, 185, 129, 0.25); border-radius:5px; padding:6px 4px; text-align:center;">
                    <div style="font-size:14px; font-weight:800; color:#34d399;">{EXPECTED_TOTAL_BACKGROUND:,}</div>
                    <div style="font-size:8.5px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.3px;">BG Samples</div>
                </div>
            </div>

            <div style="background:rgba(30, 41, 59, 0.6); border:1px solid rgba(56, 189, 248, 0.15); border-radius:6px; padding:6px 8px; margin-bottom:7px;">
                <div style="font-size:9.8px; font-weight:700; color:#f8fafc; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.4px;">
                    Regional Monitored Coverage
                </div>
                <div style="font-size:9.5px; color:#cbd5e1; line-height:1.5;">
                    <div style="display:flex; justify-content:space-between;"><span>Indian Himalayas:</span> <b>12,052 cells (29 tiles)</b></div>
                    <div style="display:flex; justify-content:space-between;"><span>Northeast India:</span> <b>14,550 cells (33 tiles)</b></div>
                    <div style="display:flex; justify-content:space-between;"><span>Western Ghats Pilot:</span> <b>4,819 cells (14 tiles pilot)</b></div>
                </div>
            </div>

            <div style="background:rgba(30, 41, 59, 0.6); border:1px solid rgba(56, 189, 248, 0.15); border-radius:6px; padding:6px 8px; margin-bottom:7px;">
                <div style="font-size:9.8px; font-weight:700; color:#f8fafc; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.4px;">
                    Standalone Regional Maps Hub
                </div>
                <div style="display:flex; flex-direction:column; gap:4px;">
                    <a href="himalayas_dynamic_risk_map.html" target="_blank" rel="noopener noreferrer"
                       style="display:flex; justify-content:space-between; align-items:center; font-size:9.5px; color:#38bdf8; text-decoration:none; background:rgba(15, 23, 42, 0.7); border:1px solid rgba(56, 189, 248, 0.25); border-radius:4px; padding:3px 6px;">
                        <span>Indian Himalayas GIS Map</span> <span>&nearr;</span>
                    </a>
                    <a href="northeast_india_dynamic_risk_map.html" target="_blank" rel="noopener noreferrer"
                       style="display:flex; justify-content:space-between; align-items:center; font-size:9.5px; color:#34d399; text-decoration:none; background:rgba(15, 23, 42, 0.7); border:1px solid rgba(16, 185, 129, 0.25); border-radius:4px; padding:3px 6px;">
                        <span>Northeast India GIS Map</span> <span>&nearr;</span>
                    </a>
                    <a href="western_ghats_dynamic_risk_map.html" target="_blank" rel="noopener noreferrer"
                       style="display:flex; justify-content:space-between; align-items:center; font-size:9.5px; color:#c084fc; text-decoration:none; background:rgba(15, 23, 42, 0.7); border:1px solid rgba(192, 132, 252, 0.25); border-radius:4px; padding:3px 6px;">
                        <span>Western Ghats Pilot GIS Map</span> <span>&nearr;</span>
                    </a>
                </div>
            </div>

            <div style="background:rgba(30, 41, 59, 0.6); border:1px solid rgba(56, 189, 248, 0.15); border-radius:6px; padding:6px 8px; margin-bottom:7px; font-size:9.5px; color:#94a3b8;">
                <div style="font-size:9.8px; font-weight:700; color:#f8fafc; margin-bottom:3px; text-transform:uppercase; letter-spacing:0.4px;">
                    Observation Window & Latency
                </div>
                <div><b>Window:</b> {safe_html(obs_start_utc)} to {safe_html(obs_end_utc)}</div>
                <div><b>Sensor:</b> NASA GPM IMERG Late Run V07C (24h)</div>
                <div><b>Latency Guard:</b> Max {max_latency_hours:.1f}h allowed</div>
            </div>

            <div style="background:rgba(192, 132, 252, 0.12); border:1px solid rgba(192, 132, 252, 0.35); border-radius:5px; padding:5px 7px; margin-bottom:5px; font-size:9.5px; color:#e9d5ff; line-height:1.35;">
                <b>Western Ghats Limited Pilot Disclosure:</b> Covers strictly 14 validated DEM tiles (4,819 cells, 182 events, 1,400 background samples); does not imply complete Western Ghats coverage.
            </div>
            <div style="background:rgba(239, 68, 68, 0.12); border:1px solid rgba(239, 68, 68, 0.35); border-radius:5px; padding:5px 7px; font-size:9.5px; color:#fca5a5; line-height:1.35;">
                <b>Disclaimer:</b> EXPERIMENTAL PROVISIONAL SCREENING ONLY — NOT AN OFFICIAL WARNING OR CALIBRATED PROBABILITY.
            </div>
            <div style="font-size:9px; color:#64748b; margin-top:4px; text-align:center;">
                Do not compare numerical scores across regions as a universal national scale.
            </div>
        </div>
    </div>
    """.strip()


def _build_client_script(
    map_name: str,
    layer_var_names: Dict[str, Dict[str, str]],
    region_cutoffs: Dict[str, Dict[str, float]],
    cells_by_region: Dict[str, List[List[Any]]],
    events_by_region: Dict[str, List[List[Any]]],
    bg_by_region: Dict[str, List[List[Any]]],
    obs_start_utc: str,
    obs_end_utc: str,
    max_latency_hours: float,
) -> str:
    """Build compact interactive JS with lazy toggleable layers, markerPane isolation, and 60s freshness guard."""
    payload_cells = json.dumps(cells_by_region, separators=(",", ":"))
    payload_events = json.dumps(events_by_region, separators=(",", ":"))
    payload_bg = json.dumps(bg_by_region, separators=(",", ":"))
    payload_cutoffs = json.dumps(region_cutoffs, separators=(",", ":"))
    payload_layers = json.dumps(layer_var_names, separators=(",", ":"))
    payload_colors = json.dumps(CATEGORY_COLORS, separators=(",", ":"))

    compute_freshness_js = build_compute_freshness_js(max_latency_hours=max_latency_hours)
    gpm_rows_js = build_gpm_rainfall_rows_js()

    return f"""
<script>
(function() {{
    var obsStartUtc = {json.dumps(obs_start_utc)};
    var obsEndUtc = {json.dumps(obs_end_utc)};
    var maxLatencyHours = {float(max_latency_hours):.6f};
    var isStale = false;
    var currentLatencyHours = 0.0;

    var REGION_META = {{
        himalayas: {{ name: 'Indian Himalayas', short: 'Himalayas', color: '#1b4f72', pilot: false, center: [31.54, 77.41], zoom: 7, scope: '29 validated DEM tiles · 12,052 cells' }},
        northeast: {{ name: 'Northeast India', short: 'Northeast India', color: '#117864', pilot: false, center: [25.72, 92.61], zoom: 7, scope: '33 validated DEM tiles · 14,550 cells' }},
        western_ghats: {{ name: 'Western Ghats (Limited Pilot)', short: 'Western Ghats Pilot', color: '#7d3c98', pilot: true, center: [14.51, 75.32], zoom: 6, scope: '14 validated DEM tiles ONLY · 4,819 cells' }}
    }};

    var REGION_CUTOFFS = {payload_cutoffs};
    var CATEGORY_COLORS = {payload_colors};
    var LAYER_VAR_NAMES = {payload_layers};

    var CELLS_DATA = {payload_cells};
    var EVENTS_DATA = {payload_events};
    var BG_DATA = {payload_bg};

    var populatedCellLayers = [];
    var populatedState = {{}};
    var groupRefs = {{}};
    var markerRenderer = null;

    function escapeHtml(val) {{
        if (val === null || val === undefined) return 'N/A';
        return String(val)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }}

    function formatVal(val, unit, decimals) {{
        if (val === null || val === undefined || val === '' || isNaN(Number(val))) return 'N/A';
        var num = Number(val);
        var txt = (decimals !== undefined && decimals !== null) ? num.toFixed(decimals) : String(num);
        return unit ? (txt + ' ' + unit) : txt;
    }}

    {compute_freshness_js}

    {gpm_rows_js}

    function getGroup(rkey, kind) {{
        var key = rkey + '_' + kind;
        if (groupRefs[key]) return groupRefs[key];
        var vname = LAYER_VAR_NAMES[rkey] && LAYER_VAR_NAMES[rkey][kind];
        if (vname && typeof window[vname] !== 'undefined') {{
            groupRefs[key] = window[vname];
            return groupRefs[key];
        }}
        return null;
    }}

    function renderCellPopupHtml(rkey, row) {{
        var meta = REGION_META[rkey];
        var cuts = REGION_CUTOFFS[rkey];
        var catName = isStale ? 'UNASSESSED' : escapeHtml(row[3]);
        var badgeColor = isStale ? '#95a5a6' : (CATEGORY_COLORS[row[3]] || '#2ecc71');
        var riskStr = isStale ? '<span style="color:#95a5a6; font-style:italic;">UNASSESSED (Stale Data)</span>' : formatVal(row[4], null, 2);
        var rcontribStr = isStale ? '<span style="color:#95a5a6; font-style:italic;">UNASSESSED (Stale Data)</span>' : formatVal(row[6], 'pts', 2);
        var statusStr = isStale ? 'STALE_DATA_UNASSESSED' : 'FRESH';
        var latencyStr = (currentLatencyHours !== null && !isNaN(currentLatencyHours)) ? currentLatencyHours.toFixed(1) + 'h latency' : 'latency unavailable';
        var pilotHtml = meta.pilot
            ? '<div style="font-size:9.5px; color:#e9d5ff; background:rgba(192, 132, 252, 0.15); border:1px solid rgba(192, 132, 252, 0.35); padding:4px 6px; border-radius:4px; margin-bottom:5px;"><b>Western Ghats Limited Pilot:</b> 14 validated DEM tiles only; does not imply complete Western Ghats coverage.</div>'
            : '';
        var gpmProxy = {{
            gpm_1h_accumulation_mean_mm: row[7],
            gpm_3h_accumulation_mean_mm: row[8],
            gpm_6h_accumulation_mean_mm: row[9],
            gpm_12h_accumulation_mean_mm: row[10],
            gpm_24h_accumulation_mean_mm: row[11]
        }};
        return '<div style="font-family:system-ui, -apple-system, sans-serif; font-size:11.5px; width:335px; line-height:1.42; color:#e2e8f0; background:#0f172a; padding:2px;">' +
            '<div style="border-bottom:2px solid ' + badgeColor + '; padding-bottom:5px; margin-bottom:7px;">' +
                '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<span style="font-size:13px; font-weight:700; color:#ffffff;">' + escapeHtml(row[1]) + '</span>' +
                    '<span style="background:' + badgeColor + '; color:white; padding:2px 8px; border-radius:12px; font-weight:700; font-size:10.5px; letter-spacing:0.3px;">' + catName + ' Risk</span>' +
                '</div>' +
                '<div style="font-size:10px; color:#94a3b8; margin-top:2px;"><b>Region:</b> ' + escapeHtml(meta.name) + ' · Tile <b>' + escapeHtml(row[2]) + '</b></div>' +
            '</div>' +
            pilotHtml +
            '<table style="width:100%; border-collapse:collapse; margin-bottom:7px; font-size:10.5px;">' +
                '<tr><td style="color:#94a3b8;">Coordinates:</td><td style="font-weight:600; text-align:right; color:#f1f5f9;">' + formatVal(row[16], '°N', 4) + ', ' + formatVal(row[17], '°E', 4) + '</td></tr>' +
                '<tr><td style="color:#94a3b8;">Elevation / Slope:</td><td style="font-weight:600; text-align:right; color:#f1f5f9;">' + formatVal(row[12], 'm', 1) + ' / ' + formatVal(row[13], '°', 1) + '</td></tr>' +
                '<tr><td style="color:#94a3b8;">Regional Dynamic Risk Index:</td><td style="font-weight:700; text-align:right; color:' + badgeColor + ';">' + riskStr + '</td></tr>' +
                '<tr><td style="color:#94a3b8;">Terrain / Rain Component:</td><td style="font-weight:600; text-align:right; color:#f1f5f9;">' + formatVal(row[5], 'pts', 2) + ' / ' + rcontribStr + '</td></tr>' +
                '<tr><td style="color:#94a3b8;">RF Terrain Prob / Baseline Tier:</td><td style="font-weight:600; text-align:right; color:#f1f5f9;">' + formatVal(row[14], null, 3) + ' (' + escapeHtml(row[15]) + ')</td></tr>' +
            '</table>' +
            '<div style="background:rgba(30, 41, 59, 0.75); border:1px solid rgba(56, 189, 248, 0.2); border-radius:6px; padding:6px 8px; margin-bottom:6px;">' +
                '<div style="font-weight:700; color:#38bdf8; margin-bottom:3px; font-size:10px; text-transform:uppercase; letter-spacing:0.4px;">NASA GPM IMERG Rainfall (24h Window)</div>' +
                renderGpmRainfallRowsHtml(gpmProxy) +
                '<div style="font-size:9.5px; color:#94a3b8; margin-top:3px;">Window: <b>' + escapeHtml(obsStartUtc) + '</b> to <b>' + escapeHtml(obsEndUtc) + '</b> (' + latencyStr + ') | <b>' + statusStr + '</b></div>' +
            '</div>' +
            '<div style="font-size:9.5px; color:#fef08a; background:rgba(234, 179, 8, 0.12); border:1px solid rgba(234, 179, 8, 0.3); padding:4px 6px; border-radius:4px;">' +
                '<b>' + escapeHtml(meta.short) + ' Thresholds:</b> P50=' + cuts.P50_low_to_moderate.toFixed(2) + ', P80=' + cuts.P80_moderate_to_high.toFixed(2) + ', P95=' + cuts.P95_high_to_very_high.toFixed(2) + ' (regional percentile scale; not comparable across regions).' +
            '</div>' +
        '</div>';
    }}

    function renderEventPopupHtml(rkey, row) {{
        var meta = REGION_META[rkey];
        var extraHtml = '';
        if (row[6]) extraHtml += '<div style="font-size:10.5px; color:#cbd5e1;"><b>Title:</b> ' + escapeHtml(row[6]) + '</div>';
        if (row[7]) extraHtml += '<div style="font-size:10.5px; color:#cbd5e1;"><b>Date:</b> ' + escapeHtml(row[7]) + (row[8] ? (' · <b>Trigger:</b> ' + escapeHtml(row[8])) : '') + '</div>';
        var note = meta.pilot
            ? 'Validated historical event (Western Ghats Limited Pilot — 14 DEM tiles only).'
            : 'Documented historical landslide event from regional evaluation catalog.';
        return '<div style="font-family:system-ui, -apple-system, sans-serif; font-size:11.5px; width:285px; color:#e2e8f0; background:#0f172a; padding:2px;">' +
            '<div style="border-bottom:2px solid #8e44ad; padding-bottom:4px; margin-bottom:6px; font-weight:700; color:#c084fc;">Historical Landslide Event · ' + escapeHtml(row[2]) + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>Region:</b> ' + escapeHtml(meta.name) + (row[9] && row[9] !== meta.name ? (' (' + escapeHtml(row[9]) + ')') : '') + '</div>' +
            extraHtml +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>DEM Tile:</b> ' + escapeHtml(row[3]) + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>Coordinates:</b> ' + formatVal(row[0], '°N', 4) + ', ' + formatVal(row[1], '°E', 4) + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>Elevation / Slope:</b> ' + formatVal(row[4], 'm', 1) + ' / ' + formatVal(row[5], '°', 1) + '</div>' +
            '<div style="font-size:9.5px; color:#94a3b8; margin-top:5px; border-top:1px solid rgba(255,255,255,0.08); padding-top:3px;">' + note + '</div>' +
        '</div>';
    }}

    function renderBgPopupHtml(rkey, row) {{
        var meta = REGION_META[rkey];
        var distHtml = row[7] ? '<div style="font-size:10.5px; color:#cbd5e1;"><b>Nearest Event Distance:</b> ' + escapeHtml(row[7]) + '</div>' : '';
        var note = meta.pilot
            ? 'Validated background control sample (Western Ghats Limited Pilot — 14 DEM tiles only).'
            : 'Provisional background sample from regional training/evaluation dataset.';
        return '<div style="font-family:system-ui, -apple-system, sans-serif; font-size:11.5px; width:285px; color:#e2e8f0; background:#0f172a; padding:2px;">' +
            '<div style="border-bottom:2px solid #16a085; padding-bottom:4px; margin-bottom:6px; font-weight:700; color:#34d399;">Background Sample · ' + escapeHtml(row[2]) + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>Region:</b> ' + escapeHtml(meta.name) + (row[6] && row[6] !== meta.name ? (' (' + escapeHtml(row[6]) + ')') : '') + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>DEM Tile:</b> ' + escapeHtml(row[3]) + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>Coordinates:</b> ' + formatVal(row[0], '°N', 4) + ', ' + formatVal(row[1], '°E', 4) + '</div>' +
            '<div style="font-size:10.5px; color:#cbd5e1;"><b>Elevation / Slope:</b> ' + formatVal(row[4], 'm', 1) + ' / ' + formatVal(row[5], '°', 1) + '</div>' +
            distHtml +
            '<div style="font-size:9.5px; color:#94a3b8; margin-top:5px; border-top:1px solid rgba(255,255,255,0.08); padding-top:3px;">' + note + '</div>' +
        '</div>';
    }}

    function ensureLayerPopulated(rkey, kind) {{
        var key = rkey + '_' + kind;
        if (populatedState[key]) return;
        var grp = getGroup(rkey, kind);
        if (!grp || typeof L === 'undefined') return;
        populatedState[key] = true;

        if (!markerRenderer && typeof L.svg === 'function') {{
            markerRenderer = L.svg({{ pane: 'markerPane' }});
        }}

        if (kind === 'cells') {{
            var rows = CELLS_DATA[rkey] || [];
            for (var i = 0; i < rows.length; i++) {{
                (function(row) {{
                    var col = isStale ? '#95a5a6' : (CATEGORY_COLORS[row[3]] || '#2ecc71');
                    var border = isStale ? '#7f8c8d' : '#2c3e50';
                    var poly = L.polygon(row[0], {{
                        color: border,
                        opacity: 0.45,
                        weight: 0.4,
                        fillColor: col,
                        fillOpacity: 0.65
                    }});
                    poly.bindTooltip(
                        isStale
                            ? (row[1] + ' (' + REGION_META[rkey].short + '): UNASSESSED (Stale)')
                            : (row[1] + ' (' + REGION_META[rkey].short + '): ' + row[3] + ' Risk (' + Number(row[4]).toFixed(1) + ')')
                    );
                    poly.bindPopup(function() {{ return renderCellPopupHtml(rkey, row); }}, {{ maxWidth: 360 }});
                    poly.addTo(grp);
                    populatedCellLayers.push({{ layer: poly, rkey: rkey, row: row }});
                }})(rows[i]);
            }}
        }} else if (kind === 'events') {{
            var evRows = EVENTS_DATA[rkey] || [];
            for (var j = 0; j < evRows.length; j++) {{
                (function(row) {{
                    var opts = {{
                        radius: 4.0,
                        color: '#ffffff',
                        weight: 0.9,
                        fillColor: '#8e44ad',
                        fillOpacity: 0.9,
                        pane: 'markerPane'
                    }};
                    if (markerRenderer) opts.renderer = markerRenderer;
                    var m = L.circleMarker([row[0], row[1]], opts);
                    m.bindTooltip('Historical Event (' + REGION_META[rkey].short + '): ' + row[2]);
                    m.bindPopup(function() {{ return renderEventPopupHtml(rkey, row); }}, {{ maxWidth: 310 }});
                    m.addTo(grp);
                }})(evRows[j]);
            }}
        }} else if (kind === 'bg') {{
            var bgRows = BG_DATA[rkey] || [];
            for (var k = 0; k < bgRows.length; k++) {{
                (function(row) {{
                    var opts = {{
                        radius: 3.0,
                        color: '#0e6251',
                        weight: 0.6,
                        fillColor: '#16a085',
                        fillOpacity: 0.68,
                        pane: 'markerPane'
                    }};
                    if (markerRenderer) opts.renderer = markerRenderer;
                    var m = L.circleMarker([row[0], row[1]], opts);
                    m.bindTooltip('Background Sample (' + REGION_META[rkey].short + '): ' + row[2]);
                    m.bindPopup(function() {{ return renderBgPopupHtml(rkey, row); }}, {{ maxWidth: 310 }});
                    m.addTo(grp);
                }})(bgRows[k]);
            }}
        }}
    }}

    function bringPointLayersToFront() {{
        var mapInst = window[{json.dumps(map_name)}];
        if (!mapInst) return;
        ['himalayas', 'northeast', 'western_ghats'].forEach(function(rk) {{
            var bgGrp = getGroup(rk, 'bg');
            var evGrp = getGroup(rk, 'events');
            if (bgGrp && mapInst.hasLayer(bgGrp) && typeof bgGrp.bringToFront === 'function') bgGrp.bringToFront();
            if (evGrp && mapInst.hasLayer(evGrp) && typeof evGrp.bringToFront === 'function') evGrp.bringToFront();
        }});
    }}

    function updateFreshnessStatus() {{
        var res = computeFreshness();
        isStale = res.isStale;
        currentLatencyHours = res.latencyHours;

        var latEl = document.getElementById('dashboard-latency-text');
        if (latEl) {{
            if (isNaN(currentLatencyHours)) {{
                latEl.textContent = 'Unavailable (' + res.statusStr + ')';
            }} else {{
                latEl.textContent = currentLatencyHours.toFixed(1) + ' hours (' + (isStale ? 'STALE' : 'FRESH') + ', limit: ' + maxLatencyHours.toFixed(1) + 'h)';
            }}
        }}
        var badgeEl = document.getElementById('dashboard-freshness-badge');
        if (badgeEl) {{
            badgeEl.textContent = isStale ? ('STALE / ' + res.statusStr) : 'FRESH (OPERATIONAL)';
            badgeEl.style.background = isStale ? '#e74c3c' : '#27ae60';
        }}

        for (var i = 0; i < populatedCellLayers.length; i++) {{
            var item = populatedCellLayers[i];
            var row = item.row;
            var rk = item.rkey;
            var col = isStale ? '#95a5a6' : (CATEGORY_COLORS[row[3]] || '#2ecc71');
            var border = isStale ? '#7f8c8d' : '#2c3e50';
            item.layer.setStyle({{ color: border, opacity: 0.45, weight: 0.4, fillColor: col, fillOpacity: 0.65 }});
            item.layer.setTooltipContent(
                isStale
                    ? (row[1] + ' (' + REGION_META[rk].short + '): UNASSESSED (Stale)')
                    : (row[1] + ' (' + REGION_META[rk].short + '): ' + row[3] + ' Risk (' + Number(row[4]).toFixed(1) + ')')
            );
        }}
    }}

    function toggleUnifiedLayer(rkey, kind, show) {{
        var mapInst = window[{json.dumps(map_name)}];
        var grp = getGroup(rkey, kind);
        if (!mapInst || !grp) return;
        if (show) {{
            ensureLayerPopulated(rkey, kind);
            if (!mapInst.hasLayer(grp)) mapInst.addLayer(grp);
            bringPointLayersToFront();
        }} else {{
            if (mapInst.hasLayer(grp)) mapInst.removeLayer(grp);
        }}
        var chk = document.getElementById('chk_' + rkey + '_' + kind);
        if (chk) chk.checked = !!show;
    }}

    function setAllCategoryLayers(kind, show) {{
        ['himalayas', 'northeast', 'western_ghats'].forEach(function(rk) {{
            toggleUnifiedLayer(rk, kind, show);
        }});
    }}

    function zoomToUnifiedRegion(target) {{
        var mapInst = window[{json.dumps(map_name)}];
        if (!mapInst) return;
        if (target === 'india') {{
            mapInst.fitBounds([[8.0, 71.5], [36.0, 97.5]]);
            return;
        }}
        var meta = REGION_META[target];
        if (meta) mapInst.setView(meta.center, meta.zoom);
    }}

    function pad2(n) {{ return (n < 10 ? '0' : '') + n; }}
    function updateLiveUtcClock() {{
        var el = document.getElementById('live-utc-clock');
        if (!el) return;
        var now = new Date();
        el.textContent = pad2(now.getUTCHours()) + ':' + pad2(now.getUTCMinutes()) + ':' + pad2(now.getUTCSeconds()) + ' UTC';
    }}

    var _unifiedInitialized = false;
    function initUnifiedMap() {{
        if (_unifiedInitialized) return;
        var mapInst = window[{json.dumps(map_name)}];
        if (typeof L === 'undefined' || !mapInst) return;
        _unifiedInitialized = true;

        // Populate initially active (show=True) layers: grid cells + historical events.
        // Background sample layers (show=False) remain lazy until toggled on.
        ['himalayas', 'northeast', 'western_ghats'].forEach(function(rk) {{
            ensureLayerPopulated(rk, 'cells');
            ensureLayerPopulated(rk, 'events');
        }});
        bringPointLayersToFront();

        mapInst.on('overlayadd', function(e) {{
            ['himalayas', 'northeast', 'western_ghats'].forEach(function(rk) {{
                ['cells', 'events', 'bg'].forEach(function(kind) {{
                    var grp = getGroup(rk, kind);
                    if (grp && e.layer === grp) {{
                        ensureLayerPopulated(rk, kind);
                        var chk = document.getElementById('chk_' + rk + '_' + kind);
                        if (chk) chk.checked = true;
                    }}
                }});
            }});
            bringPointLayersToFront();
        }});

        mapInst.on('overlayremove', function(e) {{
            ['himalayas', 'northeast', 'western_ghats'].forEach(function(rk) {{
                ['cells', 'events', 'bg'].forEach(function(kind) {{
                    var grp = getGroup(rk, kind);
                    if (grp && e.layer === grp) {{
                        var chk = document.getElementById('chk_' + rk + '_' + kind);
                        if (chk) chk.checked = false;
                    }}
                }});
            }});
        }});

        updateFreshnessStatus();
        setInterval(updateFreshnessStatus, 60000);
        updateLiveUtcClock();
        setInterval(updateLiveUtcClock, 1000);
    }}

    if (typeof window !== 'undefined') {{
        window.computeFreshness = computeFreshness;
        window.updateFreshnessStatus = updateFreshnessStatus;
        window.ensureLayerPopulated = ensureLayerPopulated;
        window.toggleUnifiedLayer = toggleUnifiedLayer;
        window.setAllCategoryLayers = setAllCategoryLayers;
        window.zoomToUnifiedRegion = zoomToUnifiedRegion;
        window.renderCellPopupHtml = renderCellPopupHtml;
        window.renderEventPopupHtml = renderEventPopupHtml;
        window.renderBgPopupHtml = renderBgPopupHtml;
        window.CELLS_DATA = CELLS_DATA;
        window.EVENTS_DATA = EVENTS_DATA;
        window.BG_DATA = BG_DATA;
        window.REGION_CUTOFFS = REGION_CUTOFFS;
        window.getIsStale = function() {{ return isStale; }};
        window.getCurrentLatencyHours = function() {{ return currentLatencyHours; }};
        window.setObservationEndTimeForTesting = function(newUtcStr) {{
            obsEndUtc = newUtcStr;
            updateFreshnessStatus();
        }};
    }}

    if (document.readyState === 'complete') {{
        initUnifiedMap();
    }} else {{
        document.addEventListener('DOMContentLoaded', initUnifiedMap);
        window.addEventListener('load', initUnifiedMap);
    }}
}})();
</script>
""".strip()


def generate_india_unified_risk_map(
    him_cells_geojson_path: Path = DEFAULT_HIM_CELLS_GEOJSON,
    him_points_geojson_path: Path = DEFAULT_HIM_POINTS_GEOJSON,
    him_map_metadata_path: Path = DEFAULT_HIM_MAP_META,
    him_risk_metadata_path: Path = DEFAULT_HIM_RISK_META,
    ne_cells_geojson_path: Path = DEFAULT_NE_CELLS_GEOJSON,
    training_csv_path: Path = DEFAULT_TRAINING_CSV,
    ne_map_metadata_path: Path = DEFAULT_NE_MAP_META,
    ne_risk_metadata_path: Path = DEFAULT_NE_RISK_META,
    wg_cells_geojson_path: Path = DEFAULT_WG_CELLS_GEOJSON,
    wg_events_csv_path: Path = DEFAULT_WG_EVENTS_CSV,
    wg_background_csv_path: Path = DEFAULT_WG_BACKGROUND_CSV,
    wg_map_metadata_path: Path = DEFAULT_WG_MAP_META,
    wg_risk_metadata_path: Path = DEFAULT_WG_RISK_META,
    output_html_path: Path = DEFAULT_OUTPUT_HTML,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Generate the Unified India Dynamic Landslide Risk Screening Map from synchronized
    regional outputs across Himalayas, Northeast India, and Western Ghats (Limited Pilot).
    """
    if max_latency_hours <= 0.0 or max_latency_hours > MAX_DATA_LATENCY_HOURS:
        raise ValueError(
            f"max_latency_hours must satisfy 0 < max_latency_hours <= {MAX_DATA_LATENCY_HOURS} "
            f"(got {max_latency_hours})."
        )

    output_html_path = Path(output_html_path)
    if output_html_path.exists() and not overwrite:
        raise FileExistsError(
            f"Target output already exists: {output_html_path}. Pass --overwrite to replace."
        )

    # 1. Validate all regional metadata & cutoffs BEFORE loading large GeoJSONs
    meta_specs = {
        "himalayas": _validate_region_metadata(
            "himalayas", him_map_metadata_path, him_risk_metadata_path
        ),
        "northeast": _validate_region_metadata(
            "northeast", ne_map_metadata_path, ne_risk_metadata_path
        ),
        "western_ghats": _validate_region_metadata(
            "western_ghats", wg_map_metadata_path, wg_risk_metadata_path
        ),
    }

    # 2. Enforce exact cross-region timestamp synchronization
    ref_start_dt = meta_specs["himalayas"]["obs_start_dt"]
    ref_end_dt = meta_specs["himalayas"]["obs_end_dt"]
    obs_start_utc = meta_specs["himalayas"]["obs_start_utc"]
    obs_end_utc = meta_specs["himalayas"]["obs_end_utc"]

    for rkey, m in meta_specs.items():
        if m["obs_start_dt"] != ref_start_dt or m["obs_end_dt"] != ref_end_dt:
            raise ValueError(
                f"Observation window mismatch across regions: Himalayas=({obs_start_utc} -> {obs_end_utc}) "
                f"vs {rkey}=({m['obs_start_utc']} -> {m['obs_end_utc']})."
            )

    # 3. Enforce 12-hour freshness guard (reject stale inputs cleanly)
    eval_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if eval_now.tzinfo is None:
        eval_now = eval_now.replace(tzinfo=timezone.utc)
    else:
        eval_now = eval_now.astimezone(timezone.utc)

    freshness = evaluate_freshness_condition(
        obs_end_utc_str=obs_end_utc,
        is_24h_complete=True,
        available_granules=48,
        missing_intervals=0,
        max_latency_hours=max_latency_hours,
        now_utc=eval_now,
    )
    if not freshness["is_fresh"]:
        raise ValueError(
            f"Freshness guard rejected unified map generation: "
            f"{freshness['status_code']} ({freshness['reason']})."
        )
    latency_hours = float(freshness["latency_hours"])

    # 4. Load & serialize all 31,421 grid cells + 1,179 historical events + 2,306 bg samples
    him_cells, him_cat_counts = _load_and_serialize_cells(
        "himalayas",
        him_cells_geojson_path,
        ref_end_dt,
        EXPECTED_COUNTS["himalayas"]["cells"],
        EXPECTED_COUNTS["himalayas"]["tiles"],
    )
    ne_cells, ne_cat_counts = _load_and_serialize_cells(
        "northeast",
        ne_cells_geojson_path,
        ref_end_dt,
        EXPECTED_COUNTS["northeast"]["cells"],
        EXPECTED_COUNTS["northeast"]["tiles"],
    )
    wg_cells, wg_cat_counts = _load_and_serialize_cells(
        "western_ghats",
        wg_cells_geojson_path,
        ref_end_dt,
        EXPECTED_COUNTS["western_ghats"]["cells"],
        EXPECTED_COUNTS["western_ghats"]["tiles"],
    )

    him_events, him_bg = _load_himalayas_points(
        him_points_geojson_path,
        ref_start_dt,
        ref_end_dt,
        EXPECTED_COUNTS["himalayas"]["events"],
        EXPECTED_COUNTS["himalayas"]["background"],
    )
    ne_events, ne_bg = _load_northeast_points(
        training_csv_path,
        EXPECTED_COUNTS["northeast"]["events"],
        EXPECTED_COUNTS["northeast"]["background"],
    )
    wg_events, wg_bg = _load_western_ghats_points(
        wg_events_csv_path,
        wg_background_csv_path,
        EXPECTED_COUNTS["western_ghats"]["events"],
        EXPECTED_COUNTS["western_ghats"]["background"],
    )

    cells_by_region = {
        "himalayas": him_cells,
        "northeast": ne_cells,
        "western_ghats": wg_cells,
    }
    events_by_region = {
        "himalayas": him_events,
        "northeast": ne_events,
        "western_ghats": wg_events,
    }
    bg_by_region = {
        "himalayas": him_bg,
        "northeast": ne_bg,
        "western_ghats": wg_bg,
    }
    region_cutoffs = {k: v["cutoffs"] for k, v in meta_specs.items()}
    category_counts_by_region = {
        "himalayas": him_cat_counts,
        "northeast": ne_cat_counts,
        "western_ghats": wg_cat_counts,
    }

    total_cells = sum(len(v) for v in cells_by_region.values())
    total_events = sum(len(v) for v in events_by_region.values())
    total_bg = sum(len(v) for v in bg_by_region.values())
    if total_cells != EXPECTED_TOTAL_CELLS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_CELLS} total cells, got {total_cells}")
    if total_events != EXPECTED_TOTAL_EVENTS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_EVENTS} total events, got {total_events}")
    if total_bg != EXPECTED_TOTAL_BACKGROUND:
        raise ValueError(f"Expected {EXPECTED_TOTAL_BACKGROUND} total background, got {total_bg}")

    # 5. Construct Folium Map centered over India
    map_obj = folium.Map(
        location=[22.5, 83.0],
        zoom_start=5,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    map_obj.fit_bounds([[8.0, 71.5], [36.0, 97.5]], padding=(20, 20))
    map_obj.get_root().header.add_child(
        folium.Element(
            "<title>India Unified Dynamic Landslide Risk Screening Map (Provisional)</title>"
        )
    )

    # Reliable Basemaps (avoid OSM 403)
    folium.TileLayer(
        tiles=ESRI_TOPO_URL,
        attr=ESRI_TOPO_ATTR,
        name="Esri World Topographic Map",
        overlay=False,
        control=True,
        show=True,
    ).add_to(map_obj)
    folium.TileLayer(
        tiles=ESRI_IMAGERY_URL,
        attr=ESRI_IMAGERY_ATTR,
        name="Esri Satellite Imagery",
        overlay=False,
        control=True,
        show=False,
    ).add_to(map_obj)
    folium.TileLayer(
        tiles=CARTO_LIGHT_URL,
        attr=CARTO_LIGHT_ATTR,
        name="CartoDB Positron (Minimal Light)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(map_obj)

    # Separate Toggleable Regional Layer Groups
    layer_var_names: Dict[str, Dict[str, str]] = {}
    layer_titles = {
        "himalayas": (
            f"Himalayas — Dynamic 5 km Risk Cells (n={len(him_cells):,})",
            f"Himalayas — Historical Landslide Events (n={len(him_events):,})",
            f"Himalayas — Provisional Background Samples (n={len(him_bg):,})",
        ),
        "northeast": (
            f"Northeast India — Dynamic 5 km Risk Cells (n={len(ne_cells):,})",
            f"Northeast India — Historical Landslide Events (n={len(ne_events):,})",
            f"Northeast India — Provisional Background Samples (n={len(ne_bg):,})",
        ),
        "western_ghats": (
            f"Western Ghats (Pilot 14 Tiles) — Dynamic 5 km Risk Cells (n={len(wg_cells):,})",
            f"Western Ghats (Pilot 14 Tiles) — Historical Landslide Events (n={len(wg_events):,})",
            f"Western Ghats (Pilot 14 Tiles) — Provisional Background Samples (n={len(wg_bg):,})",
        ),
    }
    for rkey in ("himalayas", "northeast", "western_ghats"):
        t_cells, t_events, t_bg = layer_titles[rkey]
        fg_c = folium.FeatureGroup(name=t_cells, show=True).add_to(map_obj)
        fg_e = folium.FeatureGroup(name=t_events, show=True).add_to(map_obj)
        fg_b = folium.FeatureGroup(name=t_bg, show=False).add_to(map_obj)
        layer_var_names[rkey] = {
            "cells": fg_c.get_name(),
            "events": fg_e.get_name(),
            "bg": fg_b.get_name(),
        }

    folium.LayerControl(collapsed=True).add_to(map_obj)

    gen_ts_utc = eval_now.strftime("%Y-%m-%dT%H:%M:%SZ")
    dashboard_html = _build_dashboard_legend_html(
        region_cutoffs=region_cutoffs,
        category_counts_by_region=category_counts_by_region,
        obs_start_utc=obs_start_utc,
        obs_end_utc=obs_end_utc,
        latency_hours=latency_hours,
        max_latency_hours=max_latency_hours,
        gen_ts_utc=gen_ts_utc,
        map_name=map_obj.get_name(),
        layer_names=layer_var_names,
    )
    client_script = _build_client_script(
        map_name=map_obj.get_name(),
        layer_var_names=layer_var_names,
        region_cutoffs=region_cutoffs,
        cells_by_region=cells_by_region,
        events_by_region=events_by_region,
        bg_by_region=bg_by_region,
        obs_start_utc=obs_start_utc,
        obs_end_utc=obs_end_utc,
        max_latency_hours=max_latency_hours,
    )

    map_obj.get_root().html.add_child(folium.Element(dashboard_html))
    map_obj.get_root().html.add_child(folium.Element(client_script))

    def _write_clean_html(dest: Path) -> None:
        raw_html = map_obj.get_root().render()
        clean_html = "\n".join(line.rstrip() for line in raw_html.splitlines()) + "\n"
        with open(dest, "w", encoding="utf-8") as f:
            f.write(clean_html)

    if output_html_path.exists() and overwrite:
        create_timestamped_backup([output_html_path], output_html_path.parent / "backup")

    html_size_bytes = atomic_save_map_html(output_html_path, _write_clean_html)

    return {
        "status": "SUCCESS",
        "output_html": str(output_html_path),
        "html_size_bytes": int(html_size_bytes),
        "html_size_mb": round(html_size_bytes / (1024.0 * 1024.0), 2),
        "observation_start_utc": obs_start_utc,
        "observation_end_utc": obs_end_utc,
        "generation_timestamp_utc": gen_ts_utc,
        "latency_hours": round(latency_hours, 4),
        "max_latency_hours": float(max_latency_hours),
        "counts": {
            "total_cells": total_cells,
            "total_events": total_events,
            "total_background": total_bg,
            "by_region": {
                rkey: {
                    "cells": len(cells_by_region[rkey]),
                    "events": len(events_by_region[rkey]),
                    "background": len(bg_by_region[rkey]),
                }
                for rkey in ("himalayas", "northeast", "western_ghats")
            },
        },
        "region_cutoffs": region_cutoffs,
        "disclaimer": EXPERIMENTAL_DISCLAIMER,
        "western_ghats_disclosure": WG_LIMITED_PILOT_DISCLOSURE,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Unified India Dynamic Landslide Risk Screening Map."
    )
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--max-latency-hours", type=float, default=MAX_DATA_LATENCY_HOURS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = generate_india_unified_risk_map(
        output_html_path=args.output_html,
        max_latency_hours=args.max_latency_hours,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
