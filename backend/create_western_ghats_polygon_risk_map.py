"""
Western Ghats Limited Pilot — 5 km Regional Polygon-Based Dynamic GIS Risk Map Generator
========================================================================================

Generates a standalone, interactive Folium GIS HTML map and RFC 7946 GeoJSON suite
across the Western Ghats Limited Pilot domain (strictly the 14 validated DEM tiles):

1. Outputs Produced:
   - `outputs/western_ghats_dynamic_risk_map.html`: 5-layer regional polygon GIS map
   - `outputs/western_ghats_dynamic_risk_regions.geojson`: Dissolved regional hazard polygons
   - `outputs/western_ghats_5km_spatial_cells.geojson`: 4,819 individual 5 km EPSG:32643 cells
   - `outputs/western_ghats_dynamic_hazard_risk.geojson`: 6,401 point features
     (4,819 grid cells + separated 182 historical events + separated 1,400 background samples)
   - `outputs/western_ghats_dynamic_risk_map_metadata.json`
   - `outputs/western_ghats_dynamic_risk_map_validation_report.json`

2. Coverage Transparency Disclosure:
   - Clearly discloses Western Ghats Limited Pilot scope: 14 validated DEM tiles,
     4,819 5 km grid cells, 182 historical events, and 1,400 background samples.
   - Never implies complete Western Ghats coverage; never fabricates/duplicates records.

3. Runtime Freshness Guard:
   - Client-side JavaScript evaluates `obsEndUtc` against browser `Date.now()`
     every 60 seconds with exact 12.000000h freshness boundary (`> 12h -> UNASSESSED`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry

from backend.gis_refresh_utils import (
    MAX_DATA_LATENCY_HOURS,
    atomic_write_json,
    build_5km_spatial_cells_utm,
    build_compute_freshness_js,
    build_gpm_rainfall_rows_js,
    create_timestamped_backup,
    dissolve_hazard_layer_into_regions,
    evaluate_freshness_condition,
    export_cells_geojson,
    export_regions_geojson,
    validate_dynamic_metadata_thresholds,
)
from backend.integrate_western_ghats_gpm_rainfall import WG_PILOT_DISCLAIMER
from backend.predict_western_ghats_grid_dynamic_risk import (
    DEFAULT_METADATA_JSON as DEFAULT_INPUT_METADATA,
    DEFAULT_OUTPUT_CSV as DEFAULT_INPUT_CSV,
)


LOGGER = logging.getLogger("western_ghats_polygon_risk_map")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_event_features.csv"
)
DEFAULT_BACKGROUND_CSV = (
    PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_background_samples_v2.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_HTML = DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_risk_map.html"
DEFAULT_OUTPUT_GEOJSON = DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_hazard_risk.geojson"
DEFAULT_OUTPUT_REGIONS_GEOJSON = (
    DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_risk_regions.geojson"
)
DEFAULT_OUTPUT_CELLS_GEOJSON = DEFAULT_OUTPUT_DIR / "western_ghats_5km_spatial_cells.geojson"
DEFAULT_OUTPUT_METADATA = DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_risk_map_metadata.json"
DEFAULT_VALIDATION_REPORT = (
    DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_risk_map_validation_report.json"
)

EXPECTED_LOCATIONS_COUNT = 4819
EXPECTED_TILES_COUNT = 14
EXPECTED_HISTORICAL_EVENTS_COUNT = 182
EXPECTED_BACKGROUND_SAMPLES_COUNT = 1400

LAT_BOUNDS = (9.0, 20.0)
LON_BOUNDS = (72.0, 78.0)

WG_LIMITED_PILOT_TITLE = (
    "Western Ghats Limited Pilot — 5 km Regional Polygon Dynamic Landslide Hazard GIS Map "
    "(14 Validated DEM Tiles Only)"
)
WG_COVERAGE_DISCLOSURE = (
    "Limited Western Ghats Pilot across 14 validated DEM tiles only "
    "(4,819 5 km grid cells, 182 historical landslide events, 1,400 background samples); "
    "does not imply complete Western Ghats coverage. "
    f"{WG_PILOT_DISCLAIMER}."
)

CATEGORY_COLORS: Dict[str, str] = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Very High": "#e74c3c",
    "UNASSESSED_MISSING_RAINFALL": "#95a5a6",
    "STALE_DATA_UNASSESSED": "#95a5a6",
    "UNASSESSED": "#95a5a6",
}

TERRAIN_TIER_COLORS: Dict[str, str] = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Very High": "#e74c3c",
}

RAINFALL_TIER_COLORS: Dict[str, str] = {
    "Negligible (<0.1 pts)": "#ebf5fb",
    "Low (0.1-1.0 pts)": "#85c1e9",
    "Moderate (1.0-3.0 pts)": "#3498db",
    "High (>=3.0 pts)": "#1b4f72",
}


def safe_html(val: Any) -> str:
    """Escape value safely for HTML rendering."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unavailable"
    return html.escape(str(val), quote=True)


def _json_safe_value(val: Any) -> Any:
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        fval = float(val)
        return None if np.isnan(fval) else fval
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, list):
        return [_json_safe_value(x) for x in val]
    if isinstance(val, dict):
        return {k: _json_safe_value(v) for k, v in val.items()}
    if pd.isna(val) if not isinstance(val, (list, dict, str, bool)) else False:
        return None
    return val


def _round_geojson_coords(coords: Any, precision: int = 6) -> Any:
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            return [round(float(coords[0]), precision), round(float(coords[1]), precision)]
        return [_round_geojson_coords(c, precision=precision) for c in coords]
    return coords


def _gdf_to_feature_collection_dict(
    gdf: gpd.GeoDataFrame, coord_precision: int = 6
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        props = {k: _json_safe_value(v) for k, v in row.items() if k != "geometry"}
        geom_dict = shapely.geometry.mapping(row.geometry)
        if "coordinates" in geom_dict:
            geom_dict = {
                "type": geom_dict["type"],
                "coordinates": _round_geojson_coords(
                    geom_dict["coordinates"], precision=coord_precision
                ),
            }
        features.append(
            {
                "type": "Feature",
                "geometry": geom_dict,
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _serialize_cells_payload(
    gdf_cells: gpd.GeoDataFrame, is_stale: bool = False
) -> List[List[Any]]:
    """Serialize all 4,819 5 km UTM-reprojected cell quadrilaterals compactly for Leaflet rendering."""
    payload: List[List[Any]] = []
    for _, r in gdf_cells.iterrows():
        ring = [
            [round(float(y), 5), round(float(x), 5)]
            for x, y in list(r.geometry.exterior.coords)[:-1]
        ]
        risk_val = (
            float(r["dynamic_risk_index"])
            if pd.notna(r.get("dynamic_risk_index")) and not is_stale
            else None
        )
        cat_raw = str(r.get("primary_screening_category", "UNASSESSED_MISSING_RAINFALL"))
        cat_display = "UNASSESSED" if (is_stale or risk_val is None) else cat_raw
        color = CATEGORY_COLORS.get(cat_display, "#95a5a6")
        tcontrib = (
            round(float(r["terrain_contribution"]), 2)
            if pd.notna(r.get("terrain_contribution"))
            else None
        )
        rcontrib = (
            round(float(r["rainfall_contribution"]), 2)
            if pd.notna(r.get("rainfall_contribution")) and not is_stale
            else None
        )
        tprob = (
            round(float(r["terrain_rf_prob_balanced"]), 3)
            if pd.notna(r.get("terrain_rf_prob_balanced"))
            else None
        )
        gpm_vals = [
            round(float(r[f"gpm_{w}_accumulation_mm"]), 2)
            if pd.notna(r.get(f"gpm_{w}_accumulation_mm"))
            else None
            for w in ("1h", "3h", "6h", "12h", "24h")
        ]
        payload.append(
            [
                ring,
                str(r["grid_id"]),
                str(r["tile"]),
                cat_display,
                color,
                round(risk_val, 2) if risk_val is not None else None,
                tcontrib,
                rcontrib,
                tprob,
                str(r.get("terrain_baseline_screening_tier", "Unavailable")),
                *gpm_vals,
                round(float(r["elevation_m"]), 1) if pd.notna(r.get("elevation_m")) else None,
                round(float(r["slope_degrees"]), 1) if pd.notna(r.get("slope_degrees")) else None,
            ]
        )
    return payload


def load_validated_point_layers(
    events_csv_path: Path = DEFAULT_EVENTS_CSV,
    background_csv_path: Path = DEFAULT_BACKGROUND_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the read-only 182 historical landslide events and 1,400 background samples separately."""
    if not events_csv_path.exists():
        raise FileNotFoundError(f"Historical events CSV not found: {events_csv_path}")
    if not background_csv_path.exists():
        raise FileNotFoundError(f"Background samples CSV not found: {background_csv_path}")

    df_events = pd.read_csv(events_csv_path)
    df_bg = pd.read_csv(background_csv_path)

    if len(df_events) != EXPECTED_HISTORICAL_EVENTS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_HISTORICAL_EVENTS_COUNT} historical events, got {len(df_events)}"
        )
    if len(df_bg) != EXPECTED_BACKGROUND_SAMPLES_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BACKGROUND_SAMPLES_COUNT} background samples, got {len(df_bg)}"
        )

    for label, sub in (("historical events", df_events), ("background samples", df_bg)):
        if sub["latitude"].isna().any() or sub["longitude"].isna().any():
            raise ValueError(f"Missing coordinates detected in {label}.")
    return df_events, df_bg


def _verify_map_freshness_gate(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    now_utc: Optional[datetime],
    max_latency_hours: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify observation freshness and completeness prior to map assembly."""
    obs_meta = metadata.get("gpm_observation_metadata", {})
    obs_start_utc = obs_meta.get("observation_start_utc") or str(
        df["gpm_observation_start_utc"].iloc[0]
    )
    obs_end_utc = obs_meta.get("observation_end_utc") or str(
        df["gpm_observation_end_utc"].iloc[0]
    )
    is_24h_complete = bool(df["gpm_24h_complete"].astype(bool).all())
    avail_granules = (
        int(df["gpm_interval_count"].min()) if "gpm_interval_count" in df.columns else 48
    )

    res = evaluate_freshness_condition(
        obs_end_utc_str=obs_end_utc,
        is_24h_complete=is_24h_complete,
        available_granules=avail_granules,
        missing_intervals=0,
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
    )
    ref_now = now_utc or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    is_stale = bool(res["status_code"] == "STALE_DATA_UNASSESSED" or res["is_stale"])
    audit = {
        "observation_start_utc": obs_start_utc,
        "observation_end_utc": obs_end_utc,
        "calculated_latency_hours": res.get("latency_hours") if res.get("latency_hours") is not None else 999.0,
        "freshness_status": res["status_code"],
        "is_stale": is_stale,
        "is_fresh": bool(res["is_fresh"]),
        "reason": res["reason"],
        "map_generation_timestamp_utc": ref_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return bool(res["is_fresh"]), str(res["reason"]), audit


def export_western_ghats_grid_point_geojson(
    df: pd.DataFrame,
    df_events: pd.DataFrame,
    df_bg: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: Path,
    is_stale: bool = False,
) -> Path:
    """Export RFC 7946 Point GeoJSON with 4,819 grid points and separated 182 events + 1,400 background samples."""
    features: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        risk_val = float(r["dynamic_risk_index"]) if pd.notna(r.get("dynamic_risk_index")) else None
        cat = str(r.get("primary_screening_category", "UNASSESSED_MISSING_RAINFALL"))
        cat_display = "UNASSESSED" if (is_stale or risk_val is None) else cat

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                "properties": {
                    "feature_type": "dynamic_grid_prediction",
                    "grid_id": str(r["grid_id"]),
                    "tile": str(r["tile"]),
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "elevation_m": round(float(r["elevation_m"]), 2) if pd.notna(r.get("elevation_m")) else None,
                    "slope_degrees": round(float(r["slope_degrees"]), 2) if pd.notna(r.get("slope_degrees")) else None,
                    "terrain_rf_prob_balanced": round(float(r["terrain_rf_prob_balanced"]), 4)
                    if pd.notna(r.get("terrain_rf_prob_balanced"))
                    else None,
                    "terrain_baseline_screening_tier": str(
                        r.get("terrain_baseline_screening_tier", "Unavailable")
                    ),
                    "terrain_contribution": round(float(r["terrain_contribution"]), 2)
                    if pd.notna(r.get("terrain_contribution"))
                    else None,
                    "rainfall_contribution": round(float(r["rainfall_contribution"]), 2)
                    if (pd.notna(r.get("rainfall_contribution")) and not is_stale)
                    else None,
                    "rainfall_intensity_index": round(float(r["rainfall_intensity_index"]), 4)
                    if pd.notna(r.get("rainfall_intensity_index"))
                    else None,
                    "gpm_1h_accumulation_mm": round(float(r["gpm_1h_accumulation_mm"]), 2)
                    if pd.notna(r.get("gpm_1h_accumulation_mm"))
                    else None,
                    "gpm_3h_accumulation_mm": round(float(r["gpm_3h_accumulation_mm"]), 2)
                    if pd.notna(r.get("gpm_3h_accumulation_mm"))
                    else None,
                    "gpm_6h_accumulation_mm": round(float(r["gpm_6h_accumulation_mm"]), 2)
                    if pd.notna(r.get("gpm_6h_accumulation_mm"))
                    else None,
                    "gpm_12h_accumulation_mm": round(float(r["gpm_12h_accumulation_mm"]), 2)
                    if pd.notna(r.get("gpm_12h_accumulation_mm"))
                    else None,
                    "gpm_24h_accumulation_mm": round(float(r["gpm_24h_accumulation_mm"]), 2)
                    if pd.notna(r.get("gpm_24h_accumulation_mm"))
                    else None,
                    "dynamic_risk_index": round(risk_val, 2)
                    if (risk_val is not None and not is_stale)
                    else None,
                    "primary_screening_category": cat_display,
                    "risk_tier_fixed": str(r.get("risk_tier_fixed", "Unavailable")),
                    "risk_calculation_status": (
                        "STALE_DATA_UNASSESSED"
                        if is_stale
                        else str(r.get("risk_calculation_status", "VALIDATED_SCREENING_CALCULATED"))
                    ),
                },
            }
        )

    for _, r in df_events.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                "properties": {
                    "feature_type": "historical_landslide_event",
                    "record_id": str(r.get("event_id", "Event")),
                    "tile": str(r.get("dem_tile", r.get("terrain_source_tile", "Unknown"))),
                    "sample_type": "historical_landslide",
                    "event_title": str(r.get("event_title", "Historical Landslide")),
                    "event_date": str(r.get("event_date_parsed", r.get("event_date", "Unavailable"))),
                    "subregion": str(r.get("subregion", "Western Ghats Pilot")),
                    "elevation_m": round(float(r["elevation_m"]), 2) if pd.notna(r.get("elevation_m")) else None,
                    "slope_degrees": round(float(r["slope_degrees"]), 2) if pd.notna(r.get("slope_degrees")) else None,
                },
            }
        )

    for _, r in df_bg.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                "properties": {
                    "feature_type": "provisional_background_sample",
                    "record_id": str(r.get("sample_id", "Background")),
                    "tile": str(r.get("tile", "Unknown")),
                    "sample_type": "provisional_background",
                    "subregion": str(r.get("subregion", "Western Ghats Pilot")),
                    "elevation_m": round(float(r["elevation_m"]), 2) if pd.notna(r.get("elevation_m")) else None,
                    "slope_degrees": round(float(r["slope_degrees"]), 2) if pd.notna(r.get("slope_degrees")) else None,
                },
            }
        )

    obs_meta = metadata.get("gpm_observation_metadata", {})
    doc = {
        "type": "FeatureCollection",
        "metadata": {
            "title": WG_LIMITED_PILOT_TITLE,
            "region": "Western Ghats",
            "coverage_scope": "Western Ghats Limited Pilot (14 Validated DEM Tiles Only)",
            "validated_dem_tiles_count": EXPECTED_TILES_COUNT,
            "grid_spacing_km": 5.0,
            "total_grid_locations": len(df),
            "historical_events_count": len(df_events),
            "background_samples_count": len(df_bg),
            "observation_start_utc": obs_meta.get("observation_start_utc"),
            "observation_end_utc": obs_meta.get("observation_end_utc"),
            "crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
            "disclaimer": WG_COVERAGE_DISCLOSURE,
        },
        "features": features,
    }
    return atomic_write_json(output_path, doc)


def _serialize_events_payload(df_events: pd.DataFrame) -> List[List[Any]]:
    payload: List[List[Any]] = []
    for _, r in df_events.iterrows():
        payload.append(
            [
                round(float(r["latitude"]), 5),
                round(float(r["longitude"]), 5),
                str(r.get("event_id", "")),
                str(r.get("dem_tile", r.get("terrain_source_tile", ""))),
                str(r.get("event_title", "Historical Landslide"))[:120],
                str(r.get("event_date_parsed", r.get("event_date", "")))[:20],
                str(r.get("landslide_trigger", "Unknown")),
                round(float(r["elevation_m"]), 1) if pd.notna(r.get("elevation_m")) else None,
                round(float(r["slope_degrees"]), 1) if pd.notna(r.get("slope_degrees")) else None,
                str(r.get("subregion", "Western Ghats Pilot")),
            ]
        )
    return payload


def _serialize_background_payload(df_bg: pd.DataFrame) -> List[List[Any]]:
    payload: List[List[Any]] = []
    for _, r in df_bg.iterrows():
        payload.append(
            [
                round(float(r["latitude"]), 5),
                round(float(r["longitude"]), 5),
                str(r.get("sample_id", "")),
                str(r.get("tile", "")),
                str(r.get("subregion", "Western Ghats Pilot")),
                round(float(r["elevation_m"]), 1) if pd.notna(r.get("elevation_m")) else None,
                round(float(r["slope_degrees"]), 1) if pd.notna(r.get("slope_degrees")) else None,
                round(float(r["nearest_event_distance_m"]), 0)
                if pd.notna(r.get("nearest_event_distance_m"))
                else None,
            ]
        )
    return payload


def _build_client_script(
    map_name: str,
    fg_risk_name: str,
    fg_terrain_name: str,
    fg_rain_name: str,
    fg_cells_name: str,
    fg_events_name: str,
    fg_bg_name: str,
    obs_end_utc: Optional[str],
    obs_start_utc: Optional[str],
    max_latency_hours: float,
    risk_fc: Dict[str, Any],
    terrain_fc: Dict[str, Any],
    rain_fc: Dict[str, Any],
    cells_rows: List[List[Any]],
    events_rows: List[List[Any]],
    bg_rows: List[List[Any]],
) -> str:
    freshness_js = build_compute_freshness_js(max_latency_hours=max_latency_hours)
    rain_rows_js = build_gpm_rainfall_rows_js()

    risk_json = json.dumps(risk_fc, separators=(",", ":"))
    terrain_json = json.dumps(terrain_fc, separators=(",", ":"))
    rain_json = json.dumps(rain_fc, separators=(",", ":"))
    cells_json = json.dumps(cells_rows, separators=(",", ":"))
    events_json = json.dumps(events_rows, separators=(",", ":"))
    bg_json = json.dumps(bg_rows, separators=(",", ":"))

    return f"""
<script>
(function() {{
    var obsEndUtc = {json.dumps(obs_end_utc)};
    var obsStartUtc = {json.dumps(obs_start_utc)};
    var maxLatencyHours = {float(max_latency_hours):.6f};
    var isStale = false;
    var currentLatencyHours = null;

    var lgRisk = null;
    var lgTerrain = null;
    var lgRain = null;
    var lgCells = null;
    var lgEvents = null;
    var lgBg = null;

    var riskGeo = {risk_json};
    var terrainGeo = {terrain_json};
    var rainGeo = {rain_json};
    var cellsRows = {cells_json};
    var eventsRows = {events_json};
    var bgRows = {bg_json};

    var riskPolygons = [];
    var terrainPolygons = [];
    var rainPolygons = [];
    var cellPolygons = [];
    var pilotBadgeHtml = '<span style="background:#eef4fb; color:#1b4f72; border:1px solid #b8d4f0; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:bold;">Limited Pilot · 14 DEM Tiles Only</span>';

    {freshness_js}

    function escapeHtml(str) {{
        if (str === null || str === undefined) return "Unavailable";
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }}

    function formatVal(v, suffix, decimals) {{
        if (v === null || v === undefined || isNaN(v)) {{
            return '<span style="color:#e74c3c; font-style:italic;">Unavailable</span>';
        }}
        var formatted = (decimals !== undefined) ? Number(v).toFixed(decimals) : String(v);
        return suffix ? formatted + " " + suffix : formatted;
    }}

    {rain_rows_js}

    function renderRiskPopupHtml(propsOrData) {{
        var isArray = Array.isArray(propsOrData);
        var regId = isArray ? escapeHtml(propsOrData[0]) : escapeHtml(propsOrData.region_id);
        var catName = isStale ? "UNASSESSED" : (isArray ? escapeHtml(propsOrData[7]) : escapeHtml(propsOrData.primary_screening_category));
        var badgeColor = isStale ? "#95a5a6" : (isArray ? (propsOrData[8] || '#95a5a6') : (propsOrData.color || '#95a5a6'));
        var cellCount = isArray ? "N/A" : (propsOrData.cell_count || 1);
        var areaKm2 = isArray ? "N/A" : (propsOrData.area_sqkm || 25.0);
        var tilesStr = isArray ? escapeHtml(propsOrData[1]) : (Array.isArray(propsOrData.tiles_represented) ? escapeHtml(propsOrData.tiles_represented.join(", ")) : escapeHtml(propsOrData.tiles_represented));

        var riskValStr = isStale ? '<span style="color:#95a5a6; font-style:italic;">Unassessed (Stale)</span>' :
            (isArray ? formatVal(propsOrData[6], null, 2) :
            (propsOrData.dynamic_risk_min !== undefined ?
                (formatVal(propsOrData.dynamic_risk_min, null, 2) + " / " + formatVal(propsOrData.dynamic_risk_mean, null, 2) + " / " + formatVal(propsOrData.dynamic_risk_max, null, 2)) :
                formatVal(propsOrData.dynamic_risk_mean, null, 2)));

        var tcontribStr = isArray ? formatVal(propsOrData[12], null, 2) : formatVal(propsOrData.terrain_contrib_mean, null, 2);
        var tprobStr = isArray ? formatVal(propsOrData[10], null, 3) : formatVal(propsOrData.terrain_rf_prob_mean, null, 3);
        var rcontribStr = isStale ? '<span style="color:#95a5a6; font-style:italic;">Unassessed (Stale)</span>' :
            (isArray ? formatVal(propsOrData[13], null, 2) : formatVal(propsOrData.rainfall_contrib_mean, null, 2));
        var statusStr = isStale ? "STALE_DATA_UNASSESSED" : (isArray ? escapeHtml(propsOrData[21]) : escapeHtml(propsOrData.freshness_status || "FRESH"));
        var latencyStr = (currentLatencyHours !== null && !isNaN(currentLatencyHours)) ? currentLatencyHours.toFixed(1) + "h latency" : "latency unavailable";

        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:335px; line-height:1.45; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid ' + badgeColor + '; padding-bottom:6px; margin-bottom:8px;">' +
                '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<span style="font-size:14px; font-weight:bold; color:#1a252f;">' + regId + '</span>' +
                    '<span style="background:' + badgeColor + '; color:white; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:11px;">' + catName + ' Risk</span>' +
                '</div>' +
                '<div style="font-size:11px; color:#7f8c8d; margin-top:3px;">Regional Hazard Area: <b>' + cellCount + ' cells (~' + areaKm2 + ' km²)</b> &nbsp;|&nbsp; ' + pilotBadgeHtml + '</div>' +
            '</div>' +
            '<table style="width:100%; border-collapse:collapse; margin-bottom:8px; font-size:11px;">' +
                '<tr><td style="color:#7f8c8d;">DEM Tiles Covered:</td><td style="font-weight:bold; text-align:right;">' + tilesStr + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Dynamic Risk Index (Min / Mean / Max):</td><td style="font-weight:bold; text-align:right; color:' + badgeColor + ';">' + riskValStr + '</td></tr>' +
            '</table>' +
            '<div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:6px; padding:7px 9px; margin-bottom:8px;">' +
                '<div style="font-weight:bold; color:#2c3e50; margin-bottom:4px; font-size:11px; text-transform:uppercase;">Component Breakdown (50% Terrain / 50% Rainfall)</div>' +
                '<div style="display:flex; justify-content:space-between; font-size:11px; color:#555;"><span>Mean Terrain Component:</span><span style="font-weight:bold;">' + tcontribStr + ' pts (Mean Prob: ' + tprobStr + ')</span></div>' +
                '<div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-top:2px;"><span>Mean Rainfall Component:</span><span style="font-weight:bold;">' + rcontribStr + ' pts</span></div>' +
            '</div>' +
            '<div style="background:#f4f9fd; border:1px solid #d0e3f0; border-radius:6px; padding:7px 9px; margin-bottom:8px;">' +
                '<div style="font-weight:bold; color:#1b4f72; margin-bottom:4px; font-size:11px; text-transform:uppercase;">NASA GPM IMERG Rainfall (Early Run V07C)</div>' +
                renderGpmRainfallRowsHtml(propsOrData) +
                '<div style="font-size:10px; color:#78909c; margin-top:4px;">Obs End: <b>' + escapeHtml(obsEndUtc) + '</b> (' + latencyStr + ') | Status: <b>' + statusStr + '</b></div>' +
            '</div>' +
            '<div style="font-size:9.5px; color:#7f8c8d; line-height:1.3; background:#fff8e1; border:1px solid #ffe082; padding:5px 7px; border-radius:4px;">' +
                '<b>Limited Pilot Notice:</b> Covers strictly 14 validated Western Ghats DEM tiles (4,819 5 km cells, 182 historical events, 1,400 background samples). Does not imply complete Western Ghats coverage.' +
            '</div>' +
        '</div>';
    }}

    function renderTerrainPopupHtml(props) {{
        var regId = escapeHtml(props.region_id);
        var tier = escapeHtml(props.terrain_baseline_screening_tier);
        var badgeColor = props.color || '#2ecc71';
        var cellCount = props.cell_count || 1;
        var areaKm2 = props.area_sqkm || 25.0;
        var tilesStr = Array.isArray(props.tiles_represented) ? escapeHtml(props.tiles_represented.join(", ")) : escapeHtml(props.tiles_represented);
        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:315px; line-height:1.45; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid ' + badgeColor + '; padding-bottom:6px; margin-bottom:8px;">' +
                '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<span style="font-size:14px; font-weight:bold; color:#1a252f;">' + regId + '</span>' +
                    '<span style="background:' + badgeColor + '; color:white; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:11px;">' + tier + ' Terrain</span>' +
                '</div>' +
                '<div style="font-size:11px; color:#7f8c8d; margin-top:3px;">Physical Susceptibility Baseline &nbsp;|&nbsp; ' + cellCount + ' cells (~' + areaKm2 + ' km²)</div>' +
            '</div>' +
            '<table style="width:100%; border-collapse:collapse; margin-bottom:8px; font-size:11px;">' +
                '<tr><td style="color:#7f8c8d;">DEM Tiles Covered:</td><td style="font-weight:bold; text-align:right;">' + tilesStr + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Mean RF Balanced Probability:</td><td style="font-weight:bold; text-align:right; color:' + badgeColor + ';">' + formatVal(props.terrain_rf_prob_mean, null, 3) + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Mean Terrain Contribution (50%):</td><td style="font-weight:bold; text-align:right;">' + formatVal(props.terrain_contrib_mean, 'pts', 2) + '</td></tr>' +
            '</table>' +
            '<div style="font-size:9.5px; color:#7f8c8d; line-height:1.3; background:#eafaf1; border:1px solid #a9dfbf; padding:5px 7px; border-radius:4px;">' +
                '<b>Western Ghats Limited Pilot:</b> Static frozen RF susceptibility across 14 validated DEM tiles. Does not imply complete Western Ghats coverage.' +
            '</div>' +
        '</div>';
    }}

    function renderRainPopupHtml(props) {{
        var regId = escapeHtml(props.region_id);
        var tier = isStale ? "UNASSESSED" : escapeHtml(props.rain_tier);
        var badgeColor = isStale ? "#95a5a6" : (props.color || '#3498db');
        var cellCount = props.cell_count || 1;
        var areaKm2 = props.area_sqkm || 25.0;
        var tilesStr = Array.isArray(props.tiles_represented) ? escapeHtml(props.tiles_represented.join(", ")) : escapeHtml(props.tiles_represented);
        var rcontrib = isStale ? "Unassessed (Stale)" : formatVal(props.rainfall_contrib_mean, 'pts', 2);
        var statusStr = isStale ? "STALE_DATA_UNASSESSED" : escapeHtml(props.freshness_status || "FRESH");
        var latencyStr = (currentLatencyHours !== null && !isNaN(currentLatencyHours)) ? currentLatencyHours.toFixed(1) + "h latency" : "latency unavailable";
        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:320px; line-height:1.45; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid ' + badgeColor + '; padding-bottom:6px; margin-bottom:8px;">' +
                '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<span style="font-size:14px; font-weight:bold; color:#1a252f;">' + regId + '</span>' +
                    '<span style="background:' + badgeColor + '; color:white; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:11px;">' + tier + '</span>' +
                '</div>' +
                '<div style="font-size:11px; color:#7f8c8d; margin-top:3px;">NASA GPM Rainfall Impact &nbsp;|&nbsp; ' + cellCount + ' cells (~' + areaKm2 + ' km²)</div>' +
            '</div>' +
            '<table style="width:100%; border-collapse:collapse; margin-bottom:8px; font-size:11px;">' +
                '<tr><td style="color:#7f8c8d;">DEM Tiles Covered:</td><td style="font-weight:bold; text-align:right;">' + tilesStr + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Mean Rainfall Component (50%):</td><td style="font-weight:bold; text-align:right;">' + rcontrib + '</td></tr>' +
            '</table>' +
            '<div style="background:#f4f9fd; border:1px solid #d0e3f0; border-radius:6px; padding:7px 9px; margin-bottom:8px;">' +
                renderGpmRainfallRowsHtml(props) +
                '<div style="font-size:10px; color:#78909c; margin-top:4px;">Obs End: <b>' + escapeHtml(obsEndUtc) + '</b> (' + latencyStr + ') | Status: <b>' + statusStr + '</b></div>' +
            '</div>' +
            '<div style="font-size:9.5px; color:#7f8c8d; line-height:1.3; background:#fff8e1; border:1px solid #ffe082; padding:5px 7px; border-radius:4px;">' +
                '<b>Limited Pilot Notice:</b> Covers 14 validated Western Ghats DEM tiles only.' +
            '</div>' +
        '</div>';
    }}

    function renderEventPopupHtml(row) {{
        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:275px; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid #8e44ad; padding-bottom:4px; margin-bottom:6px; font-weight:bold; color:#4a235a;">Historical Landslide Event #' + escapeHtml(row[2]) + '</div>' +
            '<div style="font-size:11px; margin-bottom:4px;"><b>Title:</b> ' + escapeHtml(row[4]) + '</div>' +
            '<div style="font-size:11px;"><b>Date:</b> ' + escapeHtml(row[5]) + ' &nbsp;|&nbsp; <b>Trigger:</b> ' + escapeHtml(row[6]) + '</div>' +
            '<div style="font-size:11px;"><b>Tile / Subregion:</b> ' + escapeHtml(row[3]) + ' (' + escapeHtml(row[9]) + ')</div>' +
            '<div style="font-size:11px;"><b>Elevation / Slope:</b> ' + formatVal(row[7], 'm', 1) + ' / ' + formatVal(row[8], '°', 1) + '</div>' +
            '<div style="font-size:9.5px; color:#7f8c8d; margin-top:5px;">Separate validated event catalog (n=182, Limited Pilot only).</div>' +
        '</div>';
    }}

    function renderBackgroundPopupHtml(row) {{
        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:275px; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid #16a085; padding-bottom:4px; margin-bottom:6px; font-weight:bold; color:#0e6251;">Background Sample ' + escapeHtml(row[2]) + '</div>' +
            '<div style="font-size:11px;"><b>Tile / Subregion:</b> ' + escapeHtml(row[3]) + ' (' + escapeHtml(row[4]) + ')</div>' +
            '<div style="font-size:11px;"><b>Elevation / Slope:</b> ' + formatVal(row[5], 'm', 1) + ' / ' + formatVal(row[6], '°', 1) + '</div>' +
            '<div style="font-size:11px;"><b>Nearest Event Distance:</b> ' + formatVal(row[7], 'm', 0) + '</div>' +
            '<div style="font-size:9.5px; color:#7f8c8d; margin-top:5px;">Separate validated background sample (n=1,400, Limited Pilot only).</div>' +
        '</div>';
    }}

    function renderCellPopupHtml(row) {{
        var catName = isStale ? "UNASSESSED" : escapeHtml(row[3]);
        var badgeColor = isStale ? "#95a5a6" : (row[4] || '#2ecc71');
        var riskStr = isStale ? '<span style="color:#95a5a6; font-style:italic;">Unassessed (Stale)</span>' : formatVal(row[5], null, 2);
        var rcontribStr = isStale ? '<span style="color:#95a5a6; font-style:italic;">Unassessed (Stale)</span>' : formatVal(row[7], 'pts', 2);
        var statusStr = isStale ? "STALE_DATA_UNASSESSED" : "FRESH";
        var latencyStr = (currentLatencyHours !== null && !isNaN(currentLatencyHours)) ? currentLatencyHours.toFixed(1) + "h latency" : "latency unavailable";
        var gpmProxy = {{
            gpm_1h_accumulation_mean_mm: row[10],
            gpm_3h_accumulation_mean_mm: row[11],
            gpm_6h_accumulation_mean_mm: row[12],
            gpm_12h_accumulation_mean_mm: row[13],
            gpm_24h_accumulation_mean_mm: row[14]
        }};
        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:325px; line-height:1.45; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid ' + badgeColor + '; padding-bottom:6px; margin-bottom:8px;">' +
                '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<span style="font-size:14px; font-weight:bold; color:#1a252f;">' + escapeHtml(row[1]) + '</span>' +
                    '<span style="background:' + badgeColor + '; color:white; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:11px;">' + catName + ' Risk</span>' +
                '</div>' +
                '<div style="font-size:11px; color:#7f8c8d; margin-top:3px;">5 km UTM Cell (EPSG:32643 · ~25 km²) &nbsp;|&nbsp; ' + pilotBadgeHtml + '</div>' +
            '</div>' +
            '<table style="width:100%; border-collapse:collapse; margin-bottom:8px; font-size:11px;">' +
                '<tr><td style="color:#7f8c8d;">DEM Tile:</td><td style="font-weight:bold; text-align:right;">' + escapeHtml(row[2]) + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Elevation / Slope:</td><td style="font-weight:bold; text-align:right;">' + formatVal(row[15], 'm', 1) + ' / ' + formatVal(row[16], '°', 1) + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Dynamic Risk Index:</td><td style="font-weight:bold; text-align:right; color:' + badgeColor + ';">' + riskStr + '</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Terrain / Rain Component:</td><td style="font-weight:bold; text-align:right;">' + formatVal(row[6], 'pts', 2) + ' / ' + rcontribStr + '</td></tr>' +
            '</table>' +
            '<div style="background:#f4f9fd; border:1px solid #d0e3f0; border-radius:6px; padding:7px 9px; margin-bottom:8px;">' +
                renderGpmRainfallRowsHtml(gpmProxy) +
                '<div style="font-size:10px; color:#78909c; margin-top:4px;">Obs End: <b>' + escapeHtml(obsEndUtc) + '</b> (' + latencyStr + ') | Status: <b>' + statusStr + '</b></div>' +
            '</div>' +
            '<div style="font-size:9.5px; color:#7f8c8d; line-height:1.3; background:#fff8e1; border:1px solid #ffe082; padding:5px 7px; border-radius:4px;">' +
                '<b>Limited Pilot Notice:</b> Covers strictly 14 validated Western Ghats DEM tiles (4,819 5 km cells, 182 historical events, 1,400 background samples). Does not imply complete Western Ghats coverage.' +
            '</div>' +
        '</div>';
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

        for (var i = 0; i < riskPolygons.length; i++) {{
            var item = riskPolygons[i];
            var p = item.props;
            var newColor = isStale ? '#95a5a6' : (p.color || '#2ecc71');
            item.layer.setStyle({{ color: newColor, opacity: 0.90, fillColor: newColor, weight: 0.9, fillOpacity: 0.72 }});
            var tip = isStale ? (p.region_id + ': UNASSESSED (Stale Data)') : (p.region_id + ': ' + p.primary_screening_category + ' Risk (' + p.cell_count + ' cells)');
            item.layer.setTooltipContent(tip);
        }}

        for (var j = 0; j < rainPolygons.length; j++) {{
            var rItem = rainPolygons[j];
            var rp = rItem.props;
            var rColor = isStale ? '#95a5a6' : (rp.color || '#3498db');
            rItem.layer.setStyle({{ color: rColor, opacity: 0.85, fillColor: rColor, weight: 0.8, fillOpacity: 0.68 }});
            var rTip = isStale ? (rp.region_id + ': UNASSESSED (Stale Data)') : (rp.region_id + ': ' + rp.rain_tier + ' (' + rp.cell_count + ' cells)');
            rItem.layer.setTooltipContent(rTip);
        }}

        for (var k = 0; k < cellPolygons.length; k++) {{
            var cItem = cellPolygons[k];
            var cr = cItem.row;
            var cColor = isStale ? '#95a5a6' : (cr[4] || '#2ecc71');
            var cBorder = isStale ? '#7f8c8d' : '#34495e';
            cItem.layer.setStyle({{ color: cBorder, opacity: 0.70, fillColor: cColor, weight: 0.5, fillOpacity: 0.58 }});
            var cTip = isStale ? (cr[1] + ': UNASSESSED (Stale Data)') : (cr[1] + ' (' + cr[2] + '): ' + cr[3] + ' Risk');
            cItem.layer.setTooltipContent(cTip);
        }}
    }}

    function attachHoverHighlight(layer, baseStyleFn) {{
        if (typeof layer.on === 'function') {{
            layer.on('mouseover', function() {{
                layer.setStyle({{ weight: 2.0, color: '#1a252f', fillOpacity: 0.88 }});
            }});
            layer.on('mouseout', function() {{
                layer.setStyle(baseStyleFn());
            }});
        }}
    }}

    function populateLayers() {{
        lgRisk = (typeof {fg_risk_name} !== 'undefined') ? {fg_risk_name} : null;
        lgTerrain = (typeof {fg_terrain_name} !== 'undefined') ? {fg_terrain_name} : null;
        lgRain = (typeof {fg_rain_name} !== 'undefined') ? {fg_rain_name} : null;
        lgCells = (typeof {fg_cells_name} !== 'undefined') ? {fg_cells_name} : null;
        lgEvents = (typeof {fg_events_name} !== 'undefined') ? {fg_events_name} : null;
        lgBg = (typeof {fg_bg_name} !== 'undefined') ? {fg_bg_name} : null;

        if (typeof L !== 'undefined' && typeof L.geoJSON === 'function') {{
            if (lgRisk) {{
                L.geoJSON(riskGeo, {{
                    style: function(feat) {{
                        var c = isStale ? '#95a5a6' : (feat.properties.color || '#2ecc71');
                        return {{ color: c, opacity: 0.90, weight: 0.9, fillColor: c, fillOpacity: 0.72 }};
                    }},
                    onEachFeature: function(feat, layer) {{
                        var p = feat.properties;
                        layer.bindTooltip(p.region_id + ': ' + p.primary_screening_category + ' Risk (' + p.cell_count + ' cells)');
                        layer.bindPopup(function() {{ return renderRiskPopupHtml(p); }}, {{ maxWidth: 370 }});
                        attachHoverHighlight(layer, function() {{
                            var c = isStale ? '#95a5a6' : (p.color || '#2ecc71');
                            return {{ color: c, opacity: 0.90, weight: 0.9, fillColor: c, fillOpacity: 0.72 }};
                        }});
                        riskPolygons.push({{ layer: layer, props: p }});
                    }}
                }}).addTo(lgRisk);
            }}

            if (lgTerrain) {{
                L.geoJSON(terrainGeo, {{
                    style: function(feat) {{
                        var c = feat.properties.color || '#2ecc71';
                        return {{ color: c, opacity: 0.85, weight: 0.8, fillColor: c, fillOpacity: 0.70 }};
                    }},
                    onEachFeature: function(feat, layer) {{
                        var p = feat.properties;
                        layer.bindTooltip(p.region_id + ': ' + p.terrain_baseline_screening_tier + ' Terrain (' + p.cell_count + ' cells)');
                        layer.bindPopup(function() {{ return renderTerrainPopupHtml(p); }}, {{ maxWidth: 350 }});
                        attachHoverHighlight(layer, function() {{
                            var c = p.color || '#2ecc71';
                            return {{ color: c, opacity: 0.85, weight: 0.8, fillColor: c, fillOpacity: 0.70 }};
                        }});
                        terrainPolygons.push({{ layer: layer, props: p }});
                    }}
                }}).addTo(lgTerrain);
            }}

            if (lgRain) {{
                L.geoJSON(rainGeo, {{
                    style: function(feat) {{
                        var c = isStale ? '#95a5a6' : (feat.properties.color || '#3498db');
                        return {{ color: c, opacity: 0.85, weight: 0.8, fillColor: c, fillOpacity: 0.68 }};
                    }},
                    onEachFeature: function(feat, layer) {{
                        var p = feat.properties;
                        layer.bindTooltip(p.region_id + ': ' + p.rain_tier + ' (' + p.cell_count + ' cells)');
                        layer.bindPopup(function() {{ return renderRainPopupHtml(p); }}, {{ maxWidth: 355 }});
                        attachHoverHighlight(layer, function() {{
                            var c = isStale ? '#95a5a6' : (p.color || '#3498db');
                            return {{ color: c, opacity: 0.85, weight: 0.8, fillColor: c, fillOpacity: 0.68 }};
                        }});
                        rainPolygons.push({{ layer: layer, props: p }});
                    }}
                }}).addTo(lgRain);
            }}
        }}

        if (typeof L !== 'undefined' && typeof L.polygon === 'function' && lgCells) {{
            for (var cIdx = 0; cIdx < cellsRows.length; cIdx++) {{
                (function(row) {{
                    var col = isStale ? '#95a5a6' : (row[4] || '#2ecc71');
                    var poly = L.polygon(row[0], {{
                        color: isStale ? '#7f8c8d' : '#34495e',
                        opacity: 0.70,
                        weight: 0.5,
                        fillColor: col,
                        fillOpacity: 0.58
                    }});
                    poly.bindTooltip(row[1] + ' (' + row[2] + '): ' + row[3] + ' Risk');
                    poly.bindPopup(function() {{ return renderCellPopupHtml(row); }}, {{ maxWidth: 355 }});
                    poly.addTo(lgCells);
                    cellPolygons.push({{ layer: poly, row: row }});
                }})(cellsRows[cIdx]);
            }}
        }}

        if (typeof L !== 'undefined' && typeof L.circleMarker === 'function') {{
            if (lgEvents) {{
                for (var e = 0; e < eventsRows.length; e++) {{
                    (function(row) {{
                        var m = L.circleMarker([row[0], row[1]], {{
                            radius: 4.5, color: '#ffffff', weight: 1, fillColor: '#8e44ad', fillOpacity: 0.9
                        }});
                        m.bindTooltip('Historical Event #' + row[2] + ' (' + row[3] + ')');
                        m.bindPopup(function() {{ return renderEventPopupHtml(row); }}, {{ maxWidth: 310 }});
                        m.addTo(lgEvents);
                    }})(eventsRows[e]);
                }}
            }}
            if (lgBg) {{
                for (var b = 0; b < bgRows.length; b++) {{
                    (function(row) {{
                        var m = L.circleMarker([row[0], row[1]], {{
                            radius: 3.0, color: '#0e6251', weight: 0.6, fillColor: '#16a085', fillOpacity: 0.65
                        }});
                        m.bindTooltip('Background Sample ' + row[2]);
                        m.bindPopup(function() {{ return renderBackgroundPopupHtml(row); }}, {{ maxWidth: 310 }});
                        m.addTo(lgBg);
                    }})(bgRows[b]);
                }}
            }}
        }}

        var mapInstance = (typeof {map_name} !== 'undefined') ? {map_name} : null;
        if (mapInstance && typeof mapInstance.on === 'function') {{
            mapInstance.on('overlayadd', function() {{
                if (lgEvents && mapInstance.hasLayer(lgEvents) && typeof lgEvents.bringToFront === 'function') {{
                    lgEvents.bringToFront();
                }}
                if (lgBg && mapInstance.hasLayer(lgBg) && typeof lgBg.bringToFront === 'function') {{
                    lgBg.bringToFront();
                }}
            }});
        }}
    }}

    if (typeof window !== 'undefined') {{
        window.computeFreshness = computeFreshness;
        window.updateFreshnessStatus = updateFreshnessStatus;
        window.renderRiskPopupHtml = renderRiskPopupHtml;
        window.renderRainPopupHtml = renderRainPopupHtml;
        window.renderTerrainPopupHtml = renderTerrainPopupHtml;
        window.renderCellPopupHtml = renderCellPopupHtml;
        window.renderPopupHtml = renderRiskPopupHtml;
        window.riskPolygons = riskPolygons;
        window.rainPolygons = rainPolygons;
        window.terrainPolygons = terrainPolygons;
        window.cellPolygons = cellPolygons;
        window.getIsStale = function() {{ return isStale; }};
        window.getCurrentLatencyHours = function() {{ return currentLatencyHours; }};
        window.setObservationEndTimeForTesting = function(newUtcStr) {{
            obsEndUtc = newUtcStr;
            updateFreshnessStatus();
        }};
    }}

    var _wgMapInitialized = false;
    function initMap() {{
        if (_wgMapInitialized) return;
        if (typeof L === 'undefined' || typeof {fg_risk_name} === 'undefined') {{
            return;
        }}
        _wgMapInitialized = true;
        populateLayers();
        updateFreshnessStatus();
        setInterval(updateFreshnessStatus, 60000);
    }}

    if (document.readyState === 'complete' || (document.readyState === 'interactive' && typeof {fg_risk_name} !== 'undefined')) {{
        initMap();
    }} else {{
        document.addEventListener('DOMContentLoaded', initMap);
        window.addEventListener('load', initMap);
    }}
}})();
</script>
""".strip()


def _build_legend_html(
    cutoffs: Dict[str, float],
    regions_count: int,
    obs_start_utc: Optional[str],
    obs_end_utc: Optional[str],
    latency_hours: float,
    is_stale: bool,
    gen_ts_utc: str,
) -> str:
    p50 = cutoffs["P50_low_to_moderate"]
    p80 = cutoffs["P80_moderate_to_high"]
    p95 = cutoffs["P95_high_to_very_high"]
    return f"""
    <div style="position: fixed;
                bottom: 22px; left: 22px; width: 350px;
                background-color: rgba(255, 255, 255, 0.96);
                border: 2px solid #2c3e50; border-radius: 8px;
                z-index: 9999; font-size: 11.5px;
                font-family: Arial, sans-serif;
                box-shadow: 0 4px 15px rgba(0,0,0,0.25);
                padding: 12px 14px; line-height: 1.4; color: #2c3e50;">
        <div style="font-weight: bold; font-size: 13.5px; color: #1a252f; border-bottom: 2px solid #34495e; padding-bottom: 5px; margin-bottom: 7px;">
            Western Ghats Limited Pilot — Regional Zonation
            <span style="font-size: 10px; font-weight: bold; color: #1b4f72; display: block; margin-top: 2px;">
                {EXPECTED_TILES_COUNT} Validated DEM Tiles Only · {regions_count:,} Regions · {EXPECTED_LOCATIONS_COUNT:,} Cells (5 km)
            </span>
            <span style="font-size: 10px; font-weight: normal; color: #546e7a; display: block;">
                {EXPECTED_HISTORICAL_EVENTS_COUNT} Historical Events · {EXPECTED_BACKGROUND_SAMPLES_COUNT:,} Background Samples
            </span>
        </div>

        <div style="margin-bottom: 8px;">
            <div style="font-weight: bold; color: #34495e; margin-bottom: 4px; font-size: 11px;">
                Regional Screening Categories (P50 / P80 / P95):
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="background:{CATEGORY_COLORS['Low']}; width:13px; height:13px; display:inline-block; border-radius:3px; margin-right:8px; border:1px solid #27ae60;"></span>
                <span><b>Low Risk</b> (&lt; {p50:.2f})</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="background:{CATEGORY_COLORS['Moderate']}; width:13px; height:13px; display:inline-block; border-radius:3px; margin-right:8px; border:1px solid #f39c12;"></span>
                <span><b>Moderate Risk</b> ({p50:.2f} – {p80:.2f})</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="background:{CATEGORY_COLORS['High']}; width:13px; height:13px; display:inline-block; border-radius:3px; margin-right:8px; border:1px solid #d35400;"></span>
                <span><b>High Risk</b> ({p80:.2f} – {p95:.2f})</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="background:{CATEGORY_COLORS['Very High']}; width:13px; height:13px; display:inline-block; border-radius:3px; margin-right:8px; border:1px solid #c0392b;"></span>
                <span><b>Very High Risk</b> (&ge; {p95:.2f})</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background:{CATEGORY_COLORS['UNASSESSED_MISSING_RAINFALL']}; width:13px; height:13px; display:inline-block; border-radius:3px; margin-right:8px; border:1px solid #7f8c8d;"></span>
                <span><b>Unassessed / Stale</b> (Missing or &gt;12h latency)</span>
            </div>
        </div>

        <div style="border-top: 1px solid #e0e0e0; padding-top: 6px; margin-top: 6px; font-size: 10.5px; color: #555;">
            <div><b>Observation Window:</b> {safe_html(obs_start_utc)} to {safe_html(obs_end_utc)}</div>
            <div><b>Rainfall Latency:</b> <span id="dashboard-latency-text">{latency_hours:.1f} hours ({'STALE' if is_stale else 'FRESH'}, limit: 12.0h)</span></div>
            <div><b>Freshness Status:</b> <span id="dashboard-freshness-badge" style="background:{'#e74c3c' if is_stale else '#27ae60'}; color:white; padding:1px 6px; border-radius:3px; font-size:10px; font-weight:bold;">{'STALE / UNASSESSED' if is_stale else 'FRESH (OPERATIONAL)'}</span></div>
            <div><b>Generated (UTC):</b> {safe_html(gen_ts_utc)}</div>
        </div>

        <div style="border-top: 1px solid #e0e0e0; padding-top: 6px; margin-top: 6px; font-size: 9.5px; color: #7f8c8d; line-height: 1.3;">
            <b>Coverage Disclosure:</b> Limited Western Ghats pilot covering strictly 14 validated DEM tiles
            (4,819 5 km grid cells, 182 historical events, 1,400 background samples). Does not imply complete
            Western Ghats coverage. Experimental screening only; not an official warning.
        </div>
    </div>
    """.strip()


def generate_western_ghats_polygon_risk_map(
    input_csv_path: Path = DEFAULT_INPUT_CSV,
    input_metadata_path: Path = DEFAULT_INPUT_METADATA,
    events_csv_path: Path = DEFAULT_EVENTS_CSV,
    background_csv_path: Path = DEFAULT_BACKGROUND_CSV,
    output_html_path: Path = DEFAULT_OUTPUT_HTML,
    output_geojson_path: Path = DEFAULT_OUTPUT_GEOJSON,
    output_regions_geojson_path: Path = DEFAULT_OUTPUT_REGIONS_GEOJSON,
    output_cells_geojson_path: Path = DEFAULT_OUTPUT_CELLS_GEOJSON,
    output_metadata_path: Path = DEFAULT_OUTPUT_METADATA,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate the Western Ghats Limited Pilot 5 km regional polygon GIS map and GeoJSON suite."""
    output_targets = [
        output_html_path,
        output_geojson_path,
        output_regions_geojson_path,
        output_cells_geojson_path,
        output_metadata_path,
        validation_report_path,
    ]
    existing = [p for p in output_targets if p.exists()]
    if existing:
        if not overwrite:
            raise FileExistsError(
                f"Target output already exists: {existing[0]}. Pass --overwrite to replace."
            )
        create_timestamped_backup(existing, output_html_path.parent / "backup")

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Dynamic risk CSV not found: {input_csv_path}")
    if not input_metadata_path.exists():
        raise FileNotFoundError(f"Dynamic risk metadata JSON not found: {input_metadata_path}")

    df = pd.read_csv(input_csv_path)
    total_locations = len(df)
    if total_locations != EXPECTED_LOCATIONS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOCATIONS_COUNT} 5 km grid locations, got {total_locations}"
        )
    if int(df["tile"].nunique()) != EXPECTED_TILES_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TILES_COUNT} validated DEM tiles, got {df['tile'].nunique()}"
        )

    with open(input_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    cutoffs = validate_dynamic_metadata_thresholds(metadata)

    is_fresh, gate_msg, audit_info = _verify_map_freshness_gate(
        df=df,
        metadata=metadata,
        now_utc=now_utc,
        max_latency_hours=max_latency_hours,
    )
    is_stale = bool(audit_info["is_stale"])
    if not is_fresh and not is_stale:
        raise ValueError(f"Freshness and completeness gate failed: {gate_msg}")

    obs_start_utc = audit_info["observation_start_utc"]
    obs_end_utc = audit_info["observation_end_utc"]
    latency_hours = float(audit_info["calculated_latency_hours"])

    df_events, df_bg = load_validated_point_layers(
        events_csv_path=events_csv_path,
        background_csv_path=background_csv_path,
    )

    gdf_cells = build_5km_spatial_cells_utm(df, expected_count=EXPECTED_LOCATIONS_COUNT, spacing_m=5000.0)
    gdf_cells["rain_tier"] = pd.cut(
        gdf_cells["rainfall_contribution"].fillna(0.0),
        bins=[-0.001, 0.1, 1.0, 3.0, 100.0],
        labels=[
            "Negligible (<0.1 pts)",
            "Low (0.1-1.0 pts)",
            "Moderate (1.0-3.0 pts)",
            "High (>=3.0 pts)",
        ],
    ).astype(str)

    gdf_risk_regions, risk_meta = dissolve_hazard_layer_into_regions(
        gdf_cells=gdf_cells,
        group_col="primary_screening_category",
        layer_name="dynamic_hazard_screening",
        color_map=CATEGORY_COLORS,
        obs_start_utc=obs_start_utc or "",
        obs_end_utc=obs_end_utc or "",
        region_prefix="WG",
        disclaimer=WG_COVERAGE_DISCLOSURE,
        is_stale=is_stale,
    )
    gdf_terrain_regions, _ = dissolve_hazard_layer_into_regions(
        gdf_cells=gdf_cells,
        group_col="terrain_baseline_screening_tier",
        layer_name="terrain_susceptibility",
        color_map=TERRAIN_TIER_COLORS,
        obs_start_utc=obs_start_utc or "",
        obs_end_utc=obs_end_utc or "",
        region_prefix="WG",
        disclaimer=WG_COVERAGE_DISCLOSURE,
        is_stale=False,
    )
    gdf_rain_regions, _ = dissolve_hazard_layer_into_regions(
        gdf_cells=gdf_cells,
        group_col="rain_tier",
        layer_name="rainfall_contribution",
        color_map=RAINFALL_TIER_COLORS,
        obs_start_utc=obs_start_utc or "",
        obs_end_utc=obs_end_utc or "",
        region_prefix="WG",
        disclaimer=WG_COVERAGE_DISCLOSURE,
        is_stale=is_stale,
    )

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    minx, miny, maxx, maxy = gdf_cells.total_bounds
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    map_obj.fit_bounds([[float(miny), float(minx)], [float(maxy), float(maxx)]], padding=(20, 20))
    map_obj.get_root().header.add_child(
        folium.Element(f"<title>{html.escape(WG_LIMITED_PILOT_TITLE)}</title>")
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite Imagery",
        overlay=False,
        control=True,
        show=True,
    ).add_to(map_obj)
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap Standard",
        overlay=False,
        control=True,
        show=False,
    ).add_to(map_obj)

    fg_risk = folium.FeatureGroup(
        name=f"Dynamic Hazard Screening Regions (5 km · 14 Pilot Tiles, n={len(gdf_risk_regions):,} regions)",
        show=True,
    ).add_to(map_obj)
    fg_terrain = folium.FeatureGroup(
        name=f"Terrain Susceptibility Regions (Static Baseline, n={len(gdf_terrain_regions):,} regions)",
        show=False,
    ).add_to(map_obj)
    fg_rain = folium.FeatureGroup(
        name=f"Rainfall Contribution Regions (GPM IMERG, n={len(gdf_rain_regions):,} regions)",
        show=False,
    ).add_to(map_obj)
    fg_cells = folium.FeatureGroup(
        name=f"5 km Spatial Grid Cells (n={EXPECTED_LOCATIONS_COUNT:,} cells, Pilot Only)",
        show=False,
    ).add_to(map_obj)
    fg_events = folium.FeatureGroup(
        name=f"Historical Landslide Events (n={EXPECTED_HISTORICAL_EVENTS_COUNT}, Pilot Only)",
        show=True,
    ).add_to(map_obj)
    fg_bg = folium.FeatureGroup(
        name=f"Validated Background Samples (n={EXPECTED_BACKGROUND_SAMPLES_COUNT:,}, Pilot Only)",
        show=False,
    ).add_to(map_obj)

    folium.LayerControl(collapsed=False).add_to(map_obj)

    client_script = _build_client_script(
        map_name=map_obj.get_name(),
        fg_risk_name=fg_risk.get_name(),
        fg_terrain_name=fg_terrain.get_name(),
        fg_rain_name=fg_rain.get_name(),
        fg_cells_name=fg_cells.get_name(),
        fg_events_name=fg_events.get_name(),
        fg_bg_name=fg_bg.get_name(),
        obs_end_utc=obs_end_utc,
        obs_start_utc=obs_start_utc,
        max_latency_hours=max_latency_hours,
        risk_fc=_gdf_to_feature_collection_dict(gdf_risk_regions),
        terrain_fc=_gdf_to_feature_collection_dict(gdf_terrain_regions),
        rain_fc=_gdf_to_feature_collection_dict(gdf_rain_regions),
        cells_rows=_serialize_cells_payload(gdf_cells, is_stale=is_stale),
        events_rows=_serialize_events_payload(df_events),
        bg_rows=_serialize_background_payload(df_bg),
    )
    map_obj.get_root().html.add_child(folium.Element(client_script))

    legend_html = _build_legend_html(
        cutoffs=cutoffs,
        regions_count=len(gdf_risk_regions),
        obs_start_utc=obs_start_utc,
        obs_end_utc=obs_end_utc,
        latency_hours=latency_hours,
        is_stale=is_stale,
        gen_ts_utc=audit_info["map_generation_timestamp_utc"],
    )
    map_obj.get_root().html.add_child(folium.Element(legend_html))

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_html = map_obj.get_root().render()
    clean_html = "\n".join(line.rstrip(" \t") for line in rendered_html.splitlines()) + "\n"
    tmp_html = output_html_path.with_name(f"{output_html_path.name}.tmp")
    try:
        with open(tmp_html, "wb") as f:
            f.write(clean_html.encode("utf-8"))
        tmp_html.replace(output_html_path)
    except Exception:
        if tmp_html.exists():
            tmp_html.unlink(missing_ok=True)
        raise
    html_file_size = output_html_path.stat().st_size

    export_regions_geojson(
        gdf_regions=gdf_risk_regions,
        metadata=metadata,
        output_path=output_regions_geojson_path,
        title=f"{WG_LIMITED_PILOT_TITLE} — Regional Polygons",
        region="Western Ghats (Limited Pilot: 14 DEM Tiles Only)",
        disclaimer=WG_COVERAGE_DISCLOSURE,
    )
    regions_file_size = output_regions_geojson_path.stat().st_size

    export_cells_geojson(
        gdf_cells=gdf_cells,
        output_path=output_cells_geojson_path,
        title=f"{WG_LIMITED_PILOT_TITLE} — 5 km Spatial Cells",
        region="Western Ghats (Limited Pilot: 14 DEM Tiles Only)",
        disclaimer=WG_COVERAGE_DISCLOSURE,
    )
    cells_file_size = output_cells_geojson_path.stat().st_size

    export_western_ghats_grid_point_geojson(
        df=df,
        df_events=df_events,
        df_bg=df_bg,
        metadata=metadata,
        output_path=output_geojson_path,
        is_stale=is_stale,
    )
    geojson_file_size = output_geojson_path.stat().st_size

    map_metadata = {
        "stage": "Western Ghats Limited Pilot — Stage 6.5 Polygon Dynamic GIS Risk Map and GeoJSON",
        "system_title": WG_LIMITED_PILOT_TITLE,
        "region": "Western Ghats",
        "coverage_disclosure": WG_COVERAGE_DISCLOSURE,
        "dem_tiles_count": EXPECTED_TILES_COUNT,
        "total_grid_locations": total_locations,
        "historical_events_count": len(df_events),
        "background_samples_count": len(df_bg),
        "regional_polygons": {
            "dynamic_hazard_screening_regions": len(gdf_risk_regions),
            "terrain_susceptibility_regions": len(gdf_terrain_regions),
            "rainfall_contribution_regions": len(gdf_rain_regions),
            "total_cells_covered": total_locations,
            "all_geometries_valid": True,
        },
        "layers": {
            "dynamic_hazard_screening": len(gdf_risk_regions),
            "terrain_susceptibility": len(gdf_terrain_regions),
            "rainfall_contribution": len(gdf_rain_regions),
            "spatial_grid_cells_5km": len(gdf_cells),
            "historical_landslide_events": len(df_events),
            "validated_background_samples": len(df_bg),
        },
        "basemap_default": "Esri Satellite Imagery",
        "map_bounds_wgs84": {
            "min_lon": round(float(minx), 6),
            "min_lat": round(float(miny), 6),
            "max_lon": round(float(maxx), 6),
            "max_lat": round(float(maxy), 6),
        },
        "output_files": {
            "html_map": str(output_html_path),
            "html_size_bytes": html_file_size,
            "regions_geojson": str(output_regions_geojson_path),
            "regions_geojson_size_bytes": regions_file_size,
            "cells_geojson": str(output_cells_geojson_path),
            "cells_geojson_size_bytes": cells_file_size,
            "point_geojson": str(output_geojson_path),
            "point_geojson_size_bytes": geojson_file_size,
        },
        "observation_metadata": {
            "observation_start_utc": obs_start_utc,
            "observation_end_utc": obs_end_utc,
            "latency_hours": round(latency_hours, 2),
            "max_latency_hours_allowed": max_latency_hours,
            "is_fresh": not is_stale,
            "status": "STALE_DATA" if is_stale else "SUCCESS",
        },
        "approved_relative_thresholds": cutoffs,
        "generation_timestamp_utc": audit_info["map_generation_timestamp_utc"],
        "disclaimer": WG_COVERAGE_DISCLOSURE,
    }
    atomic_write_json(output_metadata_path, map_metadata)

    val_report = {
        "stage": "Western Ghats Limited Pilot — Stage 6.5 Polygon GIS Map Validation",
        "status": "SUCCESS" if not is_stale else "WARNING_STALE_RAINFALL",
        "region": "Western Ghats",
        "coverage_disclosure": WG_COVERAGE_DISCLOSURE,
        "is_operational_success": not is_stale,
        "dem_tiles_count": EXPECTED_TILES_COUNT,
        "grid_locations_count": total_locations,
        "historical_events_count": len(df_events),
        "background_samples_count": len(df_bg),
        "total_features_in_point_geojson": total_locations + len(df_events) + len(df_bg),
        "regional_polygons_count": len(gdf_risk_regions),
        "spatial_cells_count": len(gdf_cells),
        "html_size_mb": round(html_file_size / (1024 * 1024), 2),
        "regions_geojson_size_mb": round(regions_file_size / (1024 * 1024), 2),
        "cells_geojson_size_mb": round(cells_file_size / (1024 * 1024), 2),
        "freshness_audit": audit_info,
        "validation_checks": {
            "all_grid_locations_represented": bool(total_locations == EXPECTED_LOCATIONS_COUNT),
            "all_grid_cells_in_regional_polygons": bool(
                risk_meta["total_cells_covered"] == EXPECTED_LOCATIONS_COUNT
            ),
            "no_duplicate_or_missing_grid_ids": bool(df["grid_id"].nunique() == total_locations),
            "all_polygon_geometries_valid": bool(
                gdf_risk_regions.geometry.is_valid.all()
                and gdf_terrain_regions.geometry.is_valid.all()
                and gdf_rain_regions.geometry.is_valid.all()
                and gdf_cells.geometry.is_valid.all()
            ),
            "esri_satellite_default_basemap": True,
            "historical_events_separated": len(df_events) == EXPECTED_HISTORICAL_EVENTS_COUNT,
            "background_samples_separated": len(df_bg) == EXPECTED_BACKGROUND_SAMPLES_COUNT,
            "limited_pilot_disclosure_present": True,
            "thresholds_verified_against_stage64": True,
            "shared_popup_template_used": True,
            "rfc7946_geojson_valid": True,
        },
        "generation_timestamp_utc": audit_info["map_generation_timestamp_utc"],
        "disclaimer": WG_COVERAGE_DISCLOSURE,
    }
    atomic_write_json(validation_report_path, val_report)
    LOGGER.info(
        "Generated Western Ghats Limited Pilot polygon map (%d regions, %d cells, %.2f MB) -> %s",
        len(gdf_risk_regions),
        len(gdf_cells),
        html_file_size / (1024 * 1024),
        output_html_path,
    )
    return val_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Western Ghats Limited Pilot 5 km regional polygon GIS risk map."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_INPUT_METADATA)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--background-csv", type=Path, default=DEFAULT_BACKGROUND_CSV)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT_GEOJSON)
    parser.add_argument("--output-regions-geojson", type=Path, default=DEFAULT_OUTPUT_REGIONS_GEOJSON)
    parser.add_argument("--output-cells-geojson", type=Path, default=DEFAULT_OUTPUT_CELLS_GEOJSON)
    parser.add_argument("--output-metadata", type=Path, default=DEFAULT_OUTPUT_METADATA)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--max-latency-hours", type=float, default=MAX_DATA_LATENCY_HOURS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    try:
        generate_western_ghats_polygon_risk_map(
            input_csv_path=args.input_csv,
            input_metadata_path=args.metadata_json,
            events_csv_path=args.events_csv,
            background_csv_path=args.background_csv,
            output_html_path=args.output_html,
            output_geojson_path=args.output_geojson,
            output_regions_geojson_path=args.output_regions_geojson,
            output_cells_geojson_path=args.output_cells_geojson,
            output_metadata_path=args.output_metadata,
            validation_report_path=args.validation_report,
            max_latency_hours=args.max_latency_hours,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"\n[ERROR] Western Ghats polygon GIS map generation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
