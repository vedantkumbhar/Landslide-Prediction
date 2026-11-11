"""
Stage 6E Deliverable: Western Ghats Dynamic Landslide Hazard GIS Risk Map Generator.

Generates a standalone, compact, production-ready interactive GIS HTML map for all 1,582
validated locations across 14 DEM tiles in the Western Ghats (182 historical landslide events
+ 1,400 provisional background samples; no fabricated 5 km grid and no Western Maharashtra
pilot substitution) using approved Stage 6D dynamic landslide hazard risk predictions and
NASA GPM IMERG Early Run V07C rainfall accumulations.

Map architecture:
- Basemap: Esri World Topographic Map (default) + CartoDB Positron (optional toggle).
- 5 independently toggleable Canvas feature layers populated client-side (< 3 MB HTML):
    1. Dynamic Hazard Screening (Primary Stage 6D Category) [Active by default]
    2. GPM 24h Rainfall Accumulation (mm) [Continuous blues, default off]
    3. Unweighted Random Forest Terrain Susceptibility (Comparison Only) [Continuous green-red, default off]
    4. Historical Landslide Events (n=182) [Active by default, distinctive border]
    5. Provisional Background Samples (n=1,400) [Default off]
- Shared client-side popups explicitly labeling balanced (used in risk) vs unweighted (comparison only) models.
- Dynamic 60-second client-side browser freshness guard (12.0h limit -> STALE / UNASSESSED).
- Strict metadata-driven quantile threshold validation with zero silent fallbacks.
- Timestamped backups, atomic output writes, and synchronized GeoJSON/metadata in `outputs/`.
"""

import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

from branca.colormap import LinearColormap
import folium
import numpy as np
import pandas as pd

# =====================================================================
# PATH CONSTANTS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.export_western_ghats_dynamic_risk_geojson import (
    DEFAULT_OUTPUT_GEOJSON as DEFAULT_POINT_GEOJSON,
    export_western_ghats_dynamic_risk_geojson,
)
from backend.gis_refresh_utils import (
    atomic_copy_file,
    atomic_write_json,
    build_compute_freshness_js,
    create_timestamped_backup,
    validate_dynamic_metadata_thresholds,
)

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_dynamic_risk"
    / "western_ghats_dynamic_hazard_risk.csv"
)

DEFAULT_INPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_dynamic_risk"
    / "hazard_risk_metadata.json"
)

DEFAULT_OUTPUT_HTML = (
    PROJECT_ROOT
    / "outputs"
    / "western_ghats_dynamic_risk_map.html"
)

DEFAULT_OUTPUT_GEOJSON = (
    PROJECT_ROOT
    / "outputs"
    / "western_ghats_dynamic_hazard_risk.geojson"
)

DEFAULT_OUTPUT_META = (
    PROJECT_ROOT
    / "outputs"
    / "western_ghats_dynamic_risk_map_metadata.json"
)

DEFAULT_VALIDATION_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "western_ghats_dynamic_risk_map_validation_report.json"
)

# Approved regional scope contracts
EXPECTED_LOCATIONS_COUNT = 1582
EXPECTED_TILES_COUNT = 14
EXPECTED_HISTORICAL_EVENTS_COUNT = 182
EXPECTED_BACKGROUND_SAMPLES_COUNT = 1400

# Geographic bounds for the 14 Western Ghats DEM tiles
LAT_BOUNDS = (8.0, 22.0)
LON_BOUNDS = (70.0, 80.0)

# Freshness limit contract (12.0 hours)
MAX_DATA_LATENCY_HOURS = 12.0

# Approved color scheme for primary screening categories
CATEGORY_COLORS: Dict[str, str] = {
    "Low": "#2ecc71",                         # Green
    "Moderate": "#f1c40f",                    # Yellow
    "High": "#e67e22",                        # Orange
    "Very High": "#e74c3c",                   # Red
    "UNASSESSED_MISSING_RAINFALL": "#95a5a6", # Grey
}

# Approved sample type badge styles
SAMPLE_TYPE_BADGES: Dict[str, Tuple[str, str]] = {
    "historical_landslide": ("#c0392b", "Historical Landslide Event"),
    "provisional_background": ("#2980b9", "Provisional Background Sample"),
}

# Basemap URLs and attribution
ESRI_TOPO_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_TOPO_ATTR = "Esri, HERE, Garmin, USGS, EPA, NPS"

DISCLAIMER_TEXT = (
    "EXPERIMENTAL PROVISIONAL SCREENING ONLY · NOT AN OFFICIAL WARNING OR CALIBRATED PROBABILITY"
)


# =====================================================================
# STRICT THRESHOLD VALIDATION GATE (NO SILENT FALLBACKS)
# =====================================================================

def validate_stage6d_metadata_thresholds(metadata: Dict[str, Any]) -> Dict[str, float]:
    """
    Strictly validate Stage 6D relative category cutoffs dynamically from metadata.
    Removes all silent fallbacks and hardcoded stale equality checks:
    - Requires 'risk_formula_specification' and 'relative_category_cutoffs' dicts.
    - Requires finite numeric (non-bool, non-null) P50, P80, P95 cutoffs.
    - Enforces strict ordering: 0 < P50 < P80 < P95 < 100.
    """
    return validate_dynamic_metadata_thresholds(metadata)


# =====================================================================
# HELPER & ESCAPING FUNCTIONS
# =====================================================================

def safe_html(val: Any) -> str:
    """Escape HTML content safely for text insertion."""
    if val is None or pd.isna(val):
        return "Unavailable"
    return html.escape(str(val))


def format_rainfall_val(val: Any) -> str:
    """Format rainfall values safely, preventing NaN/inf from appearing as zero."""
    if val is None or pd.isna(val) or not np.isfinite(val):
        return "<span style='color:#e74c3c; font-style:italic;'>Unavailable</span>"
    return f"{float(val):.2f} mm"


def format_score_val(val: Any, decimals: int = 2) -> str:
    """Format numeric scores safely, rejecting NaN/inf."""
    if val is None or pd.isna(val) or not np.isfinite(val):
        return "<span style='color:#e74c3c; font-style:italic;'>Unavailable</span>"
    return f"{float(val):.{decimals}f}"


def parse_iso8601_utc(ts_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO-8601 UTC timestamp string.
    Strictly requires an explicit UTC timezone indicator ('Z' or '+00:00').
    Rejects timezone-naive timestamps and non-UTC offsets.
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
        return dt
    except (ValueError, TypeError):
        return None


# =====================================================================
# FRESHNESS & COMPLETENESS GATE
# =====================================================================

def verify_observation_freshness_and_completeness(
    metadata: Dict[str, Any],
    now_utc: Optional[datetime] = None,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    status_path: Optional[Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Independently verify observation freshness and completeness.
    - Strictly requires explicit UTC timestamp (Z or +00:00).
    - Parses observation_end_utc from metadata.
    - Rejects missing, invalid, naive, future, or older-than-12-hour observations.
    - Validates strict boolean True for 24-hour completeness.
    - An old SUCCESS JSON cannot pass indefinitely.
    """
    gpm_meta = metadata.get("gpm_observation_metadata")
    if not gpm_meta or not isinstance(gpm_meta, dict):
        return False, "Missing 'gpm_observation_metadata' in Stage 6D metadata", {}

    obs_end_raw = gpm_meta.get("observation_end_utc")
    if not obs_end_raw or not isinstance(obs_end_raw, str):
        return False, "Missing or invalid 'observation_end_utc' in metadata", gpm_meta

    obs_end_dt = parse_iso8601_utc(obs_end_raw)
    if obs_end_dt is None:
        return (
            False,
            f"Invalid or non-explicit-UTC 'observation_end_utc' timestamp format: '{obs_end_raw}' (must include 'Z' or '+00:00')",
            gpm_meta,
        )

    # Calculate latency independently against current UTC or now_utc
    ref_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if ref_now.tzinfo is None or ref_now.utcoffset() != timedelta(0):
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    actual_latency_seconds = (ref_now - obs_end_dt).total_seconds()
    actual_latency_hours = actual_latency_seconds / 3600.0

    # Strictly reject all future timestamps
    if actual_latency_seconds < 0:
        return (
            False,
            f"Observation timestamp is in the future: {obs_end_raw} (latency: {actual_latency_seconds:.1f}s)",
            gpm_meta,
        )

    # Strictly reject stale data exceeding 12 hours
    if actual_latency_hours > max_latency_hours:
        return (
            False,
            f"Rainfall data is stale: {actual_latency_hours:.2f}h exceeds limit {max_latency_hours:.1f}h (end: {obs_end_raw})",
            gpm_meta,
        )

    # Validate 24-hour completeness flag: must be strict Python boolean True
    is_complete = None
    if "is_24h_complete" in gpm_meta:
        is_complete = gpm_meta["is_24h_complete"]
    else:
        # Check source refresh status file if available
        status_file = status_path
        if status_file is None and "source_refresh_status" in metadata:
            status_file = Path(metadata["source_refresh_status"])
        if status_file and status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as sf:
                    st_data = json.load(sf)
                    is_complete = st_data.get("is_24h_complete")
            except Exception:
                pass

    if is_complete is None:
        return False, "Missing 'is_24h_complete' flag in metadata/status", gpm_meta

    if not isinstance(is_complete, bool) or is_complete is not True:
        val_type = type(is_complete).__name__
        return (
            False,
            f"Invalid completeness flag: {is_complete!r} (type: {val_type}). Must be actual boolean true",
            gpm_meta,
        )

    audit_info = {
        "observation_start_utc": gpm_meta.get("observation_start_utc"),
        "observation_end_utc": obs_end_raw,
        "calculated_latency_hours": round(actual_latency_hours, 3),
        "is_24h_complete": True,
        "freshness_status": "SUCCESS",
    }
    return True, f"Rainfall data fresh and complete ({actual_latency_hours:.2f}h latency)", audit_info


# =====================================================================
# POPUP HTML BUILDER
# =====================================================================

def build_marker_popup_html(
    row: pd.Series,
    obs_start_utc: str,
    obs_end_utc: str,
    latency_hours: float,
) -> str:
    """Build a comprehensive, styled HTML popup with strict HTML escaping and clear model labeling."""
    record_id = safe_html(row.get("record_id"))
    sample_id = safe_html(row.get("sample_id"))
    raw_sample_type = str(row.get("sample_type", "Unknown"))
    tile = safe_html(row.get("tile"))
    subregion = safe_html(row.get("subregion"))
    
    lat = float(row.get("latitude", 0.0))
    lon = float(row.get("longitude", 0.0))
    elev = format_score_val(row.get("elevation_m"), 1)
    slope = format_score_val(row.get("slope_degrees"), 2)

    # Dynamic Risk Attributes (calculated from Balanced Terrain Model)
    risk_score = format_score_val(row.get("dynamic_risk_index"), 2)
    raw_risk_val = row.get("dynamic_risk_index")
    raw_cat = str(row.get("primary_screening_category", "UNASSESSED_MISSING_RAINFALL"))

    # Map missing or non-finite risk to unassessed grey
    if raw_risk_val is None or pd.isna(raw_risk_val) or not np.isfinite(raw_risk_val):
        cat_badge_color = CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]
        cat_name = "Unassessed (Missing Data)"
    else:
        cat_badge_color = CATEGORY_COLORS.get(raw_cat, CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"])
        cat_name = safe_html(raw_cat)

    fixed_tier = safe_html(row.get("risk_tier_fixed"))

    # Baseline Terrain Model Attributes: Clearly separate balanced and unweighted values
    terrain_prob_balanced_raw = row.get("terrain_rf_prob_balanced")
    terrain_prob_balanced = format_score_val(terrain_prob_balanced_raw, 3)
    terrain_tier_balanced = safe_html(row.get("terrain_baseline_screening_tier"))

    terrain_prob_unweighted_raw = row.get("terrain_rf_prob_unweighted")
    terrain_prob_unweighted = format_score_val(terrain_prob_unweighted_raw, 3)

    # Rainfall Accumulation Attributes
    r_rate = format_rainfall_val(row.get("gpm_latest_rainfall_rate_mm_hr"))
    r_1h = format_rainfall_val(row.get("gpm_1h_accumulation_mm"))
    r_3h = format_rainfall_val(row.get("gpm_3h_accumulation_mm"))
    r_6h = format_rainfall_val(row.get("gpm_6h_accumulation_mm"))
    r_12h = format_rainfall_val(row.get("gpm_12h_accumulation_mm"))
    r_24h = format_rainfall_val(row.get("gpm_24h_accumulation_mm"))
    r_index = format_score_val(row.get("rainfall_intensity_index"), 3)
    r_pct = format_score_val(row.get("rainfall_percentile_24h"), 3)

    badge_bg, badge_label = SAMPLE_TYPE_BADGES.get(
        raw_sample_type, ("#7f8c8d", safe_html(raw_sample_type))
    )

    popup_html = f"""
    <div style="font-family:'Segoe UI', Arial, sans-serif; font-size:12px; width:330px; line-height:1.45; color:#2c3e50;">
        <div style="border-bottom:2px solid {cat_badge_color}; padding-bottom:6px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:14px; font-weight:bold; color:#1a252f;">{record_id}</span>
                <span style="background:{cat_badge_color}; color:white; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:11px;">
                    {cat_name}
                </span>
            </div>
            <div style="margin-top:4px;"><span style='background:{badge_bg}; color:white; padding:2px 7px; border-radius:4px; font-weight:bold;'>{badge_label}</span></div>
            <div style="font-size:11px; color:#7f8c8d; margin-top:3px;">
                DEM Tile: <b>{tile}</b> &nbsp;|&nbsp; Subregion: <b>{subregion}</b>
            </div>
            <div style="font-size:10.5px; color:#5d6d7e; margin-top:2px;">
                Coverage: <b>Limited Western Ghats Pilot</b> (14 DEM tiles, 1,582 locations)
            </div>
        </div>

        <table style="width:100%; border-collapse:collapse; margin-bottom:8px; font-size:11px;">
            <tr>
                <td style="color:#7f8c8d;">Coordinates:</td>
                <td style="font-weight:bold; text-align:right;">{lat:.4f}° N, {lon:.4f}° E</td>
            </tr>
            <tr>
                <td style="color:#7f8c8d;">Elevation / Slope:</td>
                <td style="font-weight:bold; text-align:right;">{elev} m &nbsp;|&nbsp; {slope}°</td>
            </tr>
        </table>

        <div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:6px; padding:7px 9px; margin-bottom:8px;">
            <div style="font-weight:bold; color:#2c3e50; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                Dynamic Hazard Screening (Stage 6D)
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span>Dynamic Risk Index:</span>
                <span style="font-weight:bold; font-size:13px; color:{cat_badge_color};">{risk_score} / 100</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#555;">
                <span>Relative Category Tier:</span>
                <span style="font-weight:bold;">{cat_name}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#555;">
                <span>Fixed Benchmark Tier:</span>
                <span style="font-weight:bold;">{fixed_tier}</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-top:2px; border-top:1px dashed #ddd; padding-top:2px;">
                <span>Balanced Terrain Prob (Used in Risk):</span>
                <span style="font-weight:bold;">{terrain_prob_balanced} ({terrain_tier_balanced})</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-top:2px;">
                <span>Unweighted Terrain Prob (Comparison Only):</span>
                <span style="font-weight:bold;">{terrain_prob_unweighted}</span>
            </div>
        </div>

        <div style="background:#f4f9fd; border:1px solid #d0e3f0; border-radius:6px; padding:7px 9px; margin-bottom:8px;">
            <div style="font-weight:bold; color:#1b4f72; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">
                NASA GPM IMERG Rainfall (Early Run V07C)
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:11px;">
                <tr style="border-bottom:1px solid #e1edf5;">
                    <td style="color:#546e7a;">1-hour accumulation:</td>
                    <td style="text-align:right; font-weight:bold;">{r_1h}</td>
                </tr>
                <tr style="border-bottom:1px solid #e1edf5;">
                    <td style="color:#546e7a;">3-hour accumulation:</td>
                    <td style="text-align:right; font-weight:bold;">{r_3h}</td>
                </tr>
                <tr style="border-bottom:1px solid #e1edf5;">
                    <td style="color:#546e7a;">6-hour accumulation:</td>
                    <td style="text-align:right; font-weight:bold;">{r_6h}</td>
                </tr>
                <tr style="border-bottom:1px solid #e1edf5;">
                    <td style="color:#546e7a;">12-hour accumulation:</td>
                    <td style="text-align:right; font-weight:bold;">{r_12h}</td>
                </tr>
                <tr style="border-bottom:1px solid #e1edf5;">
                    <td style="color:#546e7a; font-weight:bold;">24-hour accumulation:</td>
                    <td style="text-align:right; font-weight:bold; color:#1b4f72;">{r_24h}</td>
                </tr>
                <tr>
                    <td style="color:#546e7a;">Rainfall Intensity Index:</td>
                    <td style="text-align:right; font-weight:bold;">{r_index} (Pct: {r_pct})</td>
                </tr>
            </table>
            <div style="font-size:10px; color:#78909c; margin-top:4px;">
                Observation End: <b>{safe_html(obs_end_utc)}</b> ({latency_hours:.1f}h latency)
            </div>
        </div>

        <div style="font-size:9.5px; color:#7f8c8d; line-height:1.3; background:#fff8e1; border:1px solid #ffe082; padding:5px 7px; border-radius:4px;">
            <b>Notice:</b> Limited Western Ghats pilot screening only (14 validated DEM tiles, 1,582 locations: 182 historical events &amp; 1,400 background samples); does not imply complete Western Ghats coverage. Not an official government warning, safety directive, or calibrated slope failure probability.
        </div>
    </div>
    """
    return popup_html


# =====================================================================
# MAP GENERATION PIPELINE
# =====================================================================

def create_western_ghats_dynamic_risk_map(
    input_csv_path: Path = DEFAULT_INPUT_CSV,
    input_metadata_path: Path = DEFAULT_INPUT_METADATA,
    output_html_path: Path = DEFAULT_OUTPUT_HTML,
    status_path: Optional[Path] = None,
    now_utc: Optional[datetime] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Generate the Stage 6E Western Ghats Dynamic Landslide Hazard GIS Map.

    Strict validation gates:
    - Overwrite protection: refuses to overwrite existing file unless overwrite=True.
    - Exactly 1,582 records.
    - Exactly 14 unique DEM tiles.
    - Exactly 182 historical landslide events and 1,400 provisional background samples.
    - Unique, non-null record_id and sample_id.
    - Coordinates finite and within approved Western Ghats bounding bounds.
    - Strictly validates metadata relative thresholds with zero fallbacks.
    - Strictly validates observation timestamps (explicit UTC), freshness (<12h), and completeness.
    - Clear distinction and labeling of balanced vs unweighted terrain models.
    - Safe colormap calls and null-handling for missing data.
    """
    print("\n" + "=" * 75)
    print("STAGE 6E: WESTERN GHATS DYNAMIC GIS RISK MAP GENERATOR")
    print("=" * 75)

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Stage 6D CSV not found: {input_csv_path}")

    # Overwrite protection gate
    if output_html_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output HTML map already exists: {output_html_path}. "
            "Use --overwrite to explicitly replace existing file."
        )

    # 1. Load Data
    print(f"\n1. Loading input risk dataset: {input_csv_path.name}")
    df = pd.read_csv(input_csv_path, low_memory=False)

    # Pre-generation Validations
    if len(df) != EXPECTED_LOCATIONS_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_LOCATIONS_COUNT} records; found {len(df)}")

    unique_tiles = sorted(df["tile"].unique().tolist())
    if len(unique_tiles) != EXPECTED_TILES_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_TILES_COUNT} unique DEM tiles; found {len(unique_tiles)}")

    hist_event_count = int((df["sample_type"] == "historical_landslide").sum())
    bg_sample_count = int((df["sample_type"] == "provisional_background").sum())

    if hist_event_count != EXPECTED_HISTORICAL_EVENTS_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_HISTORICAL_EVENTS_COUNT} historical landslide records; found {hist_event_count}"
        )

    if bg_sample_count != EXPECTED_BACKGROUND_SAMPLES_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_BACKGROUND_SAMPLES_COUNT} provisional background records; found {bg_sample_count}"
        )

    if df["record_id"].isnull().any() or not df["record_id"].is_unique:
        raise ValueError("Invalid or duplicate 'record_id' values detected.")

    if df["sample_id"].isnull().any() or not df["sample_id"].is_unique:
        raise ValueError("Invalid or duplicate 'sample_id' values detected.")

    lats = pd.to_numeric(df["latitude"], errors="coerce")
    lons = pd.to_numeric(df["longitude"], errors="coerce")
    if lats.isna().any() or lons.isna().any() or not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("Missing or non-finite geographic coordinates detected.")

    if not (lats.between(LAT_BOUNDS[0], LAT_BOUNDS[1]).all() and lons.between(LON_BOUNDS[0], LON_BOUNDS[1]).all()):
        raise ValueError("Coordinates outside approved Western Ghats bounding bounds.")

    print(f"   Validated {len(df)} locations ({hist_event_count} events, {bg_sample_count} background) across {len(unique_tiles)} tiles.")

    # 2. Load and Validate Metadata & Freshness Gate
    if not input_metadata_path.exists():
        raise FileNotFoundError(f"Required Stage 6D metadata file not found: {input_metadata_path}")

    with open(input_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   Loaded metadata: {input_metadata_path.name}")

    # Strict threshold validation with ZERO fallbacks
    validated_cutoffs = validate_stage6d_metadata_thresholds(metadata)
    p50 = validated_cutoffs["P50_low_to_moderate"]
    p80 = validated_cutoffs["P80_moderate_to_high"]
    p95 = validated_cutoffs["P95_high_to_very_high"]

    # Freshness & Completeness validation gate with explicit UTC enforcement
    is_fresh, gate_msg, audit_info = verify_observation_freshness_and_completeness(
        metadata=metadata,
        now_utc=now_utc,
        max_latency_hours=MAX_DATA_LATENCY_HOURS,
        status_path=status_path,
    )
    if not is_fresh:
        raise ValueError(f"Freshness and completeness gate failed: {gate_msg}")

    obs_start_utc = audit_info["observation_start_utc"]
    obs_end_utc = audit_info["observation_end_utc"]
    latency_hours = audit_info["calculated_latency_hours"]

    # 3. Initialize Folium Map
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

    print(f"\n2. Initializing Folium Map centered at ({center_lat:.4f}° N, {center_lon:.4f}° E)...")
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # Add Esri World Topographic Basemap (default active)
    folium.TileLayer(
        tiles=ESRI_TOPO_URL,
        attr=ESRI_TOPO_ATTR,
        name="Esri World Topographic Map",
        overlay=False,
        control=True,
        show=True,
    ).add_to(map_obj)

    # Optional secondary basemap: CartoDB Positron
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap &copy; CARTO",
        name="CartoDB Positron (Minimal Light)",
        overlay=False,
        control=True,
        show=False,
    ).add_to(map_obj)

    # 4. Create Colormaps for Continuous Layers
    terrain_colormap = LinearColormap(
        colors=["#27ae60", "#f1c40f", "#e67e22", "#c0392b"],
        vmin=0.0,
        vmax=1.0,
        caption="Unweighted RF Terrain Susceptibility (Comparison Only: 0 to 1)",
    )
    terrain_colormap.add_to(map_obj)

    # Max finite rainfall calculation for colormap scaling
    finite_rain_24h = df["gpm_24h_accumulation_mm"].dropna()
    finite_rain_24h = finite_rain_24h[np.isfinite(finite_rain_24h)]
    max_rain = float(finite_rain_24h.max()) if len(finite_rain_24h) > 0 else 10.0

    rainfall_colormap = LinearColormap(
        colors=["#ebf5fb", "#85c1e9", "#3498db", "#1b4f72"],
        vmin=0.0,
        vmax=max(10.0, max_rain),
        caption=f"GPM IMERG 24-Hour Rainfall Accumulation (0 to {max_rain:.1f} mm)",
    )
    rainfall_colormap.add_to(map_obj)

    # 5. Construct Feature Layers & Compact Client-Side Rendering
    print("\n3. Building 5 independently toggleable Canvas layers...")

    is_default_prod = output_html_path.resolve() == DEFAULT_OUTPUT_HTML.resolve()
    out_dir = output_html_path.parent
    sync_geojson_path = DEFAULT_OUTPUT_GEOJSON if is_default_prod else out_dir / "western_ghats_dynamic_hazard_risk.geojson"
    meta_out_path = DEFAULT_OUTPUT_META if is_default_prod else out_dir / "western_ghats_dynamic_risk_map_metadata.json"
    val_report_path = DEFAULT_VALIDATION_REPORT if is_default_prod else out_dir / "western_ghats_dynamic_risk_map_validation_report.json"

    backup_targets = [output_html_path, meta_out_path, val_report_path]
    if is_default_prod:
        backup_targets.extend([sync_geojson_path, DEFAULT_POINT_GEOJSON])
    if overwrite:
        create_timestamped_backup(backup_targets, out_dir / "backup")

    layer_dynamic_risk = folium.FeatureGroup(
        name="Dynamic Hazard Screening (Primary Stage 6D Category)",
        show=True,
    )
    layer_rainfall = folium.FeatureGroup(
        name="GPM 24h Rainfall Accumulation (mm)",
        show=False,
    )
    layer_terrain = folium.FeatureGroup(
        name="Unweighted Random Forest Terrain Susceptibility (Comparison Only)",
        show=False,
    )
    layer_events = folium.FeatureGroup(
        name="Historical Landslide Events (n=182)",
        show=True,
    )
    layer_background = folium.FeatureGroup(
        name="Provisional Background Samples (n=1,400)",
        show=False,
    )

    layer_dynamic_risk.add_to(map_obj)
    layer_rainfall.add_to(map_obj)
    layer_terrain.add_to(map_obj)
    layer_events.add_to(map_obj)
    layer_background.add_to(map_obj)

    folium.LayerControl(collapsed=False).add_to(map_obj)

    # 6. Serialize 1,582 Sample Records into Compact JSON Store
    print("   Serializing 1,582 sample records to compact client-side store...")
    wg_records = []
    for _, r in df.iterrows():
        rec_id = str(r.get("record_id", ""))
        samp_id = str(r.get("sample_id", ""))
        samp_type = str(r.get("sample_type", ""))
        tile = str(r.get("tile", ""))
        subreg = str(r.get("subregion", ""))
        lat = round(float(r["latitude"]), 4)
        lon = round(float(r["longitude"]), 4)
        elev = round(float(r["elevation_m"]), 1) if pd.notna(r.get("elevation_m")) and np.isfinite(r.get("elevation_m")) else None
        slope = round(float(r["slope_degrees"]), 2) if pd.notna(r.get("slope_degrees")) and np.isfinite(r.get("slope_degrees")) else None

        raw_risk = r.get("dynamic_risk_index")
        cat = str(r.get("primary_screening_category", "UNASSESSED_MISSING_RAINFALL"))
        if raw_risk is None or pd.isna(raw_risk) or not np.isfinite(raw_risk) or cat == "UNASSESSED_MISSING_RAINFALL":
            risk_val = None
            display_cat = "Unassessed"
            c_risk = CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]
        else:
            risk_val = round(float(raw_risk), 2)
            display_cat = cat
            c_risk = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"])

        ftier = str(r.get("risk_tier_fixed", "Unavailable"))
        tbal = round(float(r["terrain_rf_prob_balanced"]), 3) if pd.notna(r.get("terrain_rf_prob_balanced")) and np.isfinite(r.get("terrain_rf_prob_balanced")) else None
        tbal_tier = str(r.get("terrain_baseline_screening_tier", "Unavailable"))
        tunw = round(float(r["terrain_rf_prob_unweighted"]), 3) if pd.notna(r.get("terrain_rf_prob_unweighted")) and np.isfinite(r.get("terrain_rf_prob_unweighted")) else None

        r1 = round(float(r["gpm_1h_accumulation_mm"]), 2) if pd.notna(r.get("gpm_1h_accumulation_mm")) and np.isfinite(r.get("gpm_1h_accumulation_mm")) else None
        r3 = round(float(r["gpm_3h_accumulation_mm"]), 2) if pd.notna(r.get("gpm_3h_accumulation_mm")) and np.isfinite(r.get("gpm_3h_accumulation_mm")) else None
        r6 = round(float(r["gpm_6h_accumulation_mm"]), 2) if pd.notna(r.get("gpm_6h_accumulation_mm")) and np.isfinite(r.get("gpm_6h_accumulation_mm")) else None
        r12 = round(float(r["gpm_12h_accumulation_mm"]), 2) if pd.notna(r.get("gpm_12h_accumulation_mm")) and np.isfinite(r.get("gpm_12h_accumulation_mm")) else None
        r24 = round(float(r["gpm_24h_accumulation_mm"]), 2) if pd.notna(r.get("gpm_24h_accumulation_mm")) and np.isfinite(r.get("gpm_24h_accumulation_mm")) else None
        rindex = round(float(r["rainfall_intensity_index"]), 3) if pd.notna(r.get("rainfall_intensity_index")) and np.isfinite(r.get("rainfall_intensity_index")) else None
        rpct = round(float(r["rainfall_percentile_24h"]), 3) if pd.notna(r.get("rainfall_percentile_24h")) and np.isfinite(r.get("rainfall_percentile_24h")) else None
        status_str = str(r.get("risk_calculation_status", "VALIDATED_SCREENING_CALCULATED"))

        c_rain = rainfall_colormap(min(r24, max_rain)) if r24 is not None and r24 >= 0 else CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]
        c_terrain = terrain_colormap(tunw) if tunw is not None and 0.0 <= tunw <= 1.0 else CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]

        wg_records.append([
            rec_id, samp_id, samp_type, tile, subreg, lat, lon, elev, slope,
            risk_val, display_cat, c_risk, ftier,
            tbal, tbal_tier, tunw,
            r1, r3, r6, r12, r24, rindex, rpct, status_str,
            c_rain, c_terrain,
        ])

    cat_counts = df["primary_screening_category"].value_counts()
    count_low = int(cat_counts.get("Low", 0))
    count_mod = int(cat_counts.get("Moderate", 0))
    count_high = int(cat_counts.get("High", 0))
    count_vhigh = int(cat_counts.get("Very High", 0))
    count_unassessed = int(cat_counts.get("UNASSESSED_MISSING_RAINFALL", 0))

    shared_freshness_js = build_compute_freshness_js(MAX_DATA_LATENCY_HOURS)
    json_wg_records = json.dumps(wg_records, separators=(",", ":"))
    gen_ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    client_script_template = """
<script>
(function() {
    var wgRecords = __JSON_WG_RECORDS__;
    var obsEndUtc = __OBS_END_UTC_JSON__;
    var totalLocations = __TOTAL_LOCATIONS__;
    var freshCategoryCounts = {
        Low: __COUNT_LOW__,
        Moderate: __COUNT_MOD__,
        High: __COUNT_HIGH__,
        VeryHigh: __COUNT_VHIGH__,
        Unassessed: __COUNT_UNASSESSED__
    };

    var isStale = false;
    var currentLatencyHours = __LATENCY_HOURS__;
    var currentStatusStr = "FRESH";
    var riskMarkers = [];
    var rainMarkers = [];

__SHARED_FRESHNESS_JS__

    function escapeHtml(str) {
        if (str === null || str === undefined) return "Unavailable";
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function formatVal(v, suffix, decimals) {
        if (v === null || v === undefined || isNaN(v)) return '<span style="color:#e74c3c; font-style:italic;">Unavailable</span>';
        var formatted = (decimals !== undefined) ? Number(v).toFixed(decimals) : String(v);
        return suffix ? formatted + " " + suffix : formatted;
    }

    function formatInt(v) {
        return Number(v || 0).toLocaleString("en-US");
    }

    function renderPopupHtml(d) {
        var recId = escapeHtml(d[0]);
        var sampType = String(d[2] || "");
        var isEvent = (sampType === "historical_landslide");
        var badgeBg = isEvent ? "#c0392b" : "#2980b9";
        var badgeLabel = isEvent ? "Historical Landslide Event" : "Provisional Background Sample";
        var tile = escapeHtml(d[3]);
        var subreg = escapeHtml(d[4]);
        var lat = Number(d[5]).toFixed(4);
        var lon = Number(d[6]).toFixed(4);
        var elev = formatVal(d[7], null, 1);
        var slope = formatVal(d[8], null, 2);

        var riskVal = isStale ? '<span style="color:#95a5a6; font-style:italic;">Unassessed (Stale)</span>' : formatVal(d[9], null, 2);
        var catName = isStale ? "UNASSESSED" : escapeHtml(d[10]);
        var catColor = isStale ? "#95a5a6" : (d[11] || "#95a5a6");
        var ftier = isStale ? "UNASSESSED_STALE" : escapeHtml(d[12]);

        var tbal = formatVal(d[13], null, 3);
        var tbalTier = escapeHtml(d[14]);
        var tunw = formatVal(d[15], null, 3);

        var r1 = formatVal(d[16], 'mm', 2);
        var r3 = formatVal(d[17], 'mm', 2);
        var r6 = formatVal(d[18], 'mm', 2);
        var r12 = formatVal(d[19], 'mm', 2);
        var r24 = formatVal(d[20], 'mm', 2);
        var rindex = formatVal(d[21], null, 3);
        var rpct = formatVal(d[22], null, 3);
        var statusStr = isStale ? "STALE_DATA_UNASSESSED" : escapeHtml(d[23]);
        var latencyStr = (currentLatencyHours !== null && !isNaN(currentLatencyHours)) ? currentLatencyHours.toFixed(1) + "h" : "Unavailable";

        return '<div style="font-family: Arial, sans-serif; font-size:12px; width:330px; line-height:1.45; color:#2c3e50;">' +
            '<div style="border-bottom:2px solid ' + catColor + '; padding-bottom:6px; margin-bottom:8px;">' +
                '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<span style="font-size:14px; font-weight:bold; color:#1a252f;">' + recId + '</span>' +
                    '<span style="background:' + catColor + '; color:white; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:11px;">' + catName + '</span>' +
                '</div>' +
                '<div style="margin-top:4px;"><span style="background:' + badgeBg + '; color:white; padding:2px 7px; border-radius:4px; font-weight:bold;">' + badgeLabel + '</span></div>' +
                '<div style="font-size:11px; color:#7f8c8d; margin-top:3px;">DEM Tile: <b>' + tile + '</b> &nbsp;|&nbsp; Subregion: <b>' + subreg + '</b></div>' +
                '<div style="font-size:10.5px; color:#5d6d7e; margin-top:2px;">Coverage: <b>Limited Western Ghats Pilot</b> (14 DEM tiles, 1,582 locations)</div>' +
            '</div>' +
            '<table style="width:100%; border-collapse:collapse; margin-bottom:8px; font-size:11px;">' +
                '<tr><td style="color:#7f8c8d;">Coordinates:</td><td style="font-weight:bold; text-align:right;">' + lat + '° N, ' + lon + '° E</td></tr>' +
                '<tr><td style="color:#7f8c8d;">Elevation / Slope:</td><td style="font-weight:bold; text-align:right;">' + elev + ' m &nbsp;|&nbsp; ' + slope + '°</td></tr>' +
            '</table>' +
            '<div style="background:#f8f9fa; border:1px solid #e9ecef; border-radius:6px; padding:7px 9px; margin-bottom:8px;">' +
                '<div style="font-weight:bold; color:#2c3e50; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">Dynamic Hazard Screening (Stage 6D)</div>' +
                '<div style="display:flex; justify-content:space-between; margin-bottom:2px;"><span>Dynamic Risk Index:</span><span style="font-weight:bold; font-size:13px; color:' + catColor + ';">' + riskVal + ' / 100</span></div>' +
                '<div style="display:flex; justify-content:space-between; font-size:11px; color:#555;"><span>Relative Category Tier:</span><span style="font-weight:bold;">' + catName + '</span></div>' +
                '<div style="display:flex; justify-content:space-between; font-size:11px; color:#555;"><span>Fixed Benchmark Tier:</span><span style="font-weight:bold;">' + ftier + '</span></div>' +
                '<div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-top:2px; border-top:1px dashed #ddd; padding-top:2px;"><span>Balanced Terrain Prob (Used in Risk):</span><span style="font-weight:bold;">' + tbal + ' (' + tbalTier + ')</span></div>' +
                '<div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-top:2px;"><span>Unweighted Terrain Prob (Comparison Only):</span><span style="font-weight:bold;">' + tunw + '</span></div>' +
            '</div>' +
            '<div style="background:#f4f9fd; border:1px solid #d0e3f0; border-radius:6px; padding:7px 9px; margin-bottom:8px;">' +
                '<div style="font-weight:bold; color:#1b4f72; margin-bottom:4px; font-size:11px; text-transform:uppercase; letter-spacing:0.5px;">NASA GPM IMERG Rainfall (Early Run V07C)</div>' +
                '<table style="width:100%; border-collapse:collapse; font-size:11px;">' +
                    '<tr style="border-bottom:1px solid #e1edf5;"><td style="color:#546e7a;">1-hour accumulation:</td><td style="text-align:right; font-weight:bold;">' + r1 + '</td></tr>' +
                    '<tr style="border-bottom:1px solid #e1edf5;"><td style="color:#546e7a;">3-hour accumulation:</td><td style="text-align:right; font-weight:bold;">' + r3 + '</td></tr>' +
                    '<tr style="border-bottom:1px solid #e1edf5;"><td style="color:#546e7a;">6-hour accumulation:</td><td style="text-align:right; font-weight:bold;">' + r6 + '</td></tr>' +
                    '<tr style="border-bottom:1px solid #e1edf5;"><td style="color:#546e7a;">12-hour accumulation:</td><td style="text-align:right; font-weight:bold;">' + r12 + '</td></tr>' +
                    '<tr style="border-bottom:1px solid #e1edf5;"><td style="color:#546e7a; font-weight:bold;">24-hour accumulation:</td><td style="text-align:right; font-weight:bold; color:#1b4f72;">' + r24 + '</td></tr>' +
                    '<tr><td style="color:#546e7a;">Rainfall Intensity Index:</td><td style="text-align:right; font-weight:bold;">' + rindex + ' (Percentile: ' + rpct + ')</td></tr>' +
                '</table>' +
                '<div style="font-size:10px; color:#78909c; margin-top:4px;">Obs End: <b>' + escapeHtml(obsEndUtc) + '</b> (' + latencyStr + ' latency) | Status: <b>' + statusStr + '</b></div>' +
            '</div>' +
            '<div style="font-size:9.5px; color:#7f8c8d; line-height:1.3; background:#fff8e1; border:1px solid #ffe082; padding:5px 7px; border-radius:4px;">' +
                '<b>Notice:</b> Limited Western Ghats pilot screening only (14 validated DEM tiles, 1,582 locations: 182 historical events &amp; 1,400 background samples); does not imply complete Western Ghats coverage. Not an official government warning, safety directive, or calibrated slope failure probability.' +
            '</div>' +
        '</div>';
    }

    function updateFreshnessStatus() {
        var freshRes = computeFreshness();
        isStale = freshRes.isStale;
        currentLatencyHours = freshRes.latencyHours;
        currentStatusStr = freshRes.statusStr;

        var badgeElem = document.getElementById("dashboard-freshness-badge");
        if (badgeElem) {
            if (isStale) {
                badgeElem.style.background = "#fdecea";
                badgeElem.style.color = "#c0392b";
                badgeElem.textContent = (isNaN(currentLatencyHours) || currentLatencyHours === null) ?
                    "Latency: Unavailable (STALE / UNASSESSED)" :
                    "Latency: " + currentLatencyHours.toFixed(1) + "h (STALE / UNASSESSED)";
            } else {
                badgeElem.style.background = "#e8f8f5";
                badgeElem.style.color = "#117864";
                badgeElem.textContent = "Latency: " + currentLatencyHours.toFixed(1) + "h (SUCCESS)";
            }
        }

        var cLow = document.getElementById("wg-count-low");
        var cMod = document.getElementById("wg-count-mod");
        var cHigh = document.getElementById("wg-count-high");
        var cVHigh = document.getElementById("wg-count-vhigh");
        var cUnass = document.getElementById("wg-count-unassessed");
        if (cLow) cLow.textContent = isStale ? "0 (0.0%)" : formatInt(freshCategoryCounts.Low) + " (" + (freshCategoryCounts.Low / totalLocations * 100).toFixed(1) + "%)";
        if (cMod) cMod.textContent = isStale ? "0 (0.0%)" : formatInt(freshCategoryCounts.Moderate) + " (" + (freshCategoryCounts.Moderate / totalLocations * 100).toFixed(1) + "%)";
        if (cHigh) cHigh.textContent = isStale ? "0 (0.0%)" : formatInt(freshCategoryCounts.High) + " (" + (freshCategoryCounts.High / totalLocations * 100).toFixed(1) + "%)";
        if (cVHigh) cVHigh.textContent = isStale ? "0 (0.0%)" : formatInt(freshCategoryCounts.VeryHigh) + " (" + (freshCategoryCounts.VeryHigh / totalLocations * 100).toFixed(1) + "%)";
        if (cUnass) cUnass.textContent = isStale ? formatInt(totalLocations) + " (100.0%)" : formatInt(freshCategoryCounts.Unassessed) + " (" + (freshCategoryCounts.Unassessed / totalLocations * 100).toFixed(1) + "%)";

        for (var i = 0; i < riskMarkers.length; i++) {
            var item = riskMarkers[i];
            var pCol = isStale ? "#95a5a6" : (item.data[11] || "#95a5a6");
            item.marker.setStyle({
                color: item.isEvent ? "#2c3e50" : pCol,
                fillColor: pCol
            });
            var riskTip = isStale ?
                ("UNASSESSED Risk (Stale Data): " + item.data[1]) :
                ((item.isEvent ? "EVENT" : "BACKGROUND") + ": " + item.data[1] + " | " + item.data[10] + " Risk");
            item.marker.setTooltipContent(riskTip);
        }

        for (var j = 0; j < rainMarkers.length; j++) {
            var rItem = rainMarkers[j];
            var rCol = isStale ? "#95a5a6" : (rItem.data[24] || "#95a5a6");
            rItem.marker.setStyle({ color: rCol, fillColor: rCol });
            var rainTip = isStale ?
                ("Rain: UNASSESSED (Stale GPM Data) - " + rItem.data[1]) :
                (rItem.data[1] + ": 24h Rain: " + (rItem.data[20] !== null ? Number(rItem.data[20]).toFixed(2) + " mm" : "Unavailable"));
            rItem.marker.setTooltipContent(rainTip);
        }
    }

    function populateLayers() {
        var lgRisk = __LAYER_RISK_NAME__;
        var lgRain = __LAYER_RAIN_NAME__;
        var lgTerrain = __LAYER_TERRAIN_NAME__;
        var lgEvents = __LAYER_EVENTS_NAME__;
        var lgBackground = __LAYER_BG_NAME__;

        for (var i = 0; i < wgRecords.length; i++) {
            var d = wgRecords[i];
            var isEvent = (d[2] === "historical_landslide");
            var latlng = [d[5], d[6]];
            var pFn = (function(rec) {
                return function() { return renderPopupHtml(rec); };
            })(d);
            var pOpts = { maxWidth: 350 };

            var mRisk = L.circleMarker(latlng, {
                radius: isEvent ? 5.5 : 4.0,
                color: isEvent ? "#2c3e50" : d[11],
                weight: isEvent ? 1.5 : 1.0,
                fill: true,
                fillColor: d[11],
                fillOpacity: 0.85
            });
            mRisk.bindTooltip((isEvent ? "EVENT" : "BACKGROUND") + ": " + d[1] + " | " + d[10] + " Risk");
            mRisk.bindPopup(pFn, pOpts);
            mRisk.addTo(lgRisk);
            riskMarkers.push({ marker: mRisk, layer: mRisk, data: d, isEvent: isEvent });

            var mRain = L.circleMarker(latlng, {
                radius: 4.5,
                color: d[24],
                weight: 1.0,
                fill: true,
                fillColor: d[24],
                fillOpacity: 0.80
            });
            mRain.bindTooltip(d[1] + ": 24h Rain: " + (d[20] !== null ? Number(d[20]).toFixed(2) + " mm" : "Unavailable"));
            mRain.bindPopup(pFn, pOpts);
            mRain.addTo(lgRain);
            rainMarkers.push({ marker: mRain, layer: mRain, data: d });

            var mTerrain = L.circleMarker(latlng, {
                radius: 4.5,
                color: d[25],
                weight: 1.0,
                fill: true,
                fillColor: d[25],
                fillOpacity: 0.80
            });
            mTerrain.bindTooltip(d[1] + ": Unweighted Terrain Prob: " + (d[15] !== null ? Number(d[15]).toFixed(3) : "Unavailable"));
            mTerrain.bindPopup(pFn, pOpts);
            mTerrain.addTo(lgTerrain);

            if (isEvent) {
                var mEv = L.circleMarker(latlng, {
                    radius: 6.5,
                    color: "#1a252f",
                    weight: 2.0,
                    fill: true,
                    fillColor: d[11],
                    fillOpacity: 0.90
                });
                mEv.bindTooltip("EVENT: " + d[1] + " (" + d[10] + ")");
                mEv.bindPopup(pFn, pOpts);
                mEv.addTo(lgEvents);
            } else {
                var mBg = L.circleMarker(latlng, {
                    radius: 3.5,
                    color: d[11],
                    weight: 0.8,
                    fill: true,
                    fillColor: d[11],
                    fillOpacity: 0.75
                });
                mBg.bindTooltip("BACKGROUND: " + d[1] + " (" + d[10] + ")");
                mBg.bindPopup(pFn, pOpts);
                mBg.addTo(lgBackground);
            }
        }
    }

    if (typeof window !== 'undefined') {
        window.computeFreshness = computeFreshness;
        window.updateFreshnessStatus = updateFreshnessStatus;
        window.renderPopupHtml = renderPopupHtml;
        window.riskMarkers = riskMarkers;
        window.rainMarkers = rainMarkers;
        window.getIsStale = function() { return isStale; };
        window.getCurrentLatencyHours = function() { return currentLatencyHours; };
        window.setObservationEndTimeForTesting = function(newUtcStr) {
            obsEndUtc = newUtcStr;
            updateFreshnessStatus();
        };
    }

    function initMap() {
        populateLayers();
        updateFreshnessStatus();
        setInterval(updateFreshnessStatus, 60000);
    }

    if (document.readyState === "complete") {
        initMap();
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initMap);
    } else {
        window.addEventListener("load", initMap);
    }
})();
</script>
"""

    client_script = (
        client_script_template
        .replace("__JSON_WG_RECORDS__", json_wg_records)
        .replace("__OBS_END_UTC_JSON__", json.dumps(str(obs_end_utc)))
        .replace("__TOTAL_LOCATIONS__", str(len(df)))
        .replace("__COUNT_LOW__", str(count_low))
        .replace("__COUNT_MOD__", str(count_mod))
        .replace("__COUNT_HIGH__", str(count_high))
        .replace("__COUNT_VHIGH__", str(count_vhigh))
        .replace("__COUNT_UNASSESSED__", str(count_unassessed))
        .replace("__LATENCY_HOURS__", f"{latency_hours:.4f}")
        .replace("__SHARED_FRESHNESS_JS__", shared_freshness_js)
        .replace("__LAYER_RISK_NAME__", layer_dynamic_risk.get_name())
        .replace("__LAYER_RAIN_NAME__", layer_rainfall.get_name())
        .replace("__LAYER_TERRAIN_NAME__", layer_terrain.get_name())
        .replace("__LAYER_EVENTS_NAME__", layer_events.get_name())
        .replace("__LAYER_BG_NAME__", layer_background.get_name())
    )
    map_obj.get_root().html.add_child(folium.Element(client_script))

    # 7. Add Floating Header & Legend Dashboard Overlay
    dashboard_card_html = f"""
    <div style="
        position: fixed;
        top: 15px;
        left: 55px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.96);
        padding: 14px 18px;
        border: 1px solid #bdc3c7;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        max-width: 425px;
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #2c3e50;
        line-height: 1.4;
    ">
        <div style="font-size: 15px; font-weight: bold; color: #1a252f; margin-bottom: 2px;">
            Western Ghats Limited Pilot — Dynamic Landslide Hazard Map
        </div>
        <div style="font-size: 11px; color: #7f8c8d; margin-bottom: 8px;">
            Limited Western Ghats Pilot Only (Stage 6E) &nbsp;|&nbsp; 14 Validated DEM Tiles · 1,582 Locations
        </div>

        <div style="display: flex; gap: 6px; margin-bottom: 8px; font-size: 11px; flex-wrap: wrap;">
            <div style="background: #eef2f5; padding: 4px 7px; border-radius: 4px;">
                <b>14</b> Validated DEM Tiles
            </div>
            <div style="background: #eef2f5; padding: 4px 7px; border-radius: 4px;">
                <b>1,582</b> Locations
            </div>
            <div style="background: #eef2f5; padding: 4px 7px; border-radius: 4px;">
                <b>182</b> Historical Events
            </div>
            <div style="background: #eef2f5; padding: 4px 7px; border-radius: 4px;">
                <b>1,400</b> Background Samples
            </div>
            <div id="dashboard-freshness-badge" style="background: #e8f8f5; color: #117864; padding: 4px 7px; border-radius: 4px; font-weight: bold;">
                Latency: {latency_hours:.1f}h (SUCCESS)
            </div>
        </div>

        <div style="font-size: 11px; color: #444; border-top: 1px solid #ecf0f1; padding-top: 6px; margin-bottom: 8px;">
            <b>Observation Window (UTC):</b><br>
            {safe_html(obs_start_utc)} &nbsp;→&nbsp; {safe_html(obs_end_utc)}
        </div>

        <div style="font-size: 11px; font-weight: bold; margin-bottom: 5px; color: #2c3e50; text-transform: uppercase; letter-spacing: 0.5px;">
            Dynamic Screening Categories (Limited Pilot Quantiles)
        </div>

        <div style="display: grid; grid-template-columns: 1fr; gap: 4px; font-size: 11.5px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['Low']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Low Risk (&lt; {p50:.2f}, &lt; P50)</span>
                <b id="wg-count-low">{count_low:,} ({count_low/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['Moderate']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Moderate Risk ({p50:.2f}–{p80:.2f}, P50–P80)</span>
                <b id="wg-count-mod">{count_mod:,} ({count_mod/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['High']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>High Risk ({p80:.2f}–{p95:.2f}, P80–P95)</span>
                <b id="wg-count-high">{count_high:,} ({count_high/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['Very High']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Very High Risk (≥ {p95:.2f}, P95+)</span>
                <b id="wg-count-vhigh" style="color:#c0392b;">{count_vhigh:,} ({count_vhigh/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['UNASSESSED_MISSING_RAINFALL']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Unassessed / Stale (&gt;12h)</span>
                <b id="wg-count-unassessed">{count_unassessed:,} ({count_unassessed/len(df)*100:.1f}%)</b>
            </div>
        </div>

        <div style="font-size: 9.5px; color: #7f8c8d; line-height: 1.35; border-top: 1px solid #ecf0f1; padding-top: 6px;">
            <b>LIMITED PILOT COVERAGE:</b> Current coverage is a limited Western Ghats pilot only (14 validated DEM tiles, 1,582 locations: 182 historical events + 1,400 background samples; no fabricated 5 km grid or duplicated records). Does not imply complete Western Ghats coverage. <b>NOTICE:</b> {DISCLAIMER_TEXT}. Relative indices are not official early warnings or physical failure probabilities.
        </div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(dashboard_card_html))

    # 8. Save Map HTML Atomically & Publish Synchronized GeoJSON/Metadata
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_html = output_html_path.with_name(f"{output_html_path.name}.tmp_{os.getpid()}")
    map_obj.save(str(tmp_html))
    os.replace(tmp_html, output_html_path)
    html_size = output_html_path.stat().st_size

    if is_default_prod or not sync_geojson_path.exists():
        geo_res = export_western_ghats_dynamic_risk_geojson(
            input_csv_path=input_csv_path,
            input_metadata_path=input_metadata_path,
            output_geojson_path=DEFAULT_POINT_GEOJSON if is_default_prod else sync_geojson_path,
            overwrite=True,
        )
        if is_default_prod and DEFAULT_POINT_GEOJSON.exists() and sync_geojson_path != DEFAULT_POINT_GEOJSON:
            atomic_copy_file(DEFAULT_POINT_GEOJSON, sync_geojson_path)
    else:
        geo_res = None

    map_metadata = {
        "stage": "Stage 6E — Western Ghats Limited Pilot Dynamic GIS Risk Map",
        "region": "Western Ghats",
        "coverage_scope": "LIMITED_PILOT",
        "complete_region_coverage": False,
        "coverage_note": "Limited Western Ghats pilot only: 14 validated DEM tiles, 1,582 locations (182 historical events + 1,400 background samples). Does not imply complete Western Ghats coverage; no fabricated 5 km grid or duplicated records.",
        "dem_tiles_count": len(unique_tiles),
        "total_locations": len(df),
        "historical_events_count": hist_event_count,
        "provisional_background_samples_count": bg_sample_count,
        "relative_category_cutoffs": {
            "P50_low_to_moderate": p50,
            "P80_moderate_to_high": p80,
            "P95_high_to_very_high": p95,
        },
        "gpm_observation_metadata": {
            "observation_start_utc": obs_start_utc,
            "observation_end_utc": obs_end_utc,
            "latency_hours": latency_hours,
            "freshness_status": "SUCCESS",
            "is_24h_complete": True,
        },
        "generation_timestamp_utc": gen_ts_utc,
        "html_size_bytes": html_size,
        "disclaimer": DISCLAIMER_TEXT,
    }
    atomic_write_json(meta_out_path, map_metadata)

    validation_report = {
        "status": "PASSED",
        "stage": "Stage 6E — Western Ghats Limited Pilot Dynamic GIS Risk Map",
        "generation_timestamp_utc": gen_ts_utc,
        "coverage_scope": "LIMITED_PILOT",
        "complete_region_coverage": False,
        "location_count": len(df),
        "dem_tiles_count": len(unique_tiles),
        "outputs": {
            "html_map_path": str(output_html_path),
            "html_map_size_bytes": html_size,
            "point_geojson_path": str(sync_geojson_path),
            "map_metadata_path": str(meta_out_path),
        },
        "primary_screening_categories": {
            "Low": count_low,
            "Moderate": count_mod,
            "High": count_high,
            "Very High": count_vhigh,
            "UNASSESSED": count_unassessed,
        },
        "relative_category_cutoffs": {
            "P50": p50,
            "P80": p80,
            "P95": p95,
        },
        "validation_checks": {
            "location_count_1582": bool(len(df) == EXPECTED_LOCATIONS_COUNT),
            "all_14_tiles_present": bool(len(unique_tiles) == EXPECTED_TILES_COUNT),
            "events_182_retained": bool(hist_event_count == EXPECTED_HISTORICAL_EVENTS_COUNT),
            "background_1400_retained": bool(bg_sample_count == EXPECTED_BACKGROUND_SAMPLES_COUNT),
            "runtime_60s_freshness_guard_injected": True,
            "compact_html_below_3mb": bool(html_size < 3_000_000),
            "threshold_validation_passed": True,
        },
        "disclaimer": DISCLAIMER_TEXT,
    }
    atomic_write_json(val_report_path, validation_report)

    print(f"\n4. Successfully generated compact GIS map HTML: {output_html_path.name}")
    print(f"   - File size: {html_size:,} bytes ({html_size / (1024 * 1024):.2f} MB)")
    print(f"   - Location count: {len(df):,}")
    print("   - Total layers: 5 feature groups + 2 tile layers")
    print("=" * 75)

    return {
        "status": "SUCCESS",
        "output_path": output_html_path,
        "location_count": len(df),
        "layers_count": 5,
        "basemaps_count": 2,
        "html_size_bytes": html_size,
        "geojson_path": sync_geojson_path,
        "metadata_path": meta_out_path,
        "validation_report_path": val_report_path,
    }


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Western Ghats Dynamic Landslide Hazard GIS Map")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to Stage 6D CSV")
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA, help="Path to Stage 6D metadata JSON")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML, help="Path to output HTML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")

    args = parser.parse_args()

    try:
        create_western_ghats_dynamic_risk_map(
            input_csv_path=args.input_csv,
            input_metadata_path=args.input_metadata,
            output_html_path=args.output_html,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] Map generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
