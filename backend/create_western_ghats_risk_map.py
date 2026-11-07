"""
Stage 6E Deliverable: Western Ghats Dynamic Landslide Hazard GIS Risk Map Generator.

Generates a standalone, production-ready interactive GIS HTML map for all 1,582 locations
across 14 DEM tiles in the Western Ghats (182 historical landslide events + 1,400 provisional
background samples) using approved Stage 6D dynamic landslide hazard risk predictions and NASA
GPM IMERG Early Run V07C rainfall accumulations.

Map architecture:
- Basemap: Esri World Topographic Map (default) + CartoDB Positron (optional toggle).
- 5 independently toggleable feature layers:
    1. Dynamic Hazard Screening (Primary Stage 6D Category) [Active by default]
    2. GPM 24h Rainfall Accumulation (mm) [Continuous blues, default off]
    3. Unweighted Random Forest Terrain Susceptibility (Comparison Only) [Continuous green-red, default off]
    4. Historical Landslide Events (n=182) [Active by default, distinctive border]
    5. Provisional Background Samples (n=1,400) [Default off]
- Informative popups with full terrain (both balanced and unweighted explicitly labeled), rainfall,
  risk, latency, and disclaimer attributes.
- Floating dashboard overlay showing category quantile legend, observation window, latency, and legal disclaimers.
- Strict input validation, data freshness/completeness gating with explicit UTC enforcement, strict metadata
  threshold validation with zero fallbacks, missing-value protection, HTML escaping, and overwrite protection.
"""

import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import math
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

# Exact approved Stage 6D relative quantile cutoffs
APPROVED_P50 = 9.364049598402843
APPROVED_P80 = 19.79328141262924
APPROVED_P95 = 45.95278574197298
THRESHOLD_TOLERANCE = 1e-6

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
    Strictly validate Stage 6D relative category cutoffs in metadata.
    Removes all silent fallbacks:
    - Requires 'risk_formula_specification' dict.
    - Requires 'relative_category_cutoffs' dict.
    - Requires all three keys: 'P50_low_to_moderate', 'P80_moderate_to_high', 'P95_high_to_very_high'.
    - Rejects missing, null, non-numeric, or non-finite values.
    - Verifies values against approved Stage 6D thresholds within strict floating-point tolerance (1e-6).
    - Raises ValueError on any validation failure.
    - Never silently substitutes hard-coded defaults.
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
        ("P50_low_to_moderate", APPROVED_P50),
        ("P80_moderate_to_high", APPROVED_P80),
        ("P95_high_to_very_high", APPROVED_P95),
    ]

    validated_cutoffs: Dict[str, float] = {}
    for key, approved_val in required_keys:
        if key not in cutoffs:
            raise ValueError(f"Missing required threshold key in metadata: '{key}'")
        raw_val = cutoffs[key]
        if raw_val is None or isinstance(raw_val, bool) or not isinstance(raw_val, (int, float, np.floating)):
            raise ValueError(f"Invalid or non-numeric threshold value for '{key}': {raw_val!r}")
        val = float(raw_val)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite threshold value for '{key}': {val}")
        if not math.isclose(val, approved_val, rel_tol=THRESHOLD_TOLERANCE, abs_tol=THRESHOLD_TOLERANCE):
            raise ValueError(
                f"Threshold mismatch for '{key}': got {val}, expected {approved_val} (within tolerance {THRESHOLD_TOLERANCE})"
            )
        validated_cutoffs[key] = val

    return validated_cutoffs


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
            <b>Notice:</b> Experimental relative screening index only. Not an official government warning, safety directive, or calibrated slope failure probability.
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

    # 5. Construct Feature Layers
    print("\n3. Building 5 independently toggleable layers...")

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

    # 6. Populate Map Layers
    for _, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        cat = str(row.get("primary_screening_category", "UNASSESSED_MISSING_RAINFALL"))
        sample_type = str(row.get("sample_type", ""))
        is_event = (sample_type == "historical_landslide")
        sample_id = safe_html(row.get("sample_id"))

        # Color mapping with safety check for missing/unassessed
        raw_risk = row.get("dynamic_risk_index")
        if raw_risk is None or pd.isna(raw_risk) or not np.isfinite(raw_risk) or cat == "UNASSESSED_MISSING_RAINFALL":
            primary_color = CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]
            display_cat = "Unassessed"
        else:
            primary_color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"])
            display_cat = safe_html(cat)

        popup_html = build_marker_popup_html(row, obs_start_utc, obs_end_utc, latency_hours)
        popup = folium.Popup(popup_html, max_width=350)
        tooltip_text = f"{'EVENT' if is_event else 'BACKGROUND'}: {sample_id} | {display_cat} Risk"

        # Layer 1: Primary Screening Category
        folium.CircleMarker(
            location=[lat, lon],
            radius=5.5 if is_event else 4.0,
            color="#2c3e50" if is_event else primary_color,
            weight=1.5 if is_event else 1.0,
            fill=True,
            fill_color=primary_color,
            fill_opacity=0.85,
            popup=popup,
            tooltip=tooltip_text,
        ).add_to(layer_dynamic_risk)

        # Layer 2: GPM 24h Rainfall Accumulation (Safe colormap evaluation)
        val_24h = row.get("gpm_24h_accumulation_mm")
        if val_24h is not None and pd.notna(val_24h) and np.isfinite(val_24h) and val_24h >= 0:
            rain_color = rainfall_colormap(min(float(val_24h), max_rain))
        else:
            rain_color = CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]

        folium.CircleMarker(
            location=[lat, lon],
            radius=4.5,
            color=rain_color,
            weight=1.0,
            fill=True,
            fill_color=rain_color,
            fill_opacity=0.80,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{sample_id}: 24h Rain: {format_rainfall_val(val_24h)}",
        ).add_to(layer_rainfall)

        # Layer 3: Unweighted Terrain Susceptibility (Safe colormap evaluation)
        val_prob = row.get("terrain_rf_prob_unweighted")
        if val_prob is not None and pd.notna(val_prob) and np.isfinite(val_prob) and 0.0 <= val_prob <= 1.0:
            terrain_color = terrain_colormap(float(val_prob))
        else:
            terrain_color = CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"]

        folium.CircleMarker(
            location=[lat, lon],
            radius=4.5,
            color=terrain_color,
            weight=1.0,
            fill=True,
            fill_color=terrain_color,
            fill_opacity=0.80,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{sample_id}: Unweighted Terrain Prob: {format_score_val(val_prob, 3)}",
        ).add_to(layer_terrain)

        # Layer 4 & 5: Event vs Background Separation
        if is_event:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6.5,
                color="#1a252f",
                weight=2.0,
                fill=True,
                fill_color=primary_color,
                fill_opacity=0.90,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"EVENT: {sample_id} ({display_cat})",
            ).add_to(layer_events)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3.5,
                color=primary_color,
                weight=0.8,
                fill=True,
                fill_color=primary_color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"BACKGROUND: {sample_id} ({display_cat})",
            ).add_to(layer_background)

    # Add all 5 layers to the map
    layer_dynamic_risk.add_to(map_obj)
    layer_rainfall.add_to(map_obj)
    layer_terrain.add_to(map_obj)
    layer_events.add_to(map_obj)
    layer_background.add_to(map_obj)

    folium.LayerControl(collapsed=False).add_to(map_obj)

    # 7. Add Floating Header & Legend Dashboard Overlay
    cat_counts = df["primary_screening_category"].value_counts()
    count_low = cat_counts.get("Low", 0)
    count_mod = cat_counts.get("Moderate", 0)
    count_high = cat_counts.get("High", 0)
    count_vhigh = cat_counts.get("Very High", 0)

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
        max-width: 410px;
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #2c3e50;
        line-height: 1.4;
    ">
        <div style="font-size: 15px; font-weight: bold; color: #1a252f; margin-bottom: 2px;">
            Western Ghats Dynamic Landslide Hazard Map
        </div>
        <div style="font-size: 11px; color: #7f8c8d; margin-bottom: 8px;">
            Stage 6E Deliverable &nbsp;|&nbsp; 14 DEM Tiles (1,582 Locations)
        </div>

        <div style="display: flex; gap: 8px; margin-bottom: 8px; font-size: 11px;">
            <div style="background: #eef2f5; padding: 4px 7px; border-radius: 4px;">
                <b>182</b> Events
            </div>
            <div style="background: #eef2f5; padding: 4px 7px; border-radius: 4px;">
                <b>1,400</b> Background
            </div>
            <div style="background: #e8f8f5; color: #117864; padding: 4px 7px; border-radius: 4px; font-weight: bold;">
                Latency: {latency_hours:.1f}h (SUCCESS)
            </div>
        </div>

        <div style="font-size: 11px; color: #444; border-top: 1px solid #ecf0f1; padding-top: 6px; margin-bottom: 8px;">
            <b>Observation Window (UTC):</b><br>
            {safe_html(obs_start_utc)} &nbsp;→&nbsp; {safe_html(obs_end_utc)}
        </div>

        <div style="font-size: 11px; font-weight: bold; margin-bottom: 5px; color: #2c3e50; text-transform: uppercase; letter-spacing: 0.5px;">
            Dynamic Screening Categories (Regional Quantiles)
        </div>

        <div style="display: grid; grid-template-columns: 1fr; gap: 4px; font-size: 11.5px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['Low']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Low Risk (&lt; {p50:.2f}, &lt; P50)</span>
                <b>{count_low:,} ({count_low/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['Moderate']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Moderate Risk ({p50:.2f}–{p80:.2f}, P50–P80)</span>
                <b>{count_mod:,} ({count_mod/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['High']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>High Risk ({p80:.2f}–{p95:.2f}, P80–P95)</span>
                <b>{count_high:,} ({count_high/len(df)*100:.1f}%)</b>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span style="display:inline-block; width:12px; height:12px; background:{CATEGORY_COLORS['Very High']}; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>Very High Risk (≥ {p95:.2f}, P95+)</span>
                <b style="color:#c0392b;">{count_vhigh:,} ({count_vhigh/len(df)*100:.1f}%)</b>
            </div>
        </div>

        <div style="font-size: 9.5px; color: #7f8c8d; line-height: 1.35; border-top: 1px solid #ecf0f1; padding-top: 6px;">
            <b>NOTICE:</b> {DISCLAIMER_TEXT}. Relative indices are uncalibrated situational indicators and do not represent official early warnings or physical landslide failure probabilities.
        </div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(dashboard_card_html))

    # 8. Save Map HTML
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_html_path))

    print(f"\n4. Successfully generated GIS map HTML: {output_html_path.name}")
    print(f"   - File size: {output_html_path.stat().st_size:,} bytes")
    print(f"   - Location count: {len(df):,}")
    print("   - Total layers: 5 feature groups + 2 tile layers")
    print("=" * 75)

    return {
        "status": "SUCCESS",
        "output_path": output_html_path,
        "location_count": len(df),
        "layers_count": 5,
        "basemaps_count": 2,
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
