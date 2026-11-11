"""
Stage 6E Deliverable: Western Ghats Dynamic Landslide Hazard Risk GeoJSON Exporter.

Standardized GIS GeoJSON FeatureCollection export for the Western Ghats regional coverage
(14 DEM tiles, 1,582 spatial locations: 182 historical landslide events + 1,400 provisional
background samples) using validated Stage 6D dynamic landslide hazard risk predictions.

Conforms to RFC 7946 GeoJSON specification with EPSG:4326 (WGS 84) coordinate reference system.
Includes explicit validation gates, boolean/null type preservation, strict Stage 6D metadata
threshold validation with zero fallbacks, and overwrite protection.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# =====================================================================
# PATH CONSTANTS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.gis_refresh_utils import (
    atomic_write_json,
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

DEFAULT_OUTPUT_GEOJSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_dynamic_risk"
    / "western_ghats_dynamic_hazard_risk.geojson"
)

# Approved regional scope contracts
EXPECTED_LOCATIONS_COUNT = 1582
EXPECTED_TILES_COUNT = 14
EXPECTED_HISTORICAL_EVENTS_COUNT = 182
EXPECTED_BACKGROUND_SAMPLES_COUNT = 1400

# Geographic bounds for the 14 Western Ghats DEM tiles
LAT_BOUNDS = (8.0, 22.0)
LON_BOUNDS = (70.0, 80.0)


# =====================================================================
# STRICT THRESHOLD VALIDATION GATE (NO SILENT FALLBACKS)
# =====================================================================

def validate_stage6d_metadata_thresholds(metadata: Dict[str, Any]) -> Dict[str, float]:
    """
    Strictly validate Stage 6D relative category cutoffs dynamically from metadata.
    Never rejects fresh runs against stale hardcoded cutoffs.
    """
    return validate_dynamic_metadata_thresholds(metadata)


# =====================================================================
# SAFE SERIALIZATION HELPERS
# =====================================================================

def safe_value(value: Any) -> Any:
    """
    Convert numpy/pandas types to standard JSON-compatible types.
    Preserves booleans strictly as bool (true/false) before checking integers.
    Maps non-finite floats (NaN, +inf, -inf) and nulls to None (serializes to JSON null).
    Never converts missing values to zero.
    """
    if value is None or pd.isna(value):
        return None

    # Handle Python and NumPy booleans FIRST before integers (since bool is a subclass of int)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # Unwrap single-element NumPy scalar if present
    if hasattr(value, "item") and not isinstance(value, (np.ndarray, list)):
        try:
            scalar = value.item()
            if isinstance(scalar, bool):
                return bool(scalar)
            value = scalar
        except Exception:
            pass

    # Re-check boolean after item unwrap
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # Handle integers
    if isinstance(value, (int, np.integer)):
        return int(value)

    # Handle floating point numbers (reject non-finite)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return None
        return float(value)

    # Handle datetime / timestamp
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    return str(value)


# =====================================================================
# GEOJSON EXPORTER ENGINE
# =====================================================================

def export_western_ghats_dynamic_risk_geojson(
    input_csv_path: Path = DEFAULT_INPUT_CSV,
    input_metadata_path: Path = DEFAULT_INPUT_METADATA,
    output_geojson_path: Path = DEFAULT_OUTPUT_GEOJSON,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Export Stage 6D dynamic landslide hazard risk data to a validated GeoJSON FeatureCollection.

    Strict validation gates:
    - Verifies overwrite protection: refuses to overwrite existing file unless overwrite=True.
    - Exactly 1,582 records.
    - Exactly 14 unique DEM tiles.
    - Exactly 182 historical_landslide records.
    - Exactly 1,400 provisional_background records.
    - Non-null, strictly unique record_id and sample_id.
    - Finite geographic coordinates within Western Ghats bounding bounds.
    - Validates 1,582 Point features and confirms output feature count equals source row count.
    - Strictly validates Stage 6D metadata relative thresholds with zero fallbacks.
    """
    print("\n" + "=" * 75)
    print("STAGE 6E: WESTERN GHATS DYNAMIC RISK GEOJSON EXPORTER")
    print("=" * 75)

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input Stage 6D CSV not found: {input_csv_path}")

    # Overwrite & Backup protection gate
    if output_geojson_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output GeoJSON already exists: {output_geojson_path}. "
                "Use --overwrite to explicitly replace existing file."
            )
        create_timestamped_backup([output_geojson_path], output_geojson_path.parent / "backup")

    # 1. Load input dataset
    print(f"\n1. Loading Stage 6D risk dataset: {input_csv_path.name}")
    df = pd.read_csv(input_csv_path, low_memory=False)

    # 2. Strict Pre-Export Validations
    if len(df) != EXPECTED_LOCATIONS_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_LOCATIONS_COUNT} records; found {len(df)}"
        )

    unique_tiles = sorted(df["tile"].unique().tolist())
    if len(unique_tiles) != EXPECTED_TILES_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_TILES_COUNT} unique DEM tiles; found {len(unique_tiles)}: {unique_tiles}"
        )

    # Sample type validation
    if "sample_type" not in df.columns:
        raise ValueError("Missing 'sample_type' column in source CSV")

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

    # Unique identifier validation
    if "record_id" not in df.columns or "sample_id" not in df.columns:
        raise ValueError("Missing 'record_id' or 'sample_id' columns in source CSV")

    if df["record_id"].isnull().any():
        raise ValueError("Null values detected in 'record_id'")
    if df["sample_id"].isnull().any():
        raise ValueError("Null values detected in 'sample_id'")

    if not df["record_id"].is_unique:
        duplicates = df[df["record_id"].duplicated()]["record_id"].tolist()
        raise ValueError(f"Duplicate 'record_id' values detected: {duplicates[:5]}")

    if not df["sample_id"].is_unique:
        duplicates = df[df["sample_id"].duplicated()]["sample_id"].tolist()
        raise ValueError(f"Duplicate 'sample_id' values detected: {duplicates[:5]}")

    # Coordinate validation
    lats = pd.to_numeric(df["latitude"], errors="coerce")
    lons = pd.to_numeric(df["longitude"], errors="coerce")
    if lats.isna().any() or lons.isna().any():
        raise ValueError("Missing or non-numeric geographic coordinates detected.")

    if not (np.isfinite(lats).all() and np.isfinite(lons).all()):
        raise ValueError("Non-finite (inf/NaN) geographic coordinates detected.")

    if not (lats.between(LAT_BOUNDS[0], LAT_BOUNDS[1]).all() and lons.between(LON_BOUNDS[0], LON_BOUNDS[1]).all()):
        raise ValueError("Coordinates outside approved Western Ghats bounding bounds.")

    print(f"   Validated {len(df)} records ({hist_event_count} events, {bg_sample_count} background) across {len(unique_tiles)} DEM tiles.")

    # 3. Load and Strictly Validate Stage 6D Metadata
    if not input_metadata_path.exists():
        raise FileNotFoundError(f"Required Stage 6D metadata file not found: {input_metadata_path}")

    with open(input_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   Loaded metadata from: {input_metadata_path.name}")

    # Strict threshold validation with ZERO fallbacks
    validated_cutoffs = validate_stage6d_metadata_thresholds(metadata)

    # 4. Construct GeoJSON FeatureCollection
    print("\n2. Serializing 1,582 Point features to GeoJSON format...")
    features: List[Dict[str, Any]] = []

    # Property columns to include in GeoJSON feature properties
    properties_to_export = [
        "record_id",
        "sample_id",
        "sample_type",
        "label",
        "tile",
        "dem_region",
        "subregion",
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "aspect_sin",
        "aspect_cos",
        "aspect_flat",
        "nearest_event_distance_m",
        "terrain_rf_pred_balanced",
        "terrain_rf_prob_balanced",
        "terrain_rf_pred_unweighted",
        "terrain_rf_prob_unweighted",
        "terrain_baseline_screening_tier",
        "gpm_latest_rainfall_rate_mm_hr",
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_24h_accumulation_mm",
        "gpm_24h_complete",
        "gpm_matching_distance_km",
        "norm_rain_1h",
        "norm_rain_3h",
        "norm_rain_6h",
        "norm_rain_12h",
        "norm_rain_24h",
        "rainfall_intensity_index",
        "rainfall_percentile_24h",
        "dynamic_risk_index",
        "primary_screening_category",
        "risk_category_relative",
        "risk_tier_fixed",
        "risk_calculation_status",
        "risk_calculation_timestamp_utc",
        "gpm_observation_start_utc",
        "gpm_observation_end_utc",
        "screening_disclaimer",
    ]

    for _, row in df.iterrows():
        props: Dict[str, Any] = {}
        for col in properties_to_export:
            if col in row:
                props[col] = safe_value(row[col])

        # Fill observation window timestamps from metadata if missing in row
        if "gpm_observation_start_utc" not in props or props["gpm_observation_start_utc"] is None:
            props["gpm_observation_start_utc"] = metadata.get("gpm_observation_metadata", {}).get("observation_start_utc")
        if "gpm_observation_end_utc" not in props or props["gpm_observation_end_utc"] is None:
            props["gpm_observation_end_utc"] = metadata.get("gpm_observation_metadata", {}).get("observation_end_utc")

        # Explicit rainfall display status to prevent incomplete data being mistaken for zero
        if props.get("gpm_24h_accumulation_mm") is None:
            props["rainfall_display_status"] = "UNAVAILABLE"
        else:
            props["rainfall_display_status"] = "VALID_MEASUREMENT"

        lon_val = safe_value(row["longitude"])
        lat_val = safe_value(row["latitude"])

        # Validate geometry coordinate finiteness
        if lon_val is None or lat_val is None or not (LAT_BOUNDS[0] <= lat_val <= LAT_BOUNDS[1]) or not (LON_BOUNDS[0] <= lon_val <= LON_BOUNDS[1]):
            raise ValueError(f"Invalid coordinate for record {row.get('record_id')}: ({lon_val}, {lat_val})")

        feature = {
            "type": "Feature",
            "id": str(row["record_id"]),
            "geometry": {
                "type": "Point",
                "coordinates": [lon_val, lat_val],
            },
            "properties": props,
        }
        features.append(feature)

    # Validate output feature count equals source row count
    if len(features) != len(df):
        raise ValueError(f"Feature count ({len(features)}) does not match source row count ({len(df)})")

    geojson_doc: Dict[str, Any] = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "Western Ghats Limited Pilot Dynamic Landslide Hazard Risk",
            "stage": "Stage 6E Deliverable",
            "coverage_region": "Western Ghats, India (Limited Pilot — 14 Validated DEM Tiles Only)",
            "coverage_scope": "LIMITED_PILOT",
            "complete_region_coverage": False,
            "coverage_note": "Limited Western Ghats pilot: 14 validated DEM tiles, 1,582 locations (182 historical events + 1,400 background samples). Does not imply complete Western Ghats coverage; no duplicated historical records.",
            "total_locations": len(features),
            "historical_events_count": hist_event_count,
            "provisional_background_count": bg_sample_count,
            "dem_tiles_count": len(unique_tiles),
            "tiles": unique_tiles,
            "gpm_observation_metadata": metadata.get("gpm_observation_metadata", {}),
            "risk_formula": {
                "formula": "0.50 * terrain_rf_prob_balanced + 0.50 * (normalized_rainfall * 100)",
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": validated_cutoffs["P50_low_to_moderate"],
                    "P80_moderate_to_high": validated_cutoffs["P80_moderate_to_high"],
                    "P95_high_to_very_high": validated_cutoffs["P95_high_to_very_high"],
                },
            },
            "export_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "disclaimer": "EXPERIMENTAL_PROVISIONAL_SCREENING_ONLY_NOT_OFFICIAL_WARNING",
        },
        "features": features,
    }

    # 5. Save GeoJSON Atomically
    atomic_write_json(output_geojson_path, geojson_doc)

    print(f"\n3. Successfully exported GeoJSON: {output_geojson_path.name}")
    print(f"   - File size: {output_geojson_path.stat().st_size:,} bytes")
    print(f"   - Feature count: {len(features):,}")
    print("=" * 75)

    return {
        "status": "SUCCESS",
        "output_path": output_geojson_path,
        "feature_count": len(features),
        "tiles_count": len(unique_tiles),
        "historical_events_count": hist_event_count,
        "provisional_background_count": bg_sample_count,
    }


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Export Western Ghats Dynamic Hazard Risk to GeoJSON")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to Stage 6D CSV")
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA, help="Path to Stage 6D metadata JSON")
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT_GEOJSON, help="Path to output GeoJSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")

    args = parser.parse_args()

    try:
        export_western_ghats_dynamic_risk_geojson(
            input_csv_path=args.input_csv,
            input_metadata_path=args.input_metadata,
            output_geojson_path=args.output_geojson,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] GeoJSON export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
