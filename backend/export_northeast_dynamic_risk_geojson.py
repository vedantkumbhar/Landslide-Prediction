"""
Stage 7.5 Deliverable: Northeast India Dynamic Landslide Hazard Risk GeoJSON Exporter.

Standardized GIS GeoJSON FeatureCollection export for the Northeast India 5 km regular grid
(33 DEM tiles, 14,550 spatial locations) using approved Stage 7.4 dynamic landslide hazard risk predictions.

Conforms to RFC 7946 GeoJSON specification with EPSG:4326 (WGS 84) coordinate reference system.
Includes explicit validation gates, boolean/null type preservation, strict Stage 7.4 metadata
threshold validation with zero fallbacks, coordinate bounding audits, and overwrite protection.
"""

from __future__ import annotations

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
    / "northeast_dynamic_risk"
    / "northeast_india_5km_dynamic_hazard_risk.csv"
)

DEFAULT_INPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_dynamic_risk"
    / "northeast_dynamic_risk_metadata.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_dynamic_risk_map"
)

DEFAULT_OUTPUT_GEOJSON = (
    DEFAULT_OUTPUT_DIR
    / "northeast_india_dynamic_hazard_risk.geojson"
)

# Approved regional scope contracts
EXPECTED_LOCATIONS_COUNT = 14550
EXPECTED_TILES_COUNT = 33

# Geographic bounds for the 33 Northeast India DEM tiles
LAT_BOUNDS = (21.0, 30.0)
LON_BOUNDS = (87.0, 98.0)


# =====================================================================
# STRICT THRESHOLD VALIDATION GATE (NO SILENT FALLBACKS)
# =====================================================================

def validate_stage74_metadata_thresholds(metadata: Dict[str, Any]) -> Dict[str, float]:
    """
    Strictly validate Stage 7.4 relative category cutoffs dynamically from metadata.
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

def export_northeast_dynamic_risk_geojson(
    input_csv_path: Path = DEFAULT_INPUT_CSV,
    input_metadata_path: Path = DEFAULT_INPUT_METADATA,
    output_geojson_path: Path = DEFAULT_OUTPUT_GEOJSON,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Export Stage 7.4 dynamic landslide hazard risk data to a validated RFC 7946 GeoJSON FeatureCollection.

    Strict validation gates:
    - Overwrite protection: refuses to overwrite existing file unless overwrite=True.
    - Exactly 14,550 records.
    - Exactly 33 unique DEM tiles.
    - Non-null, strictly unique grid_id.
    - Finite geographic coordinates within Northeast India bounding bounds.
    - Validates 14,550 Point features and confirms output feature count equals source row count.
    - Strictly validates Stage 7.4 metadata relative thresholds with zero fallbacks.
    """
    print("\n" + "=" * 75)
    print("STAGE 7.5: NORTHEAST INDIA DYNAMIC RISK GEOJSON EXPORTER")
    print("=" * 75)

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input Stage 7.4 CSV not found: {input_csv_path}")

    # Overwrite & Backup protection gate
    if output_geojson_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output GeoJSON already exists: {output_geojson_path}. "
                "Use --overwrite to explicitly replace existing file."
            )
        create_timestamped_backup([output_geojson_path], output_geojson_path.parent / "backup")

    # 1. Load input dataset
    print(f"\n1. Loading Stage 7.4 risk dataset: {input_csv_path.name}")
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

    # Unique grid_id validation
    if "grid_id" not in df.columns:
        raise ValueError("Missing 'grid_id' column in source CSV")

    if df["grid_id"].isnull().any():
        raise ValueError("Null values detected in 'grid_id'")

    if not df["grid_id"].is_unique:
        duplicates = df[df["grid_id"].duplicated()]["grid_id"].tolist()
        raise ValueError(f"Duplicate 'grid_id' values detected: {duplicates[:5]}")

    # Coordinate validation
    lats = pd.to_numeric(df["latitude"], errors="coerce")
    lons = pd.to_numeric(df["longitude"], errors="coerce")
    if lats.isna().any() or lons.isna().any():
        raise ValueError("Missing or non-numeric geographic coordinates detected.")

    if not (np.isfinite(lats).all() and np.isfinite(lons).all()):
        raise ValueError("Non-finite (inf/NaN) geographic coordinates detected.")

    if not (lats.between(LAT_BOUNDS[0], LAT_BOUNDS[1]).all() and lons.between(LON_BOUNDS[0], LON_BOUNDS[1]).all()):
        raise ValueError("Coordinates outside approved Northeast India bounding bounds.")

    print(f"   Validated {len(df)} grid records across {len(unique_tiles)} DEM tiles.")

    # 3. Load and Strictly Validate Stage 7.4 Metadata
    if not input_metadata_path.exists():
        raise FileNotFoundError(f"Required Stage 7.4 metadata file not found: {input_metadata_path}")

    with open(input_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   Loaded metadata from: {input_metadata_path.name}")

    # Strict threshold validation with ZERO fallbacks
    validated_cutoffs = validate_stage74_metadata_thresholds(metadata)

    # 4. Construct GeoJSON FeatureCollection
    print("\n2. Serializing 14,550 Point features to GeoJSON format...")
    features: List[Dict[str, Any]] = []

    properties_to_export = [
        "grid_id",
        "tile",
        "elevation_m",
        "slope_degrees",
        "aspect_sin",
        "aspect_cos",
        "aspect_flat",
        "terrain_rf_prob_balanced",
        "terrain_rf_pred_balanced",
        "terrain_baseline_screening_tier",
        "gpm_grid_latitude",
        "gpm_grid_longitude",
        "gpm_matching_distance_km",
        "gpm_observation_start_utc",
        "gpm_observation_end_utc",
        "rainfall_refresh_status",
        "gpm_latest_rainfall_rate_mm_hr",
        "gpm_latest_quality_index",
        "gpm_latest_random_error_mm_hr",
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_24h_accumulation_mm",
        "gpm_interval_count",
        "gpm_24h_complete",
        "norm_rain_1h",
        "norm_rain_3h",
        "norm_rain_6h",
        "norm_rain_12h",
        "norm_rain_24h",
        "rainfall_intensity_index",
        "rainfall_percentile_24h",
        "terrain_contribution",
        "rainfall_contribution",
        "dynamic_risk_index",
        "dynamic_risk_interactive",
        "risk_tier_fixed",
        "risk_category_relative",
        "primary_screening_category",
        "risk_calculation_status",
        "risk_calculation_timestamp_utc",
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
            raise ValueError(f"Invalid coordinate for grid point {row.get('grid_id')}: ({lon_val}, {lat_val})")

        feature = {
            "type": "Feature",
            "id": str(row["grid_id"]),
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
            "title": "Northeast India Dynamic Landslide Hazard Risk Grid",
            "stage": "Stage 7.5 Deliverable",
            "coverage_region": "Northeast India",
            "total_locations": len(features),
            "grid_resolution": "5 km regular spatial grid",
            "dem_tiles_count": len(unique_tiles),
            "tiles": unique_tiles,
            "gpm_observation_metadata": metadata.get("gpm_observation_metadata", {}),
            "risk_formula": {
                "formula": "Dynamic_Risk_Index = 100 * (0.50 * terrain_rf_prob_balanced + 0.50 * rainfall_intensity_index)",
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": validated_cutoffs["P50_low_to_moderate"],
                    "P80_moderate_to_high": validated_cutoffs["P80_moderate_to_high"],
                    "P95_high_to_very_high": validated_cutoffs["P95_high_to_very_high"],
                },
            },
            "export_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "disclaimer": "EXPERIMENTAL_PROVISIONAL_NORTHEAST_SCREENING_ONLY_NOT_OFFICIAL_WARNING",
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
    }


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Export Northeast India Dynamic Hazard Risk to GeoJSON")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV, help="Path to Stage 7.4 CSV")
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA, help="Path to Stage 7.4 metadata JSON")
    parser.add_argument("--output-geojson", type=Path, default=DEFAULT_OUTPUT_GEOJSON, help="Path to output GeoJSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")

    args = parser.parse_args()

    try:
        export_northeast_dynamic_risk_geojson(
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
