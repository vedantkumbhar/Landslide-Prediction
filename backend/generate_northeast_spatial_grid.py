"""
Stage 7.1: Northeast India Spatial Prediction Grid Generator.

Generates a regular 5 km spatial prediction grid across the valid 30m SRTM DEM coverage
for Northeast India (33 processed DEM tiles spanning UTM zones 45N, 46N, and 47N).

Core Requirements:
1. Generate regular 5,000-meter grid points using native projected UTM CRS for each DEM tile.
2. Dynamically support all three regional UTM zones (EPSG:32645, EPSG:32646, EPSG:32647)
   without assuming a single global projection.
3. Transform final projected coordinates to WGS84 (latitude/longitude in EPSG:4326).
4. Extract the exact five terrain features mandated by the regional Random Forest contract:
   - elevation_m
   - slope_degrees
   - aspect_sin
   - aspect_cos
   - aspect_flat
5. Strictly exclude nodata (-9999.0), non-finite, out-of-range, and incomplete terrain samples.
6. Handle overlapping tile boundaries, remove duplicate spatial locations, and enforce unique grid IDs.
7. Retain unambiguous DEM tile provenance for every valid grid point.
8. Output validated CSV and comprehensive JSON metadata validation report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM_DIR = PROJECT_ROOT / "data" / "processed" / "dem" / "northeast"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "northeast_spatial_grid"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "northeast_india_5km_spatial_grid.csv"
DEFAULT_VALIDATION_REPORT = DEFAULT_OUTPUT_DIR / "northeast_spatial_grid_validation_report.json"

# Regional Model Contract
MANDATORY_TERRAIN_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

# Geographic & Configuration Constraints
GRID_SPACING_METERS = 5000.0
NODATA_VALUE_DEM = -9999.0
EXCLUDED_TILE_DIRS: Set[str] = {"N27E088_backup_before_fix"}

NORTHEAST_LAT_BOUNDS = (22.0, 29.5)
NORTHEAST_LON_BOUNDS = (88.0, 97.5)


def discover_northeast_dem_tiles(dem_dir: Path) -> List[str]:
    """Discover all active, valid Northeast India DEM tile directories."""
    if not dem_dir.exists():
        raise FileNotFoundError(f"Northeast DEM directory does not exist: {dem_dir}")

    tiles = []
    for item in sorted(dem_dir.iterdir()):
        if item.is_dir() and item.name.startswith("N") and "E" in item.name:
            if item.name not in EXCLUDED_TILE_DIRS:
                tiles.append(item.name)

    if not tiles:
        raise ValueError(f"No valid DEM tiles found in {dem_dir}")

    return sorted(tiles)


def get_tile_raster_paths(dem_dir: Path, tile: str) -> Dict[str, Path]:
    """
    Locate elevation, slope, and aspect rasters for a given tile.
    Supports standard `<tile>/metric/*.tif` as well as root `<tile>/*.tif`.
    """
    tile_dir = dem_dir / tile
    metric_dir = tile_dir / "metric"

    paths = {}
    for layer in ["elevation", "slope", "aspect"]:
        p_metric = metric_dir / f"{layer}.tif"
        p_root = tile_dir / f"{layer}.tif"

        if p_metric.exists():
            paths[layer] = p_metric
        elif p_root.exists():
            paths[layer] = p_root
        else:
            raise FileNotFoundError(f"Missing {layer}.tif for tile {tile} in both {metric_dir} and {tile_dir}")

    return paths


def compute_aspect_derivatives(aspect_deg: float) -> Tuple[float, float, int]:
    """
    Compute aspect_sin, aspect_cos, and aspect_flat matching regional training pipeline:
    - aspect < 0 (flat/undefined): sin=0.0, cos=0.0, flat=1
    - aspect >= 0: sin=sin(rad), cos=cos(rad), flat=0
    """
    if aspect_deg < 0.0:
        return 0.0, 0.0, 1

    rad = math.radians(aspect_deg)
    return float(math.sin(rad)), float(math.cos(rad)), 0


def generate_northeast_spatial_grid(
    dem_dir: Path = DEFAULT_DEM_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT,
    spacing_m: float = GRID_SPACING_METERS,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Generate 5 km spatial prediction grid across all 33 Northeast India DEM tiles.
    """
    # Overwrite protection
    if not overwrite:
        if output_csv.exists():
            raise FileExistsError(
                f"Output CSV already exists: {output_csv}. Use --overwrite to replace."
            )
        if validation_report_path.exists():
            raise FileExistsError(
                f"Validation report already exists: {validation_report_path}. Use --overwrite to replace."
            )

    print("=" * 75)
    print("STAGE 7.1: NORTHEAST INDIA SPATIAL PREDICTION GRID GENERATOR")
    print("=" * 75)

    tiles = discover_northeast_dem_tiles(dem_dir)
    print(f"\n1. Discovered {len(tiles)} active DEM tiles in {dem_dir.name}:")
    print(f"   Tiles: {', '.join(tiles[:8])} ... {', '.join(tiles[-4:])}")

    raw_candidates_total = 0
    excluded_counts = {
        "nodata_elevation": 0,
        "nodata_slope": 0,
        "nodata_aspect": 0,
        "outside_raster_bounds": 0,
        "outside_nominal_tile_footprint": 0,
        "duplicate_coordinates_removed": 0,
    }

    tile_point_counts: Dict[str, int] = {}
    utm_zones_used: Set[str] = set()
    raw_records: List[Dict[str, Any]] = []

    print(f"\n2. Sampling terrain rasters at {spacing_m / 1000.0:.1f} km grid spacing...")

    for tile in tiles:
        t_lat = int(tile[1:3])
        t_lon = int(tile[4:7])

        paths = get_tile_raster_paths(dem_dir, tile)

        with rasterio.open(paths["elevation"]) as src_elev, \
             rasterio.open(paths["slope"]) as src_slope, \
             rasterio.open(paths["aspect"]) as src_aspect:

            crs = src_elev.crs
            utm_zones_used.add(str(crs))
            bounds = src_elev.bounds
            inv_transform = ~src_elev.transform

            elev_arr = src_elev.read(1)
            slope_arr = src_slope.read(1)
            aspect_arr = src_aspect.read(1)

            nodata_e = src_elev.nodata if src_elev.nodata is not None else NODATA_VALUE_DEM
            nodata_s = src_slope.nodata if src_slope.nodata is not None else NODATA_VALUE_DEM
            nodata_a = src_aspect.nodata if src_aspect.nodata is not None else NODATA_VALUE_DEM

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

            # Generate grid coordinates aligned to round spacing_m multiples in native UTM
            xs = np.arange(math.ceil(bounds.left / spacing_m) * spacing_m, bounds.right, spacing_m)
            ys = np.arange(math.ceil(bounds.bottom / spacing_m) * spacing_m, bounds.top, spacing_m)

            tile_valid_count = 0

            for x in xs:
                for y in ys:
                    raw_candidates_total += 1
                    col, row = [int(round(v)) for v in inv_transform @ (x, y)]

                    # Validate within raster array
                    if not (0 <= row < elev_arr.shape[0] and 0 <= col < elev_arr.shape[1]):
                        excluded_counts["outside_raster_bounds"] += 1
                        continue

                    e_val = elev_arr[row, col]
                    s_val = slope_arr[row, col]
                    a_val = aspect_arr[row, col]

                    # Nodata & finiteness checks
                    if e_val == nodata_e or not np.isfinite(e_val):
                        excluded_counts["nodata_elevation"] += 1
                        continue

                    if s_val == nodata_s or not np.isfinite(s_val) or s_val < 0.0 or s_val > 90.0:
                        excluded_counts["nodata_slope"] += 1
                        continue

                    if a_val == nodata_a or not np.isfinite(a_val):
                        excluded_counts["nodata_aspect"] += 1
                        continue

                    # Transform to WGS84
                    lon, lat = transformer.transform(x, y)

                    # Nominal tile footprint assignment: [t_lon, t_lon + 1.0) x [t_lat, t_lat + 1.0)
                    # Resolves projection overlaps at tile boundaries
                    if not (t_lon <= lon < t_lon + 1.0 and t_lat <= lat < t_lat + 1.0):
                        excluded_counts["outside_nominal_tile_footprint"] += 1
                        continue

                    asp_sin, asp_cos, asp_flat = compute_aspect_derivatives(float(a_val))

                    raw_records.append({
                        "tile": tile,
                        "latitude": round(float(lat), 6),
                        "longitude": round(float(lon), 6),
                        "elevation_m": round(float(e_val), 2),
                        "slope_degrees": round(float(s_val), 2),
                        "aspect_sin": round(asp_sin, 6),
                        "aspect_cos": round(asp_cos, 6),
                        "aspect_flat": int(asp_flat),
                        "utm_zone": str(crs),
                    })
                    tile_valid_count += 1

            tile_point_counts[tile] = tile_valid_count

    df = pd.DataFrame(raw_records)
    print(f"\n3. Extracted {len(df):,} candidate grid points across {len(tiles)} tiles.")

    # Deduplication check
    initial_count = len(df)
    df = df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)
    duplicate_count = initial_count - len(df)
    excluded_counts["duplicate_coordinates_removed"] = duplicate_count

    if duplicate_count > 0:
        print(f"   Removed {duplicate_count:,} duplicate spatial locations at overlapping boundaries.")
    else:
        print("   Zero duplicate coordinates detected across tile boundaries.")

    # Assign deterministic, sequential Grid IDs
    df.insert(0, "grid_id", [f"NE_GRID_{i+1:05d}" for i in range(len(df))])

    # Reorder columns strictly
    output_cols = [
        "grid_id",
        "latitude",
        "longitude",
        "tile",
        "elevation_m",
        "slope_degrees",
        "aspect_sin",
        "aspect_cos",
        "aspect_flat",
    ]
    df_out = df[output_cols]

    # Validate output schema, finiteness and bounds
    print("\n4. Validating output dataset integrity against model contract...")
    _validate_grid_dataset(df_out)

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\n5. Successfully saved Northeast India spatial grid:")
    print(f"   - File: {output_csv}")
    print(f"   - Grid points: {len(df_out):,}")
    print(f"   - File size: {output_csv.stat().st_size:,} bytes")

    # Generate Validation Report
    stats: Dict[str, Dict[str, float]] = {}
    for col in MANDATORY_TERRAIN_FEATURES:
        series = df_out[col]
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "P25": round(float(series.quantile(0.25)), 4),
            "P50": round(float(series.quantile(0.50)), 4),
            "P75": round(float(series.quantile(0.75)), 4),
        }

    report = {
        "stage": "Stage 7.1 — Northeast India Spatial Prediction Grid",
        "status": "SUCCESS",
        "region": "Northeast India",
        "grid_spacing_km": spacing_m / 1000.0,
        "total_valid_grid_locations": len(df_out),
        "raw_candidate_points_evaluated": raw_candidates_total,
        "excluded_samples_breakdown": excluded_counts,
        "dem_tile_count": len(tiles),
        "active_dem_tiles": tiles,
        "tile_point_distribution": {
            "min_points_per_tile": int(min(tile_point_counts.values())),
            "max_points_per_tile": int(max(tile_point_counts.values())),
            "mean_points_per_tile": round(float(np.mean(list(tile_point_counts.values()))), 2),
            "counts_by_tile": tile_point_counts,
        },
        "projected_utm_zones_represented": sorted(list(utm_zones_used)),
        "geographic_bounding_box": {
            "latitude_min": float(df_out["latitude"].min()),
            "latitude_max": float(df_out["latitude"].max()),
            "longitude_min": float(df_out["longitude"].min()),
            "longitude_max": float(df_out["longitude"].max()),
        },
        "model_contract_verification": {
            "mandatory_features": MANDATORY_TERRAIN_FEATURES,
            "all_features_present": True,
            "all_values_finite": True,
            "zero_nodata_leakage": True,
            "zero_duplicate_coordinates": bool(df_out[["latitude", "longitude"]].duplicated().sum() == 0),
            "unique_grid_ids": bool(df_out["grid_id"].nunique() == len(df_out)),
        },
        "terrain_feature_statistics": stats,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "disclaimer": "EXPERIMENTAL_SPATIAL_PREDICTION_GRID_FOR_TERRAIN_SCREENING_ONLY",
    }

    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"   - Validation report saved: {validation_report_path.name}")
    print("=" * 75)

    return report


def _validate_grid_dataset(df: pd.DataFrame) -> None:
    """Enforce strict model contract and geographic bounds."""
    # Column presence
    for col in MANDATORY_TERRAIN_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory model feature: {col}")

    # No NaN or infinite values
    for col in MANDATORY_TERRAIN_FEATURES + ["latitude", "longitude"]:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values found in {col}")

    # Slope range [0, 90]
    if (df["slope_degrees"] < 0.0).any() or (df["slope_degrees"] > 90.0).any():
        raise ValueError("slope_degrees contains values outside [0.0, 90.0]")

    # Aspect sin/cos range [-1, 1]
    if (df["aspect_sin"] < -1.0001).any() or (df["aspect_sin"] > 1.0001).any():
        raise ValueError("aspect_sin contains values outside [-1.0, 1.0]")
    if (df["aspect_cos"] < -1.0001).any() or (df["aspect_cos"] > 1.0001).any():
        raise ValueError("aspect_cos contains values outside [-1.0, 1.0]")

    # Aspect flat binary
    if not set(df["aspect_flat"].unique()).issubset({0, 1}):
        raise ValueError("aspect_flat must contain only 0 and 1")

    # Geographic bounds
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    if not (NORTHEAST_LAT_BOUNDS[0] <= lat_min and lat_max <= NORTHEAST_LAT_BOUNDS[1]):
        raise ValueError(f"Latitudes out of bounds: [{lat_min}, {lat_max}]")
    if not (NORTHEAST_LON_BOUNDS[0] <= lon_min and lon_max <= NORTHEAST_LON_BOUNDS[1]):
        raise ValueError(f"Longitudes out of bounds: [{lon_min}, {lon_max}]")

    # Unique identifiers
    if df["grid_id"].nunique() != len(df):
        raise ValueError("grid_id column contains duplicate values")

    # Unique coordinates
    if df.duplicated(subset=["latitude", "longitude"]).any():
        raise ValueError("Output dataset contains duplicate spatial coordinates")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5 km spatial prediction grid for Northeast India.")
    parser.add_argument("--dem-dir", type=Path, default=DEFAULT_DEM_DIR, help="Path to Northeast DEM directory")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to output CSV")
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT, help="Path to report JSON")
    parser.add_argument("--spacing-km", type=float, default=5.0, help="Grid spacing in kilometers (default: 5.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        generate_northeast_spatial_grid(
            dem_dir=args.dem_dir,
            output_csv=args.output_csv,
            validation_report_path=args.validation_report,
            spacing_m=args.spacing_km * 1000.0,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] Grid generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
