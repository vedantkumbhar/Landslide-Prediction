"""
Western Ghats Regional NASA GPM IMERG Spatial Integration Pipeline.

Stage 6A Deliverable:
Integrates NASA GPM IMERG Early Run V07C half-hourly satellite precipitation
granules with the expanded 14 Western Ghats DEM tiles (182 historical landslide
events and 1,400 provisional background samples = 1,582 locations).

Methodological & Physical Considerations:
- Spatial Resolution: Native GPM IMERG is 0.1° x 0.1° (~10 km at equator,
  ~11.1 km lat x ~10.7 km lon in Western Ghats). Point-to-pixel mapping preserves
  this native resolution and does NOT downscale or claim 30m resolution for rainfall.
- Distance Constraint: Maximum horizontal matching distance between any point and
  its nearest GPM pixel center must be <= 8.0 km (consistent with the 0.1° grid half-diagonal).
- Accumulations: Rainfall rate is in mm/hr for 30-minute intervals (depth = rate * 0.5 hr).
  Accumulations computed across 1h (2 granules), 3h (6 granules), 6h (12 granules),
  12h (24 granules), and 24h (48 granules) windows.
- Timestamp Integrity: All timestamps are strictly UTC and formatted in ISO-8601.
- Completeness & NoData: Missing granules, NoData fill values (-9999.9), and out-of-range
  values are tracked per window. Incomplete windows produce NaN rather than silently zeroing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVENTS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_event_features.csv"
)

DEFAULT_BACKGROUND_CSV = (
    PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_background_samples_v2.csv"
)

DEFAULT_GRANULE_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "satellite" / "gpm_imerge_previous12h",
    PROJECT_ROOT / "data" / "raw" / "satellite" / "gpm_imerge_series_12h",
]

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "rainfall" / "western_ghats_regional_gpm_rainfall_features.csv"
)

EXPECTED_EVENTS_COUNT = 182
EXPECTED_BACKGROUND_COUNT = 1400
EXPECTED_TOTAL_COUNT = 1582
EXPECTED_TILE_COUNT = 14

EXPECTED_GRANULES_24H = 48
GRANULE_INTERVAL_SECONDS = 1800  # 30 minutes
EARTH_RADIUS_KM = 6371.0
MAX_GRID_MATCH_DISTANCE_KM = 8.0

GPM_PRODUCT_NAME = "NASA_GPM_IMERG_V07C_Early"
GPM_NATIVE_RESOLUTION = "0.1_deg_geographic"

ACCUMULATION_WINDOWS_HOURS = [1, 3, 6, 12, 24]


# =====================================================================
# GEODETIC & SPATIAL HELPERS
# =====================================================================

def haversine_distance_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Compute great-circle distance in kilometers using the Haversine formula."""
    r = EARTH_RADIUS_KM
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


def parse_granule_times(filename: str) -> Tuple[datetime, datetime]:
    """
    Parse UTC start and end timestamps from a standard GPM IMERG filename.
    Format: 3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.XXXX.V07C.HDF5
    """
    pattern = r"3IMERG\.(\d{8})-S(\d{6})-E(\d{6})"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Unrecognized GPM IMERG filename format: {filename}")

    date_str, start_str, end_str = match.groups()
    start_dt = datetime.strptime(date_str + start_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(date_str + end_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return start_dt, end_dt


# =====================================================================
# DATASET LOADING & UNIFICATION
# =====================================================================

def load_western_ghats_locations(
    events_path: Path,
    background_path: Path,
) -> pd.DataFrame:
    """
    Load and unify Western Ghats historical events (182) and background samples (1,400).
    Ensures ID uniqueness, schema consistency, coordinate validity, and tile completeness.
    """
    if not events_path.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_path}")
    if not background_path.exists():
        raise FileNotFoundError(f"Background CSV not found: {background_path}")

    events_raw = pd.read_csv(events_path, low_memory=False)
    bg_raw = pd.read_csv(background_path, low_memory=False)

    if len(events_raw) != EXPECTED_EVENTS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS_COUNT} historical events; found {len(events_raw)} in {events_path}"
        )
    if len(bg_raw) != EXPECTED_BACKGROUND_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BACKGROUND_COUNT} background samples; found {len(bg_raw)} in {background_path}"
        )

    # 1. Prepare Events DataFrame
    events_df = pd.DataFrame()
    events_df["record_id"] = "event_" + events_raw["event_id"].astype(str)
    events_df["sample_id"] = events_raw["event_id"].astype(str)
    events_df["event_id"] = pd.to_numeric(events_raw["event_id"], errors="coerce")
    events_df["sample_type"] = "historical_landslide"
    events_df["label"] = 1
    events_df["tile"] = events_raw["dem_tile"].astype(str)
    events_df["dem_region"] = "Western Ghats"
    events_df["subregion"] = events_raw["subregion"].astype(str)
    events_df["latitude"] = pd.to_numeric(events_raw["latitude"], errors="raise").astype(float)
    events_df["longitude"] = pd.to_numeric(events_raw["longitude"], errors="raise").astype(float)
    events_df["elevation_m"] = pd.to_numeric(events_raw["elevation_m"], errors="raise").astype(float)
    events_df["slope_degrees"] = pd.to_numeric(events_raw["slope_degrees"], errors="raise").astype(float)
    events_df["aspect_degrees"] = pd.to_numeric(events_raw["aspect_degrees"], errors="raise").astype(float)
    events_df["aspect_sin"] = pd.to_numeric(events_raw["aspect_sin"], errors="raise").astype(float)
    events_df["aspect_cos"] = pd.to_numeric(events_raw["aspect_cos"], errors="raise").astype(float)
    events_df["aspect_flat"] = pd.to_numeric(events_raw["aspect_flat"], errors="raise").astype(int)
    events_df["nearest_event_distance_m"] = 0.0

    # 2. Prepare Background DataFrame
    bg_df = pd.DataFrame()
    bg_df["record_id"] = bg_raw["sample_id"].astype(str)
    bg_df["sample_id"] = bg_raw["sample_id"].astype(str)
    bg_df["event_id"] = np.nan
    bg_df["sample_type"] = "provisional_background"
    bg_df["label"] = 0
    bg_df["tile"] = bg_raw["tile"].astype(str)
    bg_df["dem_region"] = "Western Ghats"
    bg_df["subregion"] = bg_raw["subregion"].astype(str)
    bg_df["latitude"] = pd.to_numeric(bg_raw["latitude"], errors="raise").astype(float)
    bg_df["longitude"] = pd.to_numeric(bg_raw["longitude"], errors="raise").astype(float)
    bg_df["elevation_m"] = pd.to_numeric(bg_raw["elevation_m"], errors="raise").astype(float)
    bg_df["slope_degrees"] = pd.to_numeric(bg_raw["slope_degrees"], errors="raise").astype(float)
    bg_df["aspect_degrees"] = pd.to_numeric(bg_raw["aspect_degrees"], errors="raise").astype(float)
    bg_df["aspect_sin"] = pd.to_numeric(bg_raw["aspect_sin"], errors="raise").astype(float)
    bg_df["aspect_cos"] = pd.to_numeric(bg_raw["aspect_cos"], errors="raise").astype(float)
    bg_df["aspect_flat"] = pd.to_numeric(bg_raw["aspect_flat"], errors="raise").astype(int)
    bg_df["nearest_event_distance_m"] = pd.to_numeric(
        bg_raw["nearest_event_distance_m"], errors="coerce"
    ).astype(float)

    # 3. Concatenate and Validate
    unified_df = pd.concat([events_df, bg_df], ignore_index=True)

    if len(unified_df) != EXPECTED_TOTAL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_COUNT} unified records; got {len(unified_df)}"
        )
    if unified_df["record_id"].duplicated().any():
        dups = unified_df[unified_df["record_id"].duplicated()]["record_id"].tolist()
        raise ValueError(f"Duplicate record_id detected: {dups[:5]}")
    if unified_df["tile"].nunique() != EXPECTED_TILE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TILE_COUNT} distinct tiles; found {unified_df['tile'].nunique()}"
        )

    return unified_df


# =====================================================================
# GRANULE AUDIT & VALIDATION
# =====================================================================

def load_and_validate_granules(
    granule_dirs: Sequence[Path],
    expected_count: int = EXPECTED_GRANULES_24H,
    interval_seconds: int = GRANULE_INTERVAL_SECONDS,
) -> List[Dict]:
    """
    Scan directories for HDF5 granules, sort chronologically, and verify strict
    contiguous 30-minute intervals without gaps or missing periods.
    """
    all_files: List[Path] = []
    for directory in granule_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Granule directory does not exist: {directory}")
        all_files.extend(directory.glob("*.HDF5"))

    if not all_files:
        raise FileNotFoundError(f"No .HDF5 granules found in provided directories: {granule_dirs}")

    granules: List[Dict] = []
    for filepath in all_files:
        start_dt, end_dt = parse_granule_times(filepath.name)
        granules.append({
            "path": filepath,
            "filename": filepath.name,
            "start": start_dt,
            "end": end_dt,
        })

    # Sort strictly by start timestamp
    granules.sort(key=lambda item: item["start"])

    # Deduplicate in case overlapping directories contain identical granules
    unique_granules: List[Dict] = []
    seen_starts = set()
    for item in granules:
        if item["start"] not in seen_starts:
            seen_starts.add(item["start"])
            unique_granules.append(item)

    if len(unique_granules) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} unique GPM granules; found {len(unique_granules)}."
        )

    # Validate interval continuity
    for idx, granule in enumerate(unique_granules):
        span_seconds = (granule["end"] - granule["start"]).total_seconds()
        # IMERG filenames typically specify 1799 seconds (e.g. 10:30:00 to 10:59:59)
        if span_seconds != (interval_seconds - 1):
            raise RuntimeError(
                f"Granule {granule['filename']} has invalid interval duration: {span_seconds}s"
            )

        if idx > 0:
            prev_granule = unique_granules[idx - 1]
            step_seconds = (granule["start"] - prev_granule["start"]).total_seconds()
            if step_seconds != interval_seconds:
                raise RuntimeError(
                    f"Temporal discontinuity between granule {idx} ({prev_granule['filename']}) "
                    f"and granule {idx+1} ({granule['filename']}): step = {step_seconds}s (expected {interval_seconds}s)"
                )

    return unique_granules


# =====================================================================
# SPATIAL MATCHING & EXTRACTION
# =====================================================================

def match_spatial_grid(
    points_df: pd.DataFrame,
    sample_granule_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match location coordinates to the nearest native GPM IMERG 0.1° grid cell center.
    Returns: (matched_lats, matched_lons, distances_km, lat_indices, lon_indices)
    """
    with h5py.File(sample_granule_path, "r") as hdf:
        gpm_latitudes = np.asarray(hdf["Grid/lat"][:], dtype=float)
        gpm_longitudes = np.asarray(hdf["Grid/lon"][:], dtype=float)

    pts_lat = points_df["latitude"].to_numpy(dtype=float)
    pts_lon = points_df["longitude"].to_numpy(dtype=float)

    # Vectorized nearest-neighbor coordinate matching on 1D regular axes
    lat_indices = np.abs(gpm_latitudes[:, None] - pts_lat[None, :]).argmin(axis=0)
    lon_indices = np.abs(gpm_longitudes[:, None] - pts_lon[None, :]).argmin(axis=0)

    matched_lats = gpm_latitudes[lat_indices]
    matched_lons = gpm_longitudes[lon_indices]

    distances_km = haversine_distance_km(pts_lat, pts_lon, matched_lats, matched_lons)

    max_dist = float(np.max(distances_km))
    if max_dist > MAX_GRID_MATCH_DISTANCE_KM:
        raise ValueError(
            f"Maximum matching distance {max_dist:.3f} km exceeds limit {MAX_GRID_MATCH_DISTANCE_KM} km."
        )

    return matched_lats, matched_lons, distances_km, lat_indices, lon_indices


def extract_timeseries_matrix(
    granules: List[Dict],
    lon_indices: np.ndarray,
    lat_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Extract half-hourly precipitation rates, latest quality index, and latest random error.
    Uses spatial bounding-slice indexing for sub-second efficiency.
    Returns: (rainfall_matrix, latest_quality, latest_error, time_bounds)
    """
    num_points = len(lat_indices)
    num_granules = len(granules)

    rainfall_matrix = np.full((num_points, num_granules), np.nan, dtype=float)
    latest_quality = np.full(num_points, np.nan, dtype=float)
    latest_error = np.full(num_points, np.nan, dtype=float)
    time_bounds: List[Tuple[float, float]] = []

    lon_min, lon_max = int(lon_indices.min()), int(lon_indices.max())
    lat_min, lat_max = int(lat_indices.min()), int(lat_indices.max())

    rel_lon = lon_indices - lon_min
    rel_lat = lat_indices - lat_min

    for idx, granule in enumerate(granules):
        with h5py.File(granule["path"], "r") as hdf:
            # 1. Verify HDF5 internal time bounds
            tb = np.asarray(hdf["Grid/time_bnds"][0], dtype=float).reshape(-1)
            duration = tb[1] - tb[0]
            if not np.isclose(duration, GRANULE_INTERVAL_SECONDS, atol=1.0):
                raise RuntimeError(
                    f"Granule {granule['filename']} has internal time_bnds duration {duration}s"
                )
            time_bounds.append((float(tb[0]), float(tb[1])))

            # 2. Extract precipitation chunk
            precip_dataset = hdf["Grid/precipitation"]
            chunk = precip_dataset[0, lon_min : lon_max + 1, lat_min : lat_max + 1]

            fill_val = float(precip_dataset.attrs.get("_FillValue", -9999.9))
            extracted_rates = chunk[rel_lon, rel_lat].astype(float)

            valid_mask = (
                np.isfinite(extracted_rates)
                & (extracted_rates >= 0.0)
                & (extracted_rates < 1000.0)
                & ~np.isclose(extracted_rates, fill_val, atol=1e-3)
            )

            extracted_rates[~valid_mask] = np.nan
            rainfall_matrix[:, idx] = extracted_rates

            # 3. Extract quality and error from latest granule
            if idx == num_granules - 1:
                if "Grid/precipitationQualityIndex" in hdf:
                    q_ds = hdf["Grid/precipitationQualityIndex"]
                    q_fill = float(q_ds.attrs.get("_FillValue", -9999.9))
                    q_chunk = q_ds[0, lon_min : lon_max + 1, lat_min : lat_max + 1]
                    q_vals = q_chunk[rel_lon, rel_lat].astype(float)
                    q_vals[~np.isfinite(q_vals) | np.isclose(q_vals, q_fill, atol=1e-3)] = np.nan
                    latest_quality = q_vals

                if "Grid/randomError" in hdf:
                    e_ds = hdf["Grid/randomError"]
                    e_fill = float(e_ds.attrs.get("_FillValue", -9999.9))
                    e_chunk = e_ds[0, lon_min : lon_max + 1, lat_min : lat_max + 1]
                    e_vals = e_chunk[rel_lon, rel_lat].astype(float)
                    e_vals[~np.isfinite(e_vals) | np.isclose(e_vals, e_fill, atol=1e-3)] = np.nan
                    latest_error = e_vals

    return rainfall_matrix, latest_quality, latest_error, time_bounds


# =====================================================================
# ACCUMULATION CALCULATIONS
# =====================================================================

def compute_accumulations(
    rainfall_matrix: np.ndarray,
    granules: List[Dict],
) -> Dict[str, np.ndarray]:
    """
    Compute accumulation features for 1h, 3h, 6h, 12h, and 24h windows ending at
    the common observation end time.
    Each granule has duration 0.5 hours: depth (mm) = rate (mm/hr) * 0.5 hr.
    """
    rainfall_depths = rainfall_matrix * 0.5
    num_points, num_granules = rainfall_matrix.shape

    results: Dict[str, np.ndarray] = {
        "gpm_latest_rainfall_rate_mm_hr": rainfall_matrix[:, -1],
    }

    # Window definitions: duration_hours -> granule_count
    window_specs = {
        1: 2,
        3: 6,
        6: 12,
        12: 24,
        24: 48,
    }

    for hours, count in window_specs.items():
        if count > num_granules:
            raise ValueError(
                f"Cannot compute {hours}h accumulation: requires {count} granules, only {num_granules} available."
            )

        window_depths = rainfall_depths[:, -count:]
        valid_counts = np.isfinite(window_depths).sum(axis=1)

        # Accumulation requires all intervals to be valid to avoid downward bias
        complete_mask = valid_counts == count
        accumulation = np.full(num_points, np.nan, dtype=float)
        accumulation[complete_mask] = window_depths[complete_mask].sum(axis=1)

        results[f"gpm_{hours}h_accumulation_mm"] = accumulation
        results[f"gpm_{hours}h_valid_intervals"] = valid_counts

    results["gpm_interval_count"] = results["gpm_24h_valid_intervals"]
    results["gpm_24h_complete"] = results["gpm_24h_valid_intervals"] == EXPECTED_GRANULES_24H

    return results


# =====================================================================
# INTEGRATION CLASS (MODULAR FOR STAGE 6B)
# =====================================================================

class WesternGhatsGPMIntegrator:
    """
    Modular GPM IMERG spatial integration engine for Western Ghats.
    Can be instantiated directly by downstream inference, live refresh,
    and dynamic GIS prediction pipelines (Stage 6B).
    """

    def __init__(
        self,
        granule_dirs: Optional[List[Path]] = None,
        expected_granules: int = EXPECTED_GRANULES_24H,
    ):
        self.granule_dirs = granule_dirs or DEFAULT_GRANULE_DIRS
        self.expected_granules = expected_granules
        self.granules: List[Dict] = []
        self.time_bounds: List[Tuple[float, float]] = []

    def load_granules(self) -> List[Dict]:
        """Validate and load the chronological GPM IMERG granule sequence."""
        self.granules = load_and_validate_granules(
            self.granule_dirs,
            expected_count=self.expected_granules,
            interval_seconds=GRANULE_INTERVAL_SECONDS,
        )
        return self.granules

    def extract_features_for_dataframe(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """
        Extract GPM IMERG spatial matching and accumulation features for any
        arbitrary coordinate DataFrame located within Western Ghats coverage.
        """
        if not self.granules:
            self.load_granules()

        sample_granule = self.granules[0]["path"]

        # 1. Spatial matching
        matched_lats, matched_lons, distances_km, lat_idx, lon_idx = match_spatial_grid(
            df[[lat_col, lon_col]].rename(columns={lat_col: "latitude", lon_col: "longitude"}),
            sample_granule,
        )

        # 2. Extract timeseries matrix
        rainfall_matrix, latest_quality, latest_error, time_bounds = extract_timeseries_matrix(
            self.granules,
            lon_idx,
            lat_idx,
        )
        self.time_bounds = time_bounds

        # 3. Compute window accumulations
        accum_features = compute_accumulations(rainfall_matrix, self.granules)

        # 4. Construct output DataFrame
        result = df.copy()

        result["gpm_grid_latitude"] = matched_lats
        result["gpm_grid_longitude"] = matched_lons
        result["gpm_matching_distance_km"] = distances_km
        result["gpm_spatial_resolution"] = GPM_NATIVE_RESOLUTION

        result["gpm_latest_rainfall_rate_mm_hr"] = accum_features["gpm_latest_rainfall_rate_mm_hr"]
        result["gpm_1h_accumulation_mm"] = accum_features["gpm_1h_accumulation_mm"]
        result["gpm_1h_valid_intervals"] = accum_features["gpm_1h_valid_intervals"]
        result["gpm_3h_accumulation_mm"] = accum_features["gpm_3h_accumulation_mm"]
        result["gpm_3h_valid_intervals"] = accum_features["gpm_3h_valid_intervals"]
        result["gpm_6h_accumulation_mm"] = accum_features["gpm_6h_accumulation_mm"]
        result["gpm_6h_valid_intervals"] = accum_features["gpm_6h_valid_intervals"]
        result["gpm_12h_accumulation_mm"] = accum_features["gpm_12h_accumulation_mm"]
        result["gpm_12h_valid_intervals"] = accum_features["gpm_12h_valid_intervals"]
        result["gpm_24h_accumulation_mm"] = accum_features["gpm_24h_accumulation_mm"]
        result["gpm_24h_valid_intervals"] = accum_features["gpm_24h_valid_intervals"]
        result["gpm_interval_count"] = accum_features["gpm_interval_count"]
        result["gpm_24h_complete"] = accum_features["gpm_24h_complete"]

        result["gpm_latest_quality_index"] = latest_quality
        result["gpm_latest_random_error_mm_hr"] = latest_error

        # 5. Timestamp metadata
        common_end_utc = self.granules[-1]["end"] + timedelta(seconds=1)
        common_end_iso = common_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        window_spans = {
            1: timedelta(hours=1),
            3: timedelta(hours=3),
            6: timedelta(hours=6),
            12: timedelta(hours=12),
            24: timedelta(hours=24),
        }

        for hours, span in window_spans.items():
            start_iso = (common_end_utc - span).strftime("%Y-%m-%dT%H:%M:%SZ")
            result[f"gpm_{hours}h_window_start_utc"] = start_iso
            result[f"gpm_{hours}h_window_end_utc"] = common_end_iso

        result["gpm_common_observation_end_utc"] = common_end_iso
        result["gpm_product"] = GPM_PRODUCT_NAME

        return result


# =====================================================================
# EXECUTION PIPELINE
# =====================================================================

def run_western_ghats_gpm_pipeline(
    events_csv: Path = DEFAULT_EVENTS_CSV,
    background_csv: Path = DEFAULT_BACKGROUND_CSV,
    granule_dirs: Optional[List[Path]] = None,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Execute full Stage 6A GPM IMERG spatial integration for Western Ghats.
    """
    print("\n" + "=" * 75)
    print("STAGE 6A: NASA GPM IMERG REGIONAL SPATIAL INTEGRATION (WESTERN GHATS)")
    print("=" * 75)

    if output_csv.exists() and not force_overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_csv}\n"
            "Pass --force-overwrite to replace existing output."
        )

    # 1. Load Western Ghats locations
    print(f"\n1. Loading Western Ghats locations...")
    print(f"   Events input:     {events_csv.name}")
    print(f"   Background input: {background_csv.name}")
    locations_df = load_western_ghats_locations(events_csv, background_csv)
    print(f"   Unified records:  {len(locations_df)} (182 events + 1,400 background)")
    print(f"   Distinct tiles:   {locations_df['tile'].nunique()} tiles")

    # 2. Validate granules
    print(f"\n2. Scanning and validating 48 GPM IMERG HDF5 granules...")
    integrator = WesternGhatsGPMIntegrator(granule_dirs=granule_dirs)
    granules = integrator.load_granules()
    first_granule = granules[0]
    last_granule = granules[-1]
    print(f"   Granules verified: {len(granules)}/48 contiguous half-hourly files")
    print(f"   Series start UTC:  {first_granule['start'].isoformat()}")
    print(f"   Series end UTC:    {(last_granule['end'] + timedelta(seconds=1)).isoformat()}")

    # 3. Spatial matching and extraction
    print(f"\n3. Performing spatial matching and timeseries extraction...")
    output_df = integrator.extract_features_for_dataframe(locations_df)

    # 4. Save output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"\n4. Successfully saved processed features to:")
    print(f"   {output_csv}")

    # 5. Print comprehensive extraction report
    print("\n" + "=" * 75)
    print("EXTRACTION SUMMARY & SPATIAL COVERAGE AUDIT")
    print("=" * 75)

    max_dist = output_df["gpm_matching_distance_km"].max()
    mean_dist = output_df["gpm_matching_distance_km"].mean()
    min_dist = output_df["gpm_matching_distance_km"].min()
    print(f"\nSpatial Matching Distances (Native 0.1° Grid):")
    print(f"  Minimum distance: {min_dist:.3f} km")
    print(f"  Mean distance:    {mean_dist:.3f} km")
    print(f"  Maximum distance: {max_dist:.3f} km  (All <= 8.0 km limit: {max_dist <= 8.0})")

    print(f"\nRainfall Accumulation Statistics Across All 1,582 Locations:")
    for h in ACCUMULATION_WINDOWS_HOURS:
        col = f"gpm_{h}h_accumulation_mm"
        mean_val = output_df[col].mean()
        max_val = output_df[col].max()
        valid_cnt = output_df[col].notna().sum()
        print(f"  {h:2d}-hour window: valid={valid_cnt}/{len(output_df)}, mean={mean_val:.3f} mm, max={max_val:.3f} mm")

    rate_col = "gpm_latest_rainfall_rate_mm_hr"
    print(f"  Latest rate:    valid={output_df[rate_col].notna().sum()}/{len(output_df)}, "
          f"mean={output_df[rate_col].mean():.3f} mm/hr, max={output_df[rate_col].max():.3f} mm/hr")

    print(f"\nPer-Tile Coverage and Accumulation Summary (14 DEM Tiles):")
    tile_summary = output_df.groupby("tile").agg(
        total_points=("sample_id", "count"),
        events=("label", lambda s: (s == 1).sum()),
        background=("label", lambda s: (s == 0).sum()),
        unique_gpm_pixels=("gpm_grid_latitude", lambda s: len(set(zip(s, output_df.loc[s.index, "gpm_grid_longitude"])))),
        max_dist_km=("gpm_matching_distance_km", "max"),
        mean_24h_mm=("gpm_24h_accumulation_mm", "mean"),
        max_24h_mm=("gpm_24h_accumulation_mm", "max"),
        max_6h_mm=("gpm_6h_accumulation_mm", "max"),
    )
    print(tile_summary.round(3).to_string())

    print("\n" + "=" * 75)
    print("STAGE 6A EXTRACTION COMPLETE.")
    print("=" * 75)

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract NASA GPM IMERG rainfall features for Western Ghats regional dataset."
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=DEFAULT_EVENTS_CSV,
        help="Path to approved Western Ghats historical events CSV.",
    )
    parser.add_argument(
        "--background-csv",
        type=Path,
        default=DEFAULT_BACKGROUND_CSV,
        help="Path to approved Western Ghats background samples CSV (V2).",
    )
    parser.add_argument(
        "--granule-dir",
        type=Path,
        action="append",
        dest="granule_dirs",
        help="Directory containing GPM IMERG HDF5 granules (can specify multiple).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to save processed rainfall feature CSV.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    args = parser.parse_args()

    granule_dirs = args.granule_dirs if args.granule_dirs else DEFAULT_GRANULE_DIRS

    try:
        run_western_ghats_gpm_pipeline(
            events_csv=args.events_csv,
            background_csv=args.background_csv,
            granule_dirs=granule_dirs,
            output_csv=args.output_csv,
            force_overwrite=args.force_overwrite,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
