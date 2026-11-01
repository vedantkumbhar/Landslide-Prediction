
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regional_training_dataset_30oct.csv"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "satellite"
    / "gpm_imerge_series_12h"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall"
    / "gpm_regional_timeseries_12h_01nov2026.csv"
)

EXPECTED_START = datetime(
    2026, 11, 1, 0, 0, tzinfo=timezone.utc
)

EXPECTED_INTERVALS = 24
INTERVAL_SECONDS = 1800

EARTH_RADIUS_KM = 6371.0


# ============================================================
# COORDINATE COLUMN DETECTION
# ============================================================

def find_column(df, candidates):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find coordinate column. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# GRANULE TIMESTAMP
# ============================================================

def parse_granule_timestamp(filename):

    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})-E(\d{6})",
        filename
    )

    if not match:
        raise ValueError(
            f"Cannot parse GPM timestamp: {filename}"
        )

    date_part, start_part, end_part = match.groups()

    start = datetime.strptime(
        date_part + start_part,
        "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)

    end = datetime.strptime(
        date_part + end_part,
        "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)

    return start, end


# ============================================================
# VALIDATE GRANULE SERIES
# ============================================================

def validate_granules():

    files = sorted(INPUT_DIR.glob("*.HDF5"))

    if len(files) != EXPECTED_INTERVALS:
        raise RuntimeError(
            f"Expected {EXPECTED_INTERVALS} HDF5 files, "
            f"found {len(files)}."
        )

    granules = []

    for filepath in files:

        start, end = parse_granule_timestamp(
            filepath.name
        )

        granules.append({
            "path": filepath,
            "start": start,
            "end": end
        })

    granules.sort(key=lambda item: item["start"])

    expected_starts = [
        EXPECTED_START + timedelta(
            seconds=INTERVAL_SECONDS * index
        )
        for index in range(EXPECTED_INTERVALS)
    ]

    actual_starts = [
        item["start"] for item in granules
    ]

    if actual_starts != expected_starts:
        raise RuntimeError(
            "Granule start timestamps do not match "
            "the expected 12-hour series."
        )

    for index in range(1, len(granules)):

        previous = granules[index - 1]
        current = granules[index]

        if (
            current["start"] - previous["start"]
        ).total_seconds() != INTERVAL_SECONDS:

            raise RuntimeError(
                "A gap or irregular interval exists "
                "between GPM granules."
            )

    print("Granule count: 24/24")
    print("First interval:", granules[0]["start"])
    print("Last interval:", granules[-1]["start"])
    print("Timestamp continuity: PASSED")

    return granules


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# HDF5 TIME BOUNDS VALIDATION
# ============================================================

def read_time_bounds(hdf):

    bounds_dataset = hdf["Grid/time_bnds"]

    bounds = np.asarray(
        bounds_dataset[0],
        dtype=float
    ).reshape(-1)

    if len(bounds) != 2:
        raise ValueError(
            "Unexpected GPM time_bnds structure."
        )

    duration = bounds[1] - bounds[0]

    if not np.isclose(
        duration,
        INTERVAL_SECONDS,
        atol=1
    ):
        raise ValueError(
            f"Unexpected interval duration: {duration} seconds"
        )

    return float(bounds[0]), float(bounds[1])


# ============================================================
# EXTRACT RAINFALL AT SAMPLE LOCATIONS
# ============================================================

def extract_granule(
    granule,
    sample_lon_indices,
    sample_lat_indices,
    lon_min,
    lat_min
):

    filepath = granule["path"]

    with h5py.File(filepath, "r") as hdf:

        precipitation = hdf["Grid/precipitation"]

        quality_dataset = hdf.get(
            "Grid/precipitationQualityIndex"
        )

        error_dataset = hdf.get(
            "Grid/randomError"
        )

        # Read only the regional bounding rectangle.
        lon_max = int(sample_lon_indices.max())
        lat_max = int(sample_lat_indices.max())

        rainfall_grid = precipitation[
            0,
            lon_min:lon_max + 1,
            lat_min:lat_max + 1
        ]

        rainfall_values = rainfall_grid[
            sample_lon_indices - lon_min,
            sample_lat_indices - lat_min
        ].astype(float)

        # Convert invalid and missing rainfall values to NaN.
        rainfall_fill = precipitation.attrs.get(
            "_FillValue", None
        )

        valid = np.isfinite(rainfall_values)
        valid &= rainfall_values >= 0
        valid &= rainfall_values < 1000

        if rainfall_fill is not None:
            valid &= rainfall_values != float(rainfall_fill)

        rainfall_values[~valid] = np.nan

        quality_values = np.full(
            len(sample_lon_indices),
            np.nan
        )

        error_values = np.full(
            len(sample_lon_indices),
            np.nan
        )

        if quality_dataset is not None:

            quality_grid = quality_dataset[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            quality_values = quality_grid[
                sample_lon_indices - lon_min,
                sample_lat_indices - lat_min
            ].astype(float)

            quality_values[
                ~np.isfinite(quality_values)
            ] = np.nan

        if error_dataset is not None:

            error_grid = error_dataset[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            error_values = error_grid[
                sample_lon_indices - lon_min,
                sample_lat_indices - lat_min
            ].astype(float)

            error_values[
                ~np.isfinite(error_values)
            ] = np.nan

        bounds_start, bounds_end = read_time_bounds(hdf)

    return (
        rainfall_values,
        quality_values,
        error_values,
        bounds_start,
        bounds_end
    )


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    print("\nLoading regional training locations...")

    df = pd.read_csv(INPUT_CSV)

    latitude_column = find_column(
        df,
        ["latitude", "lat", "Latitude", "LAT"]
    )

    longitude_column = find_column(
        df,
        ["longitude", "lon", "Longitude", "LON", "lng"]
    )

    latitudes = pd.to_numeric(
        df[latitude_column],
        errors="coerce"
    ).to_numpy(dtype=float)

    longitudes = pd.to_numeric(
        df[longitude_column],
        errors="coerce"
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
    ):
        raise ValueError(
            "Input contains missing or invalid coordinates."
        )

    print("Input samples:", len(df))

    # --------------------------------------------------------
    # Validate GPM granules
    # --------------------------------------------------------

    granules = validate_granules()

    # --------------------------------------------------------
    # Read GPM coordinate grids
    # --------------------------------------------------------

    first_file = granules[0]["path"]

    with h5py.File(first_file, "r") as hdf:

        grid = hdf["Grid"]

        gpm_latitudes = np.asarray(
            grid["lat"][:],
            dtype=float
        )

        gpm_longitudes = np.asarray(
            grid["lon"][:],
            dtype=float
        )

    # Find nearest grid cell for every sample.
    latitude_indices = np.abs(
        gpm_latitudes[:, None]
        - latitudes[None, :]
    ).argmin(axis=0)

    longitude_indices = np.abs(
        gpm_longitudes[:, None]
        - longitudes[None, :]
    ).argmin(axis=0)

    matched_latitudes = gpm_latitudes[
        latitude_indices
    ]

    matched_longitudes = gpm_longitudes[
        longitude_indices
    ]

    matching_distances = haversine_km(
        latitudes,
        longitudes,
        matched_latitudes,
        matched_longitudes
    )

    print(
        "Maximum grid matching distance:",
        round(float(matching_distances.max()), 3),
        "km"
    )

    # --------------------------------------------------------
    # Prepare rainfall storage
    # --------------------------------------------------------

    rainfall_matrix = np.full(
        (len(df), EXPECTED_INTERVALS),
        np.nan,
        dtype=float
    )

    latest_quality = np.full(
        len(df),
        np.nan,
        dtype=float
    )

    latest_error = np.full(
        len(df),
        np.nan,
        dtype=float
    )

    actual_bounds = []

    lon_min = int(longitude_indices.min())
    lat_min = int(latitude_indices.min())

    # --------------------------------------------------------
    # Process 24 granules
    # --------------------------------------------------------

    print("\nExtracting rainfall from GPM granules...\n")

    for index, granule in enumerate(
        granules,
        start=1
    ):

        print(
            f"[{index}/{EXPECTED_INTERVALS}] "
            f"{granule['path'].name}"
        )

        (
            rainfall,
            quality,
            error,
            bounds_start,
            bounds_end
        ) = extract_granule(
            granule,
            longitude_indices,
            latitude_indices,
            lon_min,
            lat_min
        )

        rainfall_matrix[:, index - 1] = rainfall

        actual_bounds.append(
            (bounds_start, bounds_end)
        )

        if index == EXPECTED_INTERVALS:

            latest_quality = quality
            latest_error = error

        valid_count = int(
            np.isfinite(rainfall).sum()
        )

        print(
            f"  Valid sample values: "
            f"{valid_count}/{len(df)}"
        )

    # --------------------------------------------------------
    # Validate actual HDF5 time bounds
    # --------------------------------------------------------

    for index in range(1, len(actual_bounds)):

        previous_end = actual_bounds[index - 1][1]
        current_start = actual_bounds[index][0]

        if not np.isclose(
            previous_end,
            current_start,
            atol=1
        ):
            raise RuntimeError(
                "HDF5 time bounds are not contiguous."
            )

    for start, end in actual_bounds:

        if not np.isclose(
            end - start,
            INTERVAL_SECONDS,
            atol=1
        ):
            raise RuntimeError(
                "A granule has an unexpected time-bound duration."
            )

    # --------------------------------------------------------
    # Calculate rainfall accumulations
    # --------------------------------------------------------

    # IMERG precipitation is a rate in mm/hr.
    # Each granule represents 30 minutes.
    # Therefore, rainfall depth = rate * 0.5 hours.

    rainfall_depths = rainfall_matrix * 0.5

    def accumulation(number_of_intervals):

        selected = rainfall_depths[
            :, -number_of_intervals:
        ]

        # Do not report a complete accumulation if
        # any interval in its window is missing.
        complete = np.isfinite(selected).all(axis=1)

        result = np.full(
            len(df),
            np.nan,
            dtype=float
        )

        result[complete] = np.sum(
            selected[complete],
            axis=1
        )

        return result

    output = df.copy()

    output["gpm_grid_latitude"] = matched_latitudes
    output["gpm_grid_longitude"] = matched_longitudes
    output["gpm_matching_distance_km"] = matching_distances

    output["gpm_latest_rainfall_rate_mm_hr"] = (
        rainfall_matrix[:, -1]
    )

    output["gpm_1h_accumulation_mm"] = accumulation(2)
    output["gpm_3h_accumulation_mm"] = accumulation(6)
    output["gpm_6h_accumulation_mm"] = accumulation(12)
    output["gpm_12h_accumulation_mm"] = accumulation(24)

    output["gpm_latest_quality_index"] = latest_quality
    output["gpm_latest_random_error_mm_hr"] = latest_error

    output["gpm_series_start_utc"] = (
        granules[0]["start"].isoformat()
    )

    # Use the actual final HDF5 interval bound.
    # This represents the exclusive end of the 12-hour period.
    output["gpm_series_end_utc"] = (
        granules[-1]["end"].strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    output["gpm_interval_count"] = (
        np.isfinite(rainfall_matrix).sum(axis=1)
    )

    output["gpm_12h_complete"] = (
        output["gpm_interval_count"] == 24
    )

    # --------------------------------------------------------
    # Save output without overwriting
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_CSV.exists():
        raise FileExistsError(
            "Output already exists. Refusing to overwrite:\n"
            f"{OUTPUT_CSV}"
        )

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("12-HOUR GPM EXTRACTION SUMMARY")
    print("=" * 60)

    print("Input samples:", len(df))
    print("Processed granules:", len(granules))

    print(
        "Latest rainfall valid:",
        int(
            output[
                "gpm_latest_rainfall_rate_mm_hr"
            ].notna().sum()
        ),
        "/",
        len(output)
    )

    print(
        "1-hour accumulation valid:",
        int(
            output[
                "gpm_1h_accumulation_mm"
            ].notna().sum()
        )
    )

    print(
        "3-hour accumulation valid:",
        int(
            output[
                "gpm_3h_accumulation_mm"
            ].notna().sum()
        )
    )

    print(
        "6-hour accumulation valid:",
        int(
            output[
                "gpm_6h_accumulation_mm"
            ].notna().sum()
        )
    )

    print(
        "12-hour accumulation valid:",
        int(
            output[
                "gpm_12h_accumulation_mm"
            ].notna().sum()
        )
    )

    print(
        "Complete 12-hour records:",
        int(output["gpm_12h_complete"].sum())
    )

    print("\nRegional 12-hour rainfall statistics:")

    if "region" in output.columns:

        print(
            output.groupby("region")[
                "gpm_12h_accumulation_mm"
            ].agg(
                ["count", "mean", "max"]
            ).round(4)
        )

    elif "planning_region" in output.columns:

        print(
            output.groupby("planning_region")[
                "gpm_12h_accumulation_mm"
            ].agg(
                ["count", "mean", "max"]
            ).round(4)
        )

    else:

        print(
            "Region column not found. "
            "Overall statistics are available."
        )

    print("\nSaved:", OUTPUT_CSV)
    print("Extraction completed.")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR:", error)
        sys.exit(1)
