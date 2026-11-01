
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    ROOT / "data" / "processed"
    / "regional_training_dataset_30oct.csv"
)

PREVIOUS_DIR = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_previous12h"
)

LATEST_DIR = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_series_12h"
)

OUTPUT_CSV = (
    ROOT / "data" / "processed" / "rainfall"
    / "gpm_regional_timeseries_24h_01nov2026.csv"
)

START_TIME = datetime(
    2026, 10, 31, 12, 0, tzinfo=timezone.utc
)

EXPECTED_GRANULES = 48
INTERVAL_SECONDS = 1800
EARTH_RADIUS_KM = 6371.0


# ============================================================
# COLUMN DETECTION
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
        f"Required column not found. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# GRANULE TIMESTAMPS
# ============================================================

def parse_granule_times(filename):

    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})-E(\d{6})",
        filename
    )

    if not match:
        raise ValueError(
            f"Cannot parse granule filename: {filename}"
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
# VALIDATE BOTH DOWNLOAD FOLDERS
# ============================================================

def validate_granules():

    files = []

    for folder in [PREVIOUS_DIR, LATEST_DIR]:

        if not folder.exists():
            raise FileNotFoundError(folder)

        files.extend(folder.glob("*.HDF5"))

    granules = []

    for filepath in files:

        start, end = parse_granule_times(filepath.name)

        granules.append({
            "path": filepath,
            "start": start,
            "end": end
        })

    granules.sort(key=lambda item: item["start"])

    if len(granules) != EXPECTED_GRANULES:
        raise RuntimeError(
            f"Expected 48 granules, found {len(granules)}."
        )

    for index, granule in enumerate(granules):

        expected_start = (
            START_TIME
            + timedelta(seconds=INTERVAL_SECONDS * index)
        )

        if granule["start"] != expected_start:
            raise RuntimeError(
                f"Unexpected timestamp at position {index + 1}: "
                f"{granule['start']}"
            )

        if (
            granule["end"] - granule["start"]
        ).total_seconds() != 1799:
            raise RuntimeError(
                f"Unexpected filename interval: "
                f"{granule['path'].name}"
            )

        if index > 0:

            previous = granules[index - 1]

            if (
                granule["start"] - previous["start"]
            ).total_seconds() != INTERVAL_SECONDS:

                raise RuntimeError(
                    "Gap or irregular spacing in granules."
                )

    print("Previous folder granules:", len(
        list(PREVIOUS_DIR.glob("*.HDF5"))
    ))

    print("Latest folder granules:", len(
        list(LATEST_DIR.glob("*.HDF5"))
    ))

    print("Combined granules: 48/48")
    print("First granule:", granules[0]["start"])
    print("Last granule:", granules[-1]["start"])
    print("Filename timestamp continuity: PASSED")

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
        2 * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# HDF5 TIME BOUNDS
# ============================================================

def read_time_bounds(hdf):

    bounds = np.asarray(
        hdf["Grid/time_bnds"][0],
        dtype=float
    ).reshape(-1)

    if len(bounds) != 2:
        raise ValueError("Unexpected time_bnds structure.")

    duration = bounds[1] - bounds[0]

    if not np.isclose(duration, INTERVAL_SECONDS, atol=1):
        raise ValueError(
            f"Unexpected HDF5 interval duration: {duration}"
        )

    return float(bounds[0]), float(bounds[1])


# ============================================================
# EXTRACT ONE GRANULE
# ============================================================

def extract_granule(
    granule,
    longitude_indices,
    latitude_indices,
    lon_min,
    lat_min
):

    with h5py.File(granule["path"], "r") as hdf:

        precipitation = hdf["Grid/precipitation"]

        lon_max = int(longitude_indices.max())
        lat_max = int(latitude_indices.max())

        rainfall_grid = precipitation[
            0,
            lon_min:lon_max + 1,
            lat_min:lat_max + 1
        ]

        rainfall = rainfall_grid[
            longitude_indices - lon_min,
            latitude_indices - lat_min
        ].astype(float)

        valid = np.isfinite(rainfall)
        valid &= rainfall >= 0
        valid &= rainfall < 1000

        fill_value = precipitation.attrs.get(
            "_FillValue", None
        )

        if fill_value is not None:
            valid &= rainfall != float(fill_value)

        rainfall[~valid] = np.nan

        quality = np.full(len(rainfall), np.nan)
        error = np.full(len(rainfall), np.nan)

        quality_dataset = hdf.get(
            "Grid/precipitationQualityIndex"
        )

        error_dataset = hdf.get(
            "Grid/randomError"
        )

        if quality_dataset is not None:

            quality_grid = quality_dataset[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            quality = quality_grid[
                longitude_indices - lon_min,
                latitude_indices - lat_min
            ].astype(float)

            quality[~np.isfinite(quality)] = np.nan

        if error_dataset is not None:

            error_grid = error_dataset[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            error = error_grid[
                longitude_indices - lon_min,
                latitude_indices - lat_min
            ].astype(float)

            error[~np.isfinite(error)] = np.nan

        bounds_start, bounds_end = read_time_bounds(hdf)

    return rainfall, quality, error, bounds_start, bounds_end


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite:\n"
            f"{OUTPUT_CSV}"
        )

    print("\nLoading 2,009 regional locations...")

    df = pd.read_csv(INPUT_CSV)

    lat_col = find_column(
        df, ["latitude", "lat"]
    )

    lon_col = find_column(
        df, ["longitude", "lon", "lng"]
    )

    latitudes = pd.to_numeric(
        df[lat_col], errors="coerce"
    ).to_numpy(dtype=float)

    longitudes = pd.to_numeric(
        df[lon_col], errors="coerce"
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
    ):
        raise ValueError("Invalid coordinates in input dataset.")

    print("Input samples:", len(df))

    # --------------------------------------------------------
    # Validate all 48 granules
    # --------------------------------------------------------

    granules = validate_granules()

    # --------------------------------------------------------
    # Read GPM coordinate grids
    # --------------------------------------------------------

    with h5py.File(granules[0]["path"], "r") as hdf:

        gpm_latitudes = np.asarray(
            hdf["Grid/lat"][:], dtype=float
        )

        gpm_longitudes = np.asarray(
            hdf["Grid/lon"][:], dtype=float
        )

    latitude_indices = np.abs(
        gpm_latitudes[:, None]
        - latitudes[None, :]
    ).argmin(axis=0)

    longitude_indices = np.abs(
        gpm_longitudes[:, None]
        - longitudes[None, :]
    ).argmin(axis=0)

    matched_latitudes = gpm_latitudes[latitude_indices]
    matched_longitudes = gpm_longitudes[longitude_indices]

    distances = haversine_km(
        latitudes,
        longitudes,
        matched_latitudes,
        matched_longitudes
    )

    print(
        "Maximum grid matching distance:",
        round(float(distances.max()), 3),
        "km"
    )

    # --------------------------------------------------------
    # Allocate rainfall storage
    # --------------------------------------------------------

    rainfall_matrix = np.full(
        (len(df), EXPECTED_GRANULES),
        np.nan
    )

    latest_quality = np.full(len(df), np.nan)
    latest_error = np.full(len(df), np.nan)

    actual_bounds = []

    lon_min = int(longitude_indices.min())
    lat_min = int(latitude_indices.min())

    # --------------------------------------------------------
    # Process all 48 granules
    # --------------------------------------------------------

    print("\nExtracting 24-hour rainfall...\n")

    for index, granule in enumerate(granules):

        print(
            f"[{index + 1}/48] "
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

        rainfall_matrix[:, index] = rainfall

        actual_bounds.append(
            (bounds_start, bounds_end)
        )

        if index == EXPECTED_GRANULES - 1:
            latest_quality = quality
            latest_error = error

        print(
            "  Valid values:",
            int(np.isfinite(rainfall).sum()),
            "/",
            len(df)
        )

    # --------------------------------------------------------
    # Validate actual HDF5 interval continuity
    # --------------------------------------------------------

    for index in range(len(actual_bounds)):

        start, end = actual_bounds[index]

        if not np.isclose(
            end - start,
            INTERVAL_SECONDS,
            atol=1
        ):
            raise RuntimeError(
                f"Invalid HDF5 interval at granule {index + 1}"
            )

        if index > 0:

            previous_end = actual_bounds[index - 1][1]

            if not np.isclose(
                previous_end,
                start,
                atol=1
            ):
                raise RuntimeError(
                    "HDF5 time bounds are not contiguous."
                )

    print("\nHDF5 time-bound continuity: PASSED")

    # --------------------------------------------------------
    # Calculate rainfall accumulations
    # --------------------------------------------------------

    # IMERG precipitation is in mm/hr.
    # Each granule represents 30 minutes.
    # Rainfall depth = rate * 0.5 hours.

    rainfall_depths = rainfall_matrix * 0.5

    def calculate_accumulation(interval_count):

        selected = rainfall_depths[
            :, -interval_count:
        ]

        complete = np.isfinite(selected).all(axis=1)

        result = np.full(len(df), np.nan)

        result[complete] = selected[complete].sum(axis=1)

        return result

    output = df.copy()

    output["gpm_grid_latitude"] = matched_latitudes
    output["gpm_grid_longitude"] = matched_longitudes
    output["gpm_matching_distance_km"] = distances

    output["gpm_latest_rainfall_rate_mm_hr"] = (
        rainfall_matrix[:, -1]
    )

    output["gpm_1h_accumulation_mm"] = (
        calculate_accumulation(2)
    )

    output["gpm_3h_accumulation_mm"] = (
        calculate_accumulation(6)
    )

    output["gpm_6h_accumulation_mm"] = (
        calculate_accumulation(12)
    )

    output["gpm_12h_accumulation_mm"] = (
        calculate_accumulation(24)
    )

    output["gpm_24h_accumulation_mm"] = (
        calculate_accumulation(48)
    )

    output["gpm_latest_quality_index"] = latest_quality
    output["gpm_latest_random_error_mm_hr"] = latest_error

    output["gpm_series_start_utc"] = (
        granules[0]["start"].isoformat()
    )

    output["gpm_series_end_utc"] = (
        granules[-1]["end"].strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    output["gpm_interval_count"] = (
        np.isfinite(rainfall_matrix).sum(axis=1)
    )

    output["gpm_24h_complete"] = (
        output["gpm_interval_count"] == 48
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("24-HOUR GPM EXTRACTION SUMMARY")
    print("=" * 60)

    print("Input samples:", len(df))
    print("Processed granules:", len(granules))

    for column in [
        "gpm_latest_rainfall_rate_mm_hr",
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_24h_accumulation_mm"
    ]:

        print(
            f"{column}: "
            f"{output[column].notna().sum()}/{len(output)}"
        )

    print(
        "Complete 24-hour records:",
        int(output["gpm_24h_complete"].sum())
    )

    print("\nRegional 24-hour rainfall statistics:")

    print(
        output.groupby("dem_region")[
            "gpm_24h_accumulation_mm"
        ].agg(
            samples="count",
            mean="mean",
            maximum="max"
        ).round(4)
    )

    print("\nSaved:", OUTPUT_CSV)
    print("Extraction completed successfully.")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)
