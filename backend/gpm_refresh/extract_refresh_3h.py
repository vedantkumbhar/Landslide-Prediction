
from pathlib import Path
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RUN_ID = "20261102_152742"

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT / "data" / "processed"
    / "regional_training_dataset_30oct.csv"
)

RUN_DIR = (
    ROOT / "data" / "processed" / "rainfall"
    / "refresh_runs" / RUN_ID
)

GRANULE_DIR = RUN_DIR / "downloads" / "3h"
OUTPUT_DIR = RUN_DIR / "extracted"
OUTPUT_FILE = OUTPUT_DIR / "gpm_regional_rainfall_3h.csv"

EXPECTED_COUNT = 6
INTERVAL_SECONDS = 1800
EXPECTED_SHAPE = (1, 3600, 1800)

FILL_VALUE = -9999.9

TIME_ORIGIN = datetime(
    1980, 1, 6, tzinfo=timezone.utc
)

EXPECTED_START = datetime(
    2026, 11, 2, 0, 0, tzinfo=timezone.utc
)


# ============================================================
# HELPERS
# ============================================================

def nearest_index(coordinates, value):
    return int(np.abs(coordinates - value).argmin())


def haversine_km(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean_values(values):

    values = np.asarray(values, dtype=float)

    invalid = (
        ~np.isfinite(values)
        | np.isclose(values, FILL_VALUE)
    )

    values[invalid] = np.nan

    return values


def read_fill_value(dataset):

    value = dataset.attrs.get(
        "_FillValue",
        FILL_VALUE
    )

    return float(value)


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def load_input():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    if not GRANULE_DIR.exists():
        raise FileNotFoundError(GRANULE_DIR)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_FILE}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = [
        "record_id",
        "latitude",
        "longitude",
        "dem_region"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Missing required input values."
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record IDs detected."
        )

    return df


# ============================================================
# VALIDATE AND LOAD GRANULE METADATA
# ============================================================

def validate_granules():

    files = sorted(GRANULE_DIR.glob("*.HDF5"))

    if len(files) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} granules; "
            f"found {len(files)}."
        )

    observations = []

    reference_lat = None
    reference_lon = None

    previous_end = None

    for index, file_path in enumerate(files):

        with h5py.File(file_path, "r") as hdf:

            grid = hdf["Grid"]

            bounds = grid["time_bnds"][:]

            if bounds.shape != (1, 2):
                raise RuntimeError(
                    f"Invalid time bounds: {file_path.name}"
                )

            start_seconds, end_seconds = bounds[0]

            actual_start = (
                TIME_ORIGIN
                + timedelta(seconds=int(start_seconds))
            )

            actual_end = (
                TIME_ORIGIN
                + timedelta(seconds=int(end_seconds))
            )

            expected_start = (
                EXPECTED_START
                + timedelta(seconds=index * INTERVAL_SECONDS)
            )

            if actual_start != expected_start:
                raise RuntimeError(
                    f"Unexpected start time: {actual_start}"
                )

            if actual_end - actual_start != timedelta(
                seconds=INTERVAL_SECONDS
            ):
                raise RuntimeError(
                    "Invalid granule duration."
                )

            if previous_end is not None:
                if actual_start != previous_end:
                    raise RuntimeError(
                        "Temporal gap or overlap detected."
                    )

            precipitation = grid["precipitation"]
            quality = grid["precipitationQualityIndex"]
            error = grid["randomError"]

            for name, dataset in [
                ("precipitation", precipitation),
                ("quality", quality),
                ("error", error)
            ]:

                if dataset.shape != EXPECTED_SHAPE:
                    raise RuntimeError(
                        f"Invalid {name} shape in {file_path.name}"
                    )

            units = precipitation.attrs.get("units", b"")

            if isinstance(units, bytes):
                units = units.decode("utf-8")

            if units != "mm/hr":
                raise RuntimeError(
                    f"Unexpected rainfall units: {units}"
                )

            lat = grid["lat"][:]
            lon = grid["lon"][:]

            if reference_lat is None:
                reference_lat = lat
                reference_lon = lon

            else:

                if not np.array_equal(reference_lat, lat):
                    raise RuntimeError(
                        "Latitude grid mismatch."
                    )

                if not np.array_equal(reference_lon, lon):
                    raise RuntimeError(
                        "Longitude grid mismatch."
                    )

            observations.append({
                "path": file_path,
                "start": actual_start,
                "end": actual_end,
                "lat": lat,
                "lon": lon,
                "precip_fill": read_fill_value(precipitation),
                "quality_fill": read_fill_value(quality),
                "error_fill": read_fill_value(error)
            })

            previous_end = actual_end

            print(
                "Validated:",
                file_path.name,
                actual_start,
                "to",
                actual_end
            )

    expected_end = (
        EXPECTED_START
        + timedelta(hours=3)
    )

    if previous_end != expected_end:
        raise RuntimeError(
            "The complete 3-hour window was not covered."
        )

    return observations


# ============================================================
# PREPARE REGIONAL GRID INDICES
# ============================================================

def prepare_sample_indices(df, observations):

    lat_grid = observations[0]["lat"]
    lon_grid = observations[0]["lon"]

    sample_lat = df["latitude"].to_numpy(dtype=float)
    sample_lon = df["longitude"].to_numpy(dtype=float)

    lat_indices = np.abs(
        lat_grid[None, :] - sample_lat[:, None]
    ).argmin(axis=1)

    lon_indices = np.abs(
        lon_grid[None, :] - sample_lon[:, None]
    ).argmin(axis=1)

    matched_lat = lat_grid[lat_indices]
    matched_lon = lon_grid[lon_indices]

    distances = haversine_km(
        sample_lat,
        sample_lon,
        matched_lat,
        matched_lon
    )

    if np.any(distances > 8):
        raise RuntimeError(
            "One or more samples are over 8 km "
            "from their nearest GPM grid cell."
        )

    return lat_indices, lon_indices, distances


# ============================================================
# SELECTIVE HDF5 EXTRACTION
# ============================================================

def extract_observations(
    observations,
    lat_indices,
    lon_indices
):

    rainfall = []
    quality_values = []
    error_values = []

    lat_min = int(lat_indices.min())
    lat_max = int(lat_indices.max())

    lon_min = int(lon_indices.min())
    lon_max = int(lon_indices.max())

    local_lat = lat_indices - lat_min
    local_lon = lon_indices - lon_min

    for observation in observations:

        with h5py.File(
            observation["path"], "r"
        ) as hdf:

            grid = hdf["Grid"]

            p_ds = grid["precipitation"]
            q_ds = grid["precipitationQualityIndex"]
            e_ds = grid["randomError"]

            p = p_ds[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            q = q_ds[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            e = e_ds[
                0,
                lon_min:lon_max + 1,
                lat_min:lat_max + 1
            ]

            p = np.asarray(p, dtype=float)[
                local_lon, local_lat
            ]

            q = np.asarray(q, dtype=float)[
                local_lon, local_lat
            ]

            e = np.asarray(e, dtype=float)[
                local_lon, local_lat
            ]

            for values, fill in [
                (p, observation["precip_fill"]),
                (q, observation["quality_fill"]),
                (e, observation["error_fill"])
            ]:

                values[
                    ~np.isfinite(values)
                    | np.isclose(values, fill)
                ] = np.nan

            rainfall.append(p)
            quality_values.append(q)
            error_values.append(e)

        print(
            "Extracted:",
            observation["path"].name
        )

    return (
        np.asarray(rainfall, dtype=float),
        np.asarray(quality_values, dtype=float),
        np.asarray(error_values, dtype=float)
    )


# ============================================================
# CALCULATE RAINFALL FEATURES
# ============================================================

def calculate_features(
    df,
    rainfall,
    quality,
    error,
    observations,
    distances
):

    # Six observations x regional records.
    if rainfall.shape != (EXPECTED_COUNT, len(df)):
        raise RuntimeError(
            f"Unexpected rainfall matrix shape: {rainfall.shape}"
        )

    # Convert mm/hr rates to 30-minute rainfall depth.
    rainfall_depth = rainfall * 0.5

    valid_count = np.isfinite(
        rainfall_depth
    ).sum(axis=0)

    # Latest observation rate.
    latest_rate = rainfall[-1].copy()

    # Correct latest one-hour accumulation:
    # final two 30-minute intervals.
    latest_two = rainfall_depth[-2:]

    valid_1h = np.isfinite(
        latest_two
    ).all(axis=0)

    rainfall_1h = np.full(
        len(df), np.nan
    )

    rainfall_1h[valid_1h] = (
        latest_two[:, valid_1h].sum(axis=0)
    )

    # Strict three-hour accumulation:
    # all six 30-minute intervals.
    valid_3h = np.isfinite(
        rainfall_depth
    ).all(axis=0)

    rainfall_3h = np.full(
        len(df), np.nan
    )

    rainfall_3h[valid_3h] = (
        rainfall_depth[:, valid_3h].sum(axis=0)
    )

    df["gpm_latest_rate_mm_hr"] = latest_rate

    df["gpm_1h_accumulation_mm"] = rainfall_1h

    df["gpm_3h_accumulation_mm"] = rainfall_3h

    df["gpm_valid_30min_intervals"] = valid_count

    df["gpm_latest_quality_index"] = quality[-1]

    df["gpm_latest_random_error_mm_hr"] = error[-1]

    df["gpm_grid_match_distance_km"] = distances

    df["gpm_latest_observation_utc"] = (
        observations[-1]["start"].isoformat()
    )

    df["gpm_series_start_utc"] = (
        observations[0]["start"].isoformat()
    )

    df["gpm_series_end_utc"] = (
        observations[-1]["end"].isoformat()
    )

    df["gpm_product"] = "IMERG_Early_V07C"

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nNASA GPM REFRESH: 3-HOUR EXTRACTION")
    print("=" * 65)

    df = load_input()

    print("Input records:", len(df))

    observations = validate_granules()

    print("\nAll six HDF5 granules passed validation.")

    lat_indices, lon_indices, distances = (
        prepare_sample_indices(df, observations)
    )

    print("Unique GPM cells:", len(
        set(zip(lon_indices, lat_indices))
    ))

    print(
        "Maximum grid match distance:",
        round(float(distances.max()), 3),
        "km"
    )

    rainfall, quality, error = extract_observations(
        observations,
        lat_indices,
        lon_indices
    )

    df = calculate_features(
        df,
        rainfall,
        quality,
        error,
        observations,
        distances
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_FILE}"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== EXTRACTION SUMMARY ==========")

    print("Output records:", len(df))

    print(
        "Unique record IDs:",
        df["record_id"].nunique()
    )

    print(
        "Valid latest rates:",
        df["gpm_latest_rate_mm_hr"].notna().sum()
    )

    print(
        "Valid 1-hour accumulations:",
        df["gpm_1h_accumulation_mm"].notna().sum()
    )

    print(
        "Valid 3-hour accumulations:",
        df["gpm_3h_accumulation_mm"].notna().sum()
    )

    print("\nRegional 3-hour rainfall statistics:")

    print(
        df.groupby("dem_region")[
            "gpm_3h_accumulation_mm"
        ].agg(["count", "mean", "max"])
    )

    print("\nSaved:", OUTPUT_FILE)

    print("\nGPM REFRESH EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main()
