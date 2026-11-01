
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "data" / "processed"
    / "regional_training_dataset_30oct.csv"
)

GRANULE_DIR = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_series_3h"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "gpm_regional_timeseries_3h_01nov2026.csv"
)

EXPECTED_GRANULES = 6
FILL_VALUE = -9999.9


def nearest_index(coordinates, value):
    return int(np.abs(coordinates - value).argmin())


def main():

    print("NASA GPM THREE-HOUR RAINFALL EXTRACTION")
    print("=" * 60)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    if not GRANULE_DIR.exists():
        raise FileNotFoundError(GRANULE_DIR)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {OUTPUT_FILE}"
        )

    granule_files = sorted(
        GRANULE_DIR.glob("*.HDF5")
    )

    if len(granule_files) != EXPECTED_GRANULES:
        raise RuntimeError(
            f"Expected {EXPECTED_GRANULES} granules, "
            f"found {len(granule_files)}."
        )

    df = pd.read_csv(INPUT_CSV)

    required = ["latitude", "longitude", "dem_region"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing sample coordinates.")

    print("Input samples:", len(df))
    print("Granules:", len(granule_files))

    # Load all granule observations.
    observations = []

    for file_path in granule_files:

        print("\nReading:", file_path.name)

        with h5py.File(file_path, "r") as hdf:

            grid = hdf["Grid"]

            latitudes = grid["lat"][:]
            longitudes = grid["lon"][:]

            precipitation = grid["precipitation"][0]
            quality = grid["precipitationQualityIndex"][0]
            error = grid["randomError"][0]

            time_seconds = int(grid["time"][0])

            timestamp = (
                pd.Timestamp("1980-01-06T00:00:00Z")
                + pd.Timedelta(seconds=time_seconds)
            )

            observations.append({
                "timestamp_utc": timestamp,
                "latitude_grid": latitudes,
                "longitude_grid": longitudes,
                "precipitation": precipitation,
                "quality": quality,
                "error": error,
            })

            print("Observation:", timestamp)
            print("Grid shape:", precipitation.shape)

    observations.sort(
        key=lambda item: item["timestamp_utc"]
    )

    timestamps = [
        item["timestamp_utc"]
        for item in observations
    ]

    if len(set(timestamps)) != EXPECTED_GRANULES:
        raise RuntimeError("Duplicate observation timestamps found.")

    intervals = pd.Series(timestamps).diff().dropna()

    if not (intervals == pd.Timedelta(minutes=30)).all():
        raise RuntimeError(
            "Granules are not consecutive 30-minute observations."
        )

    print("\nAll six observations are consecutive.")

    # Match every sample to its nearest GPM grid cell.
    sample_indices = []

    first = observations[0]

    for _, row in df.iterrows():

        lat_idx = nearest_index(
            first["latitude_grid"],
            float(row["latitude"])
        )

        lon_idx = nearest_index(
            first["longitude_grid"],
            float(row["longitude"])
        )

        sample_indices.append((lon_idx, lat_idx))

    rainfall_matrix = []
    quality_matrix = []
    error_matrix = []

    for observation in observations:

        rainfall_values = []
        quality_values = []
        error_values = []

        precipitation = observation["precipitation"]
        quality = observation["quality"]
        error = observation["error"]

        for lon_idx, lat_idx in sample_indices:

            # GPM dimension order: longitude, latitude
            p = float(precipitation[lon_idx, lat_idx])
            q = float(quality[lon_idx, lat_idx])
            e = float(error[lon_idx, lat_idx])

            if not np.isfinite(p) or np.isclose(p, FILL_VALUE):
                p = np.nan

            if not np.isfinite(q) or np.isclose(q, FILL_VALUE):
                q = np.nan

            if not np.isfinite(e) or np.isclose(e, FILL_VALUE):
                e = np.nan

            rainfall_values.append(p)
            quality_values.append(q)
            error_values.append(e)

        rainfall_matrix.append(rainfall_values)
        quality_matrix.append(quality_values)
        error_matrix.append(error_values)

    # Matrix shape: 6 observations x 2009 samples
    rainfall_matrix = np.asarray(rainfall_matrix, dtype=float)
    quality_matrix = np.asarray(quality_matrix, dtype=float)
    error_matrix = np.asarray(error_matrix, dtype=float)

    # Convert rainfall rates (mm/hr) into 30-minute depth (mm).
    rainfall_depth = rainfall_matrix * 0.5

    # Preserve missing observations as missing.
    valid_30min_count = np.isfinite(rainfall_depth).sum(axis=0)

    # Strict 1-hour accumulation requires two consecutive valid intervals.
    rainfall_1h = np.full(len(df), np.nan)

    for i in range(2, EXPECTED_GRANULES + 1):
        if i == 2:
            values = rainfall_depth[i - 2:i]
            valid = np.isfinite(values).all(axis=0)
            rainfall_1h[valid] = values[:, valid].sum(axis=0)

    # Strict 3-hour accumulation requires all six valid intervals.
    valid_3h = np.isfinite(rainfall_depth).all(axis=0)

    rainfall_3h = np.full(len(df), np.nan)
    rainfall_3h[valid_3h] = rainfall_depth[:, valid_3h].sum(axis=0)

    # Last observation's instantaneous rainfall rate.
    latest_rate = rainfall_matrix[-1]

    df["gpm_latest_rate_mm_hr"] = latest_rate
    df["gpm_1h_accumulation_mm"] = rainfall_1h
    df["gpm_3h_accumulation_mm"] = rainfall_3h

    df["gpm_valid_30min_intervals"] = valid_30min_count

    df["gpm_latest_quality_index"] = quality_matrix[-1]
    df["gpm_latest_random_error_mm_hr"] = error_matrix[-1]

    df["gpm_latest_observation_utc"] = timestamps[-1].isoformat()
    df["gpm_series_start_utc"] = timestamps[0].isoformat()
    df["gpm_series_end_utc"] = timestamps[-1].isoformat()

    df["gpm_product"] = "IMERG_Early_V07C"

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEXTRACTION SUMMARY")
    print("=" * 60)

    print("Output records:", len(df))

    print(
        "Valid latest rainfall rates:",
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
    print("\nGPM TIME-SERIES EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main()
