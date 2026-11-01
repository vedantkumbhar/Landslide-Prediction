
from pathlib import Path

from datetime import datetime, timedelta
import re

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
    / "gpm_imerge_series_6h"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "gpm_regional_timeseries_6h_01nov2026.csv"
)

EXPECTED_GRANULES = 12
INTERVAL_MINUTES = 30


def nearest_index(coordinates, value):
    return int(np.abs(coordinates - value).argmin())


def get_timestamp(file_path):
    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})",
        file_path.name
    )

    if not match:
        raise ValueError(
            f"Cannot identify timestamp: {file_path.name}"
        )

    date, clock = match.groups()

    return datetime.strptime(
        date + clock,
        "%Y%m%d%H%M%S"
    )


def read_missing_value(dataset):
    value = dataset.attrs.get("_FillValue", -9999.9)
    return float(np.asarray(value).item())


def extract_sample_values(dataset, sample_indices, fill_value):
    values = np.full(len(sample_indices), np.nan)

    # Read only required longitude rows.
    longitude_groups = {}

    for sample_position, (lon_idx, lat_idx) in enumerate(sample_indices):
        longitude_groups.setdefault(lon_idx, []).append(
            (sample_position, lat_idx)
        )

    for lon_idx, samples in longitude_groups.items():

        row = dataset[0, lon_idx, :]

        for sample_position, lat_idx in samples:

            value = float(row[lat_idx])

            if np.isfinite(value) and not np.isclose(
                value, fill_value
            ):
                values[sample_position] = value

    return values


def main():

    print("NASA GPM SIX-HOUR RAINFALL EXTRACTION")
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
        GRANULE_DIR.glob("*.HDF5"),
        key=get_timestamp
    )

    if len(granule_files) != EXPECTED_GRANULES:
        raise RuntimeError(
            f"Expected {EXPECTED_GRANULES} granules, "
            f"found {len(granule_files)}."
        )

    timestamps = [
        get_timestamp(path) for path in granule_files
    ]

    for i in range(1, len(timestamps)):

        difference = timestamps[i] - timestamps[i - 1]

        if difference != timedelta(minutes=INTERVAL_MINUTES):
            raise RuntimeError(
                f"Timestamp gap or duplicate between "
                f"{timestamps[i - 1]} and {timestamps[i]}"
            )

    print("Timestamp continuity: PASSED")
    print("First interval:", timestamps[0])
    print("Last interval:", timestamps[-1])
    print("Total duration: 6 hours")

    df = pd.read_csv(INPUT_CSV)

    required = ["latitude", "longitude", "dem_region"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing sample coordinates.")

    print("\nInput samples:", len(df))
    print("Granules:", len(granule_files))

    # Match sample coordinates to the GPM grid.
    with h5py.File(granule_files[0], "r") as hdf:

        grid = hdf["Grid"]

        latitudes = grid["lat"][:]
        longitudes = grid["lon"][:]

        sample_indices = []

        for _, row in df.iterrows():

            lat_idx = nearest_index(
                latitudes, float(row["latitude"])
            )

            lon_idx = nearest_index(
                longitudes, float(row["longitude"])
            )

            sample_indices.append((lon_idx, lat_idx))

    # Store only extracted sample values.
    rainfall_rates = []
    quality_values = []
    error_values = []

    for file_path in granule_files:

        print("\nProcessing:", file_path.name)

        with h5py.File(file_path, "r") as hdf:

            grid = hdf["Grid"]

            precipitation = grid["precipitation"]
            quality = grid["precipitationQualityIndex"]
            error = grid["randomError"]

            rate_values = extract_sample_values(
                precipitation,
                sample_indices,
                read_missing_value(precipitation)
            )

            q_values = extract_sample_values(
                quality,
                sample_indices,
                read_missing_value(quality)
            )

            e_values = extract_sample_values(
                error,
                sample_indices,
                read_missing_value(error)
            )

            rainfall_rates.append(rate_values)
            quality_values.append(q_values)
            error_values.append(e_values)

        print(
            "Valid rainfall samples:",
            np.isfinite(rate_values).sum()
        )

    # Shape: 12 observations x number of samples.
    rainfall_rates = np.asarray(rainfall_rates, dtype=float)
    quality_values = np.asarray(quality_values, dtype=float)
    error_values = np.asarray(error_values, dtype=float)

    # Rainfall rate is mm/hr.
    # Each observation represents a 30-minute interval.
    rainfall_depth = rainfall_rates * 0.5

    def calculate_accumulation(number_of_intervals):

        window = rainfall_depth[-number_of_intervals:, :]

        valid = np.isfinite(window).all(axis=0)

        result = np.full(len(df), np.nan)

        result[valid] = window[:, valid].sum(axis=0)

        return result

    df["gpm_latest_rate_mm_hr"] = rainfall_rates[-1]

    df["gpm_1h_accumulation_mm"] = (
        calculate_accumulation(2)
    )

    df["gpm_3h_accumulation_mm"] = (
        calculate_accumulation(6)
    )

    df["gpm_6h_accumulation_mm"] = (
        calculate_accumulation(12)
    )

    df["gpm_valid_30min_intervals"] = (
        np.isfinite(rainfall_depth).sum(axis=0)
    )

    df["gpm_latest_quality_index"] = quality_values[-1]

    df["gpm_latest_random_error_mm_hr"] = error_values[-1]

    df["gpm_series_start_utc"] = timestamps[0].isoformat() + "Z"
    df["gpm_series_end_utc"] = (
        timestamps[-1].isoformat() + "Z"
    )

    df["gpm_product"] = "IMERG_Early_V07C"

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEXTRACTION SUMMARY")
    print("=" * 60)

    print("Output records:", len(df))

    for column in [
        "gpm_latest_rate_mm_hr",
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm"
    ]:
        print(
            f"Valid {column}:",
            df[column].notna().sum()
        )

    print("\nRegional 6-hour rainfall statistics:")

    print(
        df.groupby("dem_region")[
            "gpm_6h_accumulation_mm"
        ].agg(["count", "mean", "max"])
    )

    print("\nSaved:", OUTPUT_FILE)
    print("\nGPM SIX-HOUR EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main()
