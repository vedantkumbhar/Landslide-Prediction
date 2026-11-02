from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timezone, timedelta

from backend.gpm_refresh.refresh_config import (
    PROJECT_ROOT,
    get_run_directory,
)

RUN_ID = "20261102_152742"
INPUT_CSV = PROJECT_ROOT / "data/processed/regional_training_dataset_30oct.csv"
GRANULE_DIR = get_run_directory(RUN_ID) / "downloads" / "6h"
OUTPUT_CSV = get_run_directory(RUN_ID) / "extracted" / "gpm_regional_rainfall_6h.csv"

FILL_FALLBACK = -9999.9
EXPECTED_GRANULES = 12
MAX_MATCH_DISTANCE_KM = 8.0

def read_fill(dataset):
    value = dataset.attrs.get("_FillValue", FILL_FALLBACK)
    return float(np.asarray(value).reshape(-1)[0])

def read_time_bounds(h5):
    raw = np.asarray(h5["Grid/time_bnds"])[0]
    units = h5["Grid/time"].attrs["units"]
    if isinstance(units, bytes):
        units = units.decode()
    origin = units.split("since", 1)[1].strip()
    base = pd.Timestamp(origin)
    if base.tzinfo is None:
        base = base.tz_localize("UTC")
    else:
        base = base.tz_convert("UTC")
    start = base + pd.to_timedelta(float(raw[0]), unit="s")
    end = base + pd.to_timedelta(float(raw[1]), unit="s")
    return start, end

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))

def main():
    print("\nNASA GPM REFRESH: 6-HOUR EXTRACTION")
    print("=" * 65)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    if OUTPUT_CSV.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"record_id", "latitude", "longitude", "dem_region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values in input dataset.")

    files = sorted(GRANULE_DIR.glob("*.HDF5"))
    if len(files) != EXPECTED_GRANULES:
        raise ValueError(
            f"Expected {EXPECTED_GRANULES} granules in {GRANULE_DIR}, "
            f"found {len(files)}. Download the complete 6-hour window first."
        )

    metadata = []
    reference_lat = reference_lon = None

    for path in files:
        with h5py.File(path, "r") as h5:
            start, end = read_time_bounds(h5)
            precip = h5["Grid/precipitation"]
            quality = h5["Grid/precipitationQualityIndex"]
            error = h5["Grid/randomError"]
            lat = np.asarray(h5["Grid/lat"][:], dtype=float)
            lon = np.asarray(h5["Grid/lon"][:], dtype=float)

            if precip.shape != (1, 3600, 1800):
                raise ValueError(f"Unexpected precipitation shape: {path.name}")
            if quality.shape != precip.shape or error.shape != precip.shape:
                raise ValueError(f"Quality/error shape mismatch: {path.name}")
            if end - start != pd.Timedelta(minutes=30):
                raise ValueError(f"Invalid interval duration: {path.name}")

            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not (
                np.array_equal(reference_lat, lat)
                and np.array_equal(reference_lon, lon)
            ):
                raise ValueError("GPM grid coordinates differ between granules.")

            metadata.append((path, start, end))
            print(f"Validated: {path.name} | {start} to {end}")

    metadata.sort(key=lambda item: item[1])
    for i in range(1, len(metadata)):
        if metadata[i][1] != metadata[i - 1][2]:
            raise ValueError("Gap or overlap in the 6-hour granule sequence.")

    if metadata[-1][2] - metadata[0][1] != pd.Timedelta(hours=6):
        raise ValueError("Granules do not cover exactly six hours.")

    print("\nAll 12 granules passed time and grid validation.")

    lat_idx = np.abs(
        reference_lat[:, None] - df["latitude"].to_numpy()[None, :]
    ).argmin(axis=0)
    lon_idx = np.abs(
        reference_lon[:, None] - df["longitude"].to_numpy()[None, :]
    ).argmin(axis=0)

    distances = haversine_km(
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
        reference_lat[lat_idx],
        reference_lon[lon_idx],
    )

    if np.any(distances > MAX_MATCH_DISTANCE_KM):
        raise ValueError(
            f"Grid match exceeds {MAX_MATCH_DISTANCE_KM} km: "
            f"{distances.max():.3f} km"
        )

    print(f"Unique GPM cells: {len(set(zip(lon_idx, lat_idx)))}")
    print(f"Maximum grid match distance: {distances.max():.3f} km")

    rain_depths = []
    latest_quality = latest_error = latest_rate = None

    for i, (path, start, end) in enumerate(metadata):
        with h5py.File(path, "r") as h5:
            lon_min, lon_max = lon_idx.min(), lon_idx.max()
            lat_min, lat_max = lat_idx.min(), lat_idx.max()

            def extract(dataset):
                values = np.asarray(
                    dataset[
                        0,
                        lon_min:lon_max + 1,
                        lat_min:lat_max + 1
                    ],
                    dtype=np.float64,
                )
                fill = read_fill(dataset)
                values[
                    ~np.isfinite(values) | np.isclose(values, fill)
                ] = np.nan
                return values[
                    lon_idx - lon_min,
                    lat_idx - lat_min
                ]

            rate = extract(h5["Grid/precipitation"])
            quality = extract(h5["Grid/precipitationQualityIndex"])
            error = extract(h5["Grid/randomError"])

            rain_depths.append(rate * 0.5)

            if i == len(metadata) - 1:
                latest_rate = rate
                latest_quality = quality
                latest_error = error

        print(f"Extracted: {path.name}")

    rain_depths = np.vstack(rain_depths)
    valid_count = np.isfinite(rain_depths).sum(axis=0)

    rainfall_1h = np.nansum(rain_depths[-2:], axis=0)
    rainfall_1h[~np.isfinite(rain_depths[-2:]).all(axis=0)] = np.nan

    rainfall_6h = np.nansum(rain_depths, axis=0)
    rainfall_6h[~np.isfinite(rain_depths).all(axis=0)] = np.nan

    df["gpm_latest_rate_mm_hr"] = latest_rate
    df["gpm_1h_accumulation_mm"] = rainfall_1h
    df["gpm_6h_accumulation_mm"] = rainfall_6h
    df["gpm_valid_30min_intervals"] = valid_count
    df["gpm_latest_quality_index"] = latest_quality
    df["gpm_latest_random_error_mm_hr"] = latest_error
    df["gpm_grid_match_distance_km"] = distances
    df["gpm_series_start_utc"] = metadata[0][1].isoformat()
    df["gpm_series_end_utc"] = metadata[-1][2].isoformat()
    df["gpm_product"] = "NASA GPM IMERG V07C"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n========== 6-HOUR EXTRACTION SUMMARY ==========")
    print(f"Output records: {len(df)}")
    print(f"Unique record IDs: {df['record_id'].nunique()}")
    print(f"Valid 1-hour accumulations: {df['gpm_1h_accumulation_mm'].notna().sum()}")
    print(f"Valid 6-hour accumulations: {df['gpm_6h_accumulation_mm'].notna().sum()}")
    print("\nRegional 6-hour rainfall statistics:")
    print(
        df.groupby("dem_region")["gpm_6h_accumulation_mm"]
        .agg(["count", "mean", "max"])
    )
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nGPM 6-HOUR EXTRACTION COMPLETED.")

if __name__ == "__main__":
    main()