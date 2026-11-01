
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    ROOT / "data" / "processed"
    / "regional_training_dataset_30oct.csv"
)

HDF5_FILE = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_test"
    / "3B-HHR-E.MS.MRG.3IMERG."
    "20261101-S103000-E105959.0630.V07C.HDF5"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "gpm_regional_extraction_01nov2026.csv"
)


def nearest_index(coordinates, value):
    return int(np.abs(coordinates - value).argmin())


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def main():

    print("NASA GPM REGIONAL RAINFALL EXTRACTION")
    print("=" * 55)

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    if not HDF5_FILE.exists():
        raise FileNotFoundError(HDF5_FILE)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {OUTPUT_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = ["latitude", "longitude", "dem_region"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing CSV column: {column}")

    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing latitude or longitude values.")

    print("\nInput records:", len(df))

    with h5py.File(HDF5_FILE, "r") as hdf:

        grid = hdf["Grid"]

        latitudes = grid["lat"][:]
        longitudes = grid["lon"][:]

        precipitation = grid["precipitation"][0]
        quality = grid["precipitationQualityIndex"][0]
        rainfall_error = grid["randomError"][0]

        fill_value = grid["precipitation"].attrs["_FillValue"]
        quality_fill = grid[
            "precipitationQualityIndex"
        ].attrs["_FillValue"]

        error_fill = grid["randomError"].attrs["_FillValue"]

        print("\nGPM grid shape:", precipitation.shape)
        print("Latitude cells:", len(latitudes))
        print("Longitude cells:", len(longitudes))

        extracted_rainfall = []
        extracted_quality = []
        extracted_error = []
        matched_latitudes = []
        matched_longitudes = []
        distances = []

        for _, row in df.iterrows():

            lat = float(row["latitude"])
            lon = float(row["longitude"])

            lat_idx = nearest_index(latitudes, lat)
            lon_idx = nearest_index(longitudes, lon)

            # GPM array ordering is (longitude, latitude)
            rainfall_value = float(
                precipitation[lon_idx, lat_idx]
            )

            quality_value = float(
                quality[lon_idx, lat_idx]
            )

            error_value = float(
                rainfall_error[lon_idx, lat_idx]
            )

            if np.isclose(rainfall_value, fill_value):
                rainfall_value = np.nan

            if np.isclose(quality_value, quality_fill):
                quality_value = np.nan

            if np.isclose(error_value, error_fill):
                error_value = np.nan

            grid_lat = float(latitudes[lat_idx])
            grid_lon = float(longitudes[lon_idx])

            distance = haversine_km(
                lat, lon, grid_lat, grid_lon
            )

            extracted_rainfall.append(rainfall_value)
            extracted_quality.append(quality_value)
            extracted_error.append(error_value)
            matched_latitudes.append(grid_lat)
            matched_longitudes.append(grid_lon)
            distances.append(distance)

    df["gpm_precipitation_mm_hr"] = extracted_rainfall
    df["gpm_quality_index"] = extracted_quality
    df["gpm_random_error_mm_hr"] = extracted_error

    df["gpm_grid_latitude"] = matched_latitudes
    df["gpm_grid_longitude"] = matched_longitudes
    df["gpm_grid_distance_km"] = distances

    df["gpm_granule_start_utc"] = "2026-11-01T10:30:00Z"
    df["gpm_granule_end_utc"] = "2026-11-01T10:59:59Z"
    df["gpm_product"] = "IMERG_Early_V07C"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEXTRACTION SUMMARY")
    print("-" * 55)

    print("Output records:", len(df))
    print("Valid rainfall values:",
          df["gpm_precipitation_mm_hr"].notna().sum())
    print("Missing rainfall values:",
          df["gpm_precipitation_mm_hr"].isna().sum())

    print("\nRainfall statistics (mm/hr):")
    print(df["gpm_precipitation_mm_hr"].describe())

    print("\nRecords by region:")
    print(df.groupby("dem_region").size())

    print("\nValid rainfall by region:")
    print(
        df.groupby("dem_region")[
            "gpm_precipitation_mm_hr"
        ].count()
    )

    print("\nMean rainfall by region (mm/hr):")
    print(
        df.groupby("dem_region")[
            "gpm_precipitation_mm_hr"
        ].mean()
    )

    print("\nMaximum grid matching distance (km):",
          round(df["gpm_grid_distance_km"].max(), 3))

    print("\nSaved:", OUTPUT_FILE)
    print("\nGPM EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main()
