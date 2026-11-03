from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import h5py

from backend.gpm_refresh.refresh_config import (
    PROJECT_ROOT,
    get_run_directory,
    GPM_INTERVAL_MINUTES,
    EXPECTED_GRANULE_SHAPE,
    MAX_MATCH_DISTANCE_KM,
)


DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "data/processed/regional_training_dataset_30oct.csv"
)

FILL_FALLBACK = -9999.9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract dynamic 12-hour NASA GPM IMERG rainfall."
    )

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-time", required=True)

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
    )

    parser.add_argument("--granule-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)

    parser.add_argument(
        "--expected-granules",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-match-distance-km",
        type=float,
        default=MAX_MATCH_DISTANCE_KM,
    )

    return parser.parse_args()


def parse_utc_timestamp(value):
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def read_fill(dataset):
    value = dataset.attrs.get("_FillValue", FILL_FALLBACK)
    return float(np.asarray(value).reshape(-1)[0])


def read_time_bounds(h5):
    raw = np.asarray(h5["Grid/time_bnds"])[0]

    units = h5["Grid/time"].attrs["units"]

    if isinstance(units, bytes):
        units = units.decode("utf-8")

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

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def load_input(input_csv):
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    required = [
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    if df["record_id"].isna().any():
        raise ValueError("Missing record IDs.")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs.")

    for column, minimum, maximum in [
        ("latitude", -90, 90),
        ("longitude", -180, 180),
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if (
            df[column].isna().any()
            or not np.isfinite(df[column]).all()
            or not df[column].between(minimum, maximum).all()
        ):
            raise ValueError(f"Invalid coordinates: {column}")

    if df["dem_region"].isna().any():
        raise ValueError("Missing regional assignments.")

    return df


def validate_granules(files, expected_start, expected_count):
    if len(files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} granules, found {len(files)}."
        )

    metadata = []
    reference_lat = None
    reference_lon = None

    for path in files:
        with h5py.File(path, "r") as h5:

            required_datasets = [
                "Grid/time",
                "Grid/time_bnds",
                "Grid/precipitation",
                "Grid/precipitationQualityIndex",
                "Grid/randomError",
                "Grid/lat",
                "Grid/lon",
            ]

            for dataset in required_datasets:
                if dataset not in h5:
                    raise ValueError(
                        f"Missing {dataset} in {path.name}"
                    )

            start, end = read_time_bounds(h5)

            precip = h5["Grid/precipitation"]
            quality = h5["Grid/precipitationQualityIndex"]
            error = h5["Grid/randomError"]

            if precip.shape != EXPECTED_GRANULE_SHAPE:
                raise ValueError(
                    f"Unexpected precipitation shape: {path.name}"
                )

            if quality.shape != precip.shape or error.shape != precip.shape:
                raise ValueError(
                    f"Quality/error shape mismatch: {path.name}"
                )

            if end - start != pd.Timedelta(
                minutes=GPM_INTERVAL_MINUTES
            ):
                raise ValueError(
                    f"Invalid interval duration: {path.name}"
                )

            units = precip.attrs.get("units", b"")

            if isinstance(units, bytes):
                units = units.decode("utf-8")

            if units.strip() != "mm/hr":
                raise ValueError(
                    f"Unexpected precipitation units in {path.name}: {units}"
                )

            lat = np.asarray(h5["Grid/lat"][:], dtype=float)
            lon = np.asarray(h5["Grid/lon"][:], dtype=float)

            if lat.ndim != 1 or lon.ndim != 1:
                raise ValueError("Unexpected GPM coordinate dimensions.")

            if (
                len(lon) != precip.shape[1]
                or len(lat) != precip.shape[2]
            ):
                raise ValueError("GPM grid coordinate dimensions mismatch.")

            if not np.isfinite(lat).all() or not np.isfinite(lon).all():
                raise ValueError("Non-finite GPM grid coordinates.")

            if reference_lat is None:
                reference_lat = lat
                reference_lon = lon

            elif not (
                np.array_equal(reference_lat, lat)
                and np.array_equal(reference_lon, lon)
            ):
                raise ValueError(
                    "GPM grid coordinates differ between granules."
                )

            metadata.append((path, start, end))

    metadata.sort(key=lambda item: item[1])

    for index, (_, start, end) in enumerate(metadata):
        expected_time = expected_start + pd.Timedelta(
            minutes=index * GPM_INTERVAL_MINUTES
        )

        if start != expected_time:
            raise ValueError(
                f"Unexpected granule start: {start}; "
                f"expected {expected_time}"
            )

        if index > 0 and start != metadata[index - 1][2]:
            raise ValueError("Gap or overlap in the granule sequence.")

        if end - start != pd.Timedelta(
            minutes=GPM_INTERVAL_MINUTES
        ):
            raise ValueError("Invalid granule interval.")

    expected_end = expected_start + pd.Timedelta(hours=12)

    if metadata[-1][2] != expected_end:
        raise ValueError(
            f"Expected 12-hour coverage ending at {expected_end}, "
            f"found {metadata[-1][2]}."
        )

    return metadata, reference_lat, reference_lon


def extract_dataset(dataset, lon_min, lon_max, lat_min, lat_max,
                    lon_idx, lat_idx):
    values = np.asarray(
        dataset[
            0,
            lon_min:lon_max + 1,
            lat_min:lat_max + 1,
        ],
        dtype=np.float64,
    )

    fill = read_fill(dataset)

    values[
        ~np.isfinite(values)
        | np.isclose(values, fill)
    ] = np.nan

    return values[
        lon_idx - lon_min,
        lat_idx - lat_min,
    ]


def main(args):
    if args.expected_granules != (
        12 * 60 // GPM_INTERVAL_MINUTES
    ):
        raise ValueError(
            "A 12-hour window requires exactly 24 half-hour granules."
        )

    if args.max_match_distance_km <= 0:
        raise ValueError("--max-match-distance-km must be positive.")

    expected_start = parse_utc_timestamp(args.start_time)

    run_dir = get_run_directory(args.run_id)

    granule_dir = args.granule_dir or (
        run_dir / "downloads" / "12h"
    )

    output_csv = args.output_csv or (
        run_dir / "extracted" / "gpm_regional_rainfall_12h.csv"
    )

    if output_csv.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_csv}"
        )

    df = load_input(args.input_csv)

    files = sorted(granule_dir.glob("*.HDF5"))

    metadata, reference_lat, reference_lon = validate_granules(
        files,
        expected_start,
        args.expected_granules,
    )

    print("\nNASA GPM: DYNAMIC 12-HOUR EXTRACTION")
    print("=" * 65)
    print("Run ID:", args.run_id)
    print("Input records:", len(df))
    print("Validated granules:", len(metadata))
    print("Observation start:", metadata[0][1])
    print("Observation end:", metadata[-1][2])

    record_lat = df["latitude"].to_numpy(dtype=float)
    record_lon = df["longitude"].to_numpy(dtype=float)

    lat_idx = np.abs(
        reference_lat[:, None] - record_lat[None, :]
    ).argmin(axis=0)

    lon_idx = np.abs(
        reference_lon[:, None] - record_lon[None, :]
    ).argmin(axis=0)

    distances = haversine_km(
        record_lat,
        record_lon,
        reference_lat[lat_idx],
        reference_lon[lon_idx],
    )

    if np.any(distances > args.max_match_distance_km):
        raise ValueError(
            f"GPM match exceeds {args.max_match_distance_km} km. "
            f"Maximum: {distances.max():.3f} km"
        )

    lon_min, lon_max = lon_idx.min(), lon_idx.max()
    lat_min, lat_max = lat_idx.min(), lat_idx.max()

    rain_depths = []
    latest_quality = None
    latest_error = None

    for index, (path, start, end) in enumerate(metadata):
        with h5py.File(path, "r") as h5:

            rate = extract_dataset(
                h5["Grid/precipitation"],
                lon_min, lon_max, lat_min, lat_max,
                lon_idx, lat_idx,
            )

            quality = extract_dataset(
                h5["Grid/precipitationQualityIndex"],
                lon_min, lon_max, lat_min, lat_max,
                lon_idx, lat_idx,
            )

            error = extract_dataset(
                h5["Grid/randomError"],
                lon_min, lon_max, lat_min, lat_max,
                lon_idx, lat_idx,
            )

            rain_depths.append(rate * 0.5)

            if index == len(metadata) - 1:
                latest_quality = quality
                latest_error = error

        print("Extracted:", path.name)

    rain_depths = np.vstack(rain_depths)

    valid_count = np.isfinite(rain_depths).sum(axis=0)

    latest_1h = rain_depths[-2:]
    latest_6h = rain_depths[-12:]

    rainfall_1h = np.nansum(latest_1h, axis=0)
    rainfall_1h[~np.isfinite(latest_1h).all(axis=0)] = np.nan

    rainfall_6h = np.nansum(latest_6h, axis=0)
    rainfall_6h[~np.isfinite(latest_6h).all(axis=0)] = np.nan

    rainfall_12h = np.nansum(rain_depths, axis=0)
    rainfall_12h[~np.isfinite(rain_depths).all(axis=0)] = np.nan

    df["gpm_12h_accumulation_mm"] = rainfall_12h
    df["gpm_12h_valid_30min_intervals"] = valid_count
    df["gpm_12h_latest_quality_index"] = latest_quality
    df["gpm_12h_latest_random_error_mm_hr"] = latest_error
    df["gpm_12h_grid_match_distance_km"] = distances

    df["gpm_12h_series_start_utc"] = metadata[0][1].isoformat()
    df["gpm_12h_series_end_utc"] = metadata[-1][2].isoformat()

    df["gpm_product"] = "NASA GPM IMERG V07C"

    if df["gpm_12h_accumulation_mm"].isna().any():
        raise ValueError(
            "Incomplete 12-hour rainfall values. Refusing to save."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n========== 12-HOUR EXTRACTION SUMMARY ==========")
    print("Records:", len(df))
    print("Valid 12-hour values:", df["gpm_12h_accumulation_mm"].notna().sum())
    print("Maximum grid distance:", f"{distances.max():.3f} km")
    print("Observation:", metadata[0][1], "to", metadata[-1][2])
    print("Saved:", output_csv)
    print("\n12-HOUR EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main(parse_args())