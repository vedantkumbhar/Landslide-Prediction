
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "regional_hourly_rainfall_30oct.csv"
)

# New output: previous v1 and v2 files remain untouched.
OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "regional_rainfall_features_30oct_v3.csv"
)

HOUR_NS = 3_600_000_000_000


def calculate_windows(values, timestamps, window, forward=False):
    """Calculate complete hourly rainfall windows."""

    result = np.full(len(values), np.nan, dtype=float)

    # Normalize timestamps explicitly to nanoseconds.
    timestamp_ns = timestamps.to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")

    for i in range(len(values)):

        if forward:
            start = i
            end = i + window
        else:
            start = i - window + 1
            end = i + 1

        if start < 0 or end > len(values):
            continue

        rainfall_window = values[start:end]
        time_window = timestamp_ns[start:end]

        # Require a complete window with valid rainfall values.
        if len(rainfall_window) != window:
            continue

        if not np.isfinite(rainfall_window).all():
            continue

        # Every timestamp in the window must be one hour apart.
        if len(time_window) > 1:
            differences = np.diff(time_window)

            if not np.all(differences == HOUR_NS):
                continue

        result[i] = rainfall_window.sum()

    return result


def main():

    print("\nREGIONAL RAINFALL FEATURE GENERATION V3")
    print("-" * 50)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {OUTPUT}"
        )

    df = pd.read_csv(INPUT)

    required_columns = {
        "region",
        "tile",
        "timestamp_local",
        "record_type",
        "precipitation_mm",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"],
        errors="coerce"
    )

    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"],
        errors="coerce"
    )

    if df["timestamp_local"].isna().any():
        raise ValueError("Invalid timestamps found.")

    if (df["precipitation_mm"].dropna() < 0).any():
        raise ValueError("Negative rainfall values found.")

    duplicate_count = df.duplicated(
        ["region", "tile", "timestamp_local"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    output_groups = []

    for (region, tile), group in df.groupby(
        ["region", "tile"]
    ):

        group = group.sort_values(
            "timestamp_local"
        ).copy()

        # Historical/current observations.
        observed = group[
            group["record_type"] == "historical_or_current"
        ].copy()

        if not observed.empty:

            observed = observed.sort_values(
                "timestamp_local"
            )

            values = observed[
                "precipitation_mm"
            ].to_numpy(dtype=float)

            timestamps = pd.DatetimeIndex(
                observed["timestamp_local"]
            )

            observed_windows = {
                "recent_rainfall_1h_mm": 1,
                "recent_rainfall_3h_mm": 3,
                "recent_rainfall_6h_mm": 6,
                "recent_rainfall_12h_mm": 12,
                "recent_rainfall_24h_mm": 24,
            }

            for column, hours in observed_windows.items():

                observed[column] = calculate_windows(
                    values,
                    timestamps,
                    hours,
                    forward=False
                )

            for column in [
                "forecast_rainfall_3h_mm",
                "forecast_rainfall_6h_mm",
                "forecast_rainfall_24h_mm",
            ]:
                observed[column] = np.nan

            output_groups.append(observed)

        # Forecast records.
        forecast = group[
            group["record_type"] == "forecast"
        ].copy()

        if not forecast.empty:

            forecast = forecast.sort_values(
                "timestamp_local"
            )

            values = forecast[
                "precipitation_mm"
            ].to_numpy(dtype=float)

            timestamps = pd.DatetimeIndex(
                forecast["timestamp_local"]
            )

            forecast_windows = {
                "forecast_rainfall_3h_mm": 3,
                "forecast_rainfall_6h_mm": 6,
                "forecast_rainfall_24h_mm": 24,
            }

            for column, hours in forecast_windows.items():

                forecast[column] = calculate_windows(
                    values,
                    timestamps,
                    hours,
                    forward=True
                )

            for column in [
                "recent_rainfall_1h_mm",
                "recent_rainfall_3h_mm",
                "recent_rainfall_6h_mm",
                "recent_rainfall_12h_mm",
                "recent_rainfall_24h_mm",
            ]:
                forecast[column] = np.nan

            output_groups.append(forecast)

    if not output_groups:
        raise ValueError("No rainfall records were processed.")

    result = pd.concat(
        output_groups,
        ignore_index=True
    )

    result = result.sort_values(
        ["region", "tile", "timestamp_local"]
    ).reset_index(drop=True)

    feature_columns = [
        "recent_rainfall_1h_mm",
        "recent_rainfall_3h_mm",
        "recent_rainfall_6h_mm",
        "recent_rainfall_12h_mm",
        "recent_rainfall_24h_mm",
        "forecast_rainfall_3h_mm",
        "forecast_rainfall_6h_mm",
        "forecast_rainfall_24h_mm",
    ]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_output = OUTPUT.with_suffix(".tmp")

    result.to_csv(
        temporary_output,
        index=False
    )

    temporary_output.replace(OUTPUT)

    print("\nSUCCESS")
    print("-" * 50)

    print("Saved:", OUTPUT)
    print("Dataset shape:", result.shape)

    print("\nRecords by region:")
    print(result.groupby("region").size().to_string())

    print("\nRecords by type:")
    print(result.groupby("record_type").size().to_string())

    print("\nMissing feature values:")
    print(result[feature_columns].isna().sum().to_string())

    print("\nNon-missing feature counts:")
    print(result[feature_columns].count().to_string())

    print("\nSample observed features:")

    print(
        result[
            result["record_type"] == "historical_or_current"
        ][
            [
                "region",
                "tile",
                "timestamp_local",
                "recent_rainfall_1h_mm",
                "recent_rainfall_3h_mm",
                "recent_rainfall_6h_mm",
                "recent_rainfall_12h_mm",
                "recent_rainfall_24h_mm",
            ]
        ].head(5).to_string(index=False)
    )

    print("\nSample forecast features:")

    print(
        result[
            result["record_type"] == "forecast"
        ][
            [
                "region",
                "tile",
                "timestamp_local",
                "forecast_rainfall_3h_mm",
                "forecast_rainfall_6h_mm",
                "forecast_rainfall_24h_mm",
            ]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
