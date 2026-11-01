
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "hourly_rainfall.csv"
)

OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "rainfall_features.csv"
)

RECENT_WINDOWS = [1, 3, 6, 12, 24]
FORECAST_WINDOWS = [3, 6, 24]


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    df = pd.read_csv(INPUT)
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="raise"
    )
    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"], errors="raise"
    )

    if df.duplicated(["sample_id", "timestamp_local"]).any():
        raise ValueError("Duplicate sample timestamps found.")

    if df["precipitation_mm"].isna().any():
        raise ValueError("Missing rainfall values found.")

    if not np.isfinite(df["precipitation_mm"]).all():
        raise ValueError("Non-finite rainfall values found.")

    if (df["precipitation_mm"] < 0).any():
        raise ValueError("Negative rainfall values found.")

    df = df.sort_values(
        ["sample_id", "timestamp_local"]
    ).reset_index(drop=True)

    groups = df.groupby("sample_id", sort=False)
    group_sizes = groups.size()

    if not (group_sizes == 120).all():
        raise ValueError("Every sample must have exactly 120 records.")

    # Check hourly continuity before using positional rolling windows.
    differences = groups["timestamp_local"].diff()
    invalid = differences.notna() & (
        differences != pd.Timedelta(hours=1)
    )
    if invalid.any():
        raise ValueError("Non-hourly timestamps found.")

    rain = df["precipitation_mm"]
    group_keys = df["sample_id"]

    # Recent windows include the current hour and previous hours.
    for hours in RECENT_WINDOWS:
        sums = rain.groupby(group_keys, sort=False).transform(
            lambda values: values.rolling(
                hours, min_periods=hours
            ).sum()
        )
        counts = rain.groupby(group_keys, sort=False).transform(
            lambda values: values.rolling(
                hours, min_periods=1
            ).count()
        )

        df[f"recent_rain_{hours}h_mm"] = sums
        df[f"recent_records_{hours}h"] = counts.astype(int)

    # Shift by one hour so forecast windows start strictly AFTER time T.
    future_rain = rain.groupby(group_keys, sort=False).shift(-1)

    for hours in FORECAST_WINDOWS:
        sums = future_rain.groupby(group_keys, sort=False).transform(
            lambda values: values.rolling(
                hours, min_periods=hours
            ).sum().shift(-(hours - 1))
        )

        # The shift above makes the window alignment explicit:
        # at row T, use rainfall at T+1 through T+hours.
        df[f"forecast_rain_{hours}h_mm"] = sums

        future_counts = future_rain.groupby(
            group_keys, sort=False
        ).transform(
            lambda values: values.rolling(
                hours, min_periods=1
            ).count().shift(-(hours - 1))
        )
        df[f"forecast_records_{hours}h"] = (
            future_counts.fillna(0).astype(int)
        )

    if len(df) != 78720:
        raise ValueError(f"Unexpected output row count: {len(df)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation protects an existing output.
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        df.to_csv(handle, index=False)

    print("Feature generation completed.")
    print("Rows:", len(df))
    print("Samples:", df["sample_id"].nunique())
    print("Output:", OUTPUT)

    print("\nFeature availability:")
    for hours in RECENT_WINDOWS:
        column = f"recent_rain_{hours}h_mm"
        print(f"{column}: {df[column].notna().sum()} available")

    for hours in FORECAST_WINDOWS:
        column = f"forecast_rain_{hours}h_mm"
        print(f"{column}: {df[column].notna().sum()} available")

    print("\nOriginal hourly rainfall file was not modified.")


if __name__ == "__main__":
    main()
