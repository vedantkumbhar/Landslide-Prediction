from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "hourly_rainfall.csv"
)
OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "rainfall_features.csv"
)

GROUP_KEYS = [
    "pilot_tile",
    "requested_latitude",
    "requested_longitude",
    "api_latitude",
    "api_longitude",
]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT}"
        )

    df = pd.read_csv(INPUT)

    required = GROUP_KEYS + [
        "timestamp_local",
        "precipitation_mm",
        "record_type",
        "source",
        "retrieved_at_local",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="raise"
    )
    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"], errors="raise"
    )

    if df["precipitation_mm"].isna().any():
        raise ValueError("Missing rainfall values; refusing to build features.")

    if not np.isfinite(df["precipitation_mm"].to_numpy()).all():
        raise ValueError("Non-finite rainfall values found.")

    if (df["precipitation_mm"] < 0).any():
        raise ValueError("Negative rainfall values found.")

    if not df["record_type"].isin(
        ["historical_or_current", "forecast"]
    ).all():
        raise ValueError("Unexpected record_type value.")

    if df.duplicated(GROUP_KEYS + ["timestamp_local"]).any():
        raise ValueError("Duplicate point timestamps found.")

    df = df.sort_values(
        GROUP_KEYS + ["timestamp_local"]
    ).copy()

    # Ensure every monitoring point has a regular hourly time series.
    for _, group in df.groupby(GROUP_KEYS, sort=False):
        times = group["timestamp_local"]
        if times.diff().dropna().ne(pd.Timedelta(hours=1)).any():
            raise ValueError(
                "Non-hourly or discontinuous timestamps detected."
            )

    recent = df[
        df["record_type"] == "historical_or_current"
    ].copy()
    forecast = df[df["record_type"] == "forecast"].copy()

    recent["recent_rainfall_1h_mm"] = recent["precipitation_mm"]

    for hours in [3, 6, 12, 24]:
        recent[f"recent_rainfall_{hours}h_mm"] = (
            recent.groupby(GROUP_KEYS, sort=False)["precipitation_mm"]
            .transform(
                lambda s: s.rolling(
                    hours, min_periods=hours
                ).sum()
            )
        )

    recent["recent_records_24h"] = (
        recent.groupby(GROUP_KEYS, sort=False)["precipitation_mm"]
        .transform(lambda s: s.rolling(24, min_periods=1).count())
        .astype(int)
    )

    # Forecast feature at time T sums the next N hourly values,
    # excluding the value at T itself.
    for hours in [3, 6, 24]:
        forecast[f"forecast_rainfall_{hours}h_mm"] = (
            forecast.groupby(GROUP_KEYS, sort=False)["precipitation_mm"]
            .transform(
                lambda s: s.shift(-1)
                .rolling(hours, min_periods=hours)
                .sum()
                .shift(-(hours - 1))
            )
        )

    recent["data_period"] = "historical_or_current"
    forecast["data_period"] = "forecast"

    recent_columns = GROUP_KEYS + [
        "timestamp_local",
        "data_period",
        "source",
        "retrieved_at_local",
        "recent_rainfall_1h_mm",
        "recent_rainfall_3h_mm",
        "recent_rainfall_6h_mm",
        "recent_rainfall_12h_mm",
        "recent_rainfall_24h_mm",
        "recent_records_24h",
    ]

    forecast_columns = GROUP_KEYS + [
        "timestamp_local",
        "data_period",
        "source",
        "retrieved_at_local",
        "forecast_rainfall_3h_mm",
        "forecast_rainfall_6h_mm",
        "forecast_rainfall_24h_mm",
    ]

    output = pd.concat(
        [
            recent[recent_columns],
            forecast[forecast_columns],
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(GROUP_KEYS + ["timestamp_local"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")

    try:
        output.to_csv(temp, index=False)
        check = pd.read_csv(temp)

        if len(check) != len(output):
            raise RuntimeError("Output row-count validation failed.")

        if check.duplicated(
            GROUP_KEYS + ["timestamp_local"]
        ).any():
            raise RuntimeError("Duplicate output timestamps detected.")

        with temp.open("rb") as source, OUTPUT.open("xb") as destination:
            destination.write(source.read())
    finally:
        if temp.exists():
            temp.unlink()

    print("Western Ghats rainfall features created.")
    print("Rows:", len(output))
    print(
        "Monitoring points:",
        output[GROUP_KEYS].drop_duplicates().shape[0],
    )
    print("Output:", OUTPUT)

    print("\nRows by period:")
    print(output["data_period"].value_counts().to_string())

    print("\nFeature availability:")
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
    for column in feature_columns:
        if column in output.columns:
            print(f"{column}: {output[column].notna().sum()} values")


if __name__ == "__main__":
    main()
