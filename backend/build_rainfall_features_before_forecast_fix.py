
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "rainfall" / "pilot_hourly_rainfall.csv"
OUTPUT = ROOT / "data" / "processed" / "rainfall" / "pilot_rainfall_features.csv"

GROUP_KEYS = [
    "region",
    "requested_latitude",
    "requested_longitude",
    "api_latitude",
    "api_longitude",
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

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
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="raise"
    )
    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"], errors="raise"
    )

    if df["precipitation_mm"].isna().any():
        raise ValueError("Rainfall data contains missing values.")

    if (df["precipitation_mm"] < 0).any():
        raise ValueError("Rainfall data contains negative values.")

    if not df["record_type"].isin(
        ["historical_or_current", "forecast"]
    ).all():
        raise ValueError("Unexpected record_type value.")

    if df.duplicated(GROUP_KEYS + ["timestamp_local"]).any():
        raise ValueError("Duplicate timestamps found for a monitoring point.")

    df = df.sort_values(GROUP_KEYS + ["timestamp_local"]).copy()

    # Do not combine forecast values with recent/current rainfall.
    recent = df[df["record_type"] == "historical_or_current"].copy()
    forecast = df[df["record_type"] == "forecast"].copy()

    recent["recent_rainfall_1h_mm"] = recent["precipitation_mm"]

    for hours in [3, 6, 12, 24]:
        recent[f"recent_rainfall_{hours}h_mm"] = (
            recent.groupby(GROUP_KEYS, sort=False)["precipitation_mm"]
            .transform(
                lambda s: s.rolling(hours, min_periods=hours).sum()
            )
        )

    recent["recent_records_24h"] = (
        recent.groupby(GROUP_KEYS, sort=False)["precipitation_mm"]
        .transform(lambda s: s.rolling(24, min_periods=1).count())
        .astype(int)
    )

    # Forecast features start at the first forecast timestamp for each point.
    for hours in [3, 6, 24]:
        forecast[f"forecast_rainfall_{hours}h_mm"] = (
            forecast.groupby(GROUP_KEYS, sort=False)["precipitation_mm"]
            .transform(
                lambda s: s.rolling(hours, min_periods=hours).sum()
            )
        )

    recent["data_period"] = "historical_or_current"
    forecast["data_period"] = "forecast"

    # Keep the source classification explicit in the output.
    recent_features = recent[
        GROUP_KEYS
        + [
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
    ].copy()

    forecast_features = forecast[
        GROUP_KEYS
        + [
            "timestamp_local",
            "data_period",
            "source",
            "retrieved_at_local",
            "forecast_rainfall_3h_mm",
            "forecast_rainfall_6h_mm",
            "forecast_rainfall_24h_mm",
        ]
    ].copy()

    # Combine rows without treating forecast values as observed rainfall.
    output = pd.concat(
        [recent_features, forecast_features],
        ignore_index=True,
        sort=False,
    ).sort_values(GROUP_KEYS + ["timestamp_local"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    print("Rainfall feature dataset created successfully.")
    print("Output:", OUTPUT)
    print("Rows:", len(output))
    print("Monitoring points:", output[GROUP_KEYS].drop_duplicates().shape[0])
    print("\nRows by data period:")
    print(output["data_period"].value_counts().to_string())
    print("\nRecent rainfall feature availability:")
    for col in [
        "recent_rainfall_1h_mm",
        "recent_rainfall_3h_mm",
        "recent_rainfall_6h_mm",
        "recent_rainfall_12h_mm",
        "recent_rainfall_24h_mm",
    ]:
        print(f"{col}: {output[col].notna().sum()} values")
    print("\nForecast rainfall feature availability:")
    for col in [
        "forecast_rainfall_3h_mm",
        "forecast_rainfall_6h_mm",
        "forecast_rainfall_24h_mm",
    ]:
        print(f"{col}: {output[col].notna().sum()} values")

if __name__ == "__main__":
    main()
