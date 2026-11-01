
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/rainfall/regional_hourly_rainfall_30oct.csv"
OUTPUT = ROOT / "data/processed/rainfall/regional_rainfall_features_30oct.csv"

WINDOWS = [1, 3, 6, 12, 24]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}. "
            "Refusing to overwrite."
        )

    df = pd.read_csv(INPUT)
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="coerce"
    )
    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"], errors="coerce"
    )

    required = [
        "region", "tile", "timestamp_local",
        "precipitation_mm", "record_type"
    ]

    if df[required].isna().any().any():
        raise ValueError(
            "Missing region, tile, timestamp or record type found."
        )

    if (df["precipitation_mm"].dropna() < 0).any():
        raise ValueError("Negative rainfall detected.")

    df = df.sort_values(
        ["region", "tile", "timestamp_local"]
    ).copy()

    if df.duplicated(
        ["region", "tile", "timestamp_local"]
    ).any():
        raise ValueError("Duplicate tile timestamps found.")

    outputs = []

    for (region, tile), group in df.groupby(
        ["region", "tile"], sort=False
    ):
        group = group.sort_values("timestamp_local").copy()

        observed = group[
            group["record_type"] == "historical_or_current"
        ].copy()

        forecast = group[
            group["record_type"] == "forecast"
        ].copy()

        # Rolling sums require complete hourly windows.
        observed = observed.set_index("timestamp_local")

        for hours in WINDOWS:
            observed[f"recent_rainfall_{hours}h_mm"] = (
                observed["precipitation_mm"]
                .rolling(
                    f"{hours}h",
                    closed="right",
                    min_periods=hours,
                )
                .sum()
            )

        observed = observed.reset_index()

        # Forecast accumulation: future hours only,
        # excluding the current timestamp.
        forecast = forecast.set_index("timestamp_local")

        for hours in [3, 6, 24]:
            forecast[f"forecast_rainfall_{hours}h_mm"] = (
                forecast["precipitation_mm"]
                .rolling(
                    f"{hours}h",
                    closed="right",
                    min_periods=hours,
                )
                .sum()
            )

        forecast = forecast.reset_index()

        observed_columns = [
            "region", "tile", "requested_latitude",
            "requested_longitude", "api_latitude",
            "api_longitude", "timestamp_local",
            "source", "retrieved_at_local",
            *[f"recent_rainfall_{h}h_mm" for h in WINDOWS],
        ]

        forecast_columns = [
            "region", "tile", "timestamp_local",
            *[
                f"forecast_rainfall_{h}h_mm"
                for h in [3, 6, 24]
            ],
        ]

        outputs.append(
            observed[observed_columns]
        )

        outputs.append(
            forecast[forecast_columns]
        )

    result = pd.concat(outputs, ignore_index=True, sort=False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("========== RAINFALL FEATURES ==========")
    print("Saved:", OUTPUT)
    print("Total records:", len(result))
    print("Regions:")
    print(result.groupby("region").size().to_string())

    print("\nMissing feature counts:")
    feature_columns = [
        c for c in result.columns
        if "rainfall_" in c
    ]
    print(result[feature_columns].isna().sum().to_string())

    print("\nFeature generation completed.")


if __name__ == "__main__":
    main()
