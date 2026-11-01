
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_live_risk_inputs.csv"
)

OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_experimental_risk_scores.csv"
)


def normalize_rainfall(values, scale_mm):
    return (values / scale_mm).clip(lower=0, upper=1)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "sample_id",
        "pilot_tile",
        "requested_latitude",
        "requested_longitude",
        "terrain_latitude",
        "terrain_longitude",
        "timestamp_local",
        "record_type",
        "retrieved_at_local",
        "slope_degrees",
        "recent_rain_1h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_3h_mm",
        "forecast_rain_24h_mm",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.duplicated(["sample_id", "timestamp_local"]).any():
        raise ValueError("Duplicate sample/timestamp combinations found.")

    numeric_columns = [
        "requested_latitude",
        "requested_longitude",
        "terrain_latitude",
        "terrain_longitude",
        "slope_degrees",
        "recent_rain_1h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_3h_mm",
        "forecast_rain_24h_mm",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["terrain_latitude", "terrain_longitude"]].isna().any().any():
        raise ValueError("Missing terrain coordinates.")

    if df["slope_degrees"].isna().any():
        raise ValueError("Missing slope values.")

    if (df["slope_degrees"] < 0).any():
        raise ValueError("Negative slope values detected.")

    # Illustrative, uncalibrated transformations.
    # These are not validated landslide thresholds.
    df["slope_component"] = (
        df["slope_degrees"].clip(lower=0, upper=45) / 45
    )

    df["recent_rain_component"] = (
        0.4 * normalize_rainfall(df["recent_rain_1h_mm"], 10)
        + 0.6 * normalize_rainfall(df["recent_rain_24h_mm"], 100)
    )

    df["forecast_rain_component"] = (
        0.4 * normalize_rainfall(df["forecast_rain_3h_mm"], 10)
        + 0.6 * normalize_rainfall(df["forecast_rain_24h_mm"], 100)
    )

    df["recent_risk_index"] = 100 * (
        0.4 * df["slope_component"]
        + 0.6 * df["recent_rain_component"]
    )

    df["forecast_risk_index"] = 100 * (
        0.4 * df["slope_component"]
        + 0.6 * df["forecast_rain_component"]
    )

    # Only score rows where the full 24-hour rainfall window exists.
    df["recent_score_status"] = np.where(
        df["recent_rain_24h_mm"].notna(),
        "experimental_screening_only",
        "incomplete_recent_rainfall_window",
    )

    df["forecast_score_status"] = np.where(
        df["forecast_rain_24h_mm"].notna(),
        "experimental_screening_only",
        "incomplete_forecast_window",
    )

    df.loc[
        df["recent_rain_24h_mm"].isna(), "recent_risk_index"
    ] = np.nan

    df.loc[
        df["forecast_rain_24h_mm"].isna(), "forecast_risk_index"
    ] = np.nan

    # Separate past-context rows from current/future candidate rows.
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    retrieved = pd.to_datetime(df["retrieved_at_local"], utc=True)
    local_timestamps = df["timestamp_local"].dt.tz_localize(
        "Asia/Kolkata"
    )
    df["temporal_role"] = np.where(
        local_timestamps >= retrieved.dt.tz_convert("Asia/Kolkata").dt.floor("h"),
        "current_or_forecast_candidate",
        "historical_context_only",
    )

    df["score_interpretation"] = (
        "Experimental index only; not a calibrated probability or warning"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        df.to_csv(handle, index=False)

    print("Experimental risk-score dataset created.")
    print("Rows:", len(df))
    print("Unique samples:", df["sample_id"].nunique())
    print("Tiles:", sorted(df["pilot_tile"].unique()))
    print("Output:", OUTPUT)

    print("\nTemporal roles:")
    print(df["temporal_role"].value_counts().to_string())

    print("\nRecent score availability:")
    print(df["recent_score_status"].value_counts().to_string())

    print("\nForecast score availability:")
    print(df["forecast_score_status"].value_counts().to_string())

    print("\nScore summary:")
    print(
        df[["recent_risk_index", "forecast_risk_index"]]
        .describe().round(2).to_string()
    )

    print(
        "\nNote: rainfall scales and weights are illustrative, "
        "not calibrated thresholds. Historical labels are not used "
        "to calculate either score."
    )
    print("Existing datasets and trained models were not modified.")


if __name__ == "__main__":
    main()
