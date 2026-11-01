
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_experimental_risk_scores.csv"
)

TOLERANCE = 1e-6


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def normalize_rainfall(values, scale_mm):
    return (values / scale_mm).clip(lower=0, upper=1)


def expected_recent_index(df):
    rainfall_component = (
        0.4 * normalize_rainfall(df["recent_rain_1h_mm"], 10)
        + 0.6 * normalize_rainfall(df["recent_rain_24h_mm"], 100)
    )
    return 100 * (
        0.4 * (df["slope_degrees"].clip(0, 45) / 45)
        + 0.6 * rainfall_component
    )


def expected_forecast_index(df):
    rainfall_component = (
        0.4 * normalize_rainfall(df["forecast_rain_3h_mm"], 10)
        + 0.6 * normalize_rainfall(df["forecast_rain_24h_mm"], 100)
    )
    return 100 * (
        0.4 * (df["slope_degrees"].clip(0, 45) / 45)
        + 0.6 * rainfall_component
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "sample_id",
        "pilot_tile",
        "timestamp_local",
        "slope_degrees",
        "recent_rain_1h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_3h_mm",
        "forecast_rain_24h_mm",
        "recent_risk_index",
        "forecast_risk_index",
        "recent_score_status",
        "forecast_score_status",
        "temporal_role",
    ]

    missing_columns = [c for c in required if c not in df.columns]
    check(not missing_columns, f"Required columns present: {missing_columns}")

    check(len(df) > 0, "Dataset is not empty")

    check(
        not df.duplicated(["sample_id", "timestamp_local"]).any(),
        "No duplicate sample/timestamp records",
    )

    check(
        df["sample_id"].nunique() == 656,
        "All 656 pilot samples are represented",
    )

    check(
        set(df["pilot_tile"].dropna().unique())
        == {"N09E076", "N11E076", "N19E072"},
        "Expected three pilot tiles are present",
    )

    numeric = [
        "slope_degrees",
        "recent_rain_1h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_3h_mm",
        "forecast_rain_24h_mm",
        "recent_risk_index",
        "forecast_risk_index",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    check(
        df["slope_degrees"].notna().all(),
        "Slope is available for every score record",
    )

    for col in [
        "recent_rain_1h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_3h_mm",
        "forecast_rain_24h_mm",
    ]:
        valid = df[col].dropna()
        check(
            (valid >= 0).all(),
            f"{col} contains no negative rainfall values",
        )

    # Validate score range.
    for col in ["recent_risk_index", "forecast_risk_index"]:
        valid = df[col].dropna()
        check(
            valid.between(0, 100).all(),
            f"{col} remains within 0–100",
        )

    # Check missing-window handling.
    check(
        df.loc[
            df["recent_rain_24h_mm"].isna(),
            "recent_risk_index",
        ].isna().all(),
        "Recent index stays missing when its 24-hour window is incomplete",
    )

    check(
        df.loc[
            df["forecast_rain_24h_mm"].isna(),
            "forecast_risk_index",
        ].isna().all(),
        "Forecast index stays missing when its 24-hour window is incomplete",
    )

    check(
        (
            df["recent_score_status"]
            == "incomplete_recent_rainfall_window"
        ).equals(df["recent_rain_24h_mm"].isna()),
        "Recent score status matches rainfall-window availability",
    )

    check(
        (
            df["forecast_score_status"]
            == "incomplete_forecast_window"
        ).equals(df["forecast_rain_24h_mm"].isna()),
        "Forecast score status matches rainfall-window availability",
    )

    # Independently recalculate the index from its documented formula.
    recent_mask = (
        df["recent_rain_1h_mm"].notna()
        & df["recent_rain_24h_mm"].notna()
        & df["recent_risk_index"].notna()
    )

    forecast_mask = (
        df["forecast_rain_3h_mm"].notna()
        & df["forecast_rain_24h_mm"].notna()
        & df["forecast_risk_index"].notna()
    )

    expected_recent = expected_recent_index(df.loc[recent_mask])
    expected_forecast = expected_forecast_index(df.loc[forecast_mask])

    check(
        np.allclose(
            df.loc[recent_mask, "recent_risk_index"],
            expected_recent,
            atol=TOLERANCE,
            rtol=0,
        ),
        "Recent index matches the documented scoring formula",
    )

    check(
        np.allclose(
            df.loc[forecast_mask, "forecast_risk_index"],
            expected_forecast,
            atol=TOLERANCE,
            rtol=0,
        ),
        "Forecast index matches the documented scoring formula",
    )

    # Synthetic monotonicity test: hold slope fixed and increase rainfall.
    # These synthetic cases do not modify or add to the real dataset.
    slope = 20.0

    def recent_score(rain_1h, rain_24h):
        component = (
            0.4 * min(rain_1h / 10, 1)
            + 0.6 * min(rain_24h / 100, 1)
        )
        return 100 * (
            0.4 * (min(max(slope, 0), 45) / 45)
            + 0.6 * component
        )

    def forecast_score(rain_3h, rain_24h):
        component = (
            0.4 * min(rain_3h / 10, 1)
            + 0.6 * min(rain_24h / 100, 1)
        )
        return 100 * (
            0.4 * (min(max(slope, 0), 45) / 45)
            + 0.6 * component
        )

    recent_low = recent_score(1, 10)
    recent_high = recent_score(5, 40)

    forecast_low = forecast_score(1, 10)
    forecast_high = forecast_score(5, 40)

    check(
        recent_high > recent_low,
        "Synthetic increase in recent rainfall increases recent index",
    )

    check(
        forecast_high > forecast_low,
        "Synthetic increase in forecast rainfall increases forecast index",
    )

    # Check that the temporal categories are present and meaningful.
    check(
        df["temporal_role"].isin([
            "historical_context_only",
            "current_or_forecast_candidate",
        ]).all(),
        "Temporal-role labels use the expected categories",
    )

    print("\n--- Validation summary ---")
    print("Rows:", len(df))
    print("Unique samples:", df["sample_id"].nunique())
    print("Recent index available:", int(df["recent_risk_index"].notna().sum()))
    print("Forecast index available:", int(df["forecast_risk_index"].notna().sum()))
    print("Recent index independently checked:", int(recent_mask.sum()))
    print("Forecast index independently checked:", int(forecast_mask.sum()))

    print("\nRecent index distribution:")
    print(df["recent_risk_index"].describe().round(3).to_string())

    print("\nForecast index distribution:")
    print(df["forecast_risk_index"].describe().round(3).to_string())

    print(
        "\nLIMITATION: These checks validate implementation and arithmetic, "
        "not predictive accuracy. The scoring formula remains illustrative "
        "and has not been calibrated against independent landslide outcomes."
    )

    print("\nVALIDATION COMPLETED.")


if __name__ == "__main__":
    main()
