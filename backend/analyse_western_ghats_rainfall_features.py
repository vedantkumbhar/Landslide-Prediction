
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "rainfall_features.csv"
)

RECENT = [
    "recent_rain_1h_mm",
    "recent_rain_3h_mm",
    "recent_rain_6h_mm",
    "recent_rain_12h_mm",
    "recent_rain_24h_mm",
]

FORECAST = [
    "forecast_rain_3h_mm",
    "forecast_rain_6h_mm",
    "forecast_rain_24h_mm",
]

def main():
    df = pd.read_csv(INPUT)

    required = [
        "sample_id", "pilot_tile", "label",
        "timestamp_local", "record_type",
        *RECENT, *FORECAST,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required[:5]].isna().any().any():
        raise ValueError("Missing sample metadata detected.")

    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="raise"
    )

    print("ROWS AND SAMPLES")
    print("Rows:", len(df))
    print("Samples:", df["sample_id"].nunique())
    print("Labels per sample:")
    print(
        df.groupby("sample_id")["label"].nunique()
        .value_counts().sort_index().to_string()
    )

    print("\nRECORD TYPES")
    print(df["record_type"].value_counts(dropna=False).to_string())

    print("\nFEATURE COVERAGE")
    for column in RECENT + FORECAST:
        print(
            f"{column}: available={df[column].notna().sum()}, "
            f"missing={df[column].isna().sum()}"
        )

    # Compare samples using the same timestamp so the two labels
    # are compared at equivalent times.
    metadata = (
        df.groupby("sample_id", as_index=False)
        .agg(
            pilot_tile=("pilot_tile", "first"),
            label=("label", "first"),
        )
    )

    # One sample-level summary for recent rainfall, using available
    # values across its hourly time series.
    summaries = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        row = {
            "sample_id": sample_id,
            "pilot_tile": group["pilot_tile"].iloc[0],
            "label": group["label"].iloc[0],
        }

        for column in RECENT + FORECAST:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_p95"] = group[column].quantile(0.95)
            row[f"{column}_valid_fraction"] = group[column].notna().mean()

        summaries.append(row)

    sample_summary = pd.DataFrame(summaries)

    print("\nSAMPLE-LEVEL RAINFALL SUMMARY BY LABEL")
    for column in RECENT + FORECAST:
        mean_column = f"{column}_mean"
        print(f"\n{column}")
        print(
            sample_summary.groupby("label")[mean_column]
            .agg(["count", "mean", "median", "std"])
            .round(3).to_string()
        )

    print("\nSAMPLE-LEVEL SUMMARY BY TILE AND LABEL")
    for column in [
        "recent_rain_24h_mm",
        "recent_rain_6h_mm",
        "forecast_rain_24h_mm",
    ]:
        mean_column = f"{column}_mean"
        print(f"\n{column}")
        print(
            sample_summary.groupby(["pilot_tile", "label"])[mean_column]
            .agg(["count", "mean", "median"])
            .round(3).to_string()
        )

    print("\nAVERAGE FEATURE COVERAGE BY LABEL")
    coverage_columns = [
        f"{column}_valid_fraction"
        for column in RECENT + FORECAST
    ]
    print(
        sample_summary.groupby("label")[coverage_columns]
        .mean().T.round(3).to_string()
    )

    print(
        "\nInterpretation caution: rainfall data covers "
        "October 27–31, 2026, not the historical event dates. "
        "Differences between labels do not establish causation "
        "or historical predictive skill."
    )
    print("Read-only diagnostic: no files were modified.")


if __name__ == "__main__":
    main()
