
from pathlib import Path
import pandas as pd
import numpy as np

from backend.gpm_refresh.refresh_config import get_run_directory

RUN_ID = "20261102_152742"

INPUT_CSV = (
    get_run_directory(RUN_ID)
    / "extracted"
    / "gpm_regional_rainfall_24h.csv"
)

EXPECTED_ROWS = 2009
EXPECTED_INTERVALS = 48

REQUIRED_COLUMNS = [
    "record_id",
    "latitude",
    "longitude",
    "dem_region",
    "gpm_24h_accumulation_mm",
    "gpm_24h_valid_30min_intervals",
    "gpm_24h_series_start_utc",
    "gpm_24h_series_end_utc",
]

def main():
    print("\nNASA GPM: 24-HOUR DATASET VALIDATION")
    print("=" * 60)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"Dataset: {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print("\n1. Required columns: PASSED")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(df)}"
        )

    if df["record_id"].isna().any():
        raise ValueError("Missing record IDs detected")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs detected")

    print("2. Record count and uniqueness: PASSED")

    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Missing coordinates detected")

    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values")

    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values")

    print("3. Coordinate validation: PASSED")

    rainfall = pd.to_numeric(
        df["gpm_24h_accumulation_mm"],
        errors="coerce"
    )

    if rainfall.isna().any():
        raise ValueError("Missing or invalid 24-hour rainfall")

    if not np.isfinite(rainfall).all():
        raise ValueError("Non-finite rainfall values detected")

    if (rainfall < 0).any():
        raise ValueError("Negative rainfall values detected")

    print("4. Rainfall values: PASSED")

    intervals = pd.to_numeric(
        df["gpm_24h_valid_30min_intervals"],
        errors="coerce"
    )

    if intervals.isna().any():
        raise ValueError("Missing interval counts")

    if not (intervals == EXPECTED_INTERVALS).all():
        raise ValueError("Some records do not contain 48 valid intervals")

    print("5. 48 valid half-hour intervals: PASSED")

    starts = df["gpm_24h_series_start_utc"].dropna().unique()
    ends = df["gpm_24h_series_end_utc"].dropna().unique()

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Inconsistent 24-hour time coverage")

    start = pd.to_datetime(starts[0], utc=True)
    end = pd.to_datetime(ends[0], utc=True)

    if start != pd.Timestamp("2026-11-01T03:00:00Z"):
        raise ValueError(f"Unexpected start time: {start}")

    if end != pd.Timestamp("2026-11-02T03:00:00Z"):
        raise ValueError(f"Unexpected end time: {end}")

    if end - start != pd.Timedelta(hours=24):
        raise ValueError("Time coverage is not exactly 24 hours")

    print("6. Temporal coverage: PASSED")

    print("\nREGIONAL RAINFALL SUMMARY")
    print("-" * 60)

    print(
        df.groupby("dem_region")["gpm_24h_accumulation_mm"]
        .agg(["count", "mean", "max"])
        .round(3)
    )

    print("\nALL VALIDATION CHECKS PASSED.")
    print("24-hour GPM dataset is ready for integration.")

if __name__ == "__main__":
    main()
