import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backend.gpm_refresh.refresh_config import get_run_directory


DEFAULT_RUN_ID = "20261102_152742"
DEFAULT_EXPECTED_INTERVALS = 48

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a refreshed 24-hour NASA GPM rainfall dataset."
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"Refresh run ID (default: {DEFAULT_RUN_ID}).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional explicit path to the 24-hour CSV.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Optional expected row count. Omit to accept a dynamic row count.",
    )
    parser.add_argument(
        "--expected-intervals",
        type=int,
        default=DEFAULT_EXPECTED_INTERVALS,
        help=f"Required valid half-hour interval count "
             f"(default: {DEFAULT_EXPECTED_INTERVALS}).",
    )
    parser.add_argument(
        "--expected-start",
        default=None,
        help="Optional expected UTC start timestamp, e.g. 2026-11-02T03:00:00Z.",
    )
    parser.add_argument(
        "--expected-end",
        default=None,
        help="Optional expected UTC end timestamp, e.g. 2026-11-03T03:00:00Z.",
    )
    return parser.parse_args()


def parse_utc(value, label):
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid {label} timestamp: {value}") from exc

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def main():
    args = parse_args()

    if args.expected_intervals <= 0:
        raise ValueError("--expected-intervals must be greater than zero.")

    if args.expected_rows is not None and args.expected_rows <= 0:
        raise ValueError("--expected-rows must be greater than zero.")

    if bool(args.expected_start) != bool(args.expected_end):
        raise ValueError(
            "Provide both --expected-start and --expected-end, or neither."
        )

    input_csv = (
        args.input_csv
        if args.input_csv is not None
        else get_run_directory(args.run_id)
        / "extracted"
        / "gpm_regional_rainfall_24h.csv"
    )

    print("\\nNASA GPM: 24-HOUR DATASET VALIDATION")
    print("=" * 60)
    print(f"Run ID: {args.run_id}")
    print(f"Dataset: {input_csv}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print("\\n1. Required columns: PASSED")

    if args.expected_rows is not None and len(df) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} rows, found {len(df)}"
        )

    if df["record_id"].isna().any():
        raise ValueError("Missing record IDs detected")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs detected")

    print("2. Record count and uniqueness: PASSED")

    coordinates = df[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )

    if coordinates.isna().any().any():
        raise ValueError("Missing or invalid coordinates detected")

    if not coordinates["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values")

    if not coordinates["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values")

    print("3. Coordinate validation: PASSED")

    rainfall = pd.to_numeric(
        df["gpm_24h_accumulation_mm"],
        errors="coerce",
    )

    if rainfall.isna().any():
        raise ValueError("Missing or invalid 24-hour rainfall")

    if not np.isfinite(rainfall.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite rainfall values detected")

    if (rainfall < 0).any():
        raise ValueError("Negative rainfall values detected")

    print("4. Rainfall values: PASSED")

    intervals = pd.to_numeric(
        df["gpm_24h_valid_30min_intervals"],
        errors="coerce",
    )

    if intervals.isna().any():
        raise ValueError("Missing or invalid interval counts")

    if not np.isfinite(intervals.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite interval counts detected")

    if not (intervals == args.expected_intervals).all():
        raise ValueError(
            f"Some records do not contain {args.expected_intervals} "
            "valid half-hour intervals"
        )

    print(
        f"5. {args.expected_intervals} valid half-hour intervals: PASSED"
    )

    starts = df["gpm_24h_series_start_utc"].dropna().astype(str).unique()
    ends = df["gpm_24h_series_end_utc"].dropna().astype(str).unique()

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Inconsistent 24-hour time coverage")

    start = parse_utc(starts[0], "series start")
    end = parse_utc(ends[0], "series end")

    if end - start != pd.Timedelta(hours=24):
        raise ValueError("Time coverage is not exactly 24 hours")

    if args.expected_start:
        expected_start = parse_utc(args.expected_start, "expected start")
        expected_end = parse_utc(args.expected_end, "expected end")

        if end <= start:
            raise ValueError("End timestamp must be later than start.")

        if start != expected_start:
            raise ValueError(
                f"Unexpected start time: expected {expected_start}, found {start}"
            )

        if end != expected_end:
            raise ValueError(
                f"Unexpected end time: expected {expected_end}, found {end}"
            )

        if expected_end - expected_start != pd.Timedelta(hours=24):
            raise ValueError(
                "The supplied expected start/end do not span exactly 24 hours."
            )

    print("6. Temporal coverage: PASSED")
    print("   Start UTC:", start)
    print("   End UTC:", end)

    print("\\nREGIONAL RAINFALL SUMMARY")
    print("-" * 60)

    print(
        df.groupby("dem_region")["gpm_24h_accumulation_mm"]
        .agg(["count", "mean", "max"])
        .round(3)
    )

    print("\\nALL VALIDATION CHECKS PASSED.")
    print("24-hour GPM dataset is ready for integration.")


if __name__ == "__main__":
    main()
