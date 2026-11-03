from pathlib import Path
import argparse

import pandas as pd

from backend.gpm_refresh.refresh_config import (
    PROJECT_ROOT,
    get_run_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine previous and latest 12-hour GPM rainfall windows."
    )

    parser.add_argument("--previous-run-id", required=True)
    parser.add_argument("--latest-run-id", required=True)
    parser.add_argument("--previous-csv", required=True)
    parser.add_argument("--latest-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--expected-records", type=int, default=None)
    parser.add_argument("--expected-intervals", type=int, default=48)

    return parser.parse_args()


def get_series_boundary(df, column, dataset_name):
    if column not in df.columns:
        raise ValueError(
            f"Missing timestamp column in {dataset_name}: {column}"
        )

    values = pd.to_datetime(df[column], utc=True, errors="coerce")

    if values.isna().any():
        raise ValueError(
            f"Invalid timestamps found in {dataset_name}: {column}"
        )

    unique_values = values.unique()

    if len(unique_values) != 1:
        raise ValueError(
            f"{dataset_name} contains inconsistent values in {column}."
        )

    return values.iloc[0]


def main():
    args = parse_args()

    previous_csv = Path(args.previous_csv)
    latest_csv = Path(args.latest_csv)

    if not previous_csv.is_absolute():
        previous_csv = PROJECT_ROOT / previous_csv

    if not latest_csv.is_absolute():
        latest_csv = PROJECT_ROOT / latest_csv

    if not previous_csv.exists():
        raise FileNotFoundError(
            f"Previous 12-hour CSV not found: {previous_csv}"
        )

    if not latest_csv.exists():
        raise FileNotFoundError(
            f"Latest 12-hour CSV not found: {latest_csv}"
        )

    previous = pd.read_csv(previous_csv)
    latest = pd.read_csv(latest_csv)

    print("\nPrevious records:", len(previous))
    print("Latest records:", len(latest))

    required_common = {
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
        "gpm_12h_series_start_utc",
        "gpm_12h_series_end_utc",
        "gpm_12h_accumulation_mm",
        "gpm_12h_valid_30min_intervals",
    }

    for name, frame in [
        ("Previous", previous),
        ("Latest", latest),
    ]:
        missing = required_common - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in {name} dataset: "
                f"{sorted(missing)}"
            )

        if frame.empty:
            raise ValueError(f"{name} dataset is empty.")

        if frame["record_id"].isna().any():
            raise ValueError(f"Missing record_id values in {name} dataset.")

        if frame["record_id"].duplicated().any():
            raise ValueError(
                f"Duplicate record IDs in {name} dataset."
            )

    if set(previous["record_id"]) != set(latest["record_id"]):
        raise ValueError(
            "Record IDs differ between the two datasets."
        )

    previous = previous.sort_values("record_id").reset_index(drop=True)
    latest = latest.sort_values("record_id").reset_index(drop=True)

    for column in ["latitude", "longitude", "dem_region"]:
        if not previous[column].equals(latest[column]):
            raise ValueError(
                f"Mismatch in {column} between datasets."
            )

    previous_start = get_series_boundary(
        previous, "gpm_12h_series_start_utc", "Previous"
    )
    previous_end = get_series_boundary(
        previous, "gpm_12h_series_end_utc", "Previous"
    )
    latest_start = get_series_boundary(
        latest, "gpm_12h_series_start_utc", "Latest"
    )
    latest_end = get_series_boundary(
        latest, "gpm_12h_series_end_utc", "Latest"
    )

    if previous_end - previous_start != pd.Timedelta(hours=12):
        raise ValueError(
            f"Previous dataset is not exactly 12 hours: "
            f"{previous_start} to {previous_end}."
        )

    if latest_end - latest_start != pd.Timedelta(hours=12):
        raise ValueError(
            f"Latest dataset is not exactly 12 hours: "
            f"{latest_start} to {latest_end}."
        )

    if previous_end != latest_start:
        raise ValueError(
            "The two 12-hour windows are not contiguous. "
            f"Previous ends at {previous_end}, "
            f"latest starts at {latest_start}."
        )

    if previous_end > latest_start:
        raise ValueError("The 12-hour windows overlap.")

    if args.expected_records is not None:
        if len(previous) != args.expected_records:
            raise ValueError(
                f"Previous record count mismatch: "
                f"{len(previous)} != {args.expected_records}"
            )

        if len(latest) != args.expected_records:
            raise ValueError(
                f"Latest record count mismatch: "
                f"{len(latest)} != {args.expected_records}"
            )

    combined = latest.copy()

    previous_columns = [
        column
        for column in previous.columns
        if column not in {
            "record_id",
            "latitude",
            "longitude",
            "dem_region",
        }
    ]

    for column in previous_columns:
        if column in combined.columns:
            combined[f"gpm_previous12h_{column}"] = previous[column].values
        else:
            combined[column] = previous[column].values

    previous_rain = pd.to_numeric(
        previous["gpm_12h_accumulation_mm"],
        errors="coerce",
    )

    latest_rain = pd.to_numeric(
        latest["gpm_12h_accumulation_mm"],
        errors="coerce",
    )

    if previous_rain.isna().any() or latest_rain.isna().any():
        raise ValueError("Missing rainfall values in one of the 12-hour windows.")

    if (previous_rain < 0).any() or (latest_rain < 0).any():
        raise ValueError("Negative rainfall values detected.")

    combined["gpm_24h_accumulation_mm"] = (
        previous_rain.values + latest_rain.values
    )

    previous_intervals = pd.to_numeric(
        previous["gpm_12h_valid_30min_intervals"],
        errors="coerce",
    )

    latest_intervals = pd.to_numeric(
        latest["gpm_12h_valid_30min_intervals"],
        errors="coerce",
    )

    if previous_intervals.isna().any() or latest_intervals.isna().any():
        raise ValueError("Invalid 30-minute interval counts.")

    combined["gpm_24h_valid_30min_intervals"] = (
        previous_intervals.values + latest_intervals.values
    )

    if not (
        combined["gpm_24h_valid_30min_intervals"]
        == args.expected_intervals
    ).all():
        raise ValueError(
            "24-hour interval count validation failed. "
            f"Expected {args.expected_intervals} intervals."
        )

    combined["gpm_24h_series_start_utc"] = previous_start.isoformat()
    combined["gpm_24h_series_end_utc"] = latest_end.isoformat()

    if latest_end - previous_start != pd.Timedelta(hours=24):
        raise ValueError("Combined rainfall period is not exactly 24 hours.")

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else get_run_directory(args.latest_run_id)
        / "extracted"
        / "regional_gpm_rainfall_24h.csv"
    )

    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv

    if output_csv.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_csv}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print("\n24-hour combination completed.")
    print("Output:", output_csv)
    print("Records:", len(combined))
    print("24-hour rainfall column: gpm_24h_accumulation_mm")
    print("24-hour start:", previous_start)
    print("24-hour end:", latest_end)


if __name__ == "__main__":
    main()