
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation"
)

EVENTS_FILE = BASE / "pilot_historical_events.csv"
EVENT_HOURLY_FILE = BASE / "pilot_historical_hourly_rainfall.csv"
CONTROL_HOURLY_FILE = BASE / "control_historical_hourly_rainfall.csv"
MATCH_FILE = BASE / "matched_rainfall_controls.csv"

PAIRS_OUTPUT = BASE / "paired_event_control_rainfall_analysis.csv"
SUMMARY_OUTPUT = BASE / "event_control_rainfall_comparison_summary.csv"


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_hourly(path, id_column):
    df = pd.read_csv(path)

    require_columns(
        df,
        [id_column, "timestamp_local", "precipitation_mm"],
        path.name,
    )

    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="coerce"
    )
    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"], errors="coerce"
    )

    if df["timestamp_local"].isna().any():
        raise ValueError(f"Invalid timestamps found in {path.name}")

    if df.duplicated([id_column, "timestamp_local"]).any():
        raise ValueError(f"Duplicate ID/timestamp rows in {path.name}")

    return df


def sum_complete_window(rows, start, end, expected_hours):
    window = rows[
        (rows["timestamp_local"] > start)
        & (rows["timestamp_local"] <= end)
    ]

    values = window["precipitation_mm"].dropna()

    if len(values) != expected_hours:
        return None, len(values)

    return round(float(values.sum()), 3), len(values)


def sum_calendar_days(rows, start, end, expected_hours):
    window = rows[
        (rows["timestamp_local"] >= start)
        & (rows["timestamp_local"] < end)
    ]

    values = window["precipitation_mm"].dropna()

    if len(values) != expected_hours:
        return None, len(values)

    return round(float(values.sum()), 3), len(values)


def calculate_metrics(rows, event_datetime, time_status):
    day_start = event_datetime.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    event_day, event_day_hours = sum_calendar_days(
        rows, day_start, day_end, 24
    )

    result = {
        "pre_event_24h_mm": None,
        "pre_event_72h_mm": None,
        "event_day_precipitation_mm": event_day,
        "pre_event_24h_valid_hours": 0,
        "pre_event_72h_valid_hours": 0,
        "event_day_valid_hours": event_day_hours,
    }

    if time_status == "time_present_in_event_date":
        # Use the original recorded event timestamp.
        rain24, hours24 = sum_complete_window(
            rows,
            event_datetime - pd.Timedelta(hours=24),
            event_datetime,
            24,
        )
        rain72, hours72 = sum_complete_window(
            rows,
            event_datetime - pd.Timedelta(hours=72),
            event_datetime,
            72,
        )

        result["pre_event_24h_mm"] = rain24
        result["pre_event_72h_mm"] = rain72
        result["pre_event_24h_valid_hours"] = hours24
        result["pre_event_72h_valid_hours"] = hours72

    else:
        # Midnight is ambiguous. Compare the same three complete
        # calendar days before the recorded date for both groups.
        start = day_start - pd.Timedelta(days=3)

        rain72, hours72 = sum_calendar_days(
            rows, start, day_start, 72
        )

        result["pre_event_72h_mm"] = rain72
        result["pre_event_72h_valid_hours"] = hours72

    return result


def main():
    required_files = [
        EVENTS_FILE,
        EVENT_HOURLY_FILE,
        CONTROL_HOURLY_FILE,
        MATCH_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    for path in [PAIRS_OUTPUT, SUMMARY_OUTPUT]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}"
            )

    events = pd.read_csv(EVENTS_FILE)
    matches = pd.read_csv(MATCH_FILE)

    require_columns(
        events,
        [
            "event_id", "event_date_parsed",
            "event_time_status", "terrain_source_tile",
        ],
        EVENTS_FILE.name,
    )

    require_columns(
        matches,
        [
            "matched_event_id", "control_id",
            "background_sample_id", "terrain_source_tile",
        ],
        MATCH_FILE.name,
    )

    events["event_date_parsed"] = pd.to_datetime(
        events["event_date_parsed"], errors="coerce"
    )

    if events["event_date_parsed"].isna().any():
        raise ValueError("Some original event timestamps cannot be parsed.")

    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in historical event list.")

    if matches["matched_event_id"].duplicated().any():
        raise ValueError("An event has more than one matched control.")

    event_hourly = load_hourly(EVENT_HOURLY_FILE, "event_id")
    control_hourly = load_hourly(CONTROL_HOURLY_FILE, "control_id")

    event_ids = set(events["event_id"].astype(str))
    matched_event_ids = set(matches["matched_event_id"].astype(str))

    if event_ids != matched_event_ids:
        raise ValueError(
            "Historical event IDs and matched-control event IDs differ."
        )

    event_hourly["event_id"] = event_hourly["event_id"].astype(str)
    control_hourly["matched_event_id"] = (
        control_hourly["matched_event_id"].astype(str)
    )
    matches["matched_event_id"] = matches["matched_event_id"].astype(str)

    event_lookup = {
        event_id: group.copy()
        for event_id, group in event_hourly.groupby("event_id")
    }
    control_lookup = {
        control_id: group.copy()
        for control_id, group in control_hourly.groupby("control_id")
    }

    rows = []

    for _, event in events.iterrows():
        event_id = str(event["event_id"])

        match = matches[
            matches["matched_event_id"] == event_id
        ]

        if len(match) != 1:
            raise ValueError(
                f"Expected exactly one control for event {event_id}."
            )

        match = match.iloc[0]
        control_id = match["control_id"]

        if event_id not in event_lookup:
            raise ValueError(f"No hourly rainfall for event {event_id}.")

        if control_id not in control_lookup:
            raise ValueError(f"No hourly rainfall for control {control_id}.")

        event_dt = event["event_date_parsed"]
        time_status = event["event_time_status"]

        event_metrics = calculate_metrics(
            event_lookup[event_id], event_dt, time_status
        )
        control_metrics = calculate_metrics(
            control_lookup[control_id], event_dt, time_status
        )

        record = {
            "event_id": event_id,
            "control_id": control_id,
            "background_sample_id": match["background_sample_id"],
            "terrain_source_tile": event["terrain_source_tile"],
            "event_datetime": event_dt.isoformat(sep=" "),
            "event_time_status": time_status,
            "control_status": "background_unverified",
        }

        for metric in [
            "pre_event_24h_mm",
            "pre_event_72h_mm",
            "event_day_precipitation_mm",
        ]:
            event_value = event_metrics[metric]
            control_value = control_metrics[metric]

            record[f"event_{metric}"] = event_value
            record[f"control_{metric}"] = control_value

            record[f"event_minus_control_{metric}"] = (
                round(event_value - control_value, 3)
                if event_value is not None and control_value is not None
                else None
            )

        record["event_pre_event_72h_valid_hours"] = (
            event_metrics["pre_event_72h_valid_hours"]
        )
        record["control_pre_event_72h_valid_hours"] = (
            control_metrics["pre_event_72h_valid_hours"]
        )
        record["event_pre_event_24h_valid_hours"] = (
            event_metrics["pre_event_24h_valid_hours"]
        )
        record["control_pre_event_24h_valid_hours"] = (
            control_metrics["pre_event_24h_valid_hours"]
        )

        rows.append(record)

    paired = pd.DataFrame(rows)

    metrics = [
        "pre_event_24h_mm",
        "pre_event_72h_mm",
        "event_day_precipitation_mm",
    ]

    summary_rows = []

    for tile in sorted(paired["terrain_source_tile"].unique()):
        tile_df = paired[
            paired["terrain_source_tile"] == tile
        ]

        for metric in metrics:
            event_col = f"event_{metric}"
            control_col = f"control_{metric}"
            diff_col = f"event_minus_control_{metric}"

            valid = tile_df[
                [event_col, control_col, diff_col]
            ].dropna()

            summary_rows.append({
                "terrain_source_tile": tile,
                "rainfall_metric": metric,
                "matched_pairs_total": len(tile_df),
                "complete_pairs": len(valid),
                "event_mean_mm": (
                    round(valid[event_col].mean(), 3)
                    if len(valid) else None
                ),
                "control_mean_mm": (
                    round(valid[control_col].mean(), 3)
                    if len(valid) else None
                ),
                "mean_paired_difference_mm": (
                    round(valid[diff_col].mean(), 3)
                    if len(valid) else None
                ),
                "median_paired_difference_mm": (
                    round(valid[diff_col].median(), 3)
                    if len(valid) else None
                ),
                "pairs_event_higher": (
                    int((valid[diff_col] > 0).sum())
                    if len(valid) else 0
                ),
                "pairs_equal": (
                    int((valid[diff_col] == 0).sum())
                    if len(valid) else 0
                ),
                "pairs_control_higher": (
                    int((valid[diff_col] < 0).sum())
                    if len(valid) else 0
                ),
            })

    # Also produce overall summaries.
    for metric in metrics:
        event_col = f"event_{metric}"
        control_col = f"control_{metric}"
        diff_col = f"event_minus_control_{metric}"

        valid = paired[
            [event_col, control_col, diff_col]
        ].dropna()

        summary_rows.append({
            "terrain_source_tile": "ALL_TILES",
            "rainfall_metric": metric,
            "matched_pairs_total": len(paired),
            "complete_pairs": len(valid),
            "event_mean_mm": (
                round(valid[event_col].mean(), 3)
                if len(valid) else None
            ),
            "control_mean_mm": (
                round(valid[control_col].mean(), 3)
                if len(valid) else None
            ),
            "mean_paired_difference_mm": (
                round(valid[diff_col].mean(), 3)
                if len(valid) else None
            ),
            "median_paired_difference_mm": (
                round(valid[diff_col].median(), 3)
                if len(valid) else None
            ),
            "pairs_event_higher": (
                int((valid[diff_col] > 0).sum())
                if len(valid) else 0
            ),
            "pairs_equal": (
                int((valid[diff_col] == 0).sum())
                if len(valid) else 0
            ),
            "pairs_control_higher": (
                int((valid[diff_col] < 0).sum())
                if len(valid) else 0
            ),
        })

    summary = pd.DataFrame(summary_rows)

    # Save only after all calculations succeed.
    paired.to_csv(PAIRS_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("=" * 70)
    print("PAIRED HISTORICAL RAINFALL COMPARISON")
    print("=" * 70)
    print("Matched event/control pairs:", len(paired))
    print("Event hourly rows:", len(event_hourly))
    print("Control hourly rows:", len(control_hourly))

    print("\nOverall comparison:")
    print(
        summary[summary["terrain_source_tile"] == "ALL_TILES"]
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(PAIRS_OUTPUT)
    print(SUMMARY_OUTPUT)

    print(
        "\nInterpretation: exploratory paired comparison only. "
        "Controls are unverified background samples, and rainfall "
        "is reanalysis data. Differences do not establish causality "
        "or a calibrated warning threshold."
    )


if __name__ == "__main__":
    main()
