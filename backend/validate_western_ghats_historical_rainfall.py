
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation"
)

EVENT_FILE = BASE_DIR / "pilot_historical_events.csv"
HOURLY_FILE = BASE_DIR / "pilot_historical_hourly_rainfall.csv"
SUMMARY_FILE = BASE_DIR / "pilot_historical_rainfall_summary.csv"

EXPECTED_EVENTS = 56
EXPECTED_HOURS_PER_EVENT = 144

def check(condition, message):
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)

def main():
    for path in (EVENT_FILE, HOURLY_FILE, SUMMARY_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    events = pd.read_csv(EVENT_FILE)
    hourly = pd.read_csv(HOURLY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    print("=" * 65)
    print("WESTERN GHATS HISTORICAL RAINFALL VALIDATION")
    print("=" * 65)

    checks = []

    checks.append(check(
        events["event_id"].nunique() == EXPECTED_EVENTS,
        f"Input has {events['event_id'].nunique()} unique events"
    ))

    checks.append(check(
        len(hourly) == EXPECTED_EVENTS * EXPECTED_HOURS_PER_EVENT,
        f"Hourly row count is {len(hourly)}; expected "
        f"{EXPECTED_EVENTS * EXPECTED_HOURS_PER_EVENT}"
    ))

    checks.append(check(
        hourly["event_id"].nunique() == EXPECTED_EVENTS,
        f"Hourly data covers {hourly['event_id'].nunique()} events"
    ))

    hourly["timestamp_local"] = pd.to_datetime(
        hourly["timestamp_local"], errors="coerce"
    )
    hourly["precipitation_mm"] = pd.to_numeric(
        hourly["precipitation_mm"], errors="coerce"
    )

    duplicate_pairs = hourly.duplicated(
        subset=["event_id", "timestamp_local"]
    ).sum()

    checks.append(check(
        duplicate_pairs == 0,
        f"Duplicate event/timestamp rows: {duplicate_pairs}"
    ))

    invalid_timestamps = hourly["timestamp_local"].isna().sum()
    checks.append(check(
        invalid_timestamps == 0,
        f"Invalid timestamps: {invalid_timestamps}"
    ))

    missing_rain = hourly["precipitation_mm"].isna().sum()
    checks.append(check(
        missing_rain == 0,
        f"Missing hourly precipitation values: {missing_rain}"
    ))

    negative_rain = (
        hourly["precipitation_mm"].dropna() < 0
    ).sum()

    checks.append(check(
        negative_rain == 0,
        f"Negative hourly precipitation values: {negative_rain}"
    ))

    rows_per_event = hourly.groupby("event_id").size()
    wrong_row_counts = rows_per_event[
        rows_per_event != EXPECTED_HOURS_PER_EVENT
    ]

    checks.append(check(
        len(wrong_row_counts) == 0,
        f"Events with other than {EXPECTED_HOURS_PER_EVENT} rows: "
        f"{len(wrong_row_counts)}"
    ))

    # Ensure hourly event IDs match the source event inventory.
    source_ids = set(events["event_id"].astype(str))
    hourly_ids = set(hourly["event_id"].astype(str))
    missing_ids = source_ids - hourly_ids
    unexpected_ids = hourly_ids - source_ids

    checks.append(check(
        not missing_ids and not unexpected_ids,
        f"Event ID mismatch: missing={len(missing_ids)}, "
        f"unexpected={len(unexpected_ids)}"
    ))

    print("\n--- Rainfall time coverage ---")
    if hourly["timestamp_local"].notna().any():
        print("First timestamp:", hourly["timestamp_local"].min())
        print("Last timestamp: ", hourly["timestamp_local"].max())

    print("\n--- Historical event summary coverage ---")

    expected_summary_columns = [
        "event_id",
        "event_time_status",
        "pre_event_24h_mm",
        "pre_event_72h_mm",
        "event_day_precipitation_mm",
        "pre_event_72h_valid_hours",
        "event_day_valid_hours",
    ]

    missing_columns = [
        col for col in expected_summary_columns
        if col not in summary.columns
    ]

    checks.append(check(
        not missing_columns,
        f"Summary required columns present: "
        f"{'yes' if not missing_columns else 'no - ' + str(missing_columns)}"
    ))

    if not missing_columns:
        checks.append(check(
            summary["event_id"].nunique() == EXPECTED_EVENTS
            and len(summary) == EXPECTED_EVENTS,
            f"Summary has {len(summary)} rows and "
            f"{summary['event_id'].nunique()} unique events"
        ))

        print(
            "24-hour pre-event totals available:",
            summary["pre_event_24h_mm"].notna().sum(),
            "/", len(summary)
        )
        print(
            "72-hour pre-event totals available:",
            summary["pre_event_72h_mm"].notna().sum(),
            "/", len(summary)
        )
        print(
            "Event-day totals available:",
            summary["event_day_precipitation_mm"].notna().sum(),
            "/", len(summary)
        )

        print("\nEvent-time status counts:")
        print(summary["event_time_status"].value_counts(dropna=False))

        print("\nPre-event 72-hour valid-hour counts:")
        print(
            summary["pre_event_72h_valid_hours"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        print("\n--- Pre-event rainfall summary (mm) ---")
        for column in (
            "pre_event_24h_mm",
            "pre_event_72h_mm",
            "event_day_precipitation_mm",
        ):
            values = pd.to_numeric(summary[column], errors="coerce")
            print(f"\n{column}")
            if values.notna().any():
                print(values.describe().round(3).to_string())
            else:
                print("No complete values available.")

        print("\n--- Sample event summaries ---")
        display_columns = [
            "event_id",
            "terrain_source_tile",
            "event_date",
            "event_time_status",
            "pre_event_24h_mm",
            "pre_event_72h_mm",
            "event_day_precipitation_mm",
            "pre_event_72h_valid_hours",
        ]
        display_columns = [
            c for c in display_columns if c in summary.columns
        ]
        print(summary[display_columns].head(10).to_string(index=False))

    print("\n" + "=" * 65)
    if all(checks):
        print("RESULT: All structural and integrity checks passed.")
    else:
        print("RESULT: One or more checks failed. Investigate before analysis.")

    print(
        "\nNote: passing checks confirms data structure and integrity, "
        "not rainfall accuracy or predictive skill."
    )
    print("=" * 65)

if __name__ == "__main__":
    main()
