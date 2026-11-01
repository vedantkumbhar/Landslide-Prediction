
from pathlib import Path
import time
import requests
import pandas as pd

# ---------------------------------------------------------
# Western Ghats: historical rainfall around pilot events
# Source: Open-Meteo Historical Weather API, ERA5
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation" / "pilot_historical_events.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation"
)

HOURLY_OUTPUT = OUTPUT_DIR / "pilot_historical_hourly_rainfall.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "pilot_historical_rainfall_summary.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Asia/Kolkata"
MODEL = "era5"

MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 1.2


def require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def retrieve_event(event):
    event_date = pd.to_datetime(event["event_date_parsed"], errors="coerce")

    if pd.isna(event_date):
        raise ValueError(
            f"Event {event['event_id']} has an invalid event date."
        )

    latitude = float(event["latitude"])
    longitude = float(event["longitude"])

    # Include at least 72 hours before the event date, the event day,
    # and one additional day for checking date/time boundaries.
    start_date = (event_date - pd.Timedelta(days=4)).date()
    end_date = (event_date + pd.Timedelta(days=1)).date()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "precipitation",
        "timezone": TIMEZONE,
        "models": MODEL,
        "precipitation_unit": "mm",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()

            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "API error"))

            hourly = payload.get("hourly", {})
            times = hourly.get("time", [])
            rainfall = hourly.get("precipitation", [])

            if not times or len(times) != len(rainfall):
                raise ValueError(
                    "API response has missing or mismatched hourly arrays."
                )

            actual_lat = payload.get("latitude")
            actual_lon = payload.get("longitude")
            rows = []

            for timestamp, rain_mm in zip(times, rainfall):
                rows.append({
                    "event_id": event["event_id"],
                    "terrain_source_tile": event["terrain_source_tile"],
                    "event_date": event_date.date().isoformat(),
                    "event_time_status": event["event_time_status"],
                    "event_latitude": latitude,
                    "event_longitude": longitude,
                    "weather_grid_latitude": actual_lat,
                    "weather_grid_longitude": actual_lon,
                    "timestamp_local": timestamp,
                    "precipitation_mm": safe_float(rain_mm),
                    "weather_model": MODEL,
                    "weather_source": "Open-Meteo Historical Weather API",
                    "data_type": "historical_reanalysis",
                })

            return rows, {
                "event_id": event["event_id"],
                "terrain_source_tile": event["terrain_source_tile"],
                "event_date": event_date.date().isoformat(),
                "event_time_status": event["event_time_status"],
                "latitude": latitude,
                "longitude": longitude,
                "weather_grid_latitude": actual_lat,
                "weather_grid_longitude": actual_lon,
                "request_start_date": start_date.isoformat(),
                "request_end_date": end_date.isoformat(),
                "hourly_records": len(rows),
                "valid_rainfall_hours": sum(
                    row["precipitation_mm"] is not None for row in rows
                ),
                "status": "success",
                "error": "",
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** attempt
                print(
                    f"  Retry {attempt}/{MAX_RETRIES - 1} for "
                    f"{event['event_id']} in {wait_seconds}s: {last_error}"
                )
                time.sleep(wait_seconds)

    return [], {
        "event_id": event["event_id"],
        "terrain_source_tile": event["terrain_source_tile"],
        "event_date": event_date.date().isoformat(),
        "event_time_status": event["event_time_status"],
        "latitude": latitude,
        "longitude": longitude,
        "weather_grid_latitude": None,
        "weather_grid_longitude": None,
        "request_start_date": start_date.isoformat(),
        "request_end_date": end_date.isoformat(),
        "hourly_records": 0,
        "valid_rainfall_hours": 0,
        "status": "failed",
        "error": last_error or "Unknown error",
    }


def calculate_event_summary(hourly_df, event):
    event_date = pd.to_datetime(event["event_date_parsed"], errors="coerce")
    event_id = event["event_id"]
    status = event["event_time_status"]

    result = {
        "event_id": event_id,
        "terrain_source_tile": event["terrain_source_tile"],
        "event_date": event_date.date().isoformat(),
        "event_time_status": status,
        "rainfall_model": MODEL,
        "pre_event_24h_mm": None,
        "pre_event_72h_mm": None,
        "event_day_precipitation_mm": None,
        "pre_event_72h_valid_hours": 0,
        "event_day_valid_hours": 0,
        "interpretation_note": "",
    }

    rows = hourly_df[hourly_df["event_id"] == event_id].copy()
    if rows.empty:
        result["interpretation_note"] = "No rainfall data retrieved."
        return result

    rows["timestamp_local"] = pd.to_datetime(
        rows["timestamp_local"], errors="coerce"
    )
    rows["precipitation_mm"] = pd.to_numeric(
        rows["precipitation_mm"], errors="coerce"
    )

    # Hourly timestamps are local time; precipitation is the preceding-hour sum.
    day_start = event_date.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    event_day = rows[
        (rows["timestamp_local"] >= day_start)
        & (rows["timestamp_local"] < day_end)
    ]
    valid_event_day = event_day["precipitation_mm"].dropna()

    if len(valid_event_day) == 24:
        result["event_day_precipitation_mm"] = round(
            float(valid_event_day.sum()), 3
        )
    result["event_day_valid_hours"] = len(valid_event_day)

    if status == "time_present_in_event_date":
        # Use the recorded event time only when the source marked it as present.
        event_time = pd.to_datetime(event["event_date_parsed"], errors="coerce")
        start_24 = event_time - pd.Timedelta(hours=24)
        start_72 = event_time - pd.Timedelta(hours=72)

        preceding_24 = rows[
            (rows["timestamp_local"] > start_24)
            & (rows["timestamp_local"] <= event_time)
        ]["precipitation_mm"].dropna()

        preceding_72 = rows[
            (rows["timestamp_local"] > start_72)
            & (rows["timestamp_local"] <= event_time)
        ]["precipitation_mm"].dropna()

        if len(preceding_24) == 24:
            result["pre_event_24h_mm"] = round(
                float(preceding_24.sum()), 3
            )

        if len(preceding_72) == 72:
            result["pre_event_72h_mm"] = round(
                float(preceding_72.sum()), 3
            )

        result["pre_event_72h_valid_hours"] = len(preceding_72)
        result["interpretation_note"] = (
            "Recorded non-midnight time used; verify original report "
            "before treating timestamp as exact."
        )

    else:
        # Midnight is ambiguous: don't claim an exact 24/72-hour window.
        # Summarize the three complete calendar days before the event date.
        start_3_days = day_start - pd.Timedelta(days=3)
        preceding_days = rows[
            (rows["timestamp_local"] >= start_3_days)
            & (rows["timestamp_local"] < day_start)
        ]
        valid = preceding_days["precipitation_mm"].dropna()

        if len(valid) == 72:
            result["pre_event_72h_mm"] = round(float(valid.sum()), 3)

        result["pre_event_72h_valid_hours"] = len(valid)
        result["interpretation_note"] = (
            "Midnight timestamp is ambiguous. 72-hour value, if available, "
            "covers the three calendar days before the recorded event date; "
            "24-hour value intentionally omitted."
        )

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Protect existing results. Never overwrite.
    existing = [p for p in (HOURLY_OUTPUT, SUMMARY_OUTPUT) if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output file(s):\n"
            + "\n".join(str(p) for p in existing)
            + "\nRename/move the old outputs or choose new filenames."
        )

    try:
        import requests  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "requests is required. Install it with: pip install requests"
        ) from exc

    events = pd.read_csv(INPUT_FILE)
    require_columns(
        events,
        [
            "event_id",
            "terrain_source_tile",
            "event_date_parsed",
            "event_time_status",
            "latitude",
            "longitude",
        ],
    )

    events["event_date_parsed"] = pd.to_datetime(
        events["event_date_parsed"], errors="coerce"
    )
    events = events.dropna(
        subset=["event_date_parsed", "latitude", "longitude"]
    ).copy()

    if events["event_id"].duplicated().any():
        duplicates = events.loc[
            events["event_id"].duplicated(), "event_id"
        ].tolist()
        raise ValueError(f"Duplicate event IDs found: {duplicates[:10]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Western Ghats historical rainfall retrieval")
    print(f"Input events: {len(events)}")
    print(f"Model: {MODEL}")
    print(f"Timezone: {TIMEZONE}")
    print("Each event requests a short window around its recorded date.")
    print("This is reanalysis data, not exact local rain-gauge measurement.")
    print()

    all_hourly = []
    all_status = []

    for index, (_, event) in enumerate(events.iterrows(), start=1):
        print(
            f"[{index}/{len(events)}] Event {event['event_id']} | "
            f"{event['terrain_source_tile']} | "
            f"{event['event_date_parsed'].date()}"
        )

        hourly_rows, status_row = retrieve_event(event)
        all_hourly.extend(hourly_rows)
        all_status.append(status_row)

        if status_row["status"] == "success":
            print(
                f"  Retrieved {status_row['hourly_records']} hourly records."
            )
        else:
            print(f"  FAILED: {status_row['error']}")

        time.sleep(REQUEST_DELAY_SECONDS)

    hourly_df = pd.DataFrame(all_hourly)
    status_df = pd.DataFrame(all_status)

    if not hourly_df.empty:
        hourly_df = hourly_df.sort_values(
            ["event_id", "timestamp_local"]
        ).reset_index(drop=True)

    # Create a summary for every input event, including API failures.
    summaries = []
    for _, event in events.iterrows():
        if status_df.loc[
            status_df["event_id"] == event["event_id"], "status"
        ].eq("success").any():
            summaries.append(calculate_event_summary(hourly_df, event))
        else:
            summaries.append({
                "event_id": event["event_id"],
                "terrain_source_tile": event["terrain_source_tile"],
                "event_date": event["event_date_parsed"].date().isoformat(),
                "event_time_status": event["event_time_status"],
                "rainfall_model": MODEL,
                "pre_event_24h_mm": None,
                "pre_event_72h_mm": None,
                "event_day_precipitation_mm": None,
                "pre_event_72h_valid_hours": 0,
                "event_day_valid_hours": 0,
                "interpretation_note": "API retrieval failed; see status report.",
            })

    # Save to new files only.
    hourly_df.to_csv(HOURLY_OUTPUT, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_OUTPUT, index=False)

    print("\nRetrieval complete.")
    print(f"Events attempted: {len(events)}")
    print(
        "Successful events:",
        int((status_df["status"] == "success").sum()),
    )
    print(
        "Failed events:",
        int((status_df["status"] != "success").sum()),
    )
    print(f"Hourly rows: {len(hourly_df)}")
    print(f"Hourly rainfall file: {HOURLY_OUTPUT}")
    print(f"Event summary file: {SUMMARY_OUTPUT}")

    failed = status_df[status_df["status"] != "success"]
    if not failed.empty:
        print("\nFailed event IDs:")
        print(failed[["event_id", "error"]].to_string(index=False))

    print(
        "\nImportant: rainfall totals are reanalysis estimates. "
        "They do not prove that rainfall caused a landslide, and the "
        "historical event list is too small to establish a reliable "
        "warning threshold by itself."
    )


if __name__ == "__main__":
    main()
