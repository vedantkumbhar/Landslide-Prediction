
from pathlib import Path
import time

import pandas as pd
import requests

# ============================================================
# Historical rainfall for Western Ghats background controls
# Uses the same API, model, timezone and windows as the events.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation"
)

INPUT_FILE = BASE_DIR / "matched_rainfall_controls.csv"

HOURLY_OUTPUT = BASE_DIR / "control_historical_hourly_rainfall.csv"
SUMMARY_OUTPUT = BASE_DIR / "control_historical_rainfall_summary.csv"
STATUS_OUTPUT = BASE_DIR / "control_historical_rainfall_download_status.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"
TIMEZONE = "Asia/Kolkata"
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 1.2


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def fetch_control(control):
    event_date = pd.to_datetime(
        control["matched_event_date"], errors="coerce"
    )

    if pd.isna(event_date):
        raise ValueError(
            f"Invalid matched event date for {control['control_id']}"
        )

    latitude = float(control["control_latitude"])
    longitude = float(control["control_longitude"])

    # Same date window as the historical landslide-event script:
    # event date - 4 days through event date + 1 day, inclusive.
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
                raise RuntimeError(
                    payload.get("reason", "Historical API error")
                )

            hourly = payload.get("hourly", {})
            timestamps = hourly.get("time", [])
            rainfall = hourly.get("precipitation", [])

            if not timestamps or len(timestamps) != len(rainfall):
                raise ValueError(
                    "Missing or mismatched hourly rainfall arrays."
                )

            rows = []

            for timestamp, rain_mm in zip(timestamps, rainfall):
                rows.append({
                    "control_id": control["control_id"],
                    "matched_event_id": control["matched_event_id"],
                    "background_sample_id": control["background_sample_id"],
                    "terrain_source_tile": control["terrain_source_tile"],
                    "matched_event_date": event_date.date().isoformat(),
                    "event_time_status": control["event_time_status"],
                    "control_latitude": latitude,
                    "control_longitude": longitude,
                    "weather_grid_latitude": payload.get("latitude"),
                    "weather_grid_longitude": payload.get("longitude"),
                    "timestamp_local": timestamp,
                    "precipitation_mm": (
                        float(rain_mm)
                        if rain_mm is not None
                        else None
                    ),
                    "weather_model": MODEL,
                    "weather_source": "Open-Meteo Historical Weather API",
                    "data_type": "historical_reanalysis_control",
                    "control_status": "background_unverified",
                })

            status = {
                "control_id": control["control_id"],
                "matched_event_id": control["matched_event_id"],
                "terrain_source_tile": control["terrain_source_tile"],
                "matched_event_date": event_date.date().isoformat(),
                "request_start_date": start_date.isoformat(),
                "request_end_date": end_date.isoformat(),
                "hourly_records": len(rows),
                "valid_rainfall_hours": sum(
                    r["precipitation_mm"] is not None for r in rows
                ),
                "status": "success",
                "error": "",
            }

            return rows, status

        except Exception as exc:
            last_error = str(exc)

            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(
                    f"  Retry {attempt}/{MAX_RETRIES - 1} for "
                    f"{control['control_id']} in {wait}s: {last_error}"
                )
                time.sleep(wait)

    return [], {
        "control_id": control["control_id"],
        "matched_event_id": control["matched_event_id"],
        "terrain_source_tile": control["terrain_source_tile"],
        "matched_event_date": event_date.date().isoformat(),
        "request_start_date": start_date.isoformat(),
        "request_end_date": end_date.isoformat(),
        "hourly_records": 0,
        "valid_rainfall_hours": 0,
        "status": "failed",
        "error": last_error or "Unknown error",
    }


def summarize_control(hourly_df, control):
    event_date = pd.to_datetime(control["matched_event_date"])
    control_id = control["control_id"]
    time_status = control["event_time_status"]

    result = {
        "control_id": control_id,
        "matched_event_id": control["matched_event_id"],
        "background_sample_id": control["background_sample_id"],
        "terrain_source_tile": control["terrain_source_tile"],
        "matched_event_date": event_date.date().isoformat(),
        "event_time_status": time_status,
        "control_status": "background_unverified",
        "rainfall_model": MODEL,
        "pre_event_24h_mm": None,
        "pre_event_72h_mm": None,
        "event_day_precipitation_mm": None,
        "pre_event_24h_valid_hours": 0,
        "pre_event_72h_valid_hours": 0,
        "event_day_valid_hours": 0,
        "interpretation_note": "",
    }

    rows = hourly_df[
        hourly_df["control_id"] == control_id
    ].copy()

    if rows.empty:
        result["interpretation_note"] = "No rainfall data retrieved."
        return result

    rows["timestamp_local"] = pd.to_datetime(
        rows["timestamp_local"], errors="coerce"
    )
    rows["precipitation_mm"] = pd.to_numeric(
        rows["precipitation_mm"], errors="coerce"
    )

    day_start = event_date.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    event_day = rows[
        (rows["timestamp_local"] >= day_start)
        & (rows["timestamp_local"] < day_end)
    ]
    event_day_values = event_day["precipitation_mm"].dropna()

    result["event_day_valid_hours"] = len(event_day_values)

    if len(event_day_values) == 24:
        result["event_day_precipitation_mm"] = round(
            float(event_day_values.sum()), 3
        )

    if time_status == "time_present_in_event_date":
        # Use the same recorded timestamp window as its matched event.
        event_time = event_date
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

        result["pre_event_24h_valid_hours"] = len(preceding_24)
        result["pre_event_72h_valid_hours"] = len(preceding_72)

        if len(preceding_24) == 24:
            result["pre_event_24h_mm"] = round(
                float(preceding_24.sum()), 3
            )

        if len(preceding_72) == 72:
            result["pre_event_72h_mm"] = round(
                float(preceding_72.sum()), 3
            )

        result["interpretation_note"] = (
            "Control rainfall calculated over the same recorded "
            "event-time windows as the matched event."
        )

    else:
        # Midnight is ambiguous: use three full calendar days before
        # the matched event date for both event and control comparisons.
        start_72 = day_start - pd.Timedelta(days=3)

        preceding_72 = rows[
            (rows["timestamp_local"] >= start_72)
            & (rows["timestamp_local"] < day_start)
        ]["precipitation_mm"].dropna()

        result["pre_event_72h_valid_hours"] = len(preceding_72)

        if len(preceding_72) == 72:
            result["pre_event_72h_mm"] = round(
                float(preceding_72.sum()), 3
            )

        result["interpretation_note"] = (
            "Midnight time is ambiguous. The 72-hour total uses the "
            "three calendar days before the matched event date; "
            "the 24-hour total is omitted."
        )

    return result


def main():
    for path in (INPUT_FILE, HOURLY_OUTPUT, SUMMARY_OUTPUT, STATUS_OUTPUT):
        if path == INPUT_FILE:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
        elif path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    controls = pd.read_csv(INPUT_FILE)

    require_columns(
        controls,
        [
            "control_id",
            "matched_event_id",
            "background_sample_id",
            "terrain_source_tile",
            "matched_event_date",
            "event_time_status",
            "control_latitude",
            "control_longitude",
            "control_status",
        ],
    )

    if controls["control_id"].duplicated().any():
        raise ValueError("Duplicate control IDs in input.")

    if controls["matched_event_id"].duplicated().any():
        raise ValueError("A historical event has multiple controls.")

    if len(controls) != 56:
        print(
            f"WARNING: expected 56 controls, but input contains "
            f"{len(controls)}."
        )

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("WESTERN GHATS CONTROL HISTORICAL RAINFALL RETRIEVAL")
    print("=" * 65)
    print("Controls:", len(controls))
    print("Model:", MODEL)
    print("Timezone:", TIMEZONE)
    print("Each control uses its matched event's date and time status.")
    print()

    all_hourly = []
    all_status = []

    for index, (_, control) in enumerate(controls.iterrows(), start=1):
        print(
            f"[{index}/{len(controls)}] {control['control_id']} | "
            f"Event {control['matched_event_id']} | "
            f"{control['terrain_source_tile']} | "
            f"{control['matched_event_date']}"
        )

        rows, status = fetch_control(control)
        all_hourly.extend(rows)
        all_status.append(status)

        if status["status"] == "success":
            print(f"  Retrieved {status['hourly_records']} hourly records.")
        else:
            print(f"  FAILED: {status['error']}")

        time.sleep(REQUEST_DELAY_SECONDS)

    hourly_df = pd.DataFrame(all_hourly)
    status_df = pd.DataFrame(all_status)

    if not hourly_df.empty:
        hourly_df = hourly_df.sort_values(
            ["control_id", "timestamp_local"]
        ).reset_index(drop=True)

    successful_ids = set(
        status_df.loc[
            status_df["status"] == "success", "control_id"
        ]
    )

    summaries = []

    for _, control in controls.iterrows():
        if control["control_id"] in successful_ids:
            summaries.append(summarize_control(hourly_df, control))
        else:
            summaries.append({
                "control_id": control["control_id"],
                "matched_event_id": control["matched_event_id"],
                "background_sample_id": control["background_sample_id"],
                "terrain_source_tile": control["terrain_source_tile"],
                "matched_event_date": control["matched_event_date"],
                "event_time_status": control["event_time_status"],
                "control_status": "background_unverified",
                "rainfall_model": MODEL,
                "pre_event_24h_mm": None,
                "pre_event_72h_mm": None,
                "event_day_precipitation_mm": None,
                "pre_event_24h_valid_hours": 0,
                "pre_event_72h_valid_hours": 0,
                "event_day_valid_hours": 0,
                "interpretation_note": (
                    "Rainfall retrieval failed; see download status."
                ),
            })

    summary_df = pd.DataFrame(summaries)

    # Create new outputs only.
    hourly_df.to_csv(HOURLY_OUTPUT, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    status_df.to_csv(STATUS_OUTPUT, index=False)

    print("\n" + "=" * 65)
    print("CONTROL RAINFALL RETRIEVAL COMPLETE")
    print("=" * 65)
    print("Controls attempted:", len(controls))
    print(
        "Successful:",
        int((status_df["status"] == "success").sum()),
    )
    print(
        "Failed:",
        int((status_df["status"] != "success").sum()),
    )
    print("Hourly rows:", len(hourly_df))
    print("Hourly rainfall:", HOURLY_OUTPUT)
    print("Control summary:", SUMMARY_OUTPUT)
    print("Download status:", STATUS_OUTPUT)

    failed = status_df[status_df["status"] != "success"]
    if not failed.empty:
        print("\nFailed controls:")
        print(
            failed[["control_id", "matched_event_id", "error"]]
            .to_string(index=False)
        )

    print(
        "\nReminder: background samples are unverified controls, not "
        "confirmed landslide-free locations. Rainfall is ERA5 "
        "reanalysis, not exact rain-gauge measurement."
    )


if __name__ == "__main__":
    main()
