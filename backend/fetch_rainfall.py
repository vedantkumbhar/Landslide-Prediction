
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "pilot_hourly_rainfall.csv"
)

LOCATIONS = {
    "Himalayas": [
        (30.77, 78.80),
        (30.50, 78.20),
        (31.00, 79.00),
        (29.90, 79.50),
        (32.00, 76.50),
    ],
    "Northeast_India": [
        (27.50, 88.50),
        (26.20, 92.00),
        (25.60, 91.90),
        (27.00, 94.00),
        (24.80, 93.90),
    ],
    "Western_Maharashtra": [
        (16.70, 73.70),
        (17.00, 73.80),
        (17.50, 73.90),
        (18.00, 73.70),
        (16.50, 74.00),
    ],
}

API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
PAST_DAYS = 1
FORECAST_DAYS = 3


def fetch_rainfall():
    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    records = []
    session = requests.Session()

    try:
        for region, points in LOCATIONS.items():
            for latitude, longitude in points:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": "precipitation",
                    "past_days": PAST_DAYS,
                    "forecast_days": FORECAST_DAYS,
                    "timezone": TIMEZONE,
                }

                response = session.get(
                    API_URL, params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()
                hourly = data.get("hourly", {})

                times = hourly.get("time", [])
                rainfall_values = hourly.get("precipitation", [])

                if not times or len(times) != len(rainfall_values):
                    raise RuntimeError(
                        f"Invalid hourly response for "
                        f"{region} ({latitude}, {longitude})"
                    )

                for timestamp, rainfall in zip(times, rainfall_values):
                    observation_time = datetime.fromisoformat(timestamp)

                    # A missing rainfall value must remain missing.
                    value = (
                        float(rainfall)
                        if rainfall is not None
                        else None
                    )

                    records.append({
                        "region": region,
                        "requested_latitude": latitude,
                        "requested_longitude": longitude,
                        "api_latitude": data["latitude"],
                        "api_longitude": data["longitude"],
                        "timestamp_local": timestamp,
                        "precipitation_mm": value,
                        "record_type": (
                            "historical_or_current"
                            if observation_time <= now
                            else "forecast"
                        ),
                        "source": "Open-Meteo",
                        "retrieved_at_local": now.isoformat(
                            timespec="seconds"
                        ),
                    })

                future_count = sum(
                    datetime.fromisoformat(t) > now for t in times
                )
                print(
                    f"{region}: ({latitude}, {longitude}) "
                    f"received {len(times)} records, "
                    f"{future_count} forecast hours"
                )

    finally:
        session.close()

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No rainfall records were received.")

    if df.duplicated([
        "region",
        "requested_latitude",
        "requested_longitude",
        "timestamp_local",
    ]).any():
        raise RuntimeError("Duplicate point-timestamp records detected.")

    if (
        df["precipitation_mm"].dropna() < 0
    ).any():
        raise RuntimeError("Negative rainfall values detected.")

    # Preserve the previous CSV unless the new dataset validates.
    if df["precipitation_mm"].isna().all():
        raise RuntimeError("All rainfall values are missing.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT.with_suffix(".tmp")

    try:
        df.to_csv(temporary_output, index=False)
        check = pd.read_csv(temporary_output)

        if len(check) != len(df):
            raise RuntimeError("Temporary CSV row count mismatch.")

        if check["timestamp_local"].isna().any():
            raise RuntimeError("Temporary CSV contains invalid timestamps.")

        temporary_output.replace(OUTPUT)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    print(f"\nTotal records: {len(df)}")
    print(
        "Unique sample points:",
        df[[
            "requested_latitude",
            "requested_longitude",
        ]].drop_duplicates().shape[0],
    )
    print(f"Saved to: {OUTPUT}")
    print("\nRecords by region and type:")
    print(df.groupby(["region", "record_type"]).size().to_string())


if __name__ == "__main__":
    fetch_rainfall()
