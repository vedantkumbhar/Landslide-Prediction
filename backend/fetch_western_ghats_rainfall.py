from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "hourly_rainfall.csv"
)

LOCATIONS = {
    "N09E076": [
        (9.45, 76.80),
        (9.65, 76.75),
        (9.80, 76.90),
        (9.90, 76.55),
        (9.55, 76.40),
    ],
    "N11E076": [
        (11.30, 76.65),
        (11.40, 76.85),
        (11.50, 76.55),
        (11.60, 76.75),
        (11.70, 76.40),
    ],
    "N19E072": [
        (19.10, 72.90),
        (19.25, 72.95),
        (19.40, 72.85),
        (19.55, 72.65),
        (19.65, 72.90),
    ],
}

API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
PAST_DAYS = 2
FORECAST_DAYS = 3


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT}"
        )

    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    records = []

    with requests.Session() as session:
        for tile, points in LOCATIONS.items():
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
                values = hourly.get("precipitation", [])

                if not times or len(times) != len(values):
                    raise RuntimeError(
                        f"Invalid hourly response for {tile}: "
                        f"({latitude}, {longitude})"
                    )

                print(
                    f"{tile}: requested ({latitude}, {longitude}), "
                    f"API grid ({data['latitude']}, {data['longitude']}), "
                    f"{len(times)} hourly records"
                )

                for timestamp, rainfall in zip(times, values):
                    observation_time = datetime.fromisoformat(timestamp)
                    records.append({
                        "pilot_tile": tile,
                        "requested_latitude": latitude,
                        "requested_longitude": longitude,
                        "api_latitude": data["latitude"],
                        "api_longitude": data["longitude"],
                        "timestamp_local": timestamp,
                        "precipitation_mm": (
                            float(rainfall)
                            if rainfall is not None else None
                        ),
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

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No rainfall records were returned.")

    keys = [
        "pilot_tile",
        "requested_latitude",
        "requested_longitude",
        "timestamp_local",
    ]

    if df.duplicated(keys).any():
        raise RuntimeError("Duplicate tile-point timestamps detected.")

    numeric_rain = pd.to_numeric(
        df["precipitation_mm"], errors="coerce"
    )
    if (numeric_rain.dropna() < 0).any():
        raise RuntimeError("Negative rainfall values detected.")

    if numeric_rain.notna().sum() == 0:
        raise RuntimeError("All rainfall values are missing.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")

    try:
        df.to_csv(temp, index=False)
        check = pd.read_csv(temp)

        if len(check) != len(df):
            raise RuntimeError("CSV validation failed: row count mismatch.")

        if check["timestamp_local"].isna().any():
            raise RuntimeError("CSV validation failed: missing timestamps.")

        # Exclusive creation prevents replacing an existing final file.
        with temp.open("rb") as source, OUTPUT.open("xb") as destination:
            destination.write(source.read())
    finally:
        if temp.exists():
            temp.unlink()

    print("\nWestern Ghats rainfall download completed.")
    print("Rows:", len(df))
    print("Monitoring points:", df[
        ["pilot_tile", "requested_latitude", "requested_longitude"]
    ].drop_duplicates().shape[0])
    print("Missing rainfall values:", int(numeric_rain.isna().sum()))
    print("Output:", OUTPUT)
    print("\nRecords by tile and type:")
    print(df.groupby(["pilot_tile", "record_type"]).size().to_string())


if __name__ == "__main__":
    main()
