from pathlib import Path
import time
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EVENTS_FILE = ROOT / "data/processed/pilot_landslide_event_features.csv"
OUTPUT_FILE = ROOT / "data/processed/rainfall/pilot_historical_event_rainfall.csv"
CACHE_DIR = ROOT / "data/processed/rainfall/event_cache"
API_URL = "https://archive-api.open-meteo.com/v1/archive"

def main():
    events = pd.read_csv(EVENTS_FILE)
    events["event_date_parsed"] = pd.to_datetime(
        events["event_date_parsed"], errors="coerce"
    )

    events = events.dropna(
        subset=["event_date_parsed", "latitude", "longitude"]
    ).copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    results = []

    print(f"Events to process: {len(events)}")

    for index, event in events.iterrows():
        event_id = str(event["event_id"])
        event_date = event["event_date_parsed"].date()
        lat = round(float(event["latitude"]), 5)
        lon = round(float(event["longitude"]), 5)

        cache_file = CACHE_DIR / (
            f"{event_id}_{event_date}_{lat}_{lon}.csv"
        )

        try:
            if cache_file.exists():
                hourly = pd.read_csv(cache_file)
                source_status = "cached"
            else:
                response = session.get(
                    API_URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": (event_date - pd.Timedelta(days=1)).isoformat(),
                        "end_date": event_date.isoformat(),
                        "hourly": "precipitation",
                        "timezone": "Asia/Kolkata",
                    },
                    timeout=45,
                )
                response.raise_for_status()
                data = response.json()
                hourly_data = data.get("hourly", {})
                times = hourly_data.get("time", [])
                rainfall = hourly_data.get("precipitation", [])

                if not times or len(times) != len(rainfall):
                    raise ValueError("Missing or mismatched hourly data")

                hourly = pd.DataFrame({
                    "timestamp": times,
                    "precipitation_mm": rainfall,
                })
                hourly.to_csv(cache_file, index=False)
                source_status = "downloaded"
                time.sleep(0.15)

            hourly["timestamp"] = pd.to_datetime(
                hourly["timestamp"], errors="coerce"
            )
            hourly["precipitation_mm"] = pd.to_numeric(
                hourly["precipitation_mm"], errors="coerce"
            )

            # Daily totals are calculated using local calendar dates.
            hourly["calendar_date"] = hourly["timestamp"].dt.date
            previous_day = event_date - pd.Timedelta(days=1)

            previous = hourly.loc[
                hourly["calendar_date"] == previous_day,
                "precipitation_mm",
            ]
            event_day = hourly.loc[
                hourly["calendar_date"] == event_date,
                "precipitation_mm",
            ]

            results.append({
                "event_id": event_id,
                "event_date": event_date.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "pilot_tile": event.get("pilot_tile"),
                "previous_day_rainfall_mm": (
                    previous.sum(min_count=1) if not previous.empty else float("nan")
                ),
                "event_day_rainfall_mm": (
                    event_day.sum(min_count=1) if not event_day.empty else float("nan")
                ),
                "previous_day_hourly_records": int(previous.notna().sum()),
                "event_day_hourly_records": int(event_day.notna().sum()),
                "api_latitude": data.get("latitude") if source_status == "downloaded" else float("nan"),
                "api_longitude": data.get("longitude") if source_status == "downloaded" else float("nan"),
                "source": "Open-Meteo Historical Weather API",
                "status": source_status,
                "error": "",
            })

        except Exception as exc:
            results.append({
                "event_id": event_id,
                "event_date": event_date.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "pilot_tile": event.get("pilot_tile"),
                "previous_day_rainfall_mm": float("nan"),
                "event_day_rainfall_mm": float("nan"),
                "previous_day_hourly_records": 0,
                "event_day_hourly_records": 0,
                "api_latitude": float("nan"),
                "api_longitude": float("nan"),
                "source": "Open-Meteo Historical Weather API",
                "status": "failed",
                "error": str(exc)[:250],
            })

        if len(results) % 10 == 0 or len(results) == len(events):
            print(f"Processed {len(results)}/{len(events)}")

        # Save progress after every event so interrupted runs can be recovered.
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    output = pd.DataFrame(results)
    print("\nEXTRACTION SUMMARY")
    print("Output:", OUTPUT_FILE)
    print("Total:", len(output))
    print(output["status"].value_counts(dropna=False).to_string())
    print("\nMissing event-day rainfall:",
          int(output["event_day_rainfall_mm"].isna().sum()))
    print("\nRainfall summary (mm):")
    print(output[
        ["previous_day_rainfall_mm", "event_day_rainfall_mm"]
    ].describe().round(2).to_string())
    print("\nImportant: event-day rainfall is not necessarily pre-event rainfall.")

if __name__ == "__main__":
    main()
