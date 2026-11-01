from pathlib import Path
from datetime import timedelta
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "pilot_landslide_event_features.csv"
URL = "https://archive-api.open-meteo.com/v1/archive"

df = pd.read_csv(EVENTS)
df["event_date_parsed"] = pd.to_datetime(
    df["event_date_parsed"], errors="coerce"
)
df = df.dropna(
    subset=["event_date_parsed", "latitude", "longitude"]
)

# Test one event from each available pilot tile, plus a later event.
samples = []
for _, group in df.groupby("pilot_tile", dropna=True):
    samples.append(group.iloc[len(group) // 2])

samples = samples[:2]
if len(df):
    samples.append(df.iloc[-1])

print("HISTORICAL RAINFALL API TEST")
print("Testing up to", len(samples), "events")

for _, event in pd.DataFrame(samples).iterrows():
    event_date = event["event_date_parsed"].date()
    start_date = event_date - timedelta(days=1)

    params = {
        "latitude": float(event["latitude"]),
        "longitude": float(event["longitude"]),
        "start_date": start_date.isoformat(),
        "end_date": event_date.isoformat(),
        "hourly": "precipitation",
        "timezone": "Asia/Kolkata",
    }

    print("\nEvent:", event.get("event_id", "unknown"))
    print("Tile:", event.get("pilot_tile", "unknown"))
    print("Event date:", event_date)

    try:
        response = requests.get(URL, params=params, timeout=45)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        rain = hourly.get("precipitation", [])

        if not times or not rain:
            print("RESULT: No hourly rainfall returned.")
            print("Response keys:", list(data.keys()))
            continue

        result = pd.DataFrame({
            "time": pd.to_datetime(times),
            "rainfall_mm": rain,
        })
        result["day"] = result["time"].dt.date

        print("API coordinates:",
              data.get("latitude"), data.get("longitude"))
        print("Hourly records:", len(result))
        print("Rainfall by calendar day (mm):")
        print(result.groupby("day")["rainfall_mm"].sum().round(2).to_string())
        print("Missing hourly values:", result["rainfall_mm"].isna().sum())
        print("RESULT: SUCCESS")

    except Exception as exc:
        print("RESULT: FAILED")
        print(type(exc).__name__ + ":", str(exc))

print("\nTest complete. No existing datasets were modified.")
