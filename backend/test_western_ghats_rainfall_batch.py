
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)
API_URL = "https://api.open-meteo.com/v1/forecast"


def main():
    df = pd.read_csv(TRAINING_PATH)
    samples = df[["pilot_tile", "latitude", "longitude"]].head(5)

    params = {
        "latitude": ",".join(
            samples["latitude"].map(lambda x: f"{x:.6f}")
        ),
        "longitude": ",".join(
            samples["longitude"].map(lambda x: f"{x:.6f}")
        ),
        "hourly": "precipitation",
        "past_days": 2,
        "forecast_days": 3,
        "timezone": "Asia/Kolkata",
    }

    print("Testing one request for", len(samples), "locations")

    with requests.Session() as session:
        response = session.get(API_URL, params=params, timeout=60)

    print("HTTP status:", response.status_code)
    response.raise_for_status()
    data = response.json()

    locations = data if isinstance(data, list) else [data]

    print("Locations returned:", len(locations))

    if len(locations) != len(samples):
        raise RuntimeError(
            "Returned location count does not match requested count."
        )

    for index, location in enumerate(locations):
        hourly = location.get("hourly", {})
        times = hourly.get("time", [])
        rainfall = hourly.get("precipitation", [])

        print(
            f"{samples.iloc[index]['pilot_tile']}: "
            f"API grid=({location.get('latitude')}, "
            f"{location.get('longitude')}), "
            f"records={len(times)}, rainfall_values={len(rainfall)}"
        )

        if not times or len(times) != len(rainfall):
            raise RuntimeError(
                f"Invalid hourly data for location {index + 1}"
            )

    print("\nPASS: all five locations returned valid hourly data.")
    print("No project files were modified.")


if __name__ == "__main__":
    main()
