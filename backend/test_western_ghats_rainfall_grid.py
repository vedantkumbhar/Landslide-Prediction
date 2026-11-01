
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

TRAINING_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

API_URL = "https://api.open-meteo.com/v1/forecast"

def main():
    df = pd.read_csv(TRAINING_PATH)

    # Select five representative samples, one from each
    # combination where available, without modifying any data.
    selected = (
        df.groupby(["pilot_tile", "label"], group_keys=False)
        .head(1)
        .head(5)
    )

    results = []

    with requests.Session() as session:
        for row in selected.itertuples(index=False):
            params = {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "hourly": "precipitation",
                "past_days": 2,
                "forecast_days": 3,
                "timezone": "Asia/Kolkata",
            }

            response = session.get(
                API_URL, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            results.append({
                "pilot_tile": row.pilot_tile,
                "label": row.label,
                "requested_latitude": row.latitude,
                "requested_longitude": row.longitude,
                "api_latitude": data.get("latitude"),
                "api_longitude": data.get("longitude"),
                "elevation_m": data.get("elevation"),
                "hourly_records": len(
                    data.get("hourly", {}).get("time", [])
                ),
            })

            print(
                f"{row.pilot_tile}: requested "
                f"({row.latitude:.5f}, {row.longitude:.5f}) "
                f"-> API grid "
                f"({data.get('latitude')}, {data.get('longitude')})"
            )

    result = pd.DataFrame(results)

    print("\nGrid diagnostic summary:")
    print(result.to_string(index=False))

    print("\nUnique API grid cells:", result[
        ["api_latitude", "api_longitude"]
    ].drop_duplicates().shape[0])

    print("\nDiagnostic only: no files were modified.")

if __name__ == "__main__":
    main()
