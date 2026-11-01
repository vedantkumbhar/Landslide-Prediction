
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/regional_training_dataset_30oct.csv"
OUTPUT = ROOT / "data/processed/rainfall/regional_hourly_rainfall_30oct.csv"

API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
PAST_DAYS = 2
FORECAST_DAYS = 3

REGION_MAP = {
    "Himalayas": "Himalayas",
    "Northeast India": "Northeast India",
    "Western Maharashtra": "Western Maharashtra",
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}. "
            "Rename or archive it before running again."
        )

    df = pd.read_csv(INPUT)

    # One representative location per DEM tile.
    points = (
        df.groupby(["dem_region", "tile"], as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
    )

    points = points[
        points["dem_region"].isin(REGION_MAP)
    ].copy()

    expected_tiles = {
        "Himalayas": 29,
        "Northeast India": 33,
        "Western Maharashtra": 5,
    }

    counts = points.groupby("dem_region").size().to_dict()

    for region, expected in expected_tiles.items():
        if counts.get(region, 0) != expected:
            raise RuntimeError(
                f"{region}: expected {expected} tiles, "
                f"found {counts.get(region, 0)}"
            )

    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    records = []
    failures = []

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "LandslidePredictionResearch/1.0"
        })

        for index, point in points.iterrows():
            region = point["dem_region"]
            tile = point["tile"]
            lat = float(point["latitude"])
            lon = float(point["longitude"])

            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation",
                "past_days": PAST_DAYS,
                "forecast_days": FORECAST_DAYS,
                "timezone": TIMEZONE,
            }

            try:
                response = session.get(
                    API_URL, params=params, timeout=40
                )
                response.raise_for_status()
                data = response.json()

                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                values = hourly.get("precipitation", [])

                if not times or len(times) != len(values):
                    raise ValueError("Invalid hourly response")

                for timestamp, rainfall in zip(times, values):
                    observation_time = datetime.fromisoformat(timestamp)

                    records.append({
                        "region": region,
                        "tile": tile,
                        "requested_latitude": lat,
                        "requested_longitude": lon,
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

                print(
                    f"[{index + 1}/{len(points)}] "
                    f"{region} | {tile} | {len(times)} hours"
                )

            except Exception as exc:
                failures.append({
                    "region": region,
                    "tile": tile,
                    "latitude": lat,
                    "longitude": lon,
                    "error": str(exc),
                })
                print(f"FAILED: {region} {tile}: {exc}")

            # Keep requests spaced to avoid unnecessary API pressure.
            time.sleep(0.5)

    if not records:
        raise RuntimeError("No rainfall records downloaded.")

    result = pd.DataFrame(records)

    if result.duplicated(
        ["region", "tile", "timestamp_local"]
    ).any():
        raise RuntimeError("Duplicate tile timestamps detected.")

    if (result["precipitation_mm"].dropna() < 0).any():
        raise RuntimeError("Negative rainfall values detected.")

    if result["precipitation_mm"].notna().sum() == 0:
        raise RuntimeError("All rainfall values are missing.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    temp = OUTPUT.with_suffix(".tmp")
    try:
        result.to_csv(temp, index=False)
        check = pd.read_csv(temp)

        if len(check) != len(result):
            raise RuntimeError("CSV verification failed.")

        temp.replace(OUTPUT)
    finally:
        if temp.exists():
            temp.unlink()

    failure_path = OUTPUT.with_name(
        "regional_rainfall_failures_30oct.csv"
    )

    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)

    print("\n========== DOWNLOAD SUMMARY ==========")
    print("Tiles requested:", len(points))
    print("Successful tiles:", len(points) - len(failures))
    print("Failed tiles:", len(failures))
    print("Total hourly records:", len(result))
    print("\nRecords by region:")
    print(result.groupby("region").size().to_string())
    print("\nRecords by type:")
    print(result.groupby("record_type").size().to_string())
    print("\nSaved:", OUTPUT)

    if failures:
        print("Failure report:", failure_path)


if __name__ == "__main__":
    main()
