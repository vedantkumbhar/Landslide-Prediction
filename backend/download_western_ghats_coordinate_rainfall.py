
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
)
OUTPUT = OUTPUT_DIR / "hourly_rainfall.csv"

API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"
PAST_DAYS = 2
FORECAST_DAYS = 3
BATCH_SIZE = 5
MAX_ATTEMPTS = 4


def fetch_batch(session, batch):
    params = {
        "latitude": ",".join(f"{x:.6f}" for x in batch["latitude"]),
        "longitude": ",".join(f"{x:.6f}" for x in batch["longitude"]),
        "hourly": "precipitation",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                API_URL, params=params, timeout=60
            )
            response.raise_for_status()
            data = response.json()

            locations = data if isinstance(data, list) else [data]

            if len(locations) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} locations, "
                    f"received {len(locations)}."
                )

            return locations

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Batch failed after {MAX_ATTEMPTS} attempts: {exc}"
                ) from exc

            delay = 2 ** attempt
            print(
                f"Batch attempt {attempt} failed: {exc}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    if not INPUT.exists():
        raise FileNotFoundError(f"Training dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = ["pilot_tile", "latitude", "longitude", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Input contains missing required values.")

    if df.duplicated(["latitude", "longitude"]).any():
        raise ValueError("Duplicate sample coordinates found.")

    coordinates = df[required].copy().reset_index(drop=True)

    if not coordinates["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values.")
    if not coordinates["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values.")

    retrieved_at = datetime.now(
        ZoneInfo(TIMEZONE)
    ).isoformat(timespec="seconds")

    records = []
    total_batches = (len(coordinates) + BATCH_SIZE - 1) // BATCH_SIZE

    with requests.Session() as session:
        for batch_number, start in enumerate(
            range(0, len(coordinates), BATCH_SIZE), start=1
        ):
            batch = coordinates.iloc[
                start:start + BATCH_SIZE
            ].reset_index(drop=True)

            locations = fetch_batch(session, batch)

            for index, location in enumerate(locations):
                sample = batch.iloc[index]
                hourly = location.get("hourly", {})
                times = hourly.get("time", [])
                values = hourly.get("precipitation", [])

                if not times or len(times) != len(values):
                    raise RuntimeError(
                        f"Invalid hourly data for sample row "
                        f"{start + index}."
                    )

                if pd.isna(location.get("latitude")) or pd.isna(
                    location.get("longitude")
                ):
                    raise RuntimeError("API returned missing grid coordinates.")

                if any(value is not None and float(value) < 0
                       for value in values):
                    raise RuntimeError("Negative precipitation returned.")

                for timestamp, rainfall in zip(times, values):
                    records.append({
                        "sample_id": f"WG_SAMPLE_{start + index + 1:04d}",
                        "pilot_tile": sample["pilot_tile"],
                        "label": int(sample["label"]),
                        "requested_latitude": float(sample["latitude"]),
                        "requested_longitude": float(sample["longitude"]),
                        "api_latitude": float(location["latitude"]),
                        "api_longitude": float(location["longitude"]),
                        "api_elevation_m": location.get("elevation"),
                        "timestamp_local": timestamp,
                        "precipitation_mm": (
                            float(rainfall)
                            if rainfall is not None else None
                        ),
                        "source": "Open-Meteo forecast API",
                        "record_type": (
                            "historical_or_current"
                            if datetime.fromisoformat(timestamp)
                            <= datetime.now(ZoneInfo(TIMEZONE)).replace(
                                tzinfo=None
                            )
                            else "forecast"
                        ),
                        "retrieved_at_local": retrieved_at,
                    })

            print(
                f"Completed batch {batch_number}/{total_batches}: "
                f"{len(batch)} locations; "
                f"{len(records)} hourly rows accumulated."
            )

            time.sleep(0.5)

    result = pd.DataFrame(records)

    if result.empty:
        raise RuntimeError("No rainfall data received.")

    key_columns = ["sample_id", "timestamp_local"]
    if result.duplicated(key_columns).any():
        raise RuntimeError("Duplicate sample timestamps detected.")

    if result["timestamp_local"].isna().any():
        raise RuntimeError("Missing timestamps detected.")

    expected_rows = len(coordinates) * 120
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows for 120 hours per sample; "
            f"received {len(result)}. No final file written."
        )

    if result.groupby("sample_id").size().nunique() != 1:
        raise RuntimeError("Inconsistent hourly record counts.")

    if result.groupby("sample_id")["api_latitude"].nunique().max() != 1:
        raise RuntimeError("API latitude changed within a sample.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting an existing final output.
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)

    print("\nCoordinate-level rainfall download completed.")
    print("Training samples:", len(coordinates))
    print("Unique requested coordinates:", len(coordinates))
    print("Rows downloaded:", len(result))
    print("Distinct API grid cells:", result[
        ["api_latitude", "api_longitude"]
    ].drop_duplicates().shape[0])
    print("Missing precipitation values:", int(
        result["precipitation_mm"].isna().sum()
    ))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
