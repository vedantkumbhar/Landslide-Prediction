
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "india_landslide_events.csv"
)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["event_date_parsed"] = pd.to_datetime(
        df["event_date_parsed"], errors="coerce"
    )

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    print("=" * 60)
    print("HISTORICAL LANDSLIDE LOCATION INSPECTION")
    print("=" * 60)

    print(f"\nValid event locations: {len(df):,}")
    print("\n--- OVERALL GEOGRAPHIC EXTENT ---")
    print(f"Latitude:  {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")

    print("\n--- EVENT COUNTS BY ADMINISTRATIVE REGION ---")
    print(
        df["admin_division_name"]
        .fillna("Unknown")
        .value_counts()
        .head(25)
        .to_string()
    )

    print("\n--- EVENT COUNTS BY BROAD GEOGRAPHIC ZONE ---")

    zones = {
        "Himalayas": (
            df["latitude"].between(27, 36)
            & df["longitude"].between(72, 98)
        ),
        "Northeast India": (
            df["latitude"].between(20, 30)
            & df["longitude"].between(88, 98)
        ),
        "Western Maharashtra test region": (
            df["latitude"].between(15, 19)
            & df["longitude"].between(73, 75)
        ),
    }

    for zone, mask in zones.items():
        print(f"{zone}: {int(mask.sum()):,} events")

    print("\n--- EVENT COUNTS BY YEAR ---")
    years = df["event_date_parsed"].dt.year
    print(years.value_counts().sort_index().to_string())

    print("\n--- LOCATION DUPLICATES ---")
    duplicate_coords = df.duplicated(
        subset=["latitude", "longitude"]
    ).sum()
    print(f"Repeated coordinate pairs: {duplicate_coords:,}")

    print("\nInspection complete.")
    print("Source dataset was not modified.")


if __name__ == "__main__":
    main()
