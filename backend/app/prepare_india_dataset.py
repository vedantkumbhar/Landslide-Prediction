
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "global_landslide_catalog.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "india_landslide_events.csv"
)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, low_memory=False)

    # Keep records identified as India, ignoring capitalization
    # and accidental whitespace.
    country = df["country_name"].astype("string").str.strip()

    india = df[country.str.casefold().eq("india")].copy()

    # Parse source dates. The source uses month/day/year.
    india["event_date_parsed"] = pd.to_datetime(
        india["event_date"],
        format="mixed",
        errors="coerce",
    )

    # Convert coordinates to numeric values.
    india["latitude"] = pd.to_numeric(
        india["latitude"], errors="coerce"
    )
    india["longitude"] = pd.to_numeric(
        india["longitude"], errors="coerce"
    )

    # Flag approximate India bounding-box matches for review.
    # This is a quality check, not a final geographic boundary.
    india["within_india_bbox"] = (
        india["latitude"].between(6, 38)
        & india["longitude"].between(68, 98)
    )

    # Standardize whitespace in region labels without
    # merging different spellings automatically.
    india["admin_division_name"] = (
        india["admin_division_name"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    india.to_csv(OUTPUT_PATH, index=False)

    print("=" * 55)
    print("INDIA LANDSLIDE DATA PREPARATION")
    print("=" * 55)

    print(f"Source records: {len(df):,}")
    print(f"India records: {len(india):,}")
    print(
        "Records with valid parsed dates:",
        f"{india['event_date_parsed'].notna().sum():,}",
    )
    print(
        "Records with valid coordinates:",
        f"{(india['latitude'].between(-90, 90) & india['longitude'].between(-180, 180)).sum():,}",
    )
    print(
        "Records inside approximate India bounding box:",
        f"{india['within_india_bbox'].sum():,}",
    )
    print(
        "Duplicate event IDs:",
        f"{india['event_id'].duplicated().sum():,}",
    )

    print("\nRecords by region:")
    print(
        india["admin_division_name"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("Original raw dataset was not modified.")


if __name__ == "__main__":
    main()
