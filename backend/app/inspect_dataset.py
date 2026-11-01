
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "global_landslide_catalog.csv"


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("=" * 60)
    print("LANDSLIDE DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Country coverage
    print("\n--- TOP 15 COUNTRIES ---")
    print(
        df["country_name"]
        .value_counts(dropna=False)
        .head(15)
        .to_string()
    )

    # India-specific records
    india = df[
        df["country_name"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("india")
    ].copy()

    print("\n--- INDIA RECORDS ---")
    print(f"India records: {len(india):,}")

    if not india.empty:
        print("\nIndia records by state/region:")
        print(
            india["admin_division_name"]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

        print("\nIndia landslide triggers:")
        print(
            india["landslide_trigger"]
            .value_counts(dropna=False)
            .to_string()
        )

    # Missing values
    print("\n--- MISSING VALUES ---")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(15).to_string())

    # Coordinate quality
    valid_coordinates = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    print("\n--- COORDINATE QUALITY ---")
    print(f"Records with valid coordinates: {valid_coordinates.sum():,}")
    print(f"Records missing/invalid coordinates: {(~valid_coordinates).sum():,}")

    # Date coverage
    dates = pd.to_datetime(
        df["event_date"],
        errors="coerce",
        format="mixed",
    )

    print("\n--- EVENT DATE COVERAGE ---")
    print(f"Valid event dates: {dates.notna().sum():,}")

    if dates.notna().any():
        print(f"Earliest event date: {dates.min()}")
        print(f"Latest event date: {dates.max()}")

    print("\nAnalysis complete. Original dataset was not modified.")


if __name__ == "__main__":
    main()
