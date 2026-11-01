from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "india_landslide_events.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("=" * 60)
    print("INDIA LANDSLIDE DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")

    print("\n--- DATE RANGE ---")
    dates = pd.to_datetime(df["event_date_parsed"], errors="coerce")
    print(f"Earliest event: {dates.min()}")
    print(f"Latest event:   {dates.max()}")

    print("\n--- LANDSLIDE TRIGGERS ---")
    print(df["landslide_trigger"].fillna("Missing").value_counts().to_string())

    print("\n--- LANDSLIDE CATEGORIES ---")
    print(df["landslide_category"].fillna("Missing").value_counts().to_string())

    print("\n--- LANDSLIDE SIZE ---")
    print(df["landslide_size"].fillna("Missing").value_counts().to_string())

    print("\n--- LANDSLIDE SETTINGS ---")
    print(df["landslide_setting"].fillna("Missing").value_counts().head(20).to_string())

    print("\n--- FATALITIES ---")
    fatality = pd.to_numeric(df["fatality_count"], errors="coerce")
    print(f"Records with known fatalities: {fatality.notna().sum():,}")
    print(f"Total reported fatalities: {fatality.sum():,.0f}")
    print(f"Maximum fatalities in one event: {fatality.max():.0f}")

    print("\n--- INJURIES ---")
    injury = pd.to_numeric(df["injury_count"], errors="coerce")
    print(f"Records with known injuries: {injury.notna().sum():,}")
    print(f"Total reported injuries: {injury.sum():,.0f}")
    print(f"Maximum injuries in one event: {injury.max():.0f}")

    print("\n--- TOP REGIONS ---")
    print(
        df["admin_division_name"]
        .fillna("Missing")
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\n--- MISSING VALUES ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string())

    print("\n--- COORDINATE RANGE ---")
    print(f"Latitude:  {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")

    print("\n--- DATA TYPES ---")
    print(df.dtypes.to_string())

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()