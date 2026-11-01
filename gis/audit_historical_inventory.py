
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "india_landslide_events.csv"
)


def main():
    df = pd.read_csv(EVENTS_PATH)

    print("=" * 65)
    print("HISTORICAL LANDSLIDE INVENTORY AUDIT")
    print("=" * 65)

    print(f"\nTotal records: {len(df)}")

    # 1. Event ID audit
    if "event_id" in df.columns:
        event_ids = df["event_id"].dropna()
        duplicate_ids = df[
            df["event_id"].notna()
            & df["event_id"].duplicated(keep=False)
        ]

        print("\n1. EVENT ID AUDIT")
        print(f"Missing event IDs: {df['event_id'].isna().sum()}")
        print(f"Repeated event-ID records: {len(duplicate_ids)}")

        if not duplicate_ids.empty:
            print(duplicate_ids[
                ["event_id", "event_date", "latitude", "longitude"]
            ].to_string(index=False))

        print(f"Unique non-missing event IDs: {event_ids.nunique()}")

    # 2. Date audit
    date_column = (
        "event_date_parsed"
        if "event_date_parsed" in df.columns
        else "event_date"
    )

    parsed_dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    print("\n2. DATE AUDIT")
    print(f"Unparseable or missing dates: {parsed_dates.isna().sum()}")

    valid_dates = parsed_dates.dropna()

    if not valid_dates.empty:
        print(f"Earliest event date: {valid_dates.min()}")
        print(f"Latest event date: {valid_dates.max()}")
        print(
            "Future dates relative to 2026-10-11:",
            int((valid_dates > pd.Timestamp("2026-10-11")).sum()),
        )

    # 3. Coordinate audit
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    invalid_coordinates = (
        lat.isna()
        | lon.isna()
        | ~lat.between(-90, 90)
        | ~lon.between(-180, 180)
    )

    print("\n3. COORDINATE AUDIT")
    print(f"Missing or invalid global coordinates: {int(invalid_coordinates.sum())}")

    # 4. Exact duplicate rows
    print("\n4. EXACT DUPLICATE ROW AUDIT")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    print("\nAudit complete. No records were modified.")


if __name__ == "__main__":
    main()
