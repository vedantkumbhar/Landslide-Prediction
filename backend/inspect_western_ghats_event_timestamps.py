
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVENTS = (
    ROOT / "data" / "processed"
    / "western_ghats_event_features.csv"
)


def main():
    if not EVENTS.exists():
        raise FileNotFoundError(f"Event file not found: {EVENTS}")

    df = pd.read_csv(EVENTS, low_memory=False)

    print("WESTERN GHATS EVENT-TIME INSPECTION")
    print("Read-only; no files will be modified.")
    print("Total inventory rows:", len(df))

    for column in [
        "event_id",
        "event_date",
        "event_date_parsed",
        "event_time",
        "created_date",
        "submitted_date",
        "last_edited_date",
        "latitude",
        "longitude",
        "terrain_source_tile",
        "terrain_valid",
        "terrain_status",
    ]:
        if column not in df.columns:
            print(f"\n{column}: COLUMN NOT FOUND")
            continue

        print(f"\n--- {column} ---")
        print("Data type:", df[column].dtype)
        print("Missing:", int(df[column].isna().sum()))
        print("Unique non-null values:", df[column].nunique())

        if column in [
            "event_date", "event_date_parsed", "event_time",
            "created_date", "submitted_date", "last_edited_date",
        ]:
            print("Example values:")
            print(df[column].dropna().astype(str).head(12).to_string(index=False))

    if "event_date_parsed" in df.columns:
        dates = pd.to_datetime(
            df["event_date_parsed"], errors="coerce"
        )
        print("\n--- Parsed event-date coverage ---")
        print("Parseable:", int(dates.notna().sum()))
        print("Unparseable:", int(dates.isna().sum()))
        if dates.notna().any():
            print("Earliest:", dates.min())
            print("Latest:", dates.max())

    if "terrain_source_tile" in df.columns:
        pilot = df[
            df["terrain_source_tile"].isin(
                ["N09E076", "N11E076", "N19E072"]
            )
        ].copy()

        print("\n--- Events in the three pilot tiles ---")
        print("Rows:", len(pilot))

        if "event_id" in pilot.columns:
            print("Unique event IDs:", pilot["event_id"].nunique())

        if "event_date_parsed" in pilot.columns:
            pilot_dates = pd.to_datetime(
                pilot["event_date_parsed"], errors="coerce"
            )
            print("Events with parsed dates:", int(pilot_dates.notna().sum()))

        if {"latitude", "longitude"}.issubset(pilot.columns):
            coords = pilot[["latitude", "longitude"]].apply(
                pd.to_numeric, errors="coerce"
            )
            print(
                "Rows with valid coordinate pairs:",
                int(coords.notna().all(axis=1).sum()),
            )

        print("\nSample pilot events:")
        columns = [
            c for c in [
                "event_id", "event_date", "event_date_parsed",
                "event_time", "latitude", "longitude",
                "terrain_source_tile", "landslide_trigger",
            ]
            if c in pilot.columns
        ]
        print(pilot[columns].head(15).to_string(index=False))

    print("\nINSPECTION COMPLETE.")
    print(
        "Do not assume a date-only event has a known event hour. "
        "Historical rainfall alignment must preserve that uncertainty."
    )


if __name__ == "__main__":
    main()
