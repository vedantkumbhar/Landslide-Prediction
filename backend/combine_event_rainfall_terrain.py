from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVENTS = ROOT / "data/processed/pilot_landslide_event_features.csv"
RAINFALL = ROOT / "data/processed/rainfall/pilot_historical_event_rainfall.csv"
OUTPUT = ROOT / "data/processed/pilot_event_terrain_rainfall_features.csv"

def main():
    events = pd.read_csv(EVENTS, dtype={"event_id": "string"})
    rainfall = pd.read_csv(RAINFALL, dtype={"event_id": "string"})

    # Normalize IDs before joining.
    events["event_id"] = events["event_id"].str.strip()
    rainfall["event_id"] = rainfall["event_id"].str.strip()

    if events["event_id"].isna().any():
        raise ValueError("Some events have missing event IDs.")
    if rainfall["event_id"].isna().any():
        raise ValueError("Some rainfall rows have missing event IDs.")
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in event features.")
    if rainfall["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in rainfall data.")

    rainfall_columns = [
        "event_id",
        "previous_day_rainfall_mm",
        "event_day_rainfall_mm",
        "previous_day_hourly_records",
        "event_day_hourly_records",
        "api_latitude",
        "api_longitude",
        "source",
        "status",
        "error",
    ]

    missing = set(rainfall_columns) - set(rainfall.columns)
    if missing:
        raise ValueError(f"Rainfall file is missing columns: {sorted(missing)}")

    combined = events.merge(
        rainfall[rainfall_columns],
        on="event_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    combined["rainfall_matched"] = combined["_merge"].eq("both")
    combined.drop(columns="_merge", inplace=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)

    print("\nINTEGRATION SUMMARY")
    print("Output:", OUTPUT)
    print("Event rows:", len(events))
    print("Combined rows:", len(combined))
    print("Matched rainfall rows:", int(combined["rainfall_matched"].sum()))
    print("Unmatched rows:", int((~combined["rainfall_matched"]).sum()))
    print("Valid terrain rows:",
          int(combined["terrain_valid"].astype(str).str.lower().eq("true").sum()))
    print("Missing previous-day rainfall:",
          int(combined["previous_day_rainfall_mm"].isna().sum()))
    print("Missing event-day rainfall:",
          int(combined["event_day_rainfall_mm"].isna().sum()))

    print("\nRainfall retrieval status:")
    print(combined["status"].value_counts(dropna=False).to_string())

    print("\nPreview:")
    cols = [
        "event_id", "event_date_parsed", "pilot_tile",
        "elevation_m", "slope_degrees",
        "previous_day_rainfall_mm", "event_day_rainfall_mm",
        "rainfall_matched",
    ]
    print(combined[[c for c in cols if c in combined.columns]].head(8).to_string(index=False))

    print("\nNOTE: Event-day rainfall is not guaranteed to precede the event.")
    print("This is an event-feature dataset, not yet a validated training dataset.")

if __name__ == "__main__":
    main()
