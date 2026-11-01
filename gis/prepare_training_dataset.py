from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/processed/all_region_event_features.csv"
BACKGROUND = ROOT / "data/processed/pilot_background_features.csv"
OUTPUT = ROOT / "data/processed/initial_training_dataset.csv"

FEATURES = ["elevation_m", "slope_degrees", "aspect_degrees"]

def main():
    events = pd.read_csv(EVENTS)
    background = pd.read_csv(BACKGROUND)

    events = events.loc[
        events["terrain_status"].eq("valid")
        & events[FEATURES].notna().all(axis=1)
    ].copy()
    events["label"] = 1
    events["sample_type"] = "historical_landslide"
    events["label_confidence"] = "recorded_event"

    background = background.loc[
        background["terrain_status"].eq("valid")
        & background[FEATURES].notna().all(axis=1)
        & background["nearest_recorded_event_km"].ge(5)
    ].copy()
    background["label"] = 0
    background["sample_type"] = "provisional_background"
    background["label_confidence"] = "unverified_background"

    event_cols = [
        "event_id", "latitude", "longitude", *FEATURES,
        "dem_region", "dem_tile", "label", "sample_type",
        "label_confidence",
    ]
    bg_cols = [
        "sample_id", "latitude", "longitude", *FEATURES,
        "dem_region", "pilot_tile", "nearest_recorded_event_km",
        "label", "sample_type", "label_confidence",
    ]

    events = events[event_cols].rename(columns={"dem_tile": "tile"})
    background = background[bg_cols].rename(
        columns={"pilot_tile": "tile", "sample_id": "event_id"}
    )

    combined = pd.concat([events, background], ignore_index=True, sort=False)
    combined.to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print("Dataset shape:", combined.shape)
    print("\nLabel counts:")
    print(combined["label"].value_counts().sort_index().to_string())
    print("\nSample types:")
    print(combined["sample_type"].value_counts().to_string())
    print("\nMissing model features:")
    print(combined[FEATURES].isna().sum().to_string())
    print("\nRegions:")
    print(combined["dem_region"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()

