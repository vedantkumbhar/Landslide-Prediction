
import pandas as pd
from pathlib import Path

root = Path("data/processed")

background = pd.read_csv(root / "pilot_background_samples.csv")
events = pd.read_csv(root / "pilot_landslide_event_features.csv")

print("BACKGROUND SAMPLE COLUMNS:")
print(background.columns.tolist())

print("\nBACKGROUND SAMPLES BY TILE:")
print(background.groupby("pilot_tile").size().to_string())

print("\nBACKGROUND SAMPLING METHODS:")
print(background["sampling_method"].value_counts(dropna=False).to_string())

print("\nEVENTS BY PILOT TILE:")
print(events["pilot_tile"].value_counts(dropna=False).to_string())

print("\nEVENT TERRAIN VALIDITY BY TILE:")
print(
    events.groupby("pilot_tile")["terrain_valid"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nBACKGROUND SAMPLE COORDINATE RANGES:")
print(
    background.groupby("pilot_tile")
    .agg(
        sample_count=("sample_id", "count"),
        min_latitude=("latitude", "min"),
        max_latitude=("latitude", "max"),
        min_longitude=("longitude", "min"),
        max_longitude=("longitude", "max"),
    )
    .to_string()
)

print("\nBACKGROUND LABELS:")
print(background["label"].value_counts(dropna=False).to_string())
