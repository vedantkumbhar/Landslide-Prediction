
import pandas as pd
from pathlib import Path

root = Path("data/processed")

events = pd.read_csv(root / "pilot_landslide_event_features.csv")
background = pd.read_csv(root / "pilot_background_samples.csv")

print("EVENTS:", len(events))
print("BACKGROUND SAMPLES:", len(background))

print("\nBackground samples by tile:")
print(background["pilot_tile"].value_counts().to_string())

print("\nDistance to nearest recorded event (km):")
print(
    background["nearest_recorded_event_km"]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90])
    .to_string()
)

for threshold in [1, 5, 10, 25]:
    count = (
        background["nearest_recorded_event_km"] < threshold
    ).sum()
    print(f"Samples within {threshold} km of a recorded event: {count}")

print("\nBackground labels:")
print(background["label"].value_counts(dropna=False).to_string())

print("\nEvent location accuracy:")
print(events["location_accuracy"].value_counts(dropna=False).to_string())
