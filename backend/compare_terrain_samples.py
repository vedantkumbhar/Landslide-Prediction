
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVENTS = ROOT / "data" / "processed" / "pilot_landslide_event_features.csv"
BACKGROUND = ROOT / "data" / "processed" / "pilot_background_terrain_features.csv"

FEATURES = ["elevation_m", "slope_degrees", "aspect_degrees"]


def main():
    events = pd.read_csv(EVENTS)
    background = pd.read_csv(BACKGROUND)

    event_required = FEATURES + ["terrain_valid", "pilot_tile"]
    missing = set(event_required) - set(events.columns)
    if missing:
        raise ValueError(f"Event columns missing: {sorted(missing)}")

    background_required = FEATURES + ["terrain_valid", "pilot_tile"]
    missing = set(background_required) - set(background.columns)
    if missing:
        raise ValueError(f"Background columns missing: {sorted(missing)}")

    event_data = events.loc[
        events["terrain_valid"].eq(True), FEATURES
    ].copy()

    background_data = background.loc[
        background["terrain_valid"].eq(True), FEATURES
    ].copy()

    print("EXPLORATORY TERRAIN COMPARISON")
    print("Events with valid terrain:", len(event_data))
    print("Background samples with valid terrain:", len(background_data))
    print("\nEvent terrain summary:")
    print(event_data.describe().round(2).to_string())

    print("\nBackground terrain summary:")
    print(background_data.describe().round(2).to_string())

    print("\nMean and median comparison:")
    summary = pd.DataFrame({
        "event_mean": event_data.mean(),
        "event_median": event_data.median(),
        "background_mean": background_data.mean(),
        "background_median": background_data.median(),
    })
    print(summary.round(2).to_string())

    print("\nEvent counts by tile:")
    print(events.groupby("pilot_tile").size().to_string())

    print("\nBackground counts by tile:")
    print(background.groupby("pilot_tile").size().to_string())

    print(
        "\nCaution: background samples are unverified and are not "
        "confirmed non-landslide locations. These summaries do not "
        "establish predictive performance or causal relationships."
    )


if __name__ == "__main__":
    main()
