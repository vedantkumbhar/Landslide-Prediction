
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EVENTS_FILE = (
    ROOT / "data" / "processed" / "western_ghats_event_features.csv"
)

BACKGROUND_FILE = (
    ROOT / "data" / "processed" / "western_ghats_background_samples.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "western_ghats_pilot_training_dataset.csv"
)

PILOT_TILES = {"N09E076", "N19E072", "N11E076"}

TERRAIN_COLUMNS = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]


def main():
    print("=" * 65)
    print("WESTERN GHATS PILOT TRAINING DATASET")
    print("=" * 65)

    for path in [EVENTS_FILE, BACKGROUND_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "Output already exists; refusing to overwrite:\n"
            f"{OUTPUT_FILE}"
        )

    events = pd.read_csv(EVENTS_FILE)
    background = pd.read_csv(BACKGROUND_FILE)

    # Retain only historical events assigned to the processed pilot tiles
    # and having all three valid terrain features.
    events["dem_tile"] = events["dem_tile"].astype("string").str.upper()

    event_mask = (
        events["dem_tile"].isin(PILOT_TILES)
        & events[TERRAIN_COLUMNS].notna().all(axis=1)
    )
    positive = events.loc[event_mask].copy()

    if positive.empty:
        raise ValueError("No complete historical event records found.")

    # Background candidates must have complete terrain features and
    # come only from the three intended pilot tiles.
    background["pilot_tile"] = (
        background["pilot_tile"].astype("string").str.upper()
    )

    background_mask = (
        background["pilot_tile"].isin(PILOT_TILES)
        & background[TERRAIN_COLUMNS].notna().all(axis=1)
    )
    negative = background.loc[background_mask].copy()

    if negative.empty:
        raise ValueError("No complete background records found.")

    # Standardize the shared columns while preserving original metadata.
    positive["pilot_tile"] = positive["dem_tile"]
    positive["label"] = 1
    positive["label_description"] = "historical_landslide_event"
    positive["sample_source"] = "historical_inventory"

    negative["dem_tile"] = negative["pilot_tile"]
    negative["label"] = 0
    negative["label_description"] = "background_unverified"
    negative["sample_source"] = "random_background_sampling"

    # Retain provenance and all available metadata.
    combined = pd.concat(
        [positive, negative],
        ignore_index=True,
        sort=False,
    )

    # Check numeric terrain features before writing.
    for column in TERRAIN_COLUMNS:
        combined[column] = pd.to_numeric(
            combined[column], errors="coerce"
        )

    terrain = combined[TERRAIN_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(terrain).all():
        raise ValueError("Non-finite terrain values found.")

    if not combined["slope_degrees"].between(0, 90).all():
        raise ValueError("Slope values outside the 0–90 degree range.")

    aspect = combined["aspect_degrees"]
    aspect_ok = (
        aspect.between(0, 360, inclusive="left") | aspect.eq(-1)
    )

    if not aspect_ok.all():
        raise ValueError("Unexpected aspect values found.")

    if combined.duplicated(
        subset=["pilot_tile", "longitude", "latitude"]
    ).any():
        raise ValueError(
            "Duplicate coordinates found within the combined dataset."
        )

    print(f"Historical events retained: {len(positive)}")
    print(f"Background candidates retained: {len(negative)}")
    print(f"Total records: {len(combined)}")

    print("\nLabel counts:")
    print(combined["label"].value_counts().sort_index().to_string())

    print("\nRecords by tile and label:")
    print(
        combined.groupby(["pilot_tile", "label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nLabel descriptions:")
    print(combined["label_description"].value_counts().to_string())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved new dataset:\n{OUTPUT_FILE}")
    print("Existing regional datasets were not modified.")
    print(
        "\nCaution: label 0 represents unverified background, "
        "not confirmed landslide-free terrain."
    )


if __name__ == "__main__":
    main()
