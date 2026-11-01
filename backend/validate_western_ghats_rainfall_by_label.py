
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAINING_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

FEATURES_PATH = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "rainfall_features.csv"
)

FEATURES = [
    "recent_rain_6h_mm",
    "recent_rain_24h_mm",
    "forecast_rain_6h_mm",
    "forecast_rain_24h_mm",
]


def main():
    training = pd.read_csv(TRAINING_PATH)
    rainfall = pd.read_csv(FEATURES_PATH)

    required_training = [
        "pilot_tile", "latitude", "longitude", "label"
    ]
    missing = [
        c for c in required_training if c not in training.columns
    ]
    if missing:
        raise ValueError(f"Training columns missing: {missing}")

    required_rainfall = [
        "sample_id", "pilot_tile", "requested_latitude",
        "requested_longitude", "label", *FEATURES
    ]
    missing = [
        c for c in required_rainfall if c not in rainfall.columns
    ]
    if missing:
        raise ValueError(f"Rainfall columns missing: {missing}")

    # Confirm the original sample labels and counts.
    if training.duplicated(["latitude", "longitude"]).any():
        raise ValueError("Duplicate training coordinates detected.")

    print("TRAINING DATA LABEL COUNTS")
    print(training["label"].value_counts().sort_index().to_string())

    print("\nLABEL COUNTS BY TILE")
    print(
        training.groupby(["pilot_tile", "label"])
        .size().unstack(fill_value=0).to_string()
    )

    # Compare the rainfall dataset's sample metadata with training data.
    metadata = training[
        ["pilot_tile", "latitude", "longitude", "label"]
    ].rename(columns={
        "latitude": "requested_latitude",
        "longitude": "requested_longitude",
        "label": "training_label",
    })

    sample_meta = rainfall[
        ["sample_id", "pilot_tile", "requested_latitude",
         "requested_longitude", "label"]
    ].drop_duplicates()

    if sample_meta["sample_id"].duplicated().any():
        raise ValueError("Inconsistent metadata within sample IDs.")

    matched = sample_meta.merge(
        metadata,
        on=[
            "pilot_tile", "requested_latitude",
            "requested_longitude",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (matched["_merge"] == "both").all():
        raise ValueError("Rainfall samples do not match training samples.")

    if not (matched["label"] == matched["training_label"]).all():
        raise ValueError("Rainfall labels differ from training labels.")

    print("\nPASS: all rainfall sample labels match training data.")
    print("Unique samples:", sample_meta["sample_id"].nunique())

    # Collapse hourly rows to one record per sample and feature.
    # These are time-window averages, not event-date rainfall.
    sample_features = (
        rainfall.groupby(
            ["sample_id", "pilot_tile", "label"], as_index=False
        )[FEATURES].mean()
    )

    print("\nSAMPLE-LEVEL RAINFALL MEANS BY TILE AND LABEL")
    for feature in FEATURES:
        print(f"\n{feature}")
        summary = sample_features.groupby(
            ["pilot_tile", "label"]
        )[feature].agg(["count", "mean", "median"])
        print(summary.round(4).to_string())

    print(
        "\nCaution: these rainfall windows are from October 27–31, 2026. "
        "They do not represent rainfall on the historical landslide dates."
    )
    print("Read-only validation: no files were modified.")


if __name__ == "__main__":
    main()
