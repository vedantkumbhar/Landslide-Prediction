
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAINING_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

def main():
    df = pd.read_csv(TRAINING_PATH)

    required = [
        "pilot_tile",
        "latitude",
        "longitude",
        "label",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing tile IDs, coordinates, or labels.")

    # Count unique coordinates across the complete dataset.
    unique_all = df[
        ["latitude", "longitude"]
    ].drop_duplicates()

    # Count unique coordinates within each tile.
    unique_by_tile = (
        df.groupby("pilot_tile")
        .apply(
            lambda group: group[
                ["latitude", "longitude"]
            ].drop_duplicates().shape[0],
            include_groups=False,
        )
    )

    print("Total training rows:", len(df))
    print("Unique coordinates overall:", len(unique_all))
    print("Duplicate coordinate rows:", len(df) - len(unique_all))

    print("\nUnique coordinates by tile:")
    print(unique_by_tile.to_string())

    print("\nSamples by label:")
    print(df["label"].value_counts().sort_index().to_string())

    print("\nEstimated request counts:")
    print("One request per training row:", len(df))
    print("One request per unique coordinate:", len(unique_all))
    print(
        "Minimum if coordinates can be safely shared across tiles:",
        len(unique_all),
    )

    print(
        "\nNote: distinct requested coordinates may resolve to the "
        "same Open-Meteo grid cell. This script does not contact the API, "
        "so the number of distinct grid cells is not yet known."
    )
    print("No files were modified.")

if __name__ == "__main__":
    main()
