
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "himalayas_N30E078_event_features.csv"
)


def main():
    df = pd.read_csv(FEATURES_PATH)

    duplicates = df[
        df.duplicated(
            subset=["latitude", "longitude"],
            keep=False,
        )
    ].sort_values(["latitude", "longitude"])

    print("=" * 60)
    print("DUPLICATE COORDINATE INVESTIGATION")
    print("=" * 60)

    if duplicates.empty:
        print("No duplicate coordinate pairs found.")
        return

    preferred_columns = [
        "event_id",
        "event_date",
        "event_date_parsed",
        "latitude",
        "longitude",
        "admin_division_name",
        "landslide_category",
        "landslide_trigger",
        "landslide_size",
        "elevation_m",
        "slope_deg",
        "aspect_deg",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in duplicates.columns
    ]

    print(duplicates[columns].to_string(index=False))
    print(f"\nDuplicate-coordinate records: {len(duplicates)}")
    print("No records were deleted or modified.")


if __name__ == "__main__":
    main()
