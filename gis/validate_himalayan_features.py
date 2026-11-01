
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "himalayas_N30E078_event_features.csv"
)


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}"
        )

    df = pd.read_csv(FEATURES_PATH)

    required_columns = [
        "latitude",
        "longitude",
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "terrain_data_valid",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("=" * 60)
    print("HIMALAYAN FEATURE DATASET VALIDATION")
    print("=" * 60)

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nMissing values:")
    print(
        df[required_columns]
        .isna()
        .sum()
        .to_string()
    )

    print("\nDuplicate coordinate pairs:")
    print(
        int(
            df.duplicated(
                subset=["latitude", "longitude"]
            ).sum()
        )
    )

    print("\nTerrain validity flags:")
    print(df["terrain_data_valid"].value_counts(dropna=False))

    print("\nPhysical range checks:")

    checks = {
        "Elevation is finite": np.isfinite(
            df["elevation_m"]
        ).all(),
        "Slope is between 0 and 90 degrees": (
            df["slope_deg"].between(0, 90).all()
        ),
        "Aspect is between -1 and 360 degrees": (
            df["aspect_deg"].between(-1, 360).all()
        ),
        "Elevation has no raster NoData sentinel": (
            (df["elevation_m"] != -9999).all()
        ),
        "Slope has no raster NoData sentinel": (
            (df["slope_deg"] != -9999).all()
        ),
        "Aspect has no raster NoData sentinel": (
            (df["aspect_deg"] != -9999).all()
        ),
    }

    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")

    print("\nFeature statistics:")
    print(
        df[
            ["elevation_m", "slope_deg", "aspect_deg"]
        ].describe().round(2)
    )

    print("\nValidation completed.")
    print("No source files were modified.")


if __name__ == "__main__":
    main()
