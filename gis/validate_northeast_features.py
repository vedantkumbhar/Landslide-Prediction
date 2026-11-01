
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_N27E088_event_features.csv"
)


def main():
    print("=" * 65)
    print("NORTHEAST EVENT FEATURE VALIDATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    terrain_columns = [
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
    ]

    print(f"\nTotal records: {len(df)}")
    print(f"Unique event IDs: {df['event_id'].nunique()}")

    print("\nMissing terrain values:")
    print(df[terrain_columns].isna().sum().to_string())

    print("\nPhysical range checks:")

    elevation_ok = (
        df["elevation_m"].dropna().ge(-500).all()
        and df["elevation_m"].dropna().le(9000).all()
    )

    slope_ok = (
        df["slope_degrees"].dropna().ge(0).all()
        and df["slope_degrees"].dropna().le(90).all()
    )

    aspect_values = df["aspect_degrees"].dropna()
    flat_mask = aspect_values.eq(-1)
    directional_aspect = aspect_values[~flat_mask]

    aspect_ok = (
        directional_aspect.ge(0).all()
        and directional_aspect.lt(360).all()
    )

    print(f"Elevation within broad physical limits: {elevation_ok}")
    print(f"Slope between 0 and 90 degrees: {slope_ok}")
    print(f"Aspect within expected range: {aspect_ok}")

    complete_mask = df[terrain_columns].notna().all(axis=1)

    print("\nCompleteness:")
    print(f"Complete terrain records: {int(complete_mask.sum())}")
    print(f"Incomplete terrain records: {int((~complete_mask).sum())}")

    if (~complete_mask).any():
        print("\nEvents requiring inspection:")
        columns = [
            c for c in [
                "event_id",
                "latitude",
                "longitude",
                *terrain_columns,
            ]
            if c in df.columns
        ]
        print(df.loc[~complete_mask, columns].to_string(index=False))

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
