
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_landslide_event_features.csv"
)


def main():
    print("=" * 65)
    print("COMBINED PILOT FEATURE VALIDATION")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)

    terrain = [
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
    ]

    required = ["event_id", "pilot_tile", *terrain]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"\nTotal records: {len(df)}")
    print(f"Unique event IDs: {df['event_id'].nunique()}")
    print(f"Duplicate event IDs: {int(df['event_id'].duplicated().sum())}")

    complete = df[terrain].notna().all(axis=1)
    finite = np.isfinite(df[terrain].to_numpy(dtype=float)).all(axis=1)

    print(f"Complete terrain records: {int(complete.sum())}")
    print(f"Records with finite terrain values: {int(finite.sum())}")

    elevation = df["elevation_m"].dropna()
    slope = df["slope_degrees"].dropna()
    aspect = df["aspect_degrees"].dropna()

    directional_aspect = aspect[aspect != -1]

    elevation_ok = (
        np.isfinite(elevation).all()
        and elevation.between(-500, 9000).all()
    )

    slope_ok = (
        np.isfinite(slope).all()
        and slope.between(0, 90).all()
    )

    aspect_ok = (
        np.isfinite(aspect).all()
        and directional_aspect.ge(0).all()
        and directional_aspect.lt(360).all()
    )

    print("\nTerrain range checks:")
    print(f"Elevation within broad limits: {elevation_ok}")
    print(f"Slope between 0 and 90 degrees: {slope_ok}")
    print(f"Aspect follows expected convention: {aspect_ok}")

    print("\nRecords by pilot tile:")
    print(df["pilot_tile"].value_counts().to_string())

    if "terrain_source_tile" in df.columns:
        event = df.loc[df["event_id"] == 6966]

        print("\nEvent 6966 provenance:")
        if event.empty:
            print("Event 6966 not found.")
        else:
            print(
                event[
                    [
                        "event_id",
                        "pilot_tile",
                        "terrain_source_tile",
                        *terrain,
                    ]
                ].to_string(index=False)
            )

    passed = (
        len(df) == 165
        and df["event_id"].is_unique
        and complete.all()
        and finite.all()
        and elevation_ok
        and slope_ok
        and aspect_ok
    )

    print("\n" + "=" * 65)
    print(f"OVERALL VALIDATION: {'PASS' if passed else 'CHECK REQUIRED'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
