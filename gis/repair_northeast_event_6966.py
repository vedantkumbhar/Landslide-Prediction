
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_N27E088_event_features.csv"
)

EVENT_ID = 6966

RECOVERED_VALUES = {
    "elevation_m": 229.435257,
    "slope_degrees": 29.683670,
    "aspect_degrees": 157.715973,
}


def main():
    print("=" * 65)
    print("REPAIRING NORTHEAST EVENT 6966")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)

    if "event_id" not in df.columns:
        raise ValueError("event_id column is missing.")

    matches = df.index[df["event_id"] == EVENT_ID].tolist()

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one event {EVENT_ID}; found {len(matches)}."
        )

    idx = matches[0]

    # Confirm the repair applies only to missing terrain values.
    for column, value in RECOVERED_VALUES.items():
        current = df.at[idx, column]

        if pd.notna(current):
            raise ValueError(
                f"{column} is already populated ({current}); "
                "review before overwriting."
            )

        df.at[idx, column] = value

    df.at[idx, "terrain_valid"] = True

    # Record provenance so the values are not mistaken for N27E088 samples.
    df.at[idx, "terrain_source_tile"] = "N26E088"

    df.to_csv(OUTPUT_FILE, index=False)

    terrain_columns = list(RECOVERED_VALUES.keys())
    complete = df[terrain_columns].notna().all(axis=1)

    print(f"Updated event ID: {EVENT_ID}")
    print(f"Terrain source tile: N26E088")
    print("\nRecovered values:")
    for column, value in RECOVERED_VALUES.items():
        print(f"{column}: {value:.6f}")

    print(f"\nTotal records: {len(df)}")
    print(f"Complete terrain records: {int(complete.sum())}")
    print(f"Incomplete terrain records: {int((~complete).sum())}")
    print(f"\nUpdated file: {OUTPUT_FILE}")
    print("\nRepair completed.")


if __name__ == "__main__":
    main()
