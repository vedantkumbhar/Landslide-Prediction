
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HIMALAYAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "himalayas_N30E078_event_features.csv"
)

NORTHEAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_N27E088_event_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_landslide_event_features.csv"
)

TERRAIN_COLUMNS = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]


def main():
    print("=" * 65)
    print("STANDARDIZING AND COMBINING PILOT FEATURES")
    print("=" * 65)

    for path in [HIMALAYAN_FILE, NORTHEAST_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    himalayas = pd.read_csv(HIMALAYAN_FILE)
    northeast = pd.read_csv(NORTHEAST_FILE)

    # Standardize Himalayan terrain column names.
    himalayas = himalayas.rename(
        columns={
            "slope_deg": "slope_degrees",
            "aspect_deg": "aspect_degrees",
            "terrain_data_valid": "terrain_valid",
        }
    )

    # Preserve provenance for each pilot tile.
    himalayas["pilot_tile"] = "N30E078"
    northeast["pilot_tile"] = "N27E088"

    # Record the primary terrain source for each event.
    # Event 6966 uses the neighboring N26E088 tile.
    if "terrain_source_tile" not in himalayas.columns:
        himalayas["terrain_source_tile"] = "N30E078"

    if "terrain_source_tile" not in northeast.columns:
        northeast["terrain_source_tile"] = "N27E088"
    else:
        northeast["terrain_source_tile"] = (
            northeast["terrain_source_tile"].fillna("N27E088")
        )

    # Retain all records and columns, even if their schemas differ.
    combined = pd.concat(
        [himalayas, northeast],
        ignore_index=True,
        sort=False,
    )

    # Recalculate completeness from the standardized terrain fields.
    combined["terrain_valid"] = (
        combined[TERRAIN_COLUMNS].notna().all(axis=1)
    )

    print(f"\nHimalayan records: {len(himalayas)}")
    print(f"Northeast records: {len(northeast)}")
    print(f"Combined records: {len(combined)}")
    print(f"Unique event IDs: {combined['event_id'].nunique()}")

    duplicate_ids = combined["event_id"].duplicated(keep=False)
    print(f"Records with repeated event IDs: {int(duplicate_ids.sum())}")

    if duplicate_ids.any():
        print("\nRepeated IDs requiring investigation:")
        print(
            combined.loc[
                duplicate_ids, ["event_id", "pilot_tile"]
            ].sort_values("event_id").to_string(index=False)
        )

    complete_count = int(combined["terrain_valid"].sum())
    print(f"\nComplete terrain records: {complete_count}")
    print(f"Incomplete terrain records: {len(combined) - complete_count}")

    print("\nRecords by pilot tile:")
    print(combined["pilot_tile"].value_counts().to_string())

    print("\nTerrain completeness by tile:")
    print(
        combined.groupby("pilot_tile")["terrain_valid"]
        .agg(["sum", "count"])
        .to_string()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCombined dataset saved to:\n{OUTPUT_FILE}")
    print("\nStandardization and combination completed.")


if __name__ == "__main__":
    main()
