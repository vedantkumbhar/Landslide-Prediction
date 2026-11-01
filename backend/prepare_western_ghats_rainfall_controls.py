
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Western Ghats: Prepare matched historical rainfall controls
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BACKGROUND_FILE = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

EVENT_FILE = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation" / "pilot_historical_events.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation" / "matched_rainfall_controls.csv"
)

SEED = 42

REQUIRED_TILES = {"N09E076", "N11E076", "N19E072"}


def main():
    # --------------------------------------------------------
    # 1. Validate input files and protect existing output
    # --------------------------------------------------------
    if not BACKGROUND_FILE.exists():
        raise FileNotFoundError(
            f"Background dataset not found: {BACKGROUND_FILE}"
        )

    if not EVENT_FILE.exists():
        raise FileNotFoundError(
            f"Historical event list not found: {EVENT_FILE}"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output:\n{OUTPUT_FILE}"
        )

    background = pd.read_csv(BACKGROUND_FILE)
    events = pd.read_csv(EVENT_FILE)

    required_background = {
        "sample_id",
        "sample_source",
        "label",
        "pilot_tile",
        "latitude",
        "longitude",
    }

    required_events = {
        "event_id",
        "terrain_source_tile",
        "event_date_parsed",
        "event_time_status",
        "latitude",
        "longitude",
    }

    missing_background = required_background - set(background.columns)
    missing_events = required_events - set(events.columns)

    if missing_background:
        raise ValueError(
            f"Background dataset missing columns: "
            f"{sorted(missing_background)}"
        )

    if missing_events:
        raise ValueError(
            f"Historical event dataset missing columns: "
            f"{sorted(missing_events)}"
        )

    print("=" * 65)
    print("WESTERN GHATS MATCHED RAINFALL CONTROL PREPARATION")
    print("=" * 65)

    print(f"Training dataset rows: {len(background)}")
    print(f"Historical events: {len(events)}")

    # --------------------------------------------------------
    # 2. Select the 600 background samples
    # --------------------------------------------------------
    background["label"] = pd.to_numeric(
        background["label"], errors="coerce"
    )
    background["latitude"] = pd.to_numeric(
        background["latitude"], errors="coerce"
    )
    background["longitude"] = pd.to_numeric(
        background["longitude"], errors="coerce"
    )

    # IMPORTANT:
    # Background rows have pilot_tile populated, while their
    # terrain_source_tile column is missing.
    controls = background[
        (background["label"] == 0)
        & (
            background["sample_source"]
            == "random_background_sampling"
        )
        & background["pilot_tile"].notna()
        & background["latitude"].notna()
        & background["longitude"].notna()
    ].copy()

    controls["terrain_source_tile"] = (
        controls["pilot_tile"].astype(str).str.strip()
    )

    controls["sample_id"] = controls["sample_id"].astype(str)

    print(f"\nEligible background samples: {len(controls)}")

    if controls.empty:
        raise RuntimeError(
            "No eligible background samples were found."
        )

    unexpected_tiles = (
        set(controls["terrain_source_tile"].unique())
        - REQUIRED_TILES
    )

    if unexpected_tiles:
        raise ValueError(
            f"Unexpected background tiles: {sorted(unexpected_tiles)}"
        )

    print("\nBackground samples by tile:")
    print(
        controls["terrain_source_tile"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Check background sample IDs and coordinates.
    if controls["sample_id"].duplicated().any():
        raise ValueError("Duplicate background sample IDs detected.")

    controls["_coordinate_key"] = (
        controls["latitude"].round(6).astype(str)
        + "_"
        + controls["longitude"].round(6).astype(str)
    )

    if controls["_coordinate_key"].duplicated().any():
        print(
            "\nWarning: duplicate background coordinates exist. "
            "Only one row per coordinate will be eligible."
        )

    controls = controls.drop_duplicates(
        subset=["_coordinate_key"]
    ).copy()

    # --------------------------------------------------------
    # 3. Validate historical events
    # --------------------------------------------------------
    events["event_date_parsed"] = pd.to_datetime(
        events["event_date_parsed"], errors="coerce"
    )
    events["latitude"] = pd.to_numeric(
        events["latitude"], errors="coerce"
    )
    events["longitude"] = pd.to_numeric(
        events["longitude"], errors="coerce"
    )
    events["terrain_source_tile"] = (
        events["terrain_source_tile"].astype(str).str.strip()
    )

    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate historical event IDs detected.")

    events = events.dropna(
        subset=[
            "event_date_parsed",
            "latitude",
            "longitude",
        ]
    ).copy()

    if not set(events["terrain_source_tile"]).issubset(REQUIRED_TILES):
        raise ValueError(
            "Historical events contain unexpected pilot tile names."
        )

    print(f"\nValid historical events: {len(events)}")

    # --------------------------------------------------------
    # 4. Select one unique control per event, same tile
    # --------------------------------------------------------
    rng = np.random.default_rng(SEED)

    selected = []
    used_coordinates = set()
    used_sample_ids = set()

    for _, event in events.sort_values("event_id").iterrows():
        event_id = event["event_id"]
        tile = event["terrain_source_tile"]

        candidates = controls[
            controls["terrain_source_tile"] == tile
        ].copy()

        candidates = candidates[
            ~candidates["_coordinate_key"].isin(used_coordinates)
            & ~candidates["sample_id"].isin(used_sample_ids)
        ]

        if candidates.empty:
            print(
                f"WARNING: no unused control available for event "
                f"{event_id}, tile {tile}"
            )
            continue

        chosen_index = rng.choice(candidates.index.to_numpy())
        chosen = candidates.loc[chosen_index]

        used_coordinates.add(chosen["_coordinate_key"])
        used_sample_ids.add(chosen["sample_id"])

        selected.append({
            "control_id": f"CTRL_{event_id}",
            "matched_event_id": event_id,
            "terrain_source_tile": tile,
            "matched_event_date": (
                event["event_date_parsed"].date().isoformat()
            ),
            "event_time_status": event["event_time_status"],
            "event_latitude": float(event["latitude"]),
            "event_longitude": float(event["longitude"]),
            "control_latitude": float(chosen["latitude"]),
            "control_longitude": float(chosen["longitude"]),
            "background_sample_id": chosen["sample_id"],
            "background_sample_source": chosen["sample_source"],
            "background_label": 0,
            "control_status": "background_unverified",
            "selection_seed": SEED,
        })

    result = pd.DataFrame(selected)

    if result.empty:
        raise RuntimeError("No matched controls could be selected.")

    if result["matched_event_id"].duplicated().any():
        raise RuntimeError("A historical event was matched more than once.")

    if result["background_sample_id"].duplicated().any():
        raise RuntimeError("A background sample was selected more than once.")

    if result[
        ["control_latitude", "control_longitude"]
    ].duplicated().any():
        raise RuntimeError("A control coordinate was selected more than once.")

    # --------------------------------------------------------
    # 5. Verify tile matching
    # --------------------------------------------------------
    event_counts = events["terrain_source_tile"].value_counts()
    control_counts = result["terrain_source_tile"].value_counts()

    print("\nMatched controls by tile:")
    print(control_counts.sort_index().to_string())

    print("\nHistorical events vs matched controls:")
    for tile in sorted(REQUIRED_TILES):
        print(
            f"{tile}: {int(event_counts.get(tile, 0))} events, "
            f"{int(control_counts.get(tile, 0))} controls"
        )

    # --------------------------------------------------------
    # 6. Save a new output file only
    # --------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nMatched control dataset created successfully.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Matched pairs: {len(result)}")
    print(
        "Unique control coordinates:",
        result[
            ["control_latitude", "control_longitude"]
        ].drop_duplicates().shape[0],
    )

    if len(result) != len(events):
        print(
            "\nWARNING: Not every event received a control. "
            "Review tile counts before proceeding."
        )
    else:
        print("All historical events have one matched control.")

    print(
        "\nIMPORTANT: Controls are unverified background samples, "
        "not confirmed landslide-free locations. Matching is by tile "
        "and event date, not by exact geographic distance."
    )

    print("Original datasets and models were not modified.")


if __name__ == "__main__":
    main()
