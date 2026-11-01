
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "western_ghats_event_features.csv"

OUT_DIR = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "historical_validation"
)

OUTPUT = OUT_DIR / "pilot_historical_events.csv"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT}"
        )

    df = pd.read_csv(INPUT, low_memory=False)

    required = [
        "event_id",
        "event_date",
        "event_date_parsed",
        "latitude",
        "longitude",
        "terrain_source_tile",
        "landslide_trigger",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pilot_tiles = ["N09E076", "N11E076", "N19E072"]

    df = df[df["terrain_source_tile"].isin(pilot_tiles)].copy()

    df["event_datetime"] = pd.to_datetime(
        df["event_date_parsed"], errors="coerce"
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Midnight is ambiguous: it may be a real midnight event or a
    # date-only record represented as 00:00:00. Do not assume either.
    df["event_time_status"] = df["event_datetime"].apply(
        lambda value: (
            "date_unparseable" if pd.isna(value)
            else "midnight_time_ambiguous"
            if value.hour == 0 and value.minute == 0 and value.second == 0
            else "time_present_in_event_date"
        )
    )

    df["event_date_only"] = df["event_datetime"].dt.strftime("%Y-%m-%d")

    df["coordinate_status"] = "valid"
    invalid = (
        df["latitude"].isna()
        | df["longitude"].isna()
        | ~df["latitude"].between(-90, 90)
        | ~df["longitude"].between(-180, 180)
    )
    df.loc[invalid, "coordinate_status"] = "invalid_coordinates"

    df["historical_rainfall_status"] = "not_yet_retrieved"

    output_columns = [
        "event_id",
        "event_date",
        "event_date_parsed",
        "event_datetime",
        "event_date_only",
        "event_time_status",
        "latitude",
        "longitude",
        "terrain_source_tile",
        "landslide_trigger",
        "coordinate_status",
        "historical_rainfall_status",
    ]

    result = df[output_columns].sort_values(
        ["event_date_only", "event_id"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)

    print("Historical event list created.")
    print("Output:", OUTPUT)
    print("Rows:", len(result))
    print("Unique events:", result["event_id"].nunique())

    print("\nEvents by pilot tile:")
    print(result["terrain_source_tile"].value_counts().to_string())

    print("\nEvent-time status:")
    print(result["event_time_status"].value_counts().to_string())

    print("\nCoordinate status:")
    print(result["coordinate_status"].value_counts().to_string())

    print("\nHistorical rainfall has not been retrieved by this script.")
    print("Existing datasets and models were not modified.")


if __name__ == "__main__":
    main()
