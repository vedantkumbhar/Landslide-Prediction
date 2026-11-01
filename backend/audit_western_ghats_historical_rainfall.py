
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EVENTS = ROOT / "data" / "processed" / "western_ghats_event_features.csv"

RAIN_DIR = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
)

FILES_TO_CHECK = [
    RAIN_DIR / "hourly_rainfall.csv",
    RAIN_DIR / "rainfall_features.csv",
    RAIN_DIR / "western_ghats_live_risk_inputs.csv",
    RAIN_DIR / "western_ghats_experimental_risk_scores.csv",
]


def inspect_file(path):
    print(f"\n--- {path.name} ---")

    if not path.exists():
        print("NOT FOUND")
        return

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", ", ".join(df.columns))

    time_col = next(
        (
            col for col in
            ["timestamp_local", "timestamp", "event_date", "event_time"]
            if col in df.columns
        ),
        None,
    )

    if time_col:
        times = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if not times.empty:
            print("Earliest timestamp:", times.min())
            print("Latest timestamp:", times.max())

    if "sample_id" in df.columns:
        print("Unique samples:", df["sample_id"].nunique())

    if "record_type" in df.columns:
        print("Record types:")
        print(df["record_type"].value_counts(dropna=False).to_string())


def main():
    print("WESTERN GHATS HISTORICAL RAINFALL COVERAGE AUDIT")
    print("Read-only: no files will be modified.")

    if EVENTS.exists():
        events = pd.read_csv(EVENTS)
        print("\n--- Historical event features ---")
        print("Rows:", len(events))
        print("Columns:", ", ".join(events.columns))

        for col in [
            "event_date", "event_time", "date",
            "time", "event_timestamp",
        ]:
            if col in events.columns:
                parsed = pd.to_datetime(events[col], errors="coerce")
                valid = parsed.dropna()
                print(f"{col}: {len(valid)}/{len(events)} parseable")
                if not valid.empty:
                    print("  Earliest:", valid.min())
                    print("  Latest:", valid.max())

        if "sample_source" in events.columns:
            print("\nSample sources:")
            print(events["sample_source"].value_counts().to_string())
    else:
        print("\nEvent features file not found:", EVENTS)

    for path in FILES_TO_CHECK:
        inspect_file(path)

    print("\nAUDIT COMPLETE.")
    print(
        "Note: a file containing recent model weather or forecasts "
        "does not prove that archived rainfall exists for historical "
        "landslide dates."
    )


if __name__ == "__main__":
    main()
