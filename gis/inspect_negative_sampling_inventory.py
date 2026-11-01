
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "India inventory": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "india_landslide_events.csv"
    ),
    "Pilot event features": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pilot_landslide_event_features.csv"
    ),
}

for name, path in FILES.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    for column in (
        "longitude",
        "latitude",
        "location_accuracy",
        "location_description",
        "event_date_parsed",
        "event_id",
        "pilot_tile",
    ):
        if column not in df.columns:
            continue

        print(f"\n--- {column} ---")
        print("Missing:", int(df[column].isna().sum()))

        if column in ("longitude", "latitude"):
            print(df[column].describe())
        else:
            print(df[column].value_counts(dropna=False).head(12))

print("\nInspection complete. No files were modified.")
