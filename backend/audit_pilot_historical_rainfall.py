
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Pilot events": ROOT / "data/processed/pilot_landslide_event_features.csv",
    "Historical rainfall": ROOT / "data/processed/rainfall/pilot_historical_event_rainfall.csv",
    "Terrain rainfall features": ROOT / "data/processed/pilot_event_terrain_rainfall_features.csv",
    "Model event features": ROOT / "data/processed/pilot_model_event_features.csv",
}

print("\nPILOT HISTORICAL RAINFALL AUDIT")
print("=" * 70)

data = {}

for name, path in FILES.items():
    print(f"\n{name}")
    print("-" * 70)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path, low_memory=False)
    data[name] = df

    print("File:", path.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    for col in [
        "event_id",
        "event_date",
        "event_date_parsed",
        "previous_day_rainfall_mm",
        "event_day_rainfall_mm",
        "previous_day_hourly_records",
        "event_day_hourly_records",
        "status",
        "rainfall_matched",
        "terrain_valid",
    ]:
        if col in df.columns:
            print(f"\n{col}:")
            print("  Non-null:", df[col].notna().sum())
            print("  Unique:", df[col].nunique())

# Detailed rainfall output audit
rain = data.get("Historical rainfall")

if rain is not None:
    print("\n\nRAINFALL EXTRACTION SUMMARY")
    print("=" * 70)

    if "status" in rain.columns:
        print("\nExtraction statuses:")
        print(rain["status"].value_counts(dropna=False).to_string())

    for col in [
        "previous_day_rainfall_mm",
        "event_day_rainfall_mm",
        "previous_day_hourly_records",
        "event_day_hourly_records",
    ]:
        if col in rain.columns:
            values = pd.to_numeric(rain[col], errors="coerce")
            print(f"\n{col}")
            print("Valid:", values.notna().sum())
            print("Missing:", values.isna().sum())
            print("Minimum:", values.min())
            print("Maximum:", values.max())
            print("Mean:", values.mean())

    if "event_date" in rain.columns:
        dates = pd.to_datetime(rain["event_date"], errors="coerce")
        print("\nEvent date coverage:")
        print("Earliest:", dates.min())
        print("Latest:", dates.max())

    if "event_id" in rain.columns:
        print("\nUnique event IDs:", rain["event_id"].nunique())
        print("Duplicate event IDs:", rain["event_id"].duplicated().sum())

# Compare event IDs between stages
print("\n\nEVENT ID CONSISTENCY")
print("=" * 70)

for name, df in data.items():
    if "event_id" in df.columns:
        ids = set(df["event_id"].dropna().astype(str))
        print(f"{name}: {len(ids)} unique event IDs")

print("\nAUDIT COMPLETE")
print("Read-only inspection. No files modified.")
