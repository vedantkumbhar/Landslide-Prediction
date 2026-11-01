from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/pilot_event_terrain_rainfall_features.csv"
OUTPUT = ROOT / "data/processed/pilot_model_event_features.csv"

df = pd.read_csv(INPUT)

required = [
    "event_id", "event_date_parsed", "pilot_tile",
    "latitude", "longitude", "elevation_m",
    "slope_degrees", "aspect_degrees",
    "previous_day_rainfall_mm", "event_day_rainfall_mm",
    "terrain_valid", "rainfall_matched",
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Required columns missing: {missing}")

df = df.loc[
    df["terrain_valid"].astype(str).str.lower().eq("true")
    & df["rainfall_matched"].astype(str).str.lower().eq("true")
].copy()

# Event-day rainfall may include rain after the landslide.
# Keep it for analysis, but explicitly flag the timing limitation.
df["event_day_rainfall_is_confirmed_pre_event"] = False
df["previous_day_rainfall_is_confirmed_pre_event"] = True

df["event_day_rainfall_is_confirmed_pre_event"] = False
df["rainfall_source"] = "Open-Meteo Historical Weather API"
df["label_type"] = "recorded_landslide_event"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nEVENT FEATURE DATASET")
print("Output:", OUTPUT)
print("Rows:", len(df))
print("Unique event IDs:", df["event_id"].nunique())
print("Pilot tiles:")
print(df["pilot_tile"].value_counts().to_string())
print("\nMissing feature values:")
print(df[[
    "elevation_m", "slope_degrees", "aspect_degrees",
    "previous_day_rainfall_mm", "event_day_rainfall_mm"
]].isna().sum().to_string())
print("\nNote: event-only data cannot train a standard binary classifier.")
print("Background samples remain unverified and are not used as negatives.")
