
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_landslide_event_features.csv"
)

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "event_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "terrain_valid",
    "pilot_tile",
]

print("=" * 55)
print("TRAINING FEATURE QUALITY CHECK")
print("=" * 55)

print("\n1. Required columns")
for column in required_columns:
    print(f"{column}: {'FOUND' if column in df.columns else 'MISSING'}")

if not all(column in df.columns for column in required_columns):
    raise SystemExit("\nRequired columns are missing. Stop and inspect the CSV.")

print("\n2. Missing values in required columns")
print(df[required_columns].isnull().sum().to_string())

print("\n3. Coordinate validity")
valid_coordinates = (
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
)
print(f"Valid coordinate records: {valid_coordinates.sum()}/{len(df)}")

print("\n4. Terrain validity flag")
print(df["terrain_valid"].value_counts(dropna=False).to_string())

print("\n5. Terrain feature summary")
print(
    df[["elevation_m", "slope_degrees", "aspect_degrees"]]
    .describe()
    .round(3)
    .to_string()
)

print("\n6. Physical range checks")
print(
    "Negative elevation records:",
    (df["elevation_m"] < 0).sum()
)
print(
    "Slope outside 0-90 degrees:",
    (~df["slope_degrees"].between(0, 90)).sum()
)
print(
    "Aspect outside 0-360 degrees:",
    (~df["aspect_degrees"].between(0, 360)).sum()
)

print("\n7. Records by pilot tile")
print(df["pilot_tile"].value_counts(dropna=False).to_string())

print("\nCHECK COMPLETED")
