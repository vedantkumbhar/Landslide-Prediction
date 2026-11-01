
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "western_ghats_event_features.csv"

df = pd.read_csv(INPUT_FILE)

terrain_columns = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]

print("=" * 65)
print("WESTERN GHATS EVENT FEATURE VALIDATION")
print("=" * 65)

print(f"Total rows: {len(df)}")

if "event_id" in df.columns:
    print(f"Unique event IDs: {df['event_id'].nunique()}")
    print(f"Duplicate event IDs: {df['event_id'].duplicated().sum()}")
else:
    print("WARNING: event_id column is missing.")

print("\nTerrain completeness:")
print(df[terrain_columns].notna().sum().to_string())

complete = df[terrain_columns].notna().all(axis=1)
print(f"\nComplete terrain records: {int(complete.sum())}")

valid = df.loc[complete, terrain_columns].to_numpy(dtype=float)
print(f"All complete terrain values finite: {bool(np.isfinite(valid).all())}")

if complete.any():
    print("\nRange checks:")
    print(f"Elevation minimum: {df.loc[complete, 'elevation_m'].min():.3f} m")
    print(f"Elevation maximum: {df.loc[complete, 'elevation_m'].max():.3f} m")
    print(f"Slope minimum: {df.loc[complete, 'slope_degrees'].min():.3f}°")
    print(f"Slope maximum: {df.loc[complete, 'slope_degrees'].max():.3f}°")
    print(f"Aspect minimum: {df.loc[complete, 'aspect_degrees'].min():.3f}°")
    print(f"Aspect maximum: {df.loc[complete, 'aspect_degrees'].max():.3f}°")

    slope_ok = df.loc[complete, "slope_degrees"].between(0, 90).all()
    aspect_values = df.loc[complete, "aspect_degrees"]
    aspect_ok = (
        aspect_values.between(0, 360, inclusive="left")
        | aspect_values.eq(-1)
    ).all()

    print(f"Slope within 0–90°: {slope_ok}")
    print(f"Aspect within valid range or flat sentinel: {aspect_ok}")

print("\nRecords by tile:")
print(df["dem_tile"].value_counts(dropna=False).to_string())

print("\nValidation completed. Source dataset was not modified.")
