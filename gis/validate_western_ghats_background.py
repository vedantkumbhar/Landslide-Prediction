
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "processed" / "western_ghats_background_samples.csv"

df = pd.read_csv(FILE)

terrain = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]

print("=" * 65)
print("WESTERN GHATS BACKGROUND VALIDATION")
print("=" * 65)

print(f"Total samples: {len(df)}")
print(f"Unique sample IDs: {df['sample_id'].nunique()}")
print(f"Duplicate sample IDs: {df['sample_id'].duplicated().sum()}")

duplicates = df.duplicated(["pilot_tile", "longitude", "latitude"]).sum()
print(f"Duplicate tile/coordinate records: {duplicates}")

print("\nSamples by tile:")
print(df["pilot_tile"].value_counts().to_string())

print("\nMissing values:")
print(df[["latitude", "longitude", *terrain]].isna().sum().to_string())

values = df[terrain].to_numpy(dtype=float)
print(f"\nAll terrain values finite: {bool(np.isfinite(values).all())}")

print("\nTerrain ranges:")
print(df[terrain].describe().round(3).to_string())

print("\nRange checks:")
print(
    "Slope within 0–90°:",
    bool(df["slope_degrees"].between(0, 90).all()),
)

aspect_ok = (
    df["aspect_degrees"].between(0, 360, inclusive="left")
    | df["aspect_degrees"].eq(-1)
).all()
print("Aspect in valid range or flat sentinel:", bool(aspect_ok))

print("\nDistance check:")
print(
    "Samples at least 1 km from nearest recorded event:",
    bool((df["nearest_recorded_event_km"] >= 1.0).all()),
)
print(
    "Minimum recorded event distance (km):",
    round(float(df["nearest_recorded_event_km"].min()), 3),
)

print("\nValidation finished. No files modified.")
