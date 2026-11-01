
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

df = pd.read_csv(FILE)

terrain = ["elevation_m", "slope_degrees", "aspect_degrees"]
expected_tiles = {"N09E076", "N11E076", "N19E072"}

print("=" * 65)
print("WESTERN GHATS TRAINING DATASET VALIDATION")
print("=" * 65)

print(f"Total rows: {len(df)}")
print(f"Missing terrain values: {int(df[terrain].isna().sum().sum())}")
print(
    "Non-finite terrain values:",
    int((~np.isfinite(df[terrain].to_numpy(dtype=float))).sum()),
)

print("\nLabel counts:")
print(df["label"].value_counts(dropna=False).sort_index().to_string())

print("\nTile and label counts:")
print(
    df.groupby(["pilot_tile", "label"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nDuplicate coordinates:")
print(
    int(df.duplicated(["pilot_tile", "longitude", "latitude"]).sum())
)

print("\nChecks:")
print("Expected pilot tiles present:", set(df["pilot_tile"]) == expected_tiles)
print("Only labels 0 and 1:", set(df["label"].dropna().unique()) == {0, 1})
print("Slope range valid:", bool(df["slope_degrees"].between(0, 90).all()))

aspect = df["aspect_degrees"]
aspect_valid = (
    aspect.between(0, 360, inclusive="left") | aspect.eq(-1)
)
print("Aspect range valid:", bool(aspect_valid.all()))

print("\nValidation complete. No files modified.")
