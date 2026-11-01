
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data/processed/rainfall/western_ghats/historical_validation"
    / "western_ghats_rainfall_terrain_experimental_dataset.csv"
)

OUTPUT_FILE = (
    INPUT_FILE.parent / "western_ghats_experimental_terrain_audit.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing report: {OUTPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "sample_key",
    "sample_role",
    "pilot_tile",
    "terrain_source_tile",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("=" * 75)
print("EXPERIMENTAL TERRAIN VALUE AUDIT")
print("=" * 75)

for column in ["elevation_m", "slope_degrees", "aspect_degrees"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nOverall terrain statistics:")
print(
    df[
        ["elevation_m", "slope_degrees", "aspect_degrees"]
    ].describe().to_string()
)

print("\nTerrain statistics by tile and role:")
summary = df.groupby(
    ["pilot_tile", "sample_role"]
)[
    ["elevation_m", "slope_degrees", "aspect_degrees"]
].agg(["min", "max", "mean", "count"])

print(summary.to_string())

# Flag suspicious values without modifying or deleting them.
flags = (
    df["elevation_m"].lt(0)
    | df["slope_degrees"].lt(0)
    | df["slope_degrees"].gt(90)
    | df["aspect_degrees"].lt(-1)
    | df["aspect_degrees"].ge(360)
)

flagged = df.loc[
    flags,
    [
        "sample_key",
        "sample_role",
        "pilot_tile",
        "terrain_source_tile",
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
    ],
].copy()

print(f"\nFlagged records: {len(flagged)}")

if flagged.empty:
    print("No values matched the configured screening rules.")
else:
    print(flagged.to_string(index=False))

flagged.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved flagged-record report: {OUTPUT_FILE}")

print("\nNo terrain values were changed.")
print("Negative elevation is flagged for review, not automatically invalid.")
print("Aspect -1 may represent a flat cell in the source processing.")
