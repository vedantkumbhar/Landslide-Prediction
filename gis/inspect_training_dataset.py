
from pathlib import Path
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Existing pilot dataset
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_landslide_event_features.csv"
)

print("=" * 60)
print("LANDSLIDE TRAINING DATASET INSPECTION")
print("=" * 60)

if not DATASET_PATH.exists():
    print(f"\nERROR: Dataset not found:\n{DATASET_PATH}")
    raise SystemExit(1)

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset path: {DATASET_PATH}")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\n--- COLUMN NAMES ---")
for column in df.columns:
    print(f"- {column}")

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "No missing values")

print("\n--- DUPLICATE EVENT IDs ---")
if "event_id" in df.columns:
    print(f"Duplicate event IDs: {df['event_id'].duplicated().sum()}")
else:
    print("No 'event_id' column found; check the column names.")

print("\n--- REGIONAL DISTRIBUTION ---")
if "pilot_tile" in df.columns:
    print(df["pilot_tile"].value_counts(dropna=False).to_string())
else:
    print("No 'pilot_tile' column found.")

print("\n--- SAMPLE RECORDS ---")
print(df.head(5).to_string(index=False))

print("\n--- COMPLETED ---")
