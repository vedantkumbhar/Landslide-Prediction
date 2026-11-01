
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data/processed/rainfall/western_ghats/historical_validation"
    / "western_ghats_rainfall_terrain_experimental_dataset.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = [
    "sample_key",
    "matched_event_id",
    "control_id",
    "background_sample_id",
    "sample_role",
    "target_label",
    "control_status",
    "event_time_status",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "pre_event_24h_rainfall_mm",
    "pre_event_24h_valid_hours",
    "pre_event_72h_rainfall_mm",
    "pre_event_72h_valid_hours",
    "event_day_rainfall_included",
]

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("=" * 75)
print("FINAL DATASET INTEGRITY VALIDATION")
print("=" * 75)

print(f"\nRows: {len(df)}")
print(f"Unique sample keys: {df['sample_key'].nunique()}")
print(f"Duplicate sample keys: {df['sample_key'].duplicated().sum()}")

print("\nRole counts:")
print(df["sample_role"].value_counts(dropna=False).to_string())

print("\nLabel counts by role:")
print(
    df.groupby(["sample_role", "target_label"], dropna=False)
    .size()
    .to_string()
)

# Check one-to-one matching: each event should have one control.
pairs = df[["matched_event_id", "control_id", "sample_role"]].drop_duplicates()

print("\nUnique matched events:", df["matched_event_id"].nunique())
print("Unique controls:", df["control_id"].nunique())

for role in ["historical_event", "background_control"]:
    subset = df[df["sample_role"] == role]
    print(f"{role} rows: {len(subset)}")

# Verify terrain and rainfall ranges.
for column in [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "pre_event_24h_rainfall_mm",
    "pre_event_72h_rainfall_mm",
]:
    values = pd.to_numeric(df[column], errors="coerce")
    print(
        f"\n{column}: missing={values.isna().sum()}, "
        f"min={values.min()}, max={values.max()}"
    )

# Verify complete 72-hour coverage.
rain72 = pd.to_numeric(df["pre_event_72h_rainfall_mm"], errors="coerce")
hours72 = pd.to_numeric(df["pre_event_72h_valid_hours"], errors="coerce")

invalid72 = rain72.notna() & hours72.ne(72)
print("\n72-hour totals with invalid coverage:", int(invalid72.sum()))

# Verify 24-hour completeness.
rain24 = pd.to_numeric(df["pre_event_24h_rainfall_mm"], errors="coerce")
hours24 = pd.to_numeric(df["pre_event_24h_valid_hours"], errors="coerce")

invalid24 = rain24.notna() & hours24.ne(24)
print("24-hour totals with invalid coverage:", int(invalid24.sum()))

ambiguous = df["event_time_status"].astype(str).str.contains(
    "ambiguous", case=False, na=False
)
ambiguous_with_24h = ambiguous & rain24.notna()

print(
    "Ambiguous-time rows with 24-hour rainfall retained:",
    int(ambiguous_with_24h.sum()),
)

# Event-day rainfall must not be included as a predictive feature.
print(
    "\nEvent-day rainfall included flags:",
    df["event_day_rainfall_included"].value_counts(dropna=False).to_dict(),
)

# Check paired event/control rainfall windows.
wide = df.pivot(
    index="matched_event_id",
    columns="sample_role",
    values=[
        "pre_event_24h_rainfall_mm",
        "pre_event_72h_rainfall_mm",
    ],
)

for metric in [
    "pre_event_24h_rainfall_mm",
    "pre_event_72h_rainfall_mm",
]:
    event_values = wide[(metric, "historical_event")]
    control_values = wide[(metric, "background_control")]

    complete = event_values.notna() & control_values.notna()

    differences = (
        event_values[complete] - control_values[complete]
    ).abs()

    print(
        f"\n{metric}: paired complete={int(complete.sum())}, "
        f"maximum absolute difference={differences.max() if len(differences) else 'N/A'}"
    )

print("\nValidation complete.")
print("This checks data integrity, not predictive performance.")
