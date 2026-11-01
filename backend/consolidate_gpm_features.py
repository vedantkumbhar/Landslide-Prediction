
from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "rainfall"

FILES = {
    "3H": DATA_DIR / "gpm_regional_timeseries_3h_01nov2026.csv",
    "6H": DATA_DIR / "gpm_regional_timeseries_6h_01nov2026.csv",
    "12H": DATA_DIR / "gpm_regional_timeseries_12h_01nov2026.csv",
    "24H": DATA_DIR / "gpm_regional_timeseries_24h_01nov2026.csv",
}

OUTPUT = DATA_DIR / "gpm_consolidated_features_01nov2026.csv"

KEY = "record_id"

# Actual intervals verified from HDF5 time_bnds.
WINDOWS = {
    "3H": ("2026-11-01T08:30:00Z", "2026-11-01T11:30:00Z"),
    "6H": ("2026-11-01T06:00:00Z", "2026-11-01T12:00:00Z"),
    "12H": ("2026-11-01T00:00:00Z", "2026-11-01T12:00:00Z"),
    "24H": ("2026-10-31T12:00:00Z", "2026-11-01T12:00:00Z"),
}

print("\nGPM FEATURE CONSOLIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. PROTECT EXISTING FILES
# --------------------------------------------------

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "No file was overwritten. Rename the intended output "
        "or choose a new filename before running again."
    )

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

datasets = {}

for name, path in FILES.items():
    df = pd.read_csv(path)

    if KEY not in df.columns:
        raise ValueError(f"{name}: missing {KEY}")

    if df[KEY].isna().any():
        raise ValueError(f"{name}: missing record IDs")

    if df[KEY].duplicated().any():
        raise ValueError(f"{name}: duplicate record IDs")

    datasets[name] = df

    print(f"{name}: loaded {len(df)} records")

# Use 24H as the base because it contains the latest
# common observation and the satellite matching metadata.
base = datasets["24H"].copy()

reference_ids = base[KEY].tolist()

# --------------------------------------------------
# 3. VERIFY IDENTIFIER ALIGNMENT
# --------------------------------------------------

for name, df in datasets.items():
    if set(df[KEY]) != set(reference_ids):
        raise ValueError(
            f"{name}: record IDs do not match the 24H dataset"
        )

    print(f"{name}: record ID set matches")

# --------------------------------------------------
# 4. VERIFY SPATIAL AND TERRAIN CONSISTENCY
# --------------------------------------------------

stable_columns = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "dem_region",
    "tile",
]

for name, df in datasets.items():
    comparison = base[[KEY] + stable_columns].merge(
        df[[KEY] + stable_columns],
        on=KEY,
        suffixes=("_base", "_other"),
        validate="one_to_one",
    )

    for col in stable_columns:
        left = comparison[f"{col}_base"]
        right = comparison[f"{col}_other"]

        if pd.api.types.is_numeric_dtype(left):
            equal = np.isclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=0,
                atol=1e-8,
                equal_nan=True,
            )
        else:
            equal = (
                left.fillna("<NA>").astype(str)
                == right.fillna("<NA>").astype(str)
            )

        if not equal.all():
            raise ValueError(
                f"{name}: mismatch in {col}"
            )

    print(f"{name}: spatial and terrain fields match")

# --------------------------------------------------
# 5. PREPARE BASE DATASET
# --------------------------------------------------

# The 24H dataset supplies:
# - identity, label, terrain and regional fields
# - latest rainfall rate and quality information
# - 1H and 24H rainfall accumulations
# - satellite grid matching metadata

drop_from_base = [
    "gpm_series_start_utc",
    "gpm_series_end_utc",
    "gpm_3h_accumulation_mm",
    "gpm_6h_accumulation_mm",
    "gpm_12h_accumulation_mm",
]

base = base.drop(
    columns=[
        col for col in drop_from_base
        if col in base.columns
    ]
)

# --------------------------------------------------
# 6. ADD 3H, 6H AND 12H ACCUMULATIONS
# --------------------------------------------------

feature_sources = {
    "3H": {
        "rainfall": "gpm_3h_accumulation_mm",
        "intervals": "gpm_valid_30min_intervals",
        "output_intervals": "gpm_3h_valid_intervals",
    },
    "6H": {
        "rainfall": "gpm_6h_accumulation_mm",
        "intervals": "gpm_valid_30min_intervals",
        "output_intervals": "gpm_6h_valid_intervals",
    },
    "12H": {
        "rainfall": "gpm_12h_accumulation_mm",
        "intervals": "gpm_interval_count",
        "output_intervals": "gpm_12h_valid_intervals",
    },
}

for name, config in feature_sources.items():
    df = datasets[name]

    required = [
        KEY,
        config["rainfall"],
        config["intervals"],
    ]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name}: missing columns {missing_columns}"
        )

    features = df[required].copy()

    features = features.rename(
        columns={
            config["intervals"]: config["output_intervals"]
        }
    )

    base = base.merge(
        features,
        on=KEY,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    print(f"{name}: rainfall features merged")

# --------------------------------------------------
# 7. ADD VERIFIED WINDOW METADATA
# --------------------------------------------------

for name, (start, end) in WINDOWS.items():
    prefix = f"gpm_{name.lower()}_window"

    base[f"{prefix}_start_utc"] = start
    base[f"{prefix}_end_utc"] = end

# Explicit common observation end.
base["gpm_common_observation_end_utc"] = (
    "2026-11-01T12:00:00Z"
)

# --------------------------------------------------
# 8. VALIDATE CONSOLIDATED DATASET
# --------------------------------------------------

required_rainfall_columns = [
    "gpm_latest_rainfall_rate_mm_hr",
    "gpm_1h_accumulation_mm",
    "gpm_3h_accumulation_mm",
    "gpm_6h_accumulation_mm",
    "gpm_12h_accumulation_mm",
    "gpm_24h_accumulation_mm",
]

print("\nFINAL VALIDATION")
print("-" * 70)

if len(base) != 2009:
    raise ValueError(
        f"Unexpected row count: {len(base)}"
    )

if base[KEY].duplicated().any():
    raise ValueError("Duplicate record IDs after merging")

if base[KEY].isna().any():
    raise ValueError("Missing record IDs after merging")

for col in required_rainfall_columns:
    if col not in base.columns:
        raise ValueError(f"Missing final feature: {col}")

    base[col] = pd.to_numeric(base[col], errors="coerce")

    valid = int(base[col].notna().sum())
    missing = int(base[col].isna().sum())

    print(f"\n{col}")
    print("Valid:", valid)
    print("Missing:", missing)
    print("Mean:", round(base[col].mean(), 6))
    print("Maximum:", round(base[col].max(), 6))

    if missing:
        raise ValueError(
            f"{col}: missing values found"
        )

# Accumulation nesting diagnostic.
comparisons = [
    ("3H", "gpm_3h_accumulation_mm",
     "gpm_6h_accumulation_mm"),
    ("6H", "gpm_6h_accumulation_mm",
     "gpm_12h_accumulation_mm"),
    ("12H", "gpm_12h_accumulation_mm",
     "gpm_24h_accumulation_mm"),
]

for label, shorter, longer in comparisons:
    violations = (
        base[shorter] > base[longer] + 1e-6
    ).sum()

    print(f"{label} nesting violations:", int(violations))

    if violations:
        raise ValueError(
            f"{label}: accumulation nesting failed"
        )

# --------------------------------------------------
# 9. SAVE NEW FILE
# --------------------------------------------------

# Final overwrite protection.
if OUTPUT.exists():
    raise FileExistsError(
        f"Output appeared before save: {OUTPUT}"
    )

base.to_csv(OUTPUT, index=False)

# --------------------------------------------------
# 10. FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 70)
print("GPM CONSOLIDATION COMPLETED")
print("=" * 70)

print("Output:", OUTPUT)
print("Rows:", len(base))
print("Columns:", len(base.columns))
print("Unique record IDs:", base[KEY].nunique())

print("\nRegional sample counts:")
print(base["dem_region"].value_counts().to_string())

print("\nFinal rainfall features:")
for col in required_rainfall_columns:
    print(f"- {col}")

print("\nOriginal four CSV files were not modified.")
print("No existing output was overwritten.")
