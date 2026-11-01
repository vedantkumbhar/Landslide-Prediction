
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed" / "rainfall"

FILES = {
    "3H": DATA / "gpm_regional_timeseries_3h_01nov2026.csv",
    "6H": DATA / "gpm_regional_timeseries_6h_01nov2026.csv",
    "12H": DATA / "gpm_regional_timeseries_12h_01nov2026.csv",
    "24H": DATA / "gpm_regional_timeseries_24h_01nov2026.csv",
}

RAIN_COLUMNS = {
    "3H": "gpm_3h_accumulation_mm",
    "6H": "gpm_6h_accumulation_mm",
    "12H": "gpm_12h_accumulation_mm",
    "24H": "gpm_24h_accumulation_mm",
}

datasets = {}

print("\nCORRECTED GPM ALIGNMENT AUDIT")
print("=" * 65)

# 1. Load and validate identifiers
for name, path in FILES.items():
    df = pd.read_csv(path)

    if "record_id" not in df.columns:
        raise ValueError(f"{name}: record_id is missing")

    if df["record_id"].isna().any():
        raise ValueError(f"{name}: missing record_id values")

    if df["record_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate record_id values")

    datasets[name] = df

    print(f"\n{name}: {len(df)} rows")
    print("Unique record IDs:", df["record_id"].nunique())
    print("Start:", df["gpm_series_start_utc"].iloc[0])
    print("End:", df["gpm_series_end_utc"].iloc[0])

# 2. Compare record IDs and their ordering
print("\n\nRECORD ID ALIGNMENT")
print("=" * 65)

reference = datasets["24H"]
reference_ids = reference["record_id"].tolist()

for name, df in datasets.items():
    ids = df["record_id"].tolist()

    print(f"\n{name}:")
    print("Same ID set:", set(ids) == set(reference_ids))
    print("Same row order:", ids == reference_ids)
    print("Missing IDs:", len(set(reference_ids) - set(ids)))
    print("Extra IDs:", len(set(ids) - set(reference_ids)))

# 3. Compare stable spatial and terrain fields
print("\n\nSPATIAL AND TERRAIN CONSISTENCY")
print("=" * 65)

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
    merged = reference[
        ["record_id"] + stable_columns
    ].merge(
        df[["record_id"] + stable_columns],
        on="record_id",
        suffixes=("_ref", "_current"),
        validate="one_to_one",
    )

    mismatches = {}

    for col in stable_columns:
        a = merged[f"{col}_ref"]
        b = merged[f"{col}_current"]

        if pd.api.types.is_numeric_dtype(a):
            equal = np.isclose(
                pd.to_numeric(a, errors="coerce"),
                pd.to_numeric(b, errors="coerce"),
                rtol=0,
                atol=1e-8,
                equal_nan=True,
            )
        else:
            equal = (
                a.fillna("<NA>").astype(str)
                == b.fillna("<NA>").astype(str)
            )

        mismatches[col] = int((~equal).sum())

    print(f"\n{name}:")
    for col, count in mismatches.items():
        print(f"{col} mismatches: {count}")

# 4. Check rainfall columns
print("\n\nRAINFALL FEATURE CHECK")
print("=" * 65)

for name, df in datasets.items():
    col = RAIN_COLUMNS[name]
    values = pd.to_numeric(df[col], errors="coerce")

    print(f"\n{name}: {col}")
    print("Valid:", values.notna().sum())
    print("Missing:", values.isna().sum())
    print("Negative:", int((values < 0).sum()))
    print("Mean:", round(values.mean(), 6))
    print("Maximum:", round(values.max(), 6))

# 5. Check whether longer accumulations are at least
# as large as shorter accumulations for matching locations.
# This is a diagnostic, not proof of temporal alignment.
print("\n\nACCUMULATION NESTING DIAGNOSTIC")
print("=" * 65)

rain = reference[["record_id", "gpm_24h_accumulation_mm"]].copy()

for name in ["3H", "6H", "12H"]:
    col = RAIN_COLUMNS[name]
    temp = datasets[name][["record_id", col]]

    rain = rain.merge(
        temp,
        on="record_id",
        how="inner",
        validate="one_to_one",
    )

comparisons = [
    ("3H vs 6H", "gpm_3h_accumulation_mm",
     "gpm_6h_accumulation_mm"),
    ("6H vs 12H", "gpm_6h_accumulation_mm",
     "gpm_12h_accumulation_mm"),
    ("12H vs 24H", "gpm_12h_accumulation_mm",
     "gpm_24h_accumulation_mm"),
]

for label, shorter, longer in comparisons:
    valid = rain[[shorter, longer]].dropna()
    violations = (
        valid[shorter] > valid[longer] + 1e-6
    ).sum()

    print(f"{label}:")
    print("Comparable records:", len(valid))
    print("Longer-window violations:", int(violations))

print("\nAUDIT COMPLETE")
print("No source datasets were modified.")
