
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "rainfall"

FILES = {
    "3H": DATA_DIR / "gpm_regional_timeseries_3h_01nov2026.csv",
    "6H": DATA_DIR / "gpm_regional_timeseries_6h_01nov2026.csv",
    "12H": DATA_DIR / "gpm_regional_timeseries_12h_01nov2026.csv",
    "24H": DATA_DIR / "gpm_regional_timeseries_24h_01nov2026.csv",
}

RAINFALL_COLUMNS = {
    "3H": "rainfall_3h_mm",
    "6H": "rainfall_6h_mm",
    "12H": "rainfall_12h_mm",
    "24H": "rainfall_24h_mm",
}

print("\nGPM TEMPORAL AND SPATIAL ALIGNMENT VALIDATION")
print("=" * 65)

datasets = {}

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    datasets[name] = df

    print(f"\n{name} DATASET")
    print("-" * 65)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    for col in [
        "sample_id",
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
        "gpm_series_start_utc",
        "gpm_series_end_utc",
    ]:
        if col in df.columns:
            print(f"{col}: available")
        else:
            print(f"{col}: not available")

# --------------------------------------------------
# 2. Check accumulation windows
# --------------------------------------------------

print("\n\nTEMPORAL ALIGNMENT")
print("=" * 65)

for name, df in datasets.items():
    start_col = "gpm_series_start_utc"
    end_col = "gpm_series_end_utc"

    print(f"\n{name}:")
    if start_col in df.columns:
        print("Start timestamps:")
        print(df[start_col].dropna().unique()[:5])

    if end_col in df.columns:
        print("End timestamps:")
        print(df[end_col].dropna().unique()[:5])

# --------------------------------------------------
# 3. Identify common spatial key
# --------------------------------------------------

print("\n\nSPATIAL ALIGNMENT")
print("=" * 65)

reference_name = "24H"
reference = datasets[reference_name]

key_candidates = [
    ["sample_id"],
    ["record_id"],
    ["latitude", "longitude", "dem_region"],
    ["latitude", "longitude"],
]

common_key = None

for candidate in key_candidates:
    if all(
        all(col in df.columns for col in candidate)
        for df in datasets.values()
    ):
        common_key = candidate
        break

if common_key is None:
    print("No common spatial key found.")
    print("Manual schema inspection is required.")
else:
    print("Selected spatial key:", common_key)

    for name, df in datasets.items():
        reference_keys = reference[common_key].astype(str)
        current_keys = df[common_key].astype(str)

        same_count = (
            reference_keys.isin(set(current_keys)).sum()
        )

        print(
            f"{name}: {same_count}/{len(reference)} "
            "reference records found"
        )

        if df.duplicated(common_key).any():
            print(
                "WARNING: Duplicate records found for "
                f"key {common_key}"
            )
        else:
            print("No duplicate records for selected key.")

# --------------------------------------------------
# 4. Compare rainfall values across windows
# --------------------------------------------------

print("\n\nRAINFALL VALUE CHECK")
print("=" * 65)

for name, df in datasets.items():
    column = RAINFALL_COLUMNS[name]

    if column not in df.columns:
        print(f"{name}: Missing expected column {column}")
        print("Available rainfall columns:")
        print([
            c for c in df.columns
            if "rain" in c.lower() or "precip" in c.lower()
        ])
        continue

    values = pd.to_numeric(df[column], errors="coerce")

    print(f"\n{name}: {column}")
    print("Valid:", values.notna().sum())
    print("Missing:", values.isna().sum())
    print("Negative values:", (values < 0).sum())
    print("Minimum:", values.min())
    print("Maximum:", values.max())

# --------------------------------------------------
# 5. Check duplicate coordinates
# --------------------------------------------------

print("\n\nDUPLICATE COORDINATE INVESTIGATION")
print("=" * 65)

for name, df in datasets.items():
    coordinate_columns = ["latitude", "longitude"]

    if not all(c in df.columns for c in coordinate_columns):
        print(f"{name}: Coordinate columns unavailable.")
        continue

    duplicates = df[
        df.duplicated(coordinate_columns, keep=False)
    ].sort_values(coordinate_columns)

    print(f"\n{name}:")
    print("Duplicate coordinate rows:", len(duplicates))

    if not duplicates.empty:
        display_columns = [
            c for c in [
                "sample_id",
                "record_id",
                "latitude",
                "longitude",
                "dem_region",
            ]
            if c in duplicates.columns
        ]

        print(duplicates[display_columns].head(20).to_string(index=False))

# --------------------------------------------------
# 6. Final status
# --------------------------------------------------

print("\n\nVALIDATION FINISHED")
print("=" * 65)
print("Read-only audit completed.")
print("No source CSV files were modified.")
