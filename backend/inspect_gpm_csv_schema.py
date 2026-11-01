
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed" / "rainfall"

FILES = {
    "3H": DATA_DIR / "gpm_regional_timeseries_3h_01nov2026.csv",
    "6H": DATA_DIR / "gpm_regional_timeseries_6h_01nov2026.csv",
    "12H": DATA_DIR / "gpm_regional_timeseries_12h_01nov2026.csv",
    "24H": DATA_DIR / "gpm_regional_timeseries_24h_01nov2026.csv",
}

for name, path in FILES.items():
    print("\n" + "=" * 75)
    print(f"{name} DATASET SCHEMA")
    print("=" * 75)

    df = pd.read_csv(path)

    print("\nALL COLUMNS:")
    for col in df.columns:
        print(f"  {col}")

    print("\nIDENTIFIER CHECK:")
    for col in ["sample_id", "record_id"]:
        if col in df.columns:
            print(f"\n{col}")
            print("Missing:", df[col].isna().sum())
            print("Unique non-missing:", df[col].nunique())
            print("Duplicate non-missing:",
                  df[col].dropna().duplicated().sum())
            print("First 5:", df[col].head().tolist())

    print("\nGPM / RAINFALL / TIME COLUMNS:")
    for col in df.columns:
        if any(term in col.lower() for term in [
            "gpm", "rain", "precip", "time", "rate", "accum"
        ]):
            print(f"  {col}")

    print("\nFIRST TWO ROWS:")
    print(df.head(2).to_string(index=False))

print("\nSCHEMA INSPECTION COMPLETE")
print("No source CSV files were modified.")
