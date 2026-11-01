from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "rainfall_features.csv"
)

df = pd.read_csv(PATH)

keys = [
    "pilot_tile",
    "requested_latitude",
    "requested_longitude",
    "api_latitude",
    "api_longitude",
    "timestamp_local",
]

required = keys + [
    "data_period",
    "recent_rainfall_1h_mm",
    "recent_rainfall_3h_mm",
    "recent_rainfall_6h_mm",
    "recent_rainfall_12h_mm",
    "recent_rainfall_24h_mm",
    "recent_records_24h",
    "forecast_rainfall_3h_mm",
    "forecast_rainfall_6h_mm",
    "forecast_rainfall_24h_mm",
]

missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Rows:", len(df))
print("Duplicate point-timestamps:", int(df.duplicated(keys).sum()))
print("Monitoring points:", df[keys[:5]].drop_duplicates().shape[0])
print("\nRows by tile and period:")
print(pd.crosstab(df["pilot_tile"], df["data_period"]).to_string())

feature_cols = [
    c for c in required
    if "rainfall_" in c
]

for col in feature_cols:
    values = pd.to_numeric(df[col], errors="coerce")
    finite = values.dropna()
    invalid = int((finite < 0).sum())
    print(
        f"{col}: available={len(finite)}, "
        f"missing={int(values.isna().sum())}, "
        f"negative={invalid}"
    )
    if invalid:
        raise ValueError(f"Negative values in {col}")

recent_cols = [
    "recent_rainfall_1h_mm",
    "recent_rainfall_3h_mm",
    "recent_rainfall_6h_mm",
    "recent_rainfall_12h_mm",
    "recent_rainfall_24h_mm",
]
forecast_cols = [
    "forecast_rainfall_3h_mm",
    "forecast_rainfall_6h_mm",
    "forecast_rainfall_24h_mm",
]

if df.loc[df["data_period"] == "forecast", recent_cols].notna().any().any():
    raise ValueError("Forecast rows unexpectedly contain recent rainfall features.")

if df.loc[
    df["data_period"] == "historical_or_current", forecast_cols
].notna().any().any():
    raise ValueError("Historical/current rows unexpectedly contain forecast features.")

print("\nPASS: structural, non-negative, and period-separation checks.")
print("Note: this does not independently verify the meteorological accuracy of the API data.")
