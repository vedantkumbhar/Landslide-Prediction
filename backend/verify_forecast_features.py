import pandas as pd
from pathlib import Path

folder = Path("data/processed/rainfall")
raw = pd.read_csv(folder / "pilot_hourly_rainfall.csv")
features = pd.read_csv(folder / "pilot_rainfall_features.csv")

keys = [
    "region",
    "requested_latitude",
    "requested_longitude",
]

raw["timestamp_local"] = pd.to_datetime(raw["timestamp_local"])
features["timestamp_local"] = pd.to_datetime(features["timestamp_local"])

merged = features.merge(
    raw[keys + ["timestamp_local", "precipitation_mm"]],
    on=keys + ["timestamp_local"],
    how="left",
    validate="one_to_one",
)

merged = merged.sort_values(keys + ["timestamp_local"])

total_checked = 0
total_mismatches = 0

print("FORECAST FEATURE VERIFICATION")

for _, group in merged.groupby(keys, sort=False):
    group = group.sort_values("timestamp_local")
    rainfall = group["precipitation_mm"].reset_index(drop=True)

    for hours in [3, 6, 24]:
        column = f"forecast_rainfall_{hours}h_mm"

        expected = (
            rainfall.shift(-1)
            .rolling(hours, min_periods=hours)
            .sum()
            .shift(-(hours - 1))
        )

        actual = group[column].reset_index(drop=True)
        valid = actual.notna() & expected.notna()

        mismatches = (
            (actual[valid] - expected[valid]).abs() > 1e-6
        ).sum()

        checked = int(valid.sum())
        mismatches = int(mismatches)

        print(
            f"{group['region'].iloc[0]} | "
            f"{hours}h: checked={checked}, "
            f"mismatches={mismatches}"
        )

        total_checked += checked
        total_mismatches += mismatches

print(f"\nTOTAL checked: {total_checked}")
print(f"TOTAL mismatches: {total_mismatches}")
