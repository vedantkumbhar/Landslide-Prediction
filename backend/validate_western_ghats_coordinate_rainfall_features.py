
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
)

RAW_PATH = BASE / "hourly_rainfall.csv"
FEATURE_PATH = BASE / "rainfall_features.csv"

RECENT_WINDOWS = [1, 3, 6, 12, 24]
FORECAST_WINDOWS = [3, 6, 24]


def main():
    raw = pd.read_csv(RAW_PATH)
    features = pd.read_csv(FEATURE_PATH)

    for frame in (raw, features):
        frame["timestamp_local"] = pd.to_datetime(
            frame["timestamp_local"], errors="raise"
        )

    key = ["sample_id", "timestamp_local"]

    if len(raw) != 78720 or len(features) != len(raw):
        raise ValueError("Unexpected raw or feature row count.")

    if features.duplicated(key).any():
        raise ValueError("Duplicate sample/timestamp rows.")

    check = raw[key + ["precipitation_mm"]].merge(
        features[key + [
            "precipitation_mm",
            *[f"recent_rain_{h}h_mm" for h in RECENT_WINDOWS],
            *[f"forecast_rain_{h}h_mm" for h in FORECAST_WINDOWS],
        ]],
        on=key,
        how="outer",
        suffixes=("_raw", "_feature"),
        validate="one_to_one",
        indicator=True,
    )

    if not (check["_merge"] == "both").all():
        raise ValueError("Raw and feature timestamp keys do not match.")

    if not np.allclose(
        check["precipitation_mm_raw"],
        check["precipitation_mm_feature"],
        equal_nan=True,
    ):
        raise ValueError("Raw rainfall values changed in feature output.")

    # Independently recalculate windows for representative samples.
    tested = 0

    for sample_id, group in raw.groupby("sample_id", sort=False):
        group = group.sort_values("timestamp_local").reset_index(drop=True)
        feat = features[features["sample_id"] == sample_id].sort_values(
            "timestamp_local"
        ).reset_index(drop=True)

        rain = group["precipitation_mm"].to_numpy(dtype=float)
        if len(rain) != 120:
            raise ValueError(f"{sample_id}: expected 120 records.")

        # Check the beginning, middle, and end of every sample.
        for i in sorted(set([0, 1, 2, 5, 11, 23, 48, 60, 96, 117, 118, 119])):
            for hours in RECENT_WINDOWS:
                column = f"recent_rain_{hours}h_mm"
                start = i - hours + 1

                expected = (
                    float(rain[start:i + 1].sum())
                    if start >= 0 else np.nan
                )
                actual = feat.loc[i, column]

                if pd.isna(expected):
                    if not pd.isna(actual):
                        raise ValueError(
                            f"{sample_id}, row {i}: {column} "
                            "should be missing."
                        )
                elif not np.isclose(actual, expected, atol=1e-8):
                    raise ValueError(
                        f"{sample_id}, row {i}: {column} misaligned."
                    )

            for hours in FORECAST_WINDOWS:
                column = f"forecast_rain_{hours}h_mm"
                end = i + hours

                expected = (
                    float(rain[i + 1:end + 1].sum())
                    if end < len(rain) else np.nan
                )
                actual = feat.loc[i, column]

                if pd.isna(expected):
                    if not pd.isna(actual):
                        raise ValueError(
                            f"{sample_id}, row {i}: {column} "
                            "should be missing."
                        )
                elif not np.isclose(actual, expected, atol=1e-8):
                    raise ValueError(
                        f"{sample_id}, row {i}: {column} misaligned."
                    )

            tested += 1

    print("PASS: rainfall feature alignment checks")
    print("Raw rows:", len(raw))
    print("Feature rows:", len(features))
    print("Samples:", features["sample_id"].nunique())
    print("Representative sample rows checked:", tested)

    print("\nFeature availability:")
    for hours in RECENT_WINDOWS:
        col = f"recent_rain_{hours}h_mm"
        print(f"{col}: {features[col].notna().sum()} available")

    for hours in FORECAST_WINDOWS:
        col = f"forecast_rain_{hours}h_mm"
        print(f"{col}: {features[col].notna().sum()} available")

    print("\nNo files were modified.")


if __name__ == "__main__":
    main()
