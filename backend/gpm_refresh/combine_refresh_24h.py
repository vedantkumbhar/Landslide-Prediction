
from pathlib import Path

import numpy as np
import pandas as pd

from backend.gpm_refresh.refresh_config import (
    PROJECT_ROOT,
    get_run_directory,
)


# Input files
PREVIOUS_CSV = (
    get_run_directory("20261102_previous12h")
    / "extracted"
    / "gpm_regional_rainfall_previous12h.csv"
)

LATEST_CSV = (
    get_run_directory("20261102_152742")
    / "extracted"
    / "gpm_regional_rainfall_12h.csv"
)

# Separate output file
OUTPUT_CSV = (
    get_run_directory("20261102_152742")
    / "extracted"
    / "gpm_regional_rainfall_24h.csv"
)


def main():

    print("\nNASA GPM: CONTINUOUS 24-HOUR DATASET")
    print("=" * 65)

    if not PREVIOUS_CSV.exists():
        raise FileNotFoundError(PREVIOUS_CSV)

    if not LATEST_CSV.exists():
        raise FileNotFoundError(LATEST_CSV)

    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_CSV}"
        )

    previous = pd.read_csv(PREVIOUS_CSV)
    latest = pd.read_csv(LATEST_CSV)

    print("Previous records:", len(previous))
    print("Latest records:", len(latest))

    # Validate record identifiers
    for name, frame in [
        ("Previous", previous),
        ("Latest", latest),
    ]:
        if frame["record_id"].duplicated().any():
            raise ValueError(
                f"Duplicate record IDs in {name} dataset."
            )

    if set(previous["record_id"]) != set(latest["record_id"]):
        raise ValueError(
            "Record IDs differ between the two datasets."
        )

    # Sort both datasets consistently
    previous = previous.sort_values("record_id").reset_index(drop=True)
    latest = latest.sort_values("record_id").reset_index(drop=True)

    # Verify spatial and regional alignment
    for column in [
        "latitude",
        "longitude",
        "dem_region",
    ]:
        if not previous[column].equals(latest[column]):
            raise ValueError(
                f"Mismatch in {column} between datasets."
            )

    # Verify exact temporal continuity
    previous_start = pd.Timestamp(
        previous["gpm_series_start_utc"].iloc[0]
    )

    previous_end = pd.Timestamp(
        previous["gpm_series_end_utc"].iloc[0]
    )

    latest_start = pd.Timestamp(
        latest["gpm_series_start_utc"].iloc[0]
    )

    latest_end = pd.Timestamp(
        latest["gpm_series_end_utc"].iloc[0]
    )

    if not (
        (previous_start == pd.Timestamp("2026-11-01T03:00:00Z"))
        and (previous_end == pd.Timestamp("2026-11-01T15:00:00Z"))
        and (latest_start == pd.Timestamp("2026-11-01T15:00:00Z"))
        and (latest_end == pd.Timestamp("2026-11-02T03:00:00Z"))
    ):
        raise ValueError(
            "Unexpected rainfall time windows."
        )

    if previous_end != latest_start:
        raise ValueError(
            "The two rainfall windows are not continuous."
        )

    if latest_end - previous_start != pd.Timedelta(hours=24):
        raise ValueError(
            "Combined coverage is not exactly 24 hours."
        )

    print("\nTemporal continuity: PASSED")
    print("24-hour start:", previous_start)
    print("24-hour end:", latest_end)

    # Validate required rainfall columns
    required = [
        "gpm_1h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_valid_30min_intervals",
    ]

    for column in required:
        if column not in previous.columns:
            raise ValueError(
                f"Missing previous feature: {column}"
            )

        if column not in latest.columns:
            raise ValueError(
                f"Missing latest feature: {column}"
            )

    # Start with latest-window dataset
    combined = latest.copy()

    # Preserve previous-window features with clear names
    previous_columns = [
        "gpm_latest_rate_mm_hr",
        "gpm_1h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_valid_30min_intervals",
        "gpm_latest_quality_index",
        "gpm_latest_random_error_mm_hr",
        "gpm_grid_match_distance_km",
    ]

    for column in previous_columns:
        if column in previous.columns:
            combined[
                "gpm_previous12h_" + column.replace("gpm_", "")
            ] = previous[column].to_numpy()

    # Calculate continuous 24-hour rainfall
    previous_12h = pd.to_numeric(
        previous["gpm_12h_accumulation_mm"],
        errors="coerce",
    ).to_numpy()

    latest_12h = pd.to_numeric(
        latest["gpm_12h_accumulation_mm"],
        errors="coerce",
    ).to_numpy()

    valid_24h = (
        np.isfinite(previous_12h)
        & np.isfinite(latest_12h)
    )

    rainfall_24h = np.full(len(combined), np.nan)

    rainfall_24h[valid_24h] = (
        previous_12h[valid_24h]
        + latest_12h[valid_24h]
    )

    combined["gpm_24h_accumulation_mm"] = rainfall_24h

    # Total valid half-hour intervals across both windows
    previous_intervals = pd.to_numeric(
        previous["gpm_valid_30min_intervals"],
        errors="coerce",
    ).to_numpy()

    latest_intervals = pd.to_numeric(
        latest["gpm_valid_30min_intervals"],
        errors="coerce",
    ).to_numpy()

    combined["gpm_24h_valid_30min_intervals"] = (
        previous_intervals + latest_intervals
    )

    combined["gpm_24h_series_start_utc"] = previous_start.isoformat()
    combined["gpm_24h_series_end_utc"] = latest_end.isoformat()

    combined["gpm_24h_product"] = "NASA GPM IMERG V07C"

    # Final integrity checks
    if len(combined) != 2009:
        raise ValueError(
            f"Unexpected output record count: {len(combined)}"
        )

    if combined["record_id"].nunique() != len(combined):
        raise ValueError("Duplicate record IDs in final output.")

    if (combined["gpm_24h_valid_30min_intervals"] > 48).any():
        raise ValueError("Unexpected interval count above 48.")

    if not (
        combined["gpm_24h_valid_30min_intervals"] == 48
    ).all():
        raise ValueError(
            "Not all records have 48 valid half-hour intervals."
        )

    # Save
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(OUTPUT_CSV, index=False)

    print("\n========== 24-HOUR DATASET SUMMARY ==========")

    print("Output records:", len(combined))
    print("Unique record IDs:", combined["record_id"].nunique())

    print(
        "Valid 24-hour rainfall:",
        combined["gpm_24h_accumulation_mm"].notna().sum(),
    )

    print(
        "Records with 48 valid intervals:",
        (combined["gpm_24h_valid_30min_intervals"] == 48).sum(),
    )

    print("\nRegional 24-hour rainfall statistics:")

    print(
        combined.groupby("dem_region")["gpm_24h_accumulation_mm"]
        .agg(["count", "mean", "max"])
    )

    print("\nSaved:", OUTPUT_CSV)
    print("\nCONTINUOUS 24-HOUR DATASET CREATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
