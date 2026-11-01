
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAINING_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

RAINFALL_PATH = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "hourly_rainfall.csv"
)

EXPECTED_HOURS = 120


def main():
    training = pd.read_csv(TRAINING_PATH)
    rainfall = pd.read_csv(RAINFALL_PATH)

    required = [
        "sample_id",
        "pilot_tile",
        "requested_latitude",
        "requested_longitude",
        "api_latitude",
        "api_longitude",
        "timestamp_local",
        "precipitation_mm",
        "record_type",
    ]

    missing_columns = [
        col for col in required if col not in rainfall.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing rainfall columns: {missing_columns}")

    if rainfall[required].isna().any().any():
        raise ValueError("Required rainfall fields contain missing values.")

    if rainfall.duplicated(["sample_id", "timestamp_local"]).any():
        raise ValueError("Duplicate sample timestamps found.")

    rainfall["timestamp_local"] = pd.to_datetime(
        rainfall["timestamp_local"], errors="raise"
    )
    rainfall["precipitation_mm"] = pd.to_numeric(
        rainfall["precipitation_mm"], errors="raise"
    )

    if not np.isfinite(rainfall["precipitation_mm"]).all():
        raise ValueError("Non-finite precipitation values found.")

    if (rainfall["precipitation_mm"] < 0).any():
        raise ValueError("Negative precipitation values found.")

    counts = rainfall.groupby("sample_id").size()

    if len(counts) != len(training):
        raise ValueError(
            f"Expected {len(training)} sample IDs, found {len(counts)}."
        )

    if not (counts == EXPECTED_HOURS).all():
        print("\nUnexpected records per sample:")
        print(counts[counts != EXPECTED_HOURS].to_string())
        raise ValueError("Some samples do not have exactly 120 records.")

    timestamp_gaps = []
    for sample_id, group in rainfall.groupby("sample_id"):
        times = group["timestamp_local"].sort_values()
        differences = times.diff().dropna()

        if not (differences == pd.Timedelta(hours=1)).all():
            timestamp_gaps.append(sample_id)

    if timestamp_gaps:
        raise ValueError(
            "Hourly gaps or irregular intervals found for sample IDs: "
            + ", ".join(timestamp_gaps[:10])
        )

    sample_metadata = rainfall[
        ["sample_id", "pilot_tile",
         "requested_latitude", "requested_longitude"]
    ].drop_duplicates()

    if sample_metadata["sample_id"].duplicated().any():
        raise ValueError("Sample metadata changes within a sample ID.")

    expected_ids = {
        f"WG_SAMPLE_{i:04d}"
        for i in range(1, len(training) + 1)
    }
    actual_ids = set(rainfall["sample_id"].unique())

    if actual_ids != expected_ids:
        raise ValueError("Sample IDs do not match the expected row sequence.")

    print("PASS: coordinate-level rainfall validation")
    print("Training samples:", len(training))
    print("Rainfall sample IDs:", len(counts))
    print("Total rainfall rows:", len(rainfall))
    print("Records per sample:", counts.min(), "to", counts.max())
    print(
        "Distinct API grid cells:",
        rainfall[["api_latitude", "api_longitude"]]
        .drop_duplicates().shape[0],
    )
    print(
        "Missing precipitation:",
        int(rainfall["precipitation_mm"].isna().sum()),
    )
    print("Negative precipitation:", int(
        (rainfall["precipitation_mm"] < 0).sum()
    ))
    print(
        "Timestamp range:",
        rainfall["timestamp_local"].min(),
        "to",
        rainfall["timestamp_local"].max(),
    )
    print("\nRecords by tile and type:")
    print(
        rainfall.groupby(["pilot_tile", "record_type"])
        .size().to_string()
    )
    print("\nNo files were modified.")


if __name__ == "__main__":
    main()
