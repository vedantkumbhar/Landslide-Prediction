
from pathlib import Path
import pandas as pd
import numpy as np

from backend.gpm_refresh.refresh_config import get_run_directory

# ============================================================
# CONFIGURATION
# ============================================================

RUN_ID = "20261102_152742"

OLD_PREDICTIONS = (
    Path("data/processed/rainfall")
    / "regional_gpm_risk_predictions_01nov2026.csv"
)

NEW_RAINFALL = (
    get_run_directory(RUN_ID)
    / "extracted"
    / "gpm_regional_rainfall_24h.csv"
)

OUTPUT_DIR = get_run_directory(RUN_ID) / "predictions"

OUTPUT_CSV = (
    OUTPUT_DIR
    / "regional_gpm_risk_predictions_02nov2026.csv"
)

EXPECTED_RECORDS = 2009

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\nNASA GPM: RAINFALL-PREDICTION INTEGRATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. CHECK INPUT FILES
    # --------------------------------------------------------

    if not OLD_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Existing predictions not found: {OLD_PREDICTIONS}"
        )

    if not NEW_RAINFALL.exists():
        raise FileNotFoundError(
            f"New rainfall dataset not found: {NEW_RAINFALL}"
        )

    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {OUTPUT_CSV}"
        )

    predictions = pd.read_csv(OLD_PREDICTIONS)
    rainfall = pd.read_csv(NEW_RAINFALL)

    print("Existing predictions:", len(predictions))
    print("New rainfall records:", len(rainfall))

    # --------------------------------------------------------
    # 2. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    prediction_required = [
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
        "terrain_model_score",
    ]

    rainfall_required = [
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
        "gpm_24h_accumulation_mm",
        "gpm_24h_valid_30min_intervals",
        "gpm_24h_series_start_utc",
        "gpm_24h_series_end_utc",
    ]

    for col in prediction_required:
        if col not in predictions.columns:
            raise ValueError(
                f"Prediction dataset missing column: {col}"
            )

    for col in rainfall_required:
        if col not in rainfall.columns:
            raise ValueError(
                f"Rainfall dataset missing column: {col}"
            )

    # --------------------------------------------------------
    # 3. VALIDATE RECORD IDENTIFIERS
    # --------------------------------------------------------

    for df, name in [
        (predictions, "predictions"),
        (rainfall, "rainfall"),
    ]:

        if df["record_id"].isna().any():
            raise ValueError(f"Missing record IDs in {name}")

        if df["record_id"].duplicated().any():
            raise ValueError(f"Duplicate record IDs in {name}")

    if len(predictions) != EXPECTED_RECORDS:
        raise ValueError("Unexpected prediction record count")

    if len(rainfall) != EXPECTED_RECORDS:
        raise ValueError("Unexpected rainfall record count")

    if set(predictions["record_id"]) != set(rainfall["record_id"]):
        raise ValueError(
            "Prediction and rainfall record IDs do not match"
        )

    # --------------------------------------------------------
    # 4. SORT AND ALIGN RECORDS
    # --------------------------------------------------------

    predictions = (
        predictions.sort_values("record_id")
        .reset_index(drop=True)
    )

    rainfall = (
        rainfall.sort_values("record_id")
        .reset_index(drop=True)
    )

    for col in ["latitude", "longitude"]:

        if not np.allclose(
            predictions[col].to_numpy(dtype=float),
            rainfall[col].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(f"Coordinate mismatch: {col}")

    if not predictions["dem_region"].equals(rainfall["dem_region"]):
        raise ValueError("Regional assignment mismatch")

    print("\nRecord IDs and coordinates: PASSED")

    # --------------------------------------------------------
    # 5. VALIDATE RAINFALL VALUES
    # --------------------------------------------------------

    rainfall_values = pd.to_numeric(
        rainfall["gpm_24h_accumulation_mm"],
        errors="coerce"
    )

    intervals = pd.to_numeric(
        rainfall["gpm_24h_valid_30min_intervals"],
        errors="coerce"
    )

    if rainfall_values.isna().any():
        raise ValueError("Invalid 24-hour rainfall values")

    if not np.isfinite(rainfall_values).all():
        raise ValueError("Non-finite rainfall values")

    if (rainfall_values < 0).any():
        raise ValueError("Negative rainfall values detected")

    if not (intervals == 48).all():
        raise ValueError("Incomplete 24-hour rainfall coverage")

    # --------------------------------------------------------
    # 6. VALIDATE OBSERVATION WINDOW
    # --------------------------------------------------------

    starts = rainfall[
        "gpm_24h_series_start_utc"
    ].dropna().unique()

    ends = rainfall[
        "gpm_24h_series_end_utc"
    ].dropna().unique()

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Inconsistent observation timestamps")

    start = pd.to_datetime(starts[0], utc=True)
    end = pd.to_datetime(ends[0], utc=True)

    if start != pd.Timestamp("2026-11-01T03:00:00Z"):
        raise ValueError(f"Unexpected start time: {start}")

    if end != pd.Timestamp("2026-11-02T03:00:00Z"):
        raise ValueError(f"Unexpected end time: {end}")

    if end - start != pd.Timedelta(hours=24):
        raise ValueError("Observation window is not 24 hours")

    print("Rainfall values and time coverage: PASSED")

    # --------------------------------------------------------
    # 7. PRESERVE EXISTING PREDICTION DATA
    # --------------------------------------------------------

    combined = predictions.copy()

    original_scores = predictions[
        "terrain_model_score"
    ].copy()

    # Rename previous GPM columns to preserve old observations.
    old_gpm_columns = [
        col for col in combined.columns
        if col.startswith("gpm_")
    ]

    for col in old_gpm_columns:
        combined.rename(
            columns={col: f"previous_{col}"},
            inplace=True
        )

    # --------------------------------------------------------
    # 8. ADD ONLY NEW GPM FEATURES
    # --------------------------------------------------------

    new_gpm_columns = [
        col for col in rainfall.columns
        if col.startswith("gpm_")
    ]

    if not new_gpm_columns:
        raise ValueError("No GPM features found in new dataset")

    conflicts = [
        col for col in new_gpm_columns
        if col in combined.columns
    ]

    if conflicts:
        raise ValueError(
            f"GPM column conflicts detected: {conflicts}"
        )

    combined = pd.concat(
        [
            combined.reset_index(drop=True),
            rainfall[new_gpm_columns].reset_index(drop=True),
        ],
        axis=1
    )

    print("New GPM columns integrated:", len(new_gpm_columns))

    # --------------------------------------------------------
    # 9. FINAL VALIDATION
    # --------------------------------------------------------

    if len(combined) != EXPECTED_RECORDS:
        raise ValueError("Final record count mismatch")

    if combined["record_id"].nunique() != EXPECTED_RECORDS:
        raise ValueError("Final record IDs are not unique")

    if not combined["terrain_model_score"].equals(original_scores):
        raise ValueError("Original terrain model scores changed")

    if not np.allclose(
        combined["gpm_24h_accumulation_mm"].to_numpy(dtype=float),
        rainfall_values.to_numpy(dtype=float),
    ):
        raise ValueError("Integrated rainfall values do not match source")

    # --------------------------------------------------------
    # 10. SAVE NEW DATASET
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined.to_csv(OUTPUT_CSV, index=False)

    # --------------------------------------------------------
    # 11. SUMMARY
    # --------------------------------------------------------

    print("\n========== INTEGRATION SUMMARY ==========")

    print("Total records:", len(combined))
    print("Total columns:", len(combined.columns))
    print("Unique record IDs:", combined["record_id"].nunique())

    print("Terrain model scores preserved: YES")
    print("Previous GPM observations preserved: YES")
    print("New 24-hour GPM rainfall integrated: YES")

    print("\nRegional 24-hour rainfall statistics:")

    print(
        combined.groupby("dem_region")[
            "gpm_24h_accumulation_mm"
        ].agg(["count", "mean", "max"]).round(3)
    )

    print("\nOutput saved:")
    print(OUTPUT_CSV)

    print("\nRAINFALL INTEGRATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
