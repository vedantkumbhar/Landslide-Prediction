
import pandas as pd

from backend.gpm_refresh.refresh_config import PROJECT_ROOT

RUN_DIR = (
    PROJECT_ROOT
    / "data/processed/rainfall/refresh_runs/20261102_152742"
)

PREDICTION_FILE = (
    RUN_DIR
    / "predictions"
    / "regional_gpm_risk_predictions_02nov2026.csv"
)

RAINFALL_3H_FILE = (
    RUN_DIR
    / "extracted"
    / "gpm_regional_rainfall_3h.csv"
)

OUTPUT_FILE = (
    RUN_DIR
    / "predictions"
    / "regional_gpm_risk_predictions_02nov2026_with_3h.csv"
)


def main():

    print("\nADDING CURRENT 3-HOUR GPM RAINFALL")
    print("=" * 60)

    predictions = pd.read_csv(PREDICTION_FILE)
    rainfall = pd.read_csv(RAINFALL_3H_FILE)

    # Validate required columns
    required_prediction = {
        "record_id", "latitude", "longitude", "dem_region"
    }

    required_rainfall = {
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
        "gpm_3h_accumulation_mm",
        "gpm_series_start_utc",
        "gpm_series_end_utc",
    }

    if not required_prediction.issubset(predictions.columns):
        raise ValueError("Missing prediction columns.")

    if not required_rainfall.issubset(rainfall.columns):
        raise ValueError("Missing 3-hour rainfall columns.")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    # Validate record IDs
    if predictions["record_id"].duplicated().any():
        raise ValueError("Duplicate prediction record IDs.")

    if rainfall["record_id"].duplicated().any():
        raise ValueError("Duplicate rainfall record IDs.")

    if set(predictions["record_id"]) != set(rainfall["record_id"]):
        raise ValueError("Prediction and rainfall record IDs do not match.")

    # Align rainfall rows to prediction order
    rainfall = rainfall.set_index("record_id").loc[
        predictions["record_id"]
    ].reset_index()

    # Validate coordinates and regions
    for column in ["latitude", "longitude"]:
        if not (
            predictions[column].to_numpy()
            == rainfall[column].to_numpy()
        ).all():
            raise ValueError(f"{column} mismatch.")

    if not (
        predictions["dem_region"].to_numpy()
        == rainfall["dem_region"].to_numpy()
    ).all():
        raise ValueError("Regional labels do not match.")

    # Validate observation window
    starts = rainfall["gpm_series_start_utc"].unique()
    ends = rainfall["gpm_series_end_utc"].unique()

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Inconsistent 3-hour observation windows.")

    if starts[0] != "2026-11-02T00:00:00+00:00":
        raise ValueError(f"Unexpected start time: {starts[0]}")

    if ends[0] != "2026-11-02T03:00:00+00:00":
        raise ValueError(f"Unexpected end time: {ends[0]}")

    values = pd.to_numeric(
        rainfall["gpm_3h_accumulation_mm"],
        errors="coerce"
    )

    if values.isna().any() or (~values.map(pd.notna)).any():
        raise ValueError("Missing or invalid 3-hour rainfall values.")

    if (values < 0).any():
        raise ValueError("Negative rainfall values found.")

    # Add only the required 3-hour feature and its window metadata
    predictions["gpm_3h_accumulation_mm"] = values.to_numpy()
    predictions["gpm_3h_series_start_utc"] = starts[0]
    predictions["gpm_3h_series_end_utc"] = ends[0]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_FILE, index=False)

    print("\nINTEGRATION SUMMARY")
    print("-" * 60)
    print("Prediction records:", len(predictions))
    print("Unique record IDs:", predictions["record_id"].nunique())
    print("3-hour rainfall values:", values.notna().sum())
    print("3-hour window:", starts[0], "to", ends[0])
    print("Final columns:", len(predictions.columns))

    print("\nRegional 3-hour rainfall:")
    print(
        predictions.groupby("dem_region")[
            "gpm_3h_accumulation_mm"
        ].agg(["count", "mean", "max"]).round(3)
    )

    print("\nSaved:", OUTPUT_FILE)
    print("\n3-HOUR RAINFALL INTEGRATION COMPLETED.")


if __name__ == "__main__":
    main()
