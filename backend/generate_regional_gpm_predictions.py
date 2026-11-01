from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data/processed/rainfall/gpm_consolidated_features_01nov2026.csv"
MODEL_DIR = BASE / "data/processed/regional_models_30oct"
OUTPUT = BASE / "data/processed/rainfall/regional_gpm_risk_predictions_01nov2026.csv"

MODEL_PATHS = {
    "Himalayas": MODEL_DIR / "himalayas/random_forest_model.joblib",
    "Northeast India": MODEL_DIR / "northeast_india/random_forest_model.joblib",
    "Western Maharashtra": MODEL_DIR / "western_maharashtra/random_forest_model.joblib",
}

RAIN_COLUMNS = [
    "gpm_3h_accumulation_mm",
    "gpm_6h_accumulation_mm",
    "gpm_12h_accumulation_mm",
    "gpm_24h_accumulation_mm",
]

MODEL_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "record_id", "latitude", "longitude", "dem_region",
        "elevation_m", "slope_degrees", "aspect_degrees",
        *RAIN_COLUMNS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values found.")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in required prediction fields.")

    # Reproduce the exact aspect transformation used during model training.
    result = df.copy()
    flat = result["aspect_degrees"] < 0
    radians = np.deg2rad(result["aspect_degrees"].clip(lower=0))

    result["aspect_sin"] = np.sin(radians)
    result["aspect_cos"] = np.cos(radians)
    result.loc[flat, ["aspect_sin", "aspect_cos"]] = 0.0
    result["aspect_flat"] = flat.astype(int)

    result["terrain_model_score"] = np.nan
    result["terrain_model_region"] = result["dem_region"]

    # Apply the corresponding regional model.
    for region, model_path in MODEL_PATHS.items():
        mask = result["dem_region"].eq(region)
        if not mask.any():
            print(f"WARNING: No records found for {region}")
            continue

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        expected = list(model.feature_names_in_)
        if expected != MODEL_FEATURES:
            raise ValueError(
                f"Unexpected model features for {region}: {expected}"
            )

        X = result.loc[mask, MODEL_FEATURES]
        positive_index = list(model.classes_).index(1)
        result.loc[mask, "terrain_model_score"] = model.predict_proba(X)[:, positive_index]

        print(f"{region}: {int(mask.sum())} predictions generated")

    unmapped = result["terrain_model_score"].isna()
    if unmapped.any():
        unknown = sorted(result.loc[unmapped, "dem_region"].astype(str).unique())
        raise ValueError(
            f"{int(unmapped.sum())} records have no matching regional model. "
            f"Unmapped dem_region values: {unknown}"
        )

    # Preserve raw rainfall measurements and calculate within-region
    # percentile indicators for this observation snapshot.
    for column in RAIN_COLUMNS:
        percentile_column = column.replace("_mm", "_regional_percentile")
        result[percentile_column] = (
            result.groupby("dem_region")[column]
            .rank(method="average", pct=True) * 100
        )

    result["rainfall_observation_end_utc"] = result[
        "gpm_common_observation_end_utc"
    ]
    result["prediction_type"] = "terrain_model_score_plus_separate_gpm_indicators"
    result["model_limitation"] = (
        "Terrain-only model; score is not a calibrated landslide probability. "
        "Rainfall percentiles are relative to this observation snapshot."
    )

    result.to_csv(OUTPUT, index=False)

    print("\nSUCCESS")
    print("Output:", OUTPUT)
    print("Rows:", len(result))
    print("Records by region:")
    print(result["dem_region"].value_counts().to_string())
    print("\nTerrain model score summary:")
    print(result.groupby("dem_region")["terrain_model_score"].agg(
        ["count", "mean", "min", "max"]
    ).to_string())
    print("\nRainfall columns and regional percentiles added.")
    print("Observation end:", result["rainfall_observation_end_utc"].iloc[0])


if __name__ == "__main__":
    main()
