
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/processed/regional_training_dataset_30oct.csv")
OUTPUT_DIR = Path("data/processed/regional_models_30oct")

REGIONS = [
    "Himalayas",
    "Northeast India",
    "Western Maharashtra",
]

RANDOM_STATE = 42

RAW_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]

MODEL_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def prepare_features(df):
    result = df.copy()

    aspect = result["aspect_degrees"].astype(float)

    # DEM aspect -1 represents flat terrain.
    flat = aspect < 0
    radians = np.deg2rad(aspect.clip(lower=0))

    result["aspect_sin"] = np.sin(radians)
    result["aspect_cos"] = np.cos(radians)

    # Flat terrain has no meaningful aspect direction.
    result.loc[flat, ["aspect_sin", "aspect_cos"]] = 0.0
    result["aspect_flat"] = flat.astype(int)

    return result


# --------------------------------------------------
# MODEL FACTORY
# --------------------------------------------------

def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred, y_probability):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_probability)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output folder already exists: {OUTPUT_DIR}\n"
            "Refusing to overwrite existing model artifacts."
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "latitude",
        "longitude",
        "dem_region",
        "tile",
        "label",
        *RAW_FEATURES,
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    # Exclude repeated coordinates only in the modelling copy.
    modelling_df = df.drop_duplicates(
        subset=["latitude", "longitude"],
        keep="first",
    ).copy()

    modelling_df = prepare_features(modelling_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    summary = {}

    print("\n========== REGIONAL MODEL TRAINING ==========")
    print("Source dataset:", DATA_PATH)
    print("Original records:", len(df))
    print("Unique-coordinate records:", len(modelling_df))
    print("Duplicate-coordinate rows excluded:", len(df) - len(modelling_df))

    for region in REGIONS:

        print(f"\n\n========== {region.upper()} ==========")

        region_df = modelling_df[
            modelling_df["dem_region"] == region
        ].copy()

        if region_df.empty:
            raise ValueError(f"No records found for {region}")

        region_df = region_df.reset_index(drop=True)

        X = region_df[MODEL_FEATURES].astype(float)
        y = region_df["label"].astype(int)
        groups = region_df["tile"].astype(str)

        if set(y.unique()) != {0, 1}:
            raise ValueError(
                f"{region}: both labels are required for training."
            )

        unique_tiles = sorted(groups.unique())

        if len(unique_tiles) < 2:
            raise ValueError(
                f"{region}: at least two tiles are required."
            )

        oof_predictions = np.full(len(region_df), -1, dtype=int)
        oof_probabilities = np.full(len(region_df), np.nan, dtype=float)

        fold_results = []

        # Leave one complete DEM tile out in each fold.
        for tile in unique_tiles:

            test_mask = groups == tile
            train_mask = ~test_mask

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]

            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]

            if set(y_train.unique()) != {0, 1}:
                raise ValueError(
                    f"{region}, held-out tile {tile}: "
                    "training fold does not contain both labels."
                )

            if set(y_test.unique()) != {0, 1}:
                raise ValueError(
                    f"{region}, held-out tile {tile}: "
                    "test fold does not contain both labels."
                )

            fold_model = create_model()
            fold_model.fit(X_train, y_train)

            predictions = fold_model.predict(X_test)
            probabilities = fold_model.predict_proba(X_test)[:, 1]

            oof_predictions[test_mask.to_numpy()] = predictions
            oof_probabilities[test_mask.to_numpy()] = probabilities

            fold_results.append({
                "held_out_tile": tile,
                "test_records": int(test_mask.sum()),
                "test_events": int((y_test == 1).sum()),
                "test_backgrounds": int((y_test == 0).sum()),
            })

        if (oof_predictions < 0).any() or np.isnan(oof_probabilities).any():
            raise RuntimeError(
                f"{region}: some records did not receive validation predictions."
            )

        metrics = calculate_metrics(
            y,
            oof_predictions,
            oof_probabilities,
        )

        # Train the deployment model on all unique records in this region.
        final_model = create_model()
        final_model.fit(X, y)

        region_slug = (
            region.lower()
            .replace(" ", "_")
        )

        region_dir = OUTPUT_DIR / region_slug
        region_dir.mkdir(parents=True, exist_ok=False)

        model_path = region_dir / "random_forest_model.joblib"
        predictions_path = region_dir / "spatial_validation_predictions.csv"
        report_path = region_dir / "evaluation_report.json"
        importance_path = region_dir / "feature_importance.csv"

        joblib.dump(final_model, model_path)

        prediction_output = region_df[
            [
                "latitude",
                "longitude",
                "dem_region",
                "tile",
                "label",
                "sample_type",
            ]
        ].copy()

        prediction_output["oof_prediction"] = oof_predictions
        prediction_output["oof_score"] = oof_probabilities

        prediction_output.to_csv(predictions_path, index=False)

        importance = pd.DataFrame({
            "feature": MODEL_FEATURES,
            "importance": final_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance.to_csv(importance_path, index=False)

        report = {
            "region": region,
            "model": "RandomForestClassifier",
            "random_state": RANDOM_STATE,
            "record_count": int(len(region_df)),
            "event_count": int((y == 1).sum()),
            "provisional_background_count": int((y == 0).sum()),
            "tile_count": int(len(unique_tiles)),
            "features": MODEL_FEATURES,
            "validation": "Leave-one-DEM-tile-out",
            "validation_fold_count": int(len(fold_results)),
            "metrics": metrics,
            "folds": fold_results,
            "label_caveat": (
                "Label 0 represents provisional background samples, "
                "not verified landslide-free locations."
            ),
            "interpretation_caveat": (
                "Scores distinguish recorded landslide locations from "
                "sampled background locations. They are not calibrated "
                "probabilities of actual landslide occurrence."
            ),
            "rainfall_included": False,
            "earthquake_included": False,
        }

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        summary[region] = metrics

        print("Records:", len(region_df))
        print("Events:", int((y == 1).sum()))
        print("Provisional backgrounds:", int((y == 0).sum()))
        print("DEM tiles:", len(unique_tiles))
        print("Validation folds:", len(fold_results))
        print("\nSpatial validation metrics:")

        for name, value in metrics.items():
            if name != "confusion_matrix":
                print(f"{name}: {value:.4f}")

        print("Confusion matrix:")
        print(np.array(metrics["confusion_matrix"]))

        print("\nFeature importance:")
        print(importance.to_string(index=False))

        print("\nSaved model:", model_path)
        print("Saved report:", report_path)

    summary_path = OUTPUT_DIR / "all_regions_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n\n========== ALL REGIONAL MODELS COMPLETED ==========")
    print("Output folder:", OUTPUT_DIR)
    print("Regions trained:", len(summary))
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
