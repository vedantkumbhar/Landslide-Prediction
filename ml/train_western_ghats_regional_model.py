"""
Western Ghats Regional Terrain-Only Random Forest Model Training Pipeline.

Validation Methodology:
Spatial Leave-One-DEM-Tile-Out Distal-Background Discrimination Benchmark
(minimum 5 km event-exclusion buffer).

Documentation of Methodology & Known Scope:
- Test background samples were selected using the full 182-event inventory, including held-out tile events.
- The test negatives are therefore conditioned on known event locations.
- The evaluation measures discrimination between recorded landslides and distal background terrain,
  not adjacent-slope stability or operational warning accuracy.
- The reported audit found zero training-background points in other tiles were altered by held-out tile events.
- Does not claim that an unbuffered or adjacent-slope evaluation has been completed.

Datasets:
- Events: data/processed/western_ghats_regional_event_features.csv (182 events, label 1)
- Background: data/processed/western_ghats_regional_background_samples_v2.csv (1,400 background, label 0)
- Predictors: elevation_m, slope_degrees, aspect_sin, aspect_cos, aspect_flat

Comparison:
- class_weight=None (unweighted) vs class_weight='balanced' (inverse class weighting)
- 14-fold spatial Leave-One-DEM-Tile-Out cross-validation
- Full-dataset final model artifacts saved for both configurations
"""

from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_event_features.csv"
)

BACKGROUND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_background_samples_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_model"
)

MODEL_BALANCED_PATH = (
    OUTPUT_DIR
    / "western_ghats_regional_terrain_random_forest.joblib"
)

MODEL_UNWEIGHTED_PATH = (
    OUTPUT_DIR
    / "western_ghats_regional_terrain_random_forest_unweighted.joblib"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "spatial_loto_evaluation_report.json"
)

METRICS_CSV_PATH = (
    OUTPUT_DIR
    / "spatial_loto_fold_metrics.csv"
)

OOF_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "spatial_loto_oof_predictions.csv"
)

PREDICTORS = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

TARGET = "label"
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_and_prepare_data():
    """Load both approved datasets, validate integrity, and assemble working dataset."""
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Historical events file not found: {EVENTS_PATH}")
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(f"Background samples file not found: {BACKGROUND_PATH}")

    events_df = pd.read_csv(EVENTS_PATH, low_memory=False)
    background_df = pd.read_csv(BACKGROUND_PATH, low_memory=False)

    print(f"Loaded {len(events_df)} historical events from: {EVENTS_PATH.name}")
    print(f"Loaded {len(background_df)} background samples from: {BACKGROUND_PATH.name}")

    if len(events_df) != 182:
        raise ValueError(f"Expected 182 events; found {len(events_df)}")
    if len(background_df) != 1400:
        raise ValueError(f"Expected 1,400 background samples; found {len(background_df)}")

    # Check predictor presence and nulls
    for col in PREDICTORS:
        if col not in events_df.columns:
            raise ValueError(f"Missing predictor {col} in events dataset.")
        if col not in background_df.columns:
            raise ValueError(f"Missing predictor {col} in background dataset.")
        if events_df[col].isna().any():
            raise ValueError(f"Null values found in events column {col}.")
        if background_df[col].isna().any():
            raise ValueError(f"Null values found in background column {col}.")

    # Format events subset
    ev_subset = events_df[PREDICTORS].copy()
    ev_subset["sample_id"] = events_df["event_id"].astype(str)
    ev_subset["tile"] = events_df["dem_tile"].astype(str)
    ev_subset["subregion"] = events_df["subregion"].astype(str)
    ev_subset["longitude"] = events_df["longitude"].astype(float)
    ev_subset["latitude"] = events_df["latitude"].astype(float)
    ev_subset[TARGET] = 1

    # Format background subset
    bg_subset = background_df[PREDICTORS].copy()
    bg_subset["sample_id"] = background_df["sample_id"].astype(str)
    bg_subset["tile"] = background_df["tile"].astype(str)
    bg_subset["subregion"] = background_df["subregion"].astype(str)
    bg_subset["longitude"] = background_df["longitude"].astype(float)
    bg_subset["latitude"] = background_df["latitude"].astype(float)
    bg_subset[TARGET] = 0

    combined_df = pd.concat([ev_subset, bg_subset], ignore_index=True)
    combined_df = combined_df.sort_values(by=["tile", "label", "sample_id"]).reset_index(drop=True)

    print(f"Combined regional dataset: {len(combined_df)} records (182 events, 1,400 background).")
    return combined_df, EVENTS_PATH.stat(), BACKGROUND_PATH.stat()


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics with edge-case protection."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC and PR-AUC require at least one sample of each class
    if len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_prob))
        pr = float(average_precision_score(y_true, y_prob))
    else:
        roc = float("nan")
        pr = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run_loto_evaluation(df, class_weight_setting, variant_name):
    """Execute 14-fold spatial Leave-One-DEM-Tile-Out cross-validation."""
    print(f"\n" + "=" * 70)
    print(f"RUNNING 14-FOLD SPATIAL LOTO: {variant_name} (class_weight={class_weight_setting})")
    print("=" * 70)

    tiles = sorted(df["tile"].unique())
    fold_metrics = []
    oof_predictions = np.zeros(len(df), dtype=int)
    oof_probabilities = np.zeros(len(df), dtype=float)

    start_time = time.time()

    for fold_idx, held_out_tile in enumerate(tiles, start=1):
        test_mask = df["tile"] == held_out_tile
        train_mask = ~test_mask

        X_train = df.loc[train_mask, PREDICTORS]
        y_train = df.loc[train_mask, TARGET]
        X_test = df.loc[test_mask, PREDICTORS]
        y_test = df.loc[test_mask, TARGET]

        subregion = df.loc[test_mask, "subregion"].iloc[0]

        # Verify class presence in training fold
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Fold {fold_idx} ({held_out_tile}): Training set is missing a class!")

        # Initialize and fit Random Forest
        rf = RandomForestClassifier(
            class_weight=class_weight_setting,
            **RF_PARAMS,
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)[:, 1]

        oof_predictions[test_mask] = preds
        oof_probabilities[test_mask] = probs

        metrics = compute_metrics(y_test, preds, probs)
        metrics["fold"] = fold_idx
        metrics["held_out_tile"] = held_out_tile
        metrics["subregion"] = subregion
        metrics["model_variant"] = variant_name
        metrics["class_weight"] = str(class_weight_setting)
        metrics["train_samples"] = int(len(X_train))
        metrics["test_samples"] = int(len(X_test))
        metrics["test_events"] = int((y_test == 1).sum())
        metrics["test_backgrounds"] = int((y_test == 0).sum())

        fold_metrics.append(metrics)

        print(
            f"  Fold {fold_idx:02d} [{held_out_tile:<8}]: "
            f"Events={metrics['test_events']:2d}, BG={metrics['test_backgrounds']:3d} | "
            f"BAcc={metrics['balanced_accuracy']:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | "
            f"ROC={metrics['roc_auc']:.4f} | "
            f"PR={metrics['pr_auc']:.4f} | "
            f"CM=[[{metrics['true_negatives']}, {metrics['false_positives']}], [{metrics['false_negatives']}, {metrics['true_positives']}]]"
        )

    elapsed = time.time() - start_time
    print(f"LOTO completed in {elapsed:.2f} seconds.")

    # Compute pooled Out-Of-Fold metrics
    pooled_metrics = compute_metrics(df[TARGET], oof_predictions, oof_probabilities)
    pooled_metrics["model_variant"] = variant_name
    pooled_metrics["class_weight"] = str(class_weight_setting)
    pooled_metrics["total_samples"] = len(df)
    pooled_metrics["total_events"] = int((df[TARGET] == 1).sum())
    pooled_metrics["total_backgrounds"] = int((df[TARGET] == 0).sum())

    # Compute unweighted mean across folds
    fold_df = pd.DataFrame(fold_metrics)
    unweighted_mean_metrics = {
        metric: float(fold_df[metric].mean())
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    }
    unweighted_std_metrics = {
        metric: float(fold_df[metric].std())
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    }

    print(f"\n--- {variant_name} Summary ---")
    print("Unweighted Mean across 14 folds:")
    for k, v in unweighted_mean_metrics.items():
        print(f"  {k:<20}: {v:.4f} (+/- {unweighted_std_metrics[k]:.4f})")

    print("\nPooled Global Out-Of-Fold Metrics:")
    for k in ["balanced_accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        print(f"  {k:<20}: {pooled_metrics[k]:.4f}")
    print(f"  Confusion Matrix    : {pooled_metrics['confusion_matrix']}")

    return {
        "fold_metrics": fold_metrics,
        "unweighted_mean_metrics": unweighted_mean_metrics,
        "unweighted_std_metrics": unweighted_std_metrics,
        "pooled_oof_metrics": pooled_metrics,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
    }


def main():
    print("=" * 80)
    print("STAGE 5B: WESTERN GHATS REGIONAL TERRAIN MODEL TRAINING")
    print("=" * 80)

    # 1. Output safety: ensure output directory can be created without overwriting
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (MODEL_BALANCED_PATH, MODEL_UNWEIGHTED_PATH, REPORT_PATH, METRICS_CSV_PATH, OOF_PREDICTIONS_PATH):
        if p.exists():
            raise FileExistsError(
                f"Output file already exists; refusing to overwrite:\n  {p}\n"
                "Please inspect the output directory first."
            )

    # 2. Load data and verify input immutability
    df, ev_stat_before, bg_stat_before = load_and_prepare_data()

    # 3. Run spatial LOTO for class_weight=None
    unweighted_results = run_loto_evaluation(
        df, class_weight_setting=None, variant_name="unweighted"
    )

    # 4. Run spatial LOTO for class_weight='balanced'
    balanced_results = run_loto_evaluation(
        df, class_weight_setting="balanced", variant_name="balanced"
    )

    # 5. Fit final full-dataset models
    print("\n" + "=" * 70)
    print("TRAINING FINAL FULL-DATASET REGIONAL MODELS (1,582 INSTANCES)")
    print("=" * 70)

    X_full = df[PREDICTORS]
    y_full = df[TARGET]

    print("Fitting final full-dataset Random Forest with class_weight='balanced'...")
    final_rf_balanced = RandomForestClassifier(
        class_weight="balanced",
        **RF_PARAMS,
    )
    final_rf_balanced.fit(X_full, y_full)
    joblib.dump(final_rf_balanced, MODEL_BALANCED_PATH)
    print(f"Saved: {MODEL_BALANCED_PATH.relative_to(PROJECT_ROOT)}")

    print("Fitting final full-dataset Random Forest with class_weight=None...")
    final_rf_unweighted = RandomForestClassifier(
        class_weight=None,
        **RF_PARAMS,
    )
    final_rf_unweighted.fit(X_full, y_full)
    joblib.dump(final_rf_unweighted, MODEL_UNWEIGHTED_PATH)
    print(f"Saved: {MODEL_UNWEIGHTED_PATH.relative_to(PROJECT_ROOT)}")

    # Calculate feature importances
    importances_balanced = {
        feat: float(imp)
        for feat, imp in zip(PREDICTORS, final_rf_balanced.feature_importances_)
    }
    importances_unweighted = {
        feat: float(imp)
        for feat, imp in zip(PREDICTORS, final_rf_unweighted.feature_importances_)
    }

    print("\nFeature Importances (class_weight='balanced'):")
    for feat, imp in sorted(importances_balanced.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<18}: {imp:.4f} ({imp*100:.2f}%)")

    # 6. Save Out-Of-Fold predictions CSV
    oof_df = df[["sample_id", "tile", "subregion", "label", "longitude", "latitude"] + PREDICTORS].copy()
    oof_df["pred_unweighted"] = unweighted_results["oof_predictions"]
    oof_df["prob_unweighted"] = unweighted_results["oof_probabilities"]
    oof_df["pred_balanced"] = balanced_results["oof_predictions"]
    oof_df["prob_balanced"] = balanced_results["oof_probabilities"]
    oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)
    print(f"\nSaved OOF Predictions: {OOF_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")

    # 7. Save Fold Metrics CSV
    all_fold_metrics = unweighted_results["fold_metrics"] + balanced_results["fold_metrics"]
    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    # Reorder columns for clarity
    metric_cols = [
        "fold", "held_out_tile", "subregion", "model_variant", "class_weight",
        "train_samples", "test_samples", "test_events", "test_backgrounds",
        "accuracy", "balanced_accuracy", "precision", "recall", "f1_score",
        "roc_auc", "pr_auc", "true_negatives", "false_positives", "false_negatives", "true_positives"
    ]
    fold_metrics_df = fold_metrics_df[metric_cols]
    fold_metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"Saved Fold Metrics: {METRICS_CSV_PATH.relative_to(PROJECT_ROOT)}")

    # 8. Compile Comprehensive JSON Evaluation Report
    report = {
        "benchmark_title": (
            "Spatial Leave-One-DEM-Tile-Out Distal-Background Discrimination Benchmark "
            "(minimum 5 km event-exclusion buffer)"
        ),
        "methodology_documentation": {
            "evaluation_design": (
                "14-fold spatial Leave-One-DEM-Tile-Out cross-validation. In each fold, "
                "one complete 1-degree DEM tile is held out for testing while the Random Forest "
                "is trained on the remaining 13 tiles."
            ),
            "test_background_conditioning": (
                "Test background samples were selected using the full 182-event inventory, "
                "including held-out tile events. The test negatives are therefore conditioned "
                "on known event locations."
            ),
            "evaluation_scope_and_claims": (
                "The evaluation measures discrimination between recorded landslides and distal background terrain, "
                "not adjacent-slope stability or operational warning accuracy."
            ),
            "training_independence_audit": (
                "The reported audit found zero training-background points in other tiles were altered "
                "by held-out tile events."
            ),
            "unbuffered_limitation": (
                "Do not claim that an unbuffered or adjacent-slope evaluation has been completed."
            ),
        },
        "datasets": {
            "events_path": str(EVENTS_PATH.relative_to(PROJECT_ROOT)),
            "background_path": str(BACKGROUND_PATH.relative_to(PROJECT_ROOT)),
            "total_instances": len(df),
            "total_positive_events": int((df[TARGET] == 1).sum()),
            "total_negative_background": int((df[TARGET] == 0).sum()),
            "class_ratio": f"1 : {(df[TARGET] == 0).sum() / (df[TARGET] == 1).sum():.2f}",
            "number_of_tiles": len(sorted(df["tile"].unique())),
            "tiles": sorted(df["tile"].unique()),
        },
        "features": {
            "predictors": PREDICTORS,
            "target": TARGET,
            "feature_importances_balanced": importances_balanced,
            "feature_importances_unweighted": importances_unweighted,
        },
        "model_architecture": {
            "model": "RandomForestClassifier",
            "parameters": RF_PARAMS,
            "compared_class_weights": ["None (unweighted)", "balanced (inverse class weighting)"],
        },
        "comparison_summary": {
            "unweighted": {
                "unweighted_mean_across_folds": unweighted_results["unweighted_mean_metrics"],
                "unweighted_std_across_folds": unweighted_results["unweighted_std_metrics"],
                "pooled_oof_metrics": unweighted_results["pooled_oof_metrics"],
            },
            "balanced": {
                "unweighted_mean_across_folds": balanced_results["unweighted_mean_metrics"],
                "unweighted_std_across_folds": balanced_results["unweighted_std_metrics"],
                "pooled_oof_metrics": balanced_results["pooled_oof_metrics"],
            },
        },
        "fold_results": {
            "unweighted": unweighted_results["fold_metrics"],
            "balanced": balanced_results["fold_metrics"],
        },
        "environment": {
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Evaluation Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # 9. Verify Input Invariance
    ev_stat_after = EVENTS_PATH.stat()
    bg_stat_after = BACKGROUND_PATH.stat()

    if (
        ev_stat_before.st_size != ev_stat_after.st_size
        or ev_stat_before.st_mtime != ev_stat_after.st_mtime
    ):
        raise RuntimeError("Events dataset was modified! Preserving integrity violated.")

    if (
        bg_stat_before.st_size != bg_stat_after.st_size
        or bg_stat_before.st_mtime != bg_stat_after.st_mtime
    ):
        raise RuntimeError("Background dataset was modified! Preserving integrity violated.")

    print("\nInput file preservation check: PASSED (both input CSVs are completely unchanged).")
    print("\n" + "=" * 80)
    print("STAGE 5B EXECUTION SUCCESSFULLY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
