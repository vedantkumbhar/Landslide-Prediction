from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)
OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "western_ghats_spatial_validation"
)
REPORT_PATH = OUTPUT_DIR / "leave_one_tile_out_report.json"
CSV_PATH = OUTPUT_DIR / "leave_one_tile_out_metrics.csv"

FEATURES = ["elevation_m", "slope_degrees", "aspect_degrees"]
TARGET = "label"
TILE_COLUMN = "pilot_tile"
SEED = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Never overwrite previous validation results.
    if REPORT_PATH.exists() or CSV_PATH.exists():
        raise FileExistsError(
            "Validation output already exists. Nothing was overwritten. "
            f"Inspect: {OUTPUT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET, TILE_COLUMN, "sample_source"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[FEATURES + [TARGET, TILE_COLUMN]].isna().any().any():
        raise ValueError("Missing features, labels, or tile names.")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite feature values found.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Both labels 0 and 1 are required.")

    tiles = sorted(df[TILE_COLUMN].astype(str).unique())
    if len(tiles) < 2:
        raise ValueError("At least two different tiles are required.")

    fold_results = []

    for held_out_tile in tiles:
        test_mask = df[TILE_COLUMN].astype(str) == held_out_tile
        train_mask = ~test_mask

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(
                f"Both classes must appear in training and test data "
                f"for held-out tile {held_out_tile}."
            )

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        metrics = {
            "held_out_tile": held_out_tile,
            "train_tiles": ", ".join(
                tile for tile in tiles if tile != held_out_tile
            ),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_events": int((y_test == 1).sum()),
            "test_background_unverified": int((y_test == 0).sum()),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        fold_results.append(metrics)

        print(f"\nHeld-out tile: {held_out_tile}")
        print("Training samples:", len(X_train))
        print("Test samples:", len(X_test))
        print("Test events:", int((y_test == 1).sum()))
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ):
            print(f"{name}: {metrics[name]:.4f}")
        print(
            "Confusion matrix [[TN, FP], [FN, TP]]:",
            [[tn, fp], [fn, tp]],
        )

    results_df = pd.DataFrame(fold_results)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    report = {
        "validation_method": "leave-one-tile-out",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "features": FEATURES,
        "target": TARGET,
        "random_seed": SEED,
        "tiles": tiles,
        "folds": fold_results,
        "unweighted_mean_metrics": {
            metric: float(results_df[metric].mean())
            for metric in metric_columns
        },
        "limitations": [
            "Only three pilot tiles are evaluated.",
            "Each held-out tile contains few historical events.",
            "Background samples are unverified, not confirmed negatives.",
            "Terrain features only; rainfall and time-varying conditions are absent.",
            "Tile-level holdout tests geographic transfer across these tiles, "
            "not generalization to the entire Western Ghats.",
            "Adjacent geographic tiles may still share environmental patterns.",
            "This is research validation, not an operational warning system.",
        ],
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CSV_PATH, index=False)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nMean metrics across held-out tiles:")
    for name, value in report["unweighted_mean_metrics"].items():
        print(f"{name}: {value:.4f}")

    print("\nValidation CSV saved to:", CSV_PATH)
    print("Validation report saved to:", REPORT_PATH)
    print("\nNo saved baseline model was modified.")


if __name__ == "__main__":
    main()
