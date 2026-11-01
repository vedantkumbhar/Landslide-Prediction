
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "initial_training_dataset.csv"
MODEL_PATH = (
    ROOT / "data" / "processed" / "baseline_model"
    / "terrain_random_forest.joblib"
)

OUTPUT_DIR = ROOT / "reports" / "regional_baseline_evaluation"
CSV_PATH = OUTPUT_DIR / "regional_test_metrics.csv"
JSON_PATH = OUTPUT_DIR / "regional_test_report.json"

FEATURES = ["elevation_m", "slope_degrees", "aspect_degrees"]
TARGET = "label"
REGION = "dem_region"
SEED = 42


def main():
    print("\nREGIONAL INDIA BASELINE EVALUATION")
    print("Read-only model evaluation; no model training.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    # Protect existing evaluation outputs.
    if CSV_PATH.exists() or JSON_PATH.exists():
        raise FileExistsError(
            "Regional evaluation output already exists. "
            f"Inspect this folder first: {OUTPUT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET, REGION]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing required data. Evaluation stopped.")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite feature values found.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected both labels 0 and 1.")

    # Reconstruct the exact original stratified split.
    _, _, _, _, _, test_indices = train_test_split(
        X,
        y,
        df.index,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    test_df = df.loc[test_indices].copy()
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    # Load the existing saved model.
    stored = joblib.load(MODEL_PATH)

    if not isinstance(stored, dict) or "model" not in stored:
        raise TypeError("Unexpected saved model structure.")

    model = stored["model"]
    saved_features = stored.get("features", [])

    if list(saved_features) != FEATURES:
        raise ValueError(
            f"Model feature mismatch: {saved_features}"
        )

    if len(test_df) != 335:
        raise ValueError(
            f"Expected 335 original test records, got {len(test_df)}. "
            "The dataset or original split may have changed."
        )

    # Predict once on the reconstructed original test set.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    test_df["actual_label"] = y_test
    test_df["predicted_label"] = predictions
    test_df["predicted_score"] = probabilities

    results = []

    print("\nORIGINAL TEST SET")
    print("Test records:", len(test_df))
    print("Label counts:")
    print(test_df["actual_label"].value_counts().sort_index().to_string())

    for region, group in test_df.groupby(REGION, sort=True):
        actual = group["actual_label"].astype(int)
        predicted = group["predicted_label"].astype(int)
        scores = group["predicted_score"].astype(float)

        positive = int((actual == 1).sum())
        negative = int((actual == 0).sum())

        row = {
            "region": str(region),
            "test_samples": int(len(group)),
            "positive_events": positive,
            "background_unverified": negative,
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "true_negatives": None,
            "false_positives": None,
            "false_negatives": None,
            "true_positives": None,
            "evaluation_status": None,
        }

        if positive > 0 and negative > 0:
            tn, fp, fn, tp = confusion_matrix(
                actual, predicted, labels=[0, 1]
            ).ravel()

            row.update({
                "accuracy": float(accuracy_score(actual, predicted)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(actual, predicted)
                ),
                "precision": float(
                    precision_score(actual, predicted, zero_division=0)
                ),
                "recall": float(
                    recall_score(actual, predicted, zero_division=0)
                ),
                "f1": float(
                    f1_score(actual, predicted, zero_division=0)
                ),
                "roc_auc": float(
                    roc_auc_score(actual, scores)
                ),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "evaluation_status": (
                    "Both labels present; provisional random-test metrics"
                ),
            })

        else:
            row["evaluation_status"] = (
                "Insufficient class diversity for binary evaluation"
            )

        results.append(row)

        print("\n" + "=" * 65)
        print("REGION:", region)
        print("Test samples:", len(group))
        print("Historical events:", positive)
        print("Unverified background:", negative)

        if row["accuracy"] is not None:
            print("\nMetrics:")
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ):
                print(f"{metric}: {row[metric]:.4f}")

            print(
                "\nConfusion matrix [[TN, FP], [FN, TP]]:",
                [[row["true_negatives"], row["false_positives"]],
                 [row["false_negatives"], row["true_positives"]]]
            )
        else:
            print("\nBinary metrics unavailable.")
            print(row["evaluation_status"])

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CSV_PATH, index=False)

    report = {
        "evaluation": "Regional breakdown of original India baseline test set",
        "model": "Existing India RandomForestClassifier",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "split_method": "Reconstructed stratified random split",
        "test_fraction": 0.25,
        "random_seed": SEED,
        "total_test_samples": int(len(test_df)),
        "regions": results,
        "limitations": [
            "This is not independent spatial validation.",
            "Background samples are unverified, not confirmed negatives.",
            "Nearby locations may occur in both training and test sets.",
            "Western Maharashtra has no negative samples in the source dataset.",
            "Results are preliminary terrain-only baseline metrics.",
            "Predicted scores are not calibrated landslide probabilities.",
        ],
    }

    JSON_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("REGIONAL EVALUATION COMPLETED")
    print("\nCSV:", CSV_PATH)
    print("JSON:", JSON_PATH)
    print("\nNo model was retrained or modified.")


if __name__ == "__main__":
    main()
