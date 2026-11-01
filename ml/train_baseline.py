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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "initial_training_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "baseline_model"
MODEL_PATH = OUTPUT_DIR / "terrain_random_forest.joblib"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.json"

FEATURES = ["elevation_m", "slope_degrees", "aspect_degrees"]
TARGET = "label"
SEED = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET, "sample_type", "dem_region", "tile"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing feature or label values; refusing to train.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target label must contain only 0 and 1.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required.")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
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

    report = {
        "model": "RandomForestClassifier",
        "purpose": "Preliminary terrain-only baseline; not operational",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "features": FEATURES,
        "target": TARGET,
        "random_seed": SEED,
        "split": {
            "method": "stratified random split",
            "test_fraction": 0.25,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_positive_samples": int((y_test == 1).sum()),
            "test_negative_samples": int((y_test == 0).sum()),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "feature_importance": {
            name: float(value)
            for name, value in zip(FEATURES, model.feature_importances_)
        },
        "dataset_sample_types": {
            str(k): int(v)
            for k, v in df["sample_type"].value_counts().items()
        },
        "dataset_regions": {
            str(k): int(v)
            for k, v in df["dem_region"].value_counts().items()
        },
        "limitations": [
            "Background samples are provisional and are not confirmed non-landslide locations.",
            "Western Maharashtra has no background samples in this training dataset.",
            "A random split may overestimate performance due to spatial dependence.",
            "Only elevation, slope, and aspect are used; rainfall is not included.",
            "This baseline is not a calibrated landslide probability or live warning system.",
        ],
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "random_seed": SEED},
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Baseline training completed.")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("\nEvaluation metrics:")
    for name, value in report["metrics"].items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
    print("\nFeature importance:")
    for name, value in sorted(
        report["feature_importance"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {value:.4f}")
    print("\nModel saved to:", MODEL_PATH)
    print("Report saved to:", REPORT_PATH)
    print("\nWarning: preliminary evaluation only; see limitations in the report.")


if __name__ == "__main__":
    main()
