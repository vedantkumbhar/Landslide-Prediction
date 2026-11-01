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

DATA_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "western_ghats_baseline_model"
)

MODEL_PATH = OUTPUT_DIR / "western_ghats_terrain_random_forest.joblib"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.json"

FEATURES = ["elevation_m", "slope_degrees", "aspect_degrees"]
TARGET = "label"
SEED = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Never overwrite an existing model or evaluation report.
    if MODEL_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "Western Ghats model output already exists. "
            "No files were overwritten. Inspect the output directory first: "
            f"{OUTPUT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [
        TARGET,
        "pilot_tile",
        "sample_source",
        "label_description",
    ]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing terrain features or labels; refusing to train.")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite terrain features found; refusing to train.")

    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError("Both binary classes (0 and 1) are required.")

    if (y.value_counts() < 2).any():
        raise ValueError("Each class must have at least two records.")

    X_train, X_test, y_train, y_test, idx_train, idx_test = (
        train_test_split(
            X,
            y,
            df.index,
            test_size=0.25,
            random_state=SEED,
            stratify=y,
        )
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
        "confusion_matrix_labels_0_1": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }

    report = {
        "model": "RandomForestClassifier",
        "purpose": (
            "Separate Western Ghats terrain-only pilot baseline; "
            "not operational"
        ),
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "features": FEATURES,
        "target": TARGET,
        "random_seed": SEED,
        "split": {
            "method": "stratified random split",
            "test_fraction": 0.25,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_positive_samples": int((y_train == 1).sum()),
            "train_negative_samples": int((y_train == 0).sum()),
            "test_positive_samples": int((y_test == 1).sum()),
            "test_negative_samples": int((y_test == 0).sum()),
        },
        "metrics": metrics,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["background_unverified", "historical_event"],
            output_dict=True,
            zero_division=0,
        ),
        "feature_importance": {
            name: float(value)
            for name, value in zip(FEATURES, model.feature_importances_)
        },
        "label_counts": {
            str(k): int(v) for k, v in y.value_counts().items()
        },
        "sample_sources": {
            str(k): int(v)
            for k, v in df["sample_source"].fillna("missing")
            .value_counts().items()
        },
        "pilot_tiles": {
            str(k): int(v)
            for k, v in df["pilot_tile"].fillna("missing")
            .value_counts().items()
        },
        "limitations": [
            "Background samples are unverified and are not confirmed negatives.",
            "Only 56 historical events are represented in this pilot dataset.",
            "Only elevation, slope, and aspect are used; rainfall is not included.",
            "Aspect is treated as a numeric feature, although it is circular.",
            "A random split may overestimate performance due to spatial dependence.",
            "The three pilot tiles do not represent the entire Western Ghats.",
            "Predicted scores are not calibrated probabilities or live warnings.",
            "This model is for research and baseline comparison only.",
        ],
        "software": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "random_seed": SEED,
            "region": "Western Ghats pilot",
        },
        MODEL_PATH,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Western Ghats baseline training completed.")
    print("Dataset shape:", df.shape)
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("Test label counts:")
    print(y_test.value_counts().sort_index().to_string())

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
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
    print("\nWarning: preliminary terrain-only evaluation.")
    print("Background samples are unverified; rainfall is not included.")


if __name__ == "__main__":
    main()
