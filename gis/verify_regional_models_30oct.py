
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path("data/processed/regional_models_30oct")

REGIONS = {
    "Himalayas": "himalayas",
    "Northeast India": "northeast_india",
    "Western Maharashtra": "western_maharashtra",
}

FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

for region, folder in REGIONS.items():
    print(f"\n========== {region.upper()} ==========")

    model_path = BASE / folder / "random_forest_model.joblib"
    report_path = BASE / folder / "evaluation_report.json"

    if not model_path.exists() or not report_path.exists():
        raise FileNotFoundError(f"Missing model or report for {region}")

    model = joblib.load(model_path)
    report = pd.read_json(report_path, typ="series")

    # Representative terrain input for a smoke test only.
    test_input = pd.DataFrame([{
        "elevation_m": 1500.0,
        "slope_degrees": 20.0,
        "aspect_sin": 0.5,
        "aspect_cos": 0.8660254,
        "aspect_flat": 0,
    }], columns=FEATURES)

    prediction = int(model.predict(test_input)[0])
    score = float(model.predict_proba(test_input)[0, 1])

    if prediction not in (0, 1):
        raise RuntimeError("Unexpected prediction label.")

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError("Invalid model score.")

    print("Model loaded: SUCCESS")
    print("Prediction generated: SUCCESS")
    print("Output label:", prediction)
    print("Model score:", round(score, 4))
    print("Report loaded: SUCCESS")
    print("Validation ROC-AUC:", report["metrics"]["roc_auc"])

print("\nAll three regional models passed the smoke test.")
print("No model files were modified.")
