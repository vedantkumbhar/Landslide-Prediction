
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

models = {
    "India baseline": DATA / "baseline_model" / "terrain_random_forest.joblib",
    "Western Ghats baseline": (
        DATA / "western_ghats_baseline_model"
        / "western_ghats_terrain_random_forest.joblib"
    ),
}

print("\n========== MODEL CONTENTS ==========")

for name, path in models.items():
    print(f"\n--- {name} ---")

    if not path.exists():
        print("File not found:", path)
        continue

    obj = joblib.load(path)
    print("Stored object type:", type(obj))

    if isinstance(obj, dict):
        print("Dictionary keys:", list(obj.keys()))

        for key, value in obj.items():
            print(f"\nKey: {key}")
            print("Value type:", type(value))

            if hasattr(value, "n_features_in_"):
                print("Expected features:", value.n_features_in_)
                print(
                    "Feature names:",
                    getattr(value, "feature_names_in_", "Not stored")
                )

            if hasattr(value, "classes_"):
                print("Classes:", value.classes_)

            if hasattr(value, "steps"):
                print("Pipeline steps:", value.named_steps.keys())

    elif hasattr(obj, "n_features_in_"):
        print("Expected features:", obj.n_features_in_)
        print("Feature names:", getattr(obj, "feature_names_in_", "Not stored"))
        print("Classes:", getattr(obj, "classes_", "Unknown"))


print("\n========== INDIA DATASET REGIONAL LABELS ==========")

path = DATA / "initial_training_dataset.csv"

if path.exists():
    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    if "dem_region" in df.columns:
        print("\nRecords by region and label:")
        print(pd.crosstab(df["dem_region"], df["label"]))

    if "tile" in df.columns:
        print("\nRecords by tile and label:")
        print(pd.crosstab(df["tile"], df["label"]))

    if "label_confidence" in df.columns:
        print("\nLabel confidence:")
        print(pd.crosstab(df["label"], df["label_confidence"]))

print("\n========== INSPECTION COMPLETE ==========")
