
from pathlib import Path
import warnings

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

FILES = {
    "India training dataset": DATA / "initial_training_dataset.csv",
    "Himalayas events": DATA / "himalayas_N30E078_event_features.csv",
    "Northeast events": DATA / "northeast_N27E088_event_features.csv",
    "Western Ghats training": DATA / "western_ghats_pilot_training_dataset.csv",
    "Western Ghats rainfall experiment": (
        DATA / "rainfall" / "western_ghats"
        / "historical_validation"
        / "western_ghats_rainfall_terrain_experimental_dataset.csv"
    ),
}

MODELS = {
    "India baseline": (
        DATA / "baseline_model" / "terrain_random_forest.joblib"
    ),
    "Western Ghats baseline": (
        DATA / "western_ghats_baseline_model"
        / "western_ghats_terrain_random_forest.joblib"
    ),
}


def inspect_dataset(name, path):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    if not path.exists():
        print("MISSING:", path)
        return

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    for col in [
        "label", "label_type", "sample_source",
        "sample_type", "region", "pilot_tile",
        "terrain_valid", "rainfall_matched"
    ]:
        if col in df.columns:
            print(f"\n{col} counts:")
            print(df[col].value_counts(dropna=False).head(15))

    print("\nMissing values in first 20 columns:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(20) if not missing.empty else "No missing values")

    terrain = [
        c for c in
        ["elevation_m", "slope_degrees", "aspect_degrees"]
        if c in df.columns
    ]

    if terrain:
        print("\nTerrain summary:")
        print(df[terrain].describe().round(3))

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))


def inspect_model(name, path):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    if not path.exists():
        print("MISSING:", path)
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)

        print("Model type:", type(model).__name__)

        if hasattr(model, "n_features_in_"):
            print("Expected feature count:", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            print("Expected feature names:")
            print(list(model.feature_names_in_))

        if hasattr(model, "classes_"):
            print("Model classes:", model.classes_)

        if hasattr(model, "steps"):
            print("Pipeline steps:")
            print([name for name, _ in model.steps])

        if hasattr(model, "get_params"):
            params = model.get_params()
            print("Model parameters:")
            print({
                key: params[key]
                for key in [
                    "n_estimators",
                    "max_depth",
                    "random_state",
                    "class_weight"
                ]
                if key in params
            })

    except Exception as exc:
        print("MODEL LOAD ERROR:", repr(exc))


def main():
    print("LANDSLIDE MODEL READINESS INSPECTION")
    print("Read-only inspection. No files will be modified.")

    for name, path in FILES.items():
        inspect_dataset(name, path)

    for name, path in MODELS.items():
        inspect_model(name, path)

    print("\n" + "=" * 75)
    print("INSPECTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
