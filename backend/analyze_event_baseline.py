from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/pilot_model_event_features.csv"
OUT = ROOT / "reports/terrain_rainfall_baseline"

FEATURES = [
    "elevation_m",
    "slope_degrees",
    "previous_day_rainfall_mm",
    "event_day_rainfall_mm",
]

def main():
    df = pd.read_csv(INPUT)

    required = ["pilot_tile"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    OUT.mkdir(parents=True, exist_ok=True)

    summary = df.groupby("pilot_tile")[FEATURES].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(2)

    summary.to_csv(OUT / "tile_feature_summary.csv")

    print("\nEVENT COUNTS BY PILOT TILE")
    print(df["pilot_tile"].value_counts().to_string())

    print("\nOVERALL FEATURE SUMMARY")
    print(df[FEATURES].describe().round(2).to_string())

    print("\nTERRAIN AND RAINFALL SUMMARY BY TILE")
    print(summary.to_string())

    print("\nREPORT FILES")
    print(OUT / "tile_feature_summary.csv")

    print("\nCAUTIONS")
    print("- This dataset contains recorded landslide events only.")
    print("- It cannot estimate landslide probability on its own.")
    print("- Event-day rainfall may include rain after the landslide.")
    print("- These are exploratory summaries, not model validation.")

if __name__ == "__main__":
    main()
