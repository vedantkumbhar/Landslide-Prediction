from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
)

INPUT_CSV = BASE / "western_ghats_experimental_risk_scores.csv"
INPUT_GEOJSON = BASE / "western_ghats_temporal_risk.geojson"


def main():
    df = pd.read_csv(INPUT_CSV)

    print("\n========== RISK SCORE VALIDATION ==========")
    print("Total rows:", len(df))
    print("Unique samples:", df["sample_id"].nunique())

    print("\nTemporal roles:")
    print(df["temporal_role"].value_counts(dropna=False))

    print("\nRecord types:")
    print(df["record_type"].value_counts(dropna=False))

    print("\nRecent risk score availability:")
    print(df["recent_score_status"].value_counts(dropna=False))

    print("\nForecast risk score availability:")
    print(df["forecast_score_status"].value_counts(dropna=False))

    print("\nRisk score statistics:")
    print(
        df[["recent_risk_index", "forecast_risk_index"]]
        .describe()
        .round(2)
    )

    print("\n========== GEOJSON VALIDATION ==========")

    with INPUT_GEOJSON.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson.get("features", [])
    print("GeoJSON features:", len(features))
    print("Metadata:", geojson.get("metadata", {}))

    if not features:
        raise ValueError("GeoJSON contains no features.")

    print("\nSample GeoJSON properties:")
    print(features[0]["properties"])

    print("\n========== VALIDATION COMPLETE ==========")


if __name__ == "__main__":
    main()