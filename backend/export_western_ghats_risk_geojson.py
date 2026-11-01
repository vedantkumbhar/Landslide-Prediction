
from pathlib import Path
import json
import math

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_experimental_risk_scores.csv"
)

OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_temporal_risk.geojson"
)


def safe_value(value):
    """Convert pandas/numpy values to valid JSON-compatible values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {OUTPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "sample_id",
        "pilot_tile",
        "terrain_latitude",
        "terrain_longitude",
        "timestamp_local",
        "record_type",
        "temporal_role",
        "slope_degrees",
        "elevation_m",
        "recent_rain_6h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_6h_mm",
        "forecast_rain_24h_mm",
        "recent_risk_index",
        "forecast_risk_index",
        "recent_score_status",
        "forecast_score_status",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Export only current/forecast candidate timestamps.
    df = df[
        df["temporal_role"] == "current_or_forecast_candidate"
    ].copy()

    if df.empty:
        raise ValueError("No current/forecast candidate records found.")

    for column in ["terrain_latitude", "terrain_longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["terrain_latitude", "terrain_longitude"]].isna().any().any():
        raise ValueError("Missing terrain coordinates.")

    if not df["terrain_latitude"].between(-90, 90).all():
        raise ValueError("Latitude outside valid geographic bounds.")

    if not df["terrain_longitude"].between(-180, 180).all():
        raise ValueError("Longitude outside valid geographic bounds.")

    for column in ["recent_risk_index", "forecast_risk_index"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        valid = df[column].dropna()
        if not valid.between(0, 100).all():
            raise ValueError(f"{column} contains values outside 0–100.")

    properties_columns = [
        "sample_id",
        "pilot_tile",
        "timestamp_local",
        "record_type",
        "temporal_role",
        "elevation_m",
        "slope_degrees",
        "recent_rain_6h_mm",
        "recent_rain_24h_mm",
        "forecast_rain_6h_mm",
        "forecast_rain_24h_mm",
        "recent_risk_index",
        "forecast_risk_index",
        "recent_score_status",
        "forecast_score_status",
        "score_interpretation",
    ]

    # Never expose the training label as a GIS prediction attribute.
    if "score_interpretation" not in df.columns:
        df["score_interpretation"] = (
            "Experimental index; not a calibrated probability or warning"
        )

    features = []

    for row in df.itertuples(index=False):
        record = row._asdict()

        properties = {
            column: safe_value(record.get(column))
            for column in properties_columns
            if column in record
        }

        # An explicit status prevents incomplete windows being mistaken
        # for a zero-risk result.
        if properties["recent_risk_index"] is None:
            properties["recent_display_status"] = "incomplete"
        else:
            properties["recent_display_status"] = "experimental_index"

        if properties["forecast_risk_index"] is None:
            properties["forecast_display_status"] = "incomplete"
        else:
            properties["forecast_display_status"] = "experimental_index"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    safe_value(record["terrain_longitude"]),
                    safe_value(record["terrain_latitude"]),
                ],
            },
            "properties": properties,
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "Western_Ghats_Experimental_Temporal_Risk",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            },
        },
        "metadata": {
            "purpose": "Experimental pilot screening layer",
            "pilot_tiles": sorted(df["pilot_tile"].unique().tolist()),
            "feature_count": len(features),
            "interpretation": (
                "Indices are illustrative, uncalibrated screening values. "
                "They are not landslide probabilities, official warnings, "
                "or evidence that a location is safe."
            ),
            "coverage_note": (
                "Point samples only; not a continuous map of the Western Ghats."
            ),
        },
        "features": features,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(geojson, handle, ensure_ascii=False, allow_nan=False)

    print("GeoJSON export completed.")
    print("Features:", len(features))
    print("Unique sample locations:", df["sample_id"].nunique())
    print("Tiles:", sorted(df["pilot_tile"].unique()))
    print("Recent score missing:", df["recent_risk_index"].isna().sum())
    print("Forecast score missing:", df["forecast_risk_index"].isna().sum())
    print("Output:", OUTPUT)
    print("CRS: WGS 84 longitude/latitude (EPSG:4326)")
    print("Existing datasets and models were not modified.")


if __name__ == "__main__":
    main()
