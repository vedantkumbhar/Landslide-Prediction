
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TERRAIN_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

RAINFALL_PATH = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "rainfall_features.csv"
)

OUTPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_live_risk_inputs.csv"
)

RAIN_FEATURES = [
    "recent_rain_1h_mm",
    "recent_rain_3h_mm",
    "recent_rain_6h_mm",
    "recent_rain_12h_mm",
    "recent_rain_24h_mm",
    "forecast_rain_3h_mm",
    "forecast_rain_6h_mm",
    "forecast_rain_24h_mm",
]

TERRAIN_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    terrain = pd.read_csv(TERRAIN_PATH)
    rainfall = pd.read_csv(RAINFALL_PATH)

    required_terrain = [
        "pilot_tile", "latitude", "longitude",
        "label", *TERRAIN_FEATURES,
    ]
    required_rainfall = [
        "sample_id", "pilot_tile",
        "requested_latitude", "requested_longitude",
        "api_latitude", "api_longitude",
        "timestamp_local", "record_type",
        "precipitation_mm", *RAIN_FEATURES,
    ]

    for name, frame, required in [
        ("terrain", terrain, required_terrain),
        ("rainfall", rainfall, required_rainfall),
    ]:
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

    if terrain.duplicated(["pilot_tile", "latitude", "longitude"]).any():
        raise ValueError("Duplicate terrain sample coordinates detected.")

    if rainfall.duplicated(["sample_id", "timestamp_local"]).any():
        raise ValueError("Duplicate rainfall sample timestamps detected.")

    # Match coordinates rounded to the precision used in API requests.
    terrain = terrain.copy()
    rainfall = rainfall.copy()

    terrain["_lat_key"] = terrain["latitude"].round(6)
    terrain["_lon_key"] = terrain["longitude"].round(6)
    rainfall["_lat_key"] = rainfall["requested_latitude"].round(6)
    rainfall["_lon_key"] = rainfall["requested_longitude"].round(6)

    terrain_columns = [
        "pilot_tile", "_lat_key", "_lon_key",
        "latitude", "longitude", "label",
        *TERRAIN_FEATURES,
    ]
    terrain_lookup = terrain[terrain_columns].rename(columns={
        "latitude": "terrain_latitude",
        "longitude": "terrain_longitude",
        "label": "historical_label_reference_only",
    })

    merged = rainfall.merge(
        terrain_lookup,
        on=["pilot_tile", "_lat_key", "_lon_key"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} rainfall rows could not be matched to terrain."
        )

    merged = merged.drop(columns=["_merge", "_lat_key", "_lon_key"])

    if len(merged) != len(rainfall):
        raise ValueError("Merge changed the rainfall row count.")

    if merged[TERRAIN_FEATURES].isna().any().any():
        raise ValueError("Missing terrain features after joining.")

    numeric_features = TERRAIN_FEATURES + [
        "precipitation_mm", *RAIN_FEATURES
    ]
    for column in numeric_features:
        merged[column] = pd.to_numeric(
            merged[column], errors="coerce"
        )

    # Missing rolling windows at time-series boundaries are expected.
    # Terrain and current hourly precipitation must always be present.
    essential = TERRAIN_FEATURES + ["precipitation_mm"]
    if merged[essential].isna().any().any():
        raise ValueError("Missing essential terrain/rainfall values.")

    if not np.isfinite(merged[essential].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite essential input features detected.")

    merged["timestamp_local"] = pd.to_datetime(
        merged["timestamp_local"], errors="raise"
    )

    # Keep labels only as traceability metadata; never use them as
    # live predictors or interpret them as event-time rainfall labels.
    merged["historical_label_reference_only"] = (
        merged["historical_label_reference_only"].astype(int)
    )

    merged = merged.sort_values(
        ["pilot_tile", "sample_id", "timestamp_local"]
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation protects existing files.
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        merged.to_csv(handle, index=False)

    print("Western Ghats live-risk input dataset created.")
    print("Terrain samples:", terrain.shape[0])
    print("Rainfall input rows:", rainfall.shape[0])
    print("Output rows:", merged.shape[0])
    print("Unique samples:", merged["sample_id"].nunique())
    print("Tiles:", sorted(merged["pilot_tile"].unique()))
    print("Missing essential features:", int(
        merged[essential].isna().sum().sum()
    ))

    print("\nRainfall feature availability:")
    for column in RAIN_FEATURES:
        print(
            f"{column}: {merged[column].notna().sum()} available, "
            f"{merged[column].isna().sum()} missing"
        )

    print("\nRows by tile and record type:")
    print(
        merged.groupby(["pilot_tile", "record_type"])
        .size().to_string()
    )

    print("\nOutput:", OUTPUT)
    print("Historical labels are reference metadata, not live predictors.")
    print("Existing terrain models and rainfall datasets were not modified.")


if __name__ == "__main__":
    main()
