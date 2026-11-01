from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]

TERRAIN_PATH = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)
RAINFALL_PATH = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "hourly_rainfall.csv"
)

terrain = pd.read_csv(TERRAIN_PATH)
rainfall = pd.read_csv(RAINFALL_PATH)

required_terrain = ["pilot_tile", "latitude", "longitude", "label"]
required_rainfall = [
    "pilot_tile",
    "requested_latitude",
    "requested_longitude",
]

if terrain[required_terrain].isna().any().any():
    raise ValueError("Terrain dataset has missing coordinates or tile IDs.")

if rainfall[required_rainfall].isna().any().any():
    raise ValueError("Rainfall dataset has missing coordinates or tile IDs.")

geod = Geod(ellps="WGS84")
assignments = []

for tile, samples in terrain.groupby("pilot_tile"):
    points = rainfall[
        rainfall["pilot_tile"] == tile
    ][["requested_latitude", "requested_longitude"]].drop_duplicates()

    if points.empty:
        raise ValueError(f"No rainfall points for tile {tile}")

    point_lats = points["requested_latitude"].to_numpy()
    point_lons = points["requested_longitude"].to_numpy()

    for row in samples.itertuples(index=False):
        distances = []

        for lat, lon in zip(point_lats, point_lons):
            _, _, distance_m = geod.inv(
                row.longitude, row.latitude, lon, lat
            )
            distances.append(distance_m)

        nearest = int(np.argmin(distances))

        assignments.append({
            "pilot_tile": tile,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "label": row.label,
            "nearest_rainfall_latitude": point_lats[nearest],
            "nearest_rainfall_longitude": point_lons[nearest],
            "nearest_distance_km": distances[nearest] / 1000.0,
        })

result = pd.DataFrame(assignments)

print("Terrain samples:", len(result))
print("\nNearest rainfall-point distance by tile (km):")
print(
    result.groupby("pilot_tile")["nearest_distance_km"]
    .agg(["min", "median", "mean", "max"])
    .round(2)
    .to_string()
)

print("\nDistance by label:")
print(
    result.groupby("label")["nearest_distance_km"]
    .agg(["count", "median", "mean", "max"])
    .round(2)
    .to_string()
)

print("\nSample counts beyond distance thresholds:")
for threshold in [10, 25, 50, 100]:
    count = int((result["nearest_distance_km"] > threshold).sum())
    print(f"> {threshold} km: {count} / {len(result)}")

print("\nDiagnostic only: no files were modified.")
