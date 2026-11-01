from pathlib import Path
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/pilot_background_samples.csv"
PLAN = ROOT / "data/processed/dem/dem_tile_plan.csv"
OUTPUT = ROOT / "data/processed/pilot_background_features.csv"

FEATURES = {
    "elevation_m": "elevation.tif",
    "slope_degrees": "slope.tif",
    "aspect_degrees": "aspect.tif",
}

def find_raster(tile, region, filename):
    folder_map = {
        "Himalayas": "himalayas",
        "Northeast India": "northeast",
        "Western Maharashtra": "western_maharashtra",
    }
    base = ROOT / "data/processed/dem" / folder_map[region] / tile
    candidates = [base / "metric" / filename, base / filename]
    return next((p for p in candidates if p.exists()), None)

def sample_value(ds, x, y):
    try:
        value = next(ds.sample([(x, y)], masked=True))[0]
        if np.ma.is_masked(value):
            return np.nan
        value = float(value)
        if not np.isfinite(value) or value == ds.nodata:
            return np.nan
        return value
    except (ValueError, IndexError, StopIteration):
        return np.nan

def main():
    samples = pd.read_csv(INPUT)
    plan = pd.read_csv(PLAN)
    tile_to_region = (
        plan.drop_duplicates("tile").set_index("tile")["region"].to_dict()
    )

    cache = {}
    rows = []

    for _, sample in samples.iterrows():
        row = sample.to_dict()
        tile = str(sample["pilot_tile"]).strip().upper()
        region = tile_to_region.get(tile)
        row["dem_region"] = region
        row["terrain_status"] = "missing_tile_or_region"

        if region is None:
            for feature in FEATURES:
                row[feature] = np.nan
            rows.append(row)
            continue

        lat, lon = float(sample["latitude"]), float(sample["longitude"])
        values = {}

        for feature, filename in FEATURES.items():
            key = (tile, feature)
            if key not in cache:
                path = find_raster(tile, region, filename)
                cache[key] = rasterio.open(path) if path else None

            ds = cache[key]
            if ds is None:
                values[feature] = np.nan
            else:
                x, y = transform("EPSG:4326", ds.crs, [lon], [lat])
                values[feature] = sample_value(ds, x[0], y[0])

        row.update(values)
        row["terrain_status"] = (
            "valid" if all(np.isfinite(values[f]) for f in FEATURES)
            else "invalid_or_nodata_sample"
        )
        rows.append(row)

    for ds in cache.values():
        if ds is not None:
            ds.close()

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print("Shape:", output.shape)
    print("\nTerrain status:")
    print(output["terrain_status"].value_counts(dropna=False).to_string())
    print("\nMissing terrain values:")
    print(output[list(FEATURES)].isna().sum().to_string())

if __name__ == "__main__":
    main()
