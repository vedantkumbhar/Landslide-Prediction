from pathlib import Path
import math
import re
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords

ROOT = Path(__file__).resolve().parents[1]
EVENTS_FILE = ROOT / "data/processed/india_landslide_events.csv"
PLAN_FILE = ROOT / "data/processed/dem/dem_tile_plan.csv"
OUTPUT_FILE = ROOT / "data/processed/all_region_event_features.csv"
DEM_ROOT = ROOT / "data/processed/dem"

FEATURE_FILES = {
    "elevation_m": "elevation.tif",
    "slope_degrees": "slope.tif",
    "aspect_degrees": "aspect.tif",
}

REGION_FOLDERS = {
    "Himalayas": ["himalayas"],
    "Northeast India": ["northeast", "northeast_india", "northeast-india"],
    "Western Maharashtra": [
        "western_maharashtra",
        "western-maharashtra",
        "westernmaharashtra",
        "western_maharashtra_dem",
    ],
}

def find_column(df, candidates, description):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Cannot identify {description}. Available columns: {df.columns.tolist()}"
    )

def find_tile_directory(tile, region):
    allowed_folders = REGION_FOLDERS.get(region, [])
    matches = [
        p for p in DEM_ROOT.rglob(tile)
        if p.is_dir()
        and p.name.upper() == tile.upper()
        and "backup" not in str(p).lower()
        and any(folder in p.parts for folder in allowed_folders)
    ]
    if not matches:
        return None
    # Prefer the shortest matching path.
    return sorted(matches, key=lambda p: len(p.parts))[0]

def find_raster(tile_dir, filename):
    if tile_dir is None:
        return None
    matches = [
        p for p in tile_dir.rglob(filename)
        if p.is_file() and "backup" not in str(p).lower()
    ]
    if not matches:
        return None
    # Prefer rasters in a metric directory when available.
    matches.sort(key=lambda p: ("metric" not in [part.lower() for part in p.parts], len(p.parts)))
    return matches[0]

def tile_from_coordinates(latitude, longitude):
    # Standard one-degree HGT tile naming for positive Indian coordinates.
    lat_band = math.floor(latitude)
    lon_band = math.floor(longitude)
    ns = "N" if lat_band >= 0 else "S"
    ew = "E" if lon_band >= 0 else "W"
    return f"{ns}{abs(lat_band):02d}{ew}{abs(lon_band):03d}"

def valid_sample(dataset, x, y):
    try:
        sample = next(dataset.sample([(x, y)], masked=True))
        if np.ma.is_masked(sample[0]):
            return None
        value = float(sample[0])
        if not np.isfinite(value) or value == -9999:
            return None
        return value
    except (StopIteration, ValueError, IndexError, TypeError):
        return None

def main():
    print("Loading event dataset and DEM tile plan...")
    events = pd.read_csv(EVENTS_FILE)
    plan = pd.read_csv(PLAN_FILE)

    lat_col = find_column(events, ["latitude", "lat"], "latitude column")
    lon_col = find_column(events, ["longitude", "lon", "long"], "longitude column")

    required_plan_columns = {"region", "tile"}
    if not required_plan_columns.issubset(plan.columns):
        raise ValueError(f"Tile plan must contain {required_plan_columns}")

    plan["tile"] = plan["tile"].astype(str).str.strip().str.upper()
    plan_lookup = plan.drop_duplicates("tile").set_index("tile")["region"].to_dict()

    tile_cache = {}
    raster_cache = {}
    output_rows = []

    print(f"Events: {len(events)} | Planned tiles: {len(plan_lookup)}")

    for index, row in events.iterrows():
        result = row.to_dict()
        result["dem_tile"] = None
        result["dem_region"] = None
        result["terrain_status"] = "not_processed"
        for feature in FEATURE_FILES:
            result[feature] = np.nan

        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            if not np.isfinite(lat) or not np.isfinite(lon):
                result["terrain_status"] = "invalid_coordinates"
                output_rows.append(result)
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                result["terrain_status"] = "invalid_coordinates"
                output_rows.append(result)
                continue
        except (TypeError, ValueError):
            result["terrain_status"] = "invalid_coordinates"
            output_rows.append(result)
            continue

        tile = tile_from_coordinates(lat, lon)
        result["dem_tile"] = tile

        if tile not in plan_lookup:
            result["terrain_status"] = "tile_not_in_plan"
            output_rows.append(result)
            continue

        region = plan_lookup[tile]
        result["dem_region"] = region

        if tile not in tile_cache:
            tile_dir = find_tile_directory(tile, region)
            paths = {
                key: find_raster(tile_dir, filename)
                for key, filename in FEATURE_FILES.items()
            }
            tile_cache[tile] = paths

        paths = tile_cache[tile]
        missing = [name for name, path in paths.items() if path is None]
        if missing:
            result["terrain_status"] = "missing_raster:" + ",".join(missing)
            output_rows.append(result)
            continue

        try:
            # Open the three rasters together and check their grid alignment.
            with rasterio.open(paths["elevation_m"]) as elev, \
                 rasterio.open(paths["slope_degrees"]) as slope, \
                 rasterio.open(paths["aspect_degrees"]) as aspect:

                datasets = {
                    "elevation_m": elev,
                    "slope_degrees": slope,
                    "aspect_degrees": aspect,
                }

                grids_match = all(
                    ds.crs == elev.crs
                    and ds.transform == elev.transform
                    and ds.width == elev.width
                    and ds.height == elev.height
                    for ds in datasets.values()
                )
                if not grids_match:
                    result["terrain_status"] = "raster_grid_mismatch"
                    output_rows.append(result)
                    continue

                if elev.crs is None:
                    result["terrain_status"] = "missing_crs"
                    output_rows.append(result)
                    continue

                if str(elev.crs) != "EPSG:4326":
                    xs, ys = transform_coords("EPSG:4326", elev.crs, [lon], [lat])
                    x, y = xs[0], ys[0]
                else:
                    x, y = lon, lat

                if not (elev.bounds.left <= x <= elev.bounds.right
                        and elev.bounds.bottom <= y <= elev.bounds.top):
                    result["terrain_status"] = "outside_raster_bounds"
                    output_rows.append(result)
                    continue

                for feature, dataset in datasets.items():
                    result[feature] = valid_sample(dataset, x, y)

                if all(pd.notna(result[f]) for f in FEATURE_FILES):
                    result["terrain_status"] = "valid"
                else:
                    result["terrain_status"] = "invalid_or_nodata_sample"

        except Exception as exc:
            result["terrain_status"] = f"sampling_error:{type(exc).__name__}"

        output_rows.append(result)

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(events)} events")

    output = pd.DataFrame(output_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Input events: {len(events)}")
    print(f"Output rows: {len(output)}")
    print("\nTerrain status:")
    print(output["terrain_status"].value_counts(dropna=False).to_string())
    print("\nValid features by region:")
    print(
        output[output["terrain_status"] == "valid"]
        .groupby("dem_region").size().to_string()
    )
    print(f"\nSaved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
