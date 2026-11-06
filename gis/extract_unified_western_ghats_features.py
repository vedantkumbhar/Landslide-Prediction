"""
Unified Western Ghats Event Terrain Feature Extraction.

Extracts terrain metrics (elevation, slope, aspect) for all historical landslide
events located within the 8 existing processed DEM tiles:
- Western Maharashtra Pilot (5 tiles): N15E073, N15E074, N16E073, N17E073, N18E073
- Existing Western Ghats (3 tiles): N09E076, N11E076, N19E072

Preserves all original event attributes, geographic coordinates, and regional metadata.
Validates that all 112 historical events obtain complete, finite terrain values.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "india_landslide_events.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_unified_event_features.csv"
)

# Directory locations for the 8 existing processed DEM tiles
TILE_DIRECTORIES = {
    "N15E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N15E073" / "metric",
    "N15E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N15E074" / "metric",
    "N16E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N16E073" / "metric",
    "N17E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N17E073" / "metric",
    "N18E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N18E073" / "metric",
    "N09E076": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N09E076" / "metric",
    "N11E076": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N11E076" / "metric",
    "N19E072": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N19E072" / "metric",
}

TILE_SUBREGIONS = {
    "N15E073": "Western Maharashtra Pilot",
    "N15E074": "Western Maharashtra Pilot",
    "N16E073": "Western Maharashtra Pilot",
    "N17E073": "Western Maharashtra Pilot",
    "N18E073": "Western Maharashtra Pilot",
    "N09E076": "Southern Western Ghats (Kerala)",
    "N11E076": "Southern Western Ghats (Kerala / Tamil Nadu)",
    "N19E072": "Northern Western Ghats (Maharashtra)",
}

RASTER_FILES = {
    "elevation_m": "elevation.tif",
    "slope_degrees": "slope.tif",
    "aspect_degrees": "aspect.tif",
}


def tile_from_coordinates(latitude: float, longitude: float) -> str:
    """Return standard 1-degree SRTM tile name for coordinates."""
    lat_band = int(math.floor(latitude))
    lon_band = int(math.floor(longitude))
    ns = "N" if lat_band >= 0 else "S"
    ew = "E" if lon_band >= 0 else "W"
    return f"{ns}{abs(lat_band):02d}{ew}{abs(lon_band):03d}"


def load_tile_rasters(tile: str, tile_dir: Path):
    """Open and validate the three aligned terrain rasters for a tile."""
    paths = {
        feature: tile_dir / filename
        for feature, filename in RASTER_FILES.items()
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{tile}: missing required raster(s):\n" + "\n".join(missing)
        )

    datasets = {
        feature: rasterio.open(path)
        for feature, path in paths.items()
    }

    reference = datasets["elevation_m"]

    for feature, ds in datasets.items():
        if ds.crs is None:
            raise ValueError(f"{tile}: {feature} raster has no CRS.")

        if (
            ds.crs != reference.crs
            or ds.transform != reference.transform
            or ds.width != reference.width
            or ds.height != reference.height
        ):
            raise ValueError(f"{tile}: terrain rasters are not aligned.")

    return datasets


def sample_value(dataset, row: int, col: int) -> float:
    """Return a valid raster value or np.nan for NoData / invalid pixels."""
    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        return np.nan

    value = dataset.read(
        1,
        window=((row, row + 1), (col, col + 1)),
        masked=True,
    )[0, 0]

    if np.ma.is_masked(value):
        return np.nan

    number = float(value)

    if not np.isfinite(number):
        return np.nan

    if dataset.nodata is not None and number == dataset.nodata:
        return np.nan

    return number


def main():
    print("=" * 70)
    print("UNIFIED WESTERN GHATS EVENT FEATURE EXTRACTION (8 PROCESSED TILES)")
    print("=" * 70)

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Input events file not found: {EVENTS_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output file already exists; refusing to overwrite: {OUTPUT_FILE}"
        )

    # 1. Load historical landslide inventory
    events = pd.read_csv(EVENTS_FILE, low_memory=False)
    print(f"Total historical India landslide records loaded: {len(events)}")

    for col in ["latitude", "longitude"]:
        if col not in events.columns:
            raise ValueError(f"Missing coordinate column: {col}")
        events[col] = pd.to_numeric(events[col], errors="coerce")

    # Filter events with valid coordinates
    valid_coords = (
        events["latitude"].between(-90, 90)
        & events["longitude"].between(-180, 180)
    )
    events = events[valid_coords].copy()

    # Determine 1-degree tile for each event
    events["dem_tile"] = events.apply(
        lambda r: tile_from_coordinates(r["latitude"], r["longitude"]),
        axis=1,
    )

    # 2. Filter for events in the 8 existing processed tiles
    target_tiles = set(TILE_DIRECTORIES.keys())
    wg_subset = events[events["dem_tile"].isin(target_tiles)].copy()

    print(f"\nEvents located within the 8 target tiles: {len(wg_subset)}")

    # 3. Validation: duplicate event IDs
    dup_ids = wg_subset[wg_subset["event_id"].duplicated(keep=False)]
    if len(dup_ids) > 0:
        print(f"WARNING: Found {len(dup_ids)} duplicate event ID records:")
        print(dup_ids[["event_id", "latitude", "longitude", "dem_tile"]])
        raise ValueError("Duplicate event IDs detected in target subset.")
    else:
        print("Duplicate event ID check: PASSED (0 duplicates).")

    # 4. Initialize terrain columns
    wg_subset["dem_region"] = "Western Ghats"
    wg_subset["subregion"] = wg_subset["dem_tile"].map(TILE_SUBREGIONS)
    wg_subset["terrain_source_tile"] = wg_subset["dem_tile"]
    wg_subset["elevation_m"] = np.nan
    wg_subset["slope_degrees"] = np.nan
    wg_subset["aspect_degrees"] = np.nan
    wg_subset["aspect_is_flat"] = False
    wg_subset["aspect_sin"] = np.nan
    wg_subset["aspect_cos"] = np.nan
    wg_subset["aspect_flat"] = 0
    wg_subset["terrain_valid"] = False
    wg_subset["terrain_status"] = "pending_extraction"

    # 5. Extract raster values tile-by-tile
    raster_cache = {}
    tile_event_counts = {}

    try:
        for tile, tile_dir in TILE_DIRECTORIES.items():
            tile_indices = wg_subset.index[wg_subset["dem_tile"] == tile].tolist()
            tile_event_counts[tile] = len(tile_indices)
            print(f"\n--- Processing Tile: {tile} ({len(tile_indices)} events) ---")
            print(f"  Raster path: {tile_dir.relative_to(PROJECT_ROOT)}")

            if not tile_indices:
                print(f"  No events found in tile {tile}.")
                continue

            datasets = load_tile_rasters(tile, tile_dir)
            raster_cache[tile] = datasets
            reference = datasets["elevation_m"]

            # Transform from EPSG:4326 to tile raster CRS (e.g. UTM 43N)
            transformer = Transformer.from_crs(
                "EPSG:4326",
                reference.crs,
                always_xy=True,
            )

            lons = wg_subset.loc[tile_indices, "longitude"].to_numpy(dtype=float)
            lats = wg_subset.loc[tile_indices, "latitude"].to_numpy(dtype=float)

            xs, ys = transformer.transform(lons, lats)
            rows, cols = rasterio.transform.rowcol(reference.transform, xs, ys)

            valid_in_tile = 0
            for idx, row, col in zip(tile_indices, rows, cols):
                elev = sample_value(datasets["elevation_m"], row, col)
                slope = sample_value(datasets["slope_degrees"], row, col)
                aspect = sample_value(datasets["aspect_degrees"], row, col)

                wg_subset.at[idx, "elevation_m"] = elev
                wg_subset.at[idx, "slope_degrees"] = slope
                wg_subset.at[idx, "aspect_degrees"] = aspect

                if (
                    np.isfinite(elev)
                    and np.isfinite(slope)
                    and np.isfinite(aspect)
                ):
                    valid_in_tile += 1
                    wg_subset.at[idx, "terrain_valid"] = True
                    wg_subset.at[idx, "terrain_status"] = "valid"

                    # Calculate continuous aspect features
                    is_flat = aspect < 0
                    wg_subset.at[idx, "aspect_is_flat"] = is_flat
                    wg_subset.at[idx, "aspect_flat"] = int(is_flat)

                    if is_flat:
                        wg_subset.at[idx, "aspect_sin"] = 0.0
                        wg_subset.at[idx, "aspect_cos"] = 0.0
                    else:
                        rad = np.deg2rad(aspect)
                        wg_subset.at[idx, "aspect_sin"] = float(np.sin(rad))
                        wg_subset.at[idx, "aspect_cos"] = float(np.cos(rad))
                else:
                    wg_subset.at[idx, "terrain_valid"] = False
                    wg_subset.at[idx, "terrain_status"] = "invalid_or_nodata"

            print(f"  Valid terrain points extracted: {valid_in_tile}/{len(tile_indices)}")

    finally:
        for datasets in raster_cache.values():
            for ds in datasets.values():
                ds.close()

    # 6. Comprehensive Verification
    print("\n" + "=" * 70)
    print("VALIDATION AND QUALITY AUDIT")
    print("=" * 70)

    total_records = len(wg_subset)
    unique_ids = wg_subset["event_id"].nunique()
    valid_count = wg_subset["terrain_valid"].sum()
    invalid_count = total_records - valid_count

    print(f"Total events in output subset: {total_records}")
    print(f"Unique event IDs: {unique_ids}")
    print(f"Valid terrain records: {valid_count}")
    print(f"Invalid or missing terrain records: {invalid_count}")

    if total_records != 112:
        raise ValueError(
            f"Expected exactly 112 events across 8 tiles; got {total_records}"
        )

    if unique_ids != 112:
        raise ValueError(
            f"Expected 112 unique event IDs; got {unique_ids}"
        )

    if invalid_count > 0:
        invalid_rows = wg_subset[~wg_subset["terrain_valid"]]
        print("\nInvalid records detail:")
        print(invalid_rows[["event_id", "dem_tile", "latitude", "longitude", "terrain_status"]])
        raise ValueError(f"Found {invalid_count} invalid terrain records.")

    # Range assertions
    elev_min = wg_subset["elevation_m"].min()
    elev_max = wg_subset["elevation_m"].max()
    slope_min = wg_subset["slope_degrees"].min()
    slope_max = wg_subset["slope_degrees"].max()
    aspect_min = wg_subset["aspect_degrees"].min()
    aspect_max = wg_subset["aspect_degrees"].max()

    print(f"\nElevation range: {elev_min:.1f} m to {elev_max:.1f} m")
    print(f"Slope range:     {slope_min:.2f} deg to {slope_max:.2f} deg")
    print(f"Aspect range:    {aspect_min:.2f} deg to {aspect_max:.2f} deg")

    if not (slope_min >= 0 and slope_max <= 90):
        raise ValueError(f"Slope values out of bounds [0, 90]: min={slope_min}, max={slope_max}")

    if not (aspect_min >= -1 and aspect_max <= 360):
        raise ValueError(f"Aspect values out of bounds [-1, 360]: min={aspect_min}, max={aspect_max}")

    # 7. Write output CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wg_subset.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS: Output saved to:\n  {OUTPUT_FILE}")

    print("\nTile-wise event summary:")
    print("-" * 50)
    for tile, count in sorted(tile_event_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tile:<10} : {count:2d} events ({TILE_SUBREGIONS[tile]})")
    print("-" * 50)
    print(f"  {'TOTAL':<10} : {sum(tile_event_counts.values()):2d} events")


if __name__ == "__main__":
    main()
