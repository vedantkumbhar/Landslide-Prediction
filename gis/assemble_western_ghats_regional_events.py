"""
Western Ghats Regional Landslide Event Dataset Assembly.

Extracts terrain metrics (elevation, slope, aspect) for historical landslide events
in the 6 newly processed Tier 1 DEM tiles:
- N19E073, N10E076, N10E077, N12E074, N14E074, N12E075

Combines them with the 112 historical events from the Stage 1 unified dataset:
- Western Maharashtra Pilot (5 tiles): N15E073, N15E074, N16E073, N17E073, N18E073
- Existing Western Ghats (3 tiles): N09E076, N11E076, N19E072

Generates the unified 14-tile regional event dataset:
- Output: data/processed/western_ghats_regional_event_features.csv
- Expected total: approximately 182 unique historical events across 14 tiles.

Strict safety:
- Never overwrites existing files.
- Preserves all original event attributes.
- Validates coordinate validity, ID uniqueness, and terrain completeness.
- Properly handles flat-terrain sentinel (-1) and circular aspect trigonometry.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "india_landslide_events.csv"
STAGE1_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_unified_event_features.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_event_features.csv"
)

# Directory locations for the 6 newly processed Tier 1 DEM tiles
TIER1_DIRECTORIES = {
    "N19E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N19E073" / "metric",
    "N10E076": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N10E076" / "metric",
    "N10E077": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N10E077" / "metric",
    "N12E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N12E074" / "metric",
    "N14E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N14E074" / "metric",
    "N12E075": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N12E075" / "metric",
}

TIER1_SUBREGIONS = {
    "N19E073": "Northern Western Ghats (Maharashtra)",
    "N10E076": "Southern Western Ghats (Kerala)",
    "N10E077": "Southern Western Ghats (Tamil Nadu / Kerala)",
    "N12E074": "Central Western Ghats (Karnataka / Kerala)",
    "N14E074": "Central Western Ghats (Karnataka)",
    "N12E075": "Central Western Ghats (Karnataka)",
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


def extract_tier1_features() -> pd.DataFrame:
    """Extract terrain features for events in the 6 Tier 1 DEM tiles."""
    if not MASTER_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Input events file not found: {MASTER_EVENTS_FILE}")

    events = pd.read_csv(MASTER_EVENTS_FILE, low_memory=False)
    print(f"Total master India landslide records loaded: {len(events)}")

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

    # Filter for events in the 6 Tier 1 tiles
    target_tiles = set(TIER1_DIRECTORIES.keys())
    t1_subset = events[events["dem_tile"].isin(target_tiles)].copy()

    print(f"Events located within the 6 Tier 1 tiles: {len(t1_subset)}")

    # Validation: duplicate event IDs
    dup_ids = t1_subset[t1_subset["event_id"].duplicated(keep=False)]
    if len(dup_ids) > 0:
        raise ValueError(f"Duplicate event IDs detected in Tier 1 subset: {len(dup_ids)}")

    # Initialize terrain columns
    t1_subset["dem_region"] = "Western Ghats"
    t1_subset["subregion"] = t1_subset["dem_tile"].map(TIER1_SUBREGIONS)
    t1_subset["terrain_source_tile"] = t1_subset["dem_tile"]
    t1_subset["elevation_m"] = np.nan
    t1_subset["slope_degrees"] = np.nan
    t1_subset["aspect_degrees"] = np.nan
    t1_subset["aspect_is_flat"] = False
    t1_subset["aspect_sin"] = np.nan
    t1_subset["aspect_cos"] = np.nan
    t1_subset["aspect_flat"] = 0
    t1_subset["terrain_valid"] = False
    t1_subset["terrain_status"] = "pending_extraction"

    raster_cache = {}
    tile_event_counts = {}

    try:
        for tile, tile_dir in TIER1_DIRECTORIES.items():
            tile_indices = t1_subset.index[t1_subset["dem_tile"] == tile].tolist()
            tile_event_counts[tile] = len(tile_indices)
            print(f"\n--- Extracting Tier 1 Tile: {tile} ({len(tile_indices)} events) ---")
            print(f"  Raster path: {tile_dir.relative_to(PROJECT_ROOT)}")

            if not tile_indices:
                print(f"  No events found in tile {tile}.")
                continue

            datasets = load_tile_rasters(tile, tile_dir)
            raster_cache[tile] = datasets
            reference = datasets["elevation_m"]

            transformer = Transformer.from_crs(
                "EPSG:4326",
                reference.crs,
                always_xy=True,
            )

            lons = t1_subset.loc[tile_indices, "longitude"].to_numpy(dtype=float)
            lats = t1_subset.loc[tile_indices, "latitude"].to_numpy(dtype=float)

            xs, ys = transformer.transform(lons, lats)
            rows, cols = rasterio.transform.rowcol(reference.transform, xs, ys)

            valid_in_tile = 0
            for idx, row, col in zip(tile_indices, rows, cols):
                elev = sample_value(datasets["elevation_m"], row, col)
                slope = sample_value(datasets["slope_degrees"], row, col)
                aspect = sample_value(datasets["aspect_degrees"], row, col)

                t1_subset.at[idx, "elevation_m"] = elev
                t1_subset.at[idx, "slope_degrees"] = slope
                t1_subset.at[idx, "aspect_degrees"] = aspect

                if (
                    np.isfinite(elev)
                    and np.isfinite(slope)
                    and np.isfinite(aspect)
                ):
                    valid_in_tile += 1
                    t1_subset.at[idx, "terrain_valid"] = True
                    t1_subset.at[idx, "terrain_status"] = "valid"

                    is_flat = aspect < 0
                    t1_subset.at[idx, "aspect_is_flat"] = is_flat
                    t1_subset.at[idx, "aspect_flat"] = int(is_flat)

                    if is_flat:
                        t1_subset.at[idx, "aspect_sin"] = 0.0
                        t1_subset.at[idx, "aspect_cos"] = 0.0
                    else:
                        rad = np.deg2rad(aspect)
                        t1_subset.at[idx, "aspect_sin"] = float(np.sin(rad))
                        t1_subset.at[idx, "aspect_cos"] = float(np.cos(rad))
                else:
                    t1_subset.at[idx, "terrain_valid"] = False
                    t1_subset.at[idx, "terrain_status"] = "invalid_or_nodata"

            print(f"  Valid terrain points extracted: {valid_in_tile}/{len(tile_indices)}")

    finally:
        for datasets in raster_cache.values():
            for ds in datasets.values():
                ds.close()

    print("\nTier 1 extraction complete:")
    for tile, count in sorted(tile_event_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tile:<10} : {count:2d} events ({TIER1_SUBREGIONS[tile]})")

    return t1_subset


def main():
    print("=" * 75)
    print("STAGE 4: WESTERN GHATS REGIONAL DATASET ASSEMBLY (14 DEM TILES)")
    print("=" * 75)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output file already exists; refusing to overwrite: {OUTPUT_FILE}"
        )

    if not STAGE1_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Stage 1 unified event dataset not found: {STAGE1_DATASET_FILE}"
        )

    # 1. Load Stage 1 dataset (112 events across 8 tiles)
    print(f"\nLoading Stage 1 unified dataset: {STAGE1_DATASET_FILE.relative_to(PROJECT_ROOT)}")
    stage1_df = pd.read_csv(STAGE1_DATASET_FILE, low_memory=False)
    print(f"Stage 1 events loaded: {len(stage1_df)}")

    if len(stage1_df) != 112:
        raise ValueError(f"Expected 112 Stage 1 events; found {len(stage1_df)}")

    # 2. Extract terrain features for the 6 Tier 1 tiles
    print("\nExtracting terrain features for 6 Tier 1 tiles...")
    tier1_df = extract_tier1_features()
    print(f"\nTier 1 events extracted: {len(tier1_df)}")

    if len(tier1_df) != 70:
        raise ValueError(f"Expected 70 Tier 1 events; found {len(tier1_df)}")

    # 3. Verify column parity
    stage1_cols = list(stage1_df.columns)
    tier1_cols = list(tier1_df.columns)

    if stage1_cols != tier1_cols:
        missing_in_t1 = set(stage1_cols) - set(tier1_cols)
        missing_in_s1 = set(tier1_cols) - set(stage1_cols)
        raise ValueError(
            f"Column mismatch between Stage 1 and Tier 1 datasets.\n"
            f"Missing in Tier 1: {missing_in_t1}\n"
            f"Missing in Stage 1: {missing_in_s1}"
        )

    # 4. Check for overlapping event IDs
    overlap_ids = set(stage1_df["event_id"]).intersection(set(tier1_df["event_id"]))
    if overlap_ids:
        raise ValueError(f"Found overlapping event IDs between Stage 1 and Tier 1: {overlap_ids}")

    # 5. Concatenate into regional dataset
    regional_df = pd.concat([stage1_df, tier1_df], ignore_index=True)
    print(f"\nRegional combined event count: {len(regional_df)}")

    # 6. Quality Audit & Integrity Assertions
    print("\n" + "=" * 75)
    print("REGIONAL DATASET QUALITY AUDIT & INTEGRITY VERIFICATION")
    print("=" * 75)

    total_events = len(regional_df)
    unique_ids = regional_df["event_id"].nunique()
    valid_terrain_count = regional_df["terrain_valid"].sum()
    invalid_terrain_count = total_events - valid_terrain_count

    print(f"Total historical events: {total_events}")
    print(f"Unique event IDs:        {unique_ids}")
    print(f"Valid terrain records:   {valid_terrain_count}")
    print(f"Invalid terrain records: {invalid_terrain_count}")

    if total_events != 182:
        raise ValueError(f"Expected exactly 182 regional events; got {total_events}")

    if unique_ids != 182:
        raise ValueError(f"Expected 182 unique event IDs; got {unique_ids}")

    if invalid_terrain_count > 0:
        invalid_rows = regional_df[~regional_df["terrain_valid"]]
        print("\nInvalid terrain records detail:")
        print(invalid_rows[["event_id", "dem_tile", "latitude", "longitude", "terrain_status"]])
        raise ValueError(f"Found {invalid_terrain_count} invalid terrain records.")

    # Range assertions
    elev_min = float(regional_df["elevation_m"].min())
    elev_max = float(regional_df["elevation_m"].max())
    slope_min = float(regional_df["slope_degrees"].min())
    slope_max = float(regional_df["slope_degrees"].max())
    aspect_min = float(regional_df["aspect_degrees"].min())
    aspect_max = float(regional_df["aspect_degrees"].max())

    print(f"\nRegional Elevation range: {elev_min:.1f} m to {elev_max:.1f} m")
    print(f"Regional Slope range:     {slope_min:.2f} deg to {slope_max:.2f} deg")
    print(f"Regional Aspect range:    {aspect_min:.2f} deg to {aspect_max:.2f} deg")

    if not (slope_min >= 0 and slope_max <= 90):
        raise ValueError(f"Slope values out of bounds [0, 90]: min={slope_min}, max={slope_max}")

    if not (aspect_min >= -1 and aspect_max < 360):
        raise ValueError(f"Aspect values out of bounds [-1, 360): min={aspect_min}, max={aspect_max}")

    # Verify aspect_sin and aspect_cos bounds
    sin_min = float(regional_df["aspect_sin"].min())
    sin_max = float(regional_df["aspect_sin"].max())
    cos_min = float(regional_df["aspect_cos"].min())
    cos_max = float(regional_df["aspect_cos"].max())
    print(f"Regional Aspect sin range: [{sin_min:.4f}, {sin_max:.4f}]")
    print(f"Regional Aspect cos range: [{cos_min:.4f}, {cos_max:.4f}]")

    assert -1.0 <= sin_min and sin_max <= 1.0, f"aspect_sin out of [-1, 1]: [{sin_min}, {sin_max}]"
    assert -1.0 <= cos_min and cos_max <= 1.0, f"aspect_cos out of [-1, 1]: [{cos_min}, {cos_max}]"

    # Verify all 14 tiles are present and report breakdown
    tile_counts = regional_df["dem_tile"].value_counts().to_dict()
    print("\nTile-wise historical event counts across all 14 tiles:")
    print("-" * 60)
    for tile in sorted(tile_counts.keys()):
        subregion = regional_df.loc[regional_df["dem_tile"] == tile, "subregion"].iloc[0]
        print(f"  {tile:<10} : {tile_counts[tile]:2d} events ({subregion})")
    print("-" * 60)
    print(f"  {'TOTAL':<10} : {sum(tile_counts.values()):2d} events across {len(tile_counts)} tiles")

    if len(tile_counts) != 14:
        raise ValueError(f"Expected 14 DEM tiles; found {len(tile_counts)}")

    # 7. Write output CSV exclusively
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    regional_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS: Regional event dataset successfully written to:\n  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
