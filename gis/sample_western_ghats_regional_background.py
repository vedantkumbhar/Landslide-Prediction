"""
Western Ghats Regional Background Sampling.

Generates 100 provisional background samples per tile across all 14 DEM tiles:
- Pilot Tiles (5): N15E073, N15E074, N16E073, N17E073, N18E073
- Previous Western Ghats (3): N09E076, N11E076, N19E072
- Tier 1 Western Ghats (6): N19E073, N10E076, N10E077, N12E074, N14E074, N12E075
Total target: 1,400 provisional background samples.

Key constraints:
1. Excludes ocean/water pixels using Natural Earth 10m land polygons (ne_10m_land.shp).
2. Preserves low-slope coastal land; never uses slope as a land/water exclusion filter.
3. Enforces horizontal geodesic distance >= 5,000 m from all 182 regional historical events.
4. Enforces horizontal geodesic separation >= 500 m between accepted background samples.
5. Calculates horizontal geodesic distance via pyproj.Geod(ellps='WGS84').inv (no terrain elevation inflation).
6. Uses fixed random seed (42) for reproducible sampling.
7. Refuses to overwrite existing files (data/processed/western_ghats_regional_background_samples.csv).
8. Labels all samples as 'provisional_background'.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import box, Point
from pyproj import Transformer, Geod

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_event_features.csv"
)

LAND_SHP = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
    / "naturalearth_land"
    / "ne_10m_land.shp"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_background_samples.csv"
)

TILE_DIRS = {
    # 5 Western Maharashtra Pilot tiles
    "N15E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N15E073" / "metric",
    "N15E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N15E074" / "metric",
    "N16E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N16E073" / "metric",
    "N17E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N17E073" / "metric",
    "N18E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N18E073" / "metric",
    # 3 Previous Western Ghats tiles
    "N09E076": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N09E076" / "metric",
    "N11E076": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N11E076" / "metric",
    "N19E072": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N19E072" / "metric",
    # 6 Tier 1 Western Ghats tiles
    "N19E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N19E073" / "metric",
    "N10E076": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N10E076" / "metric",
    "N10E077": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N10E077" / "metric",
    "N12E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N12E074" / "metric",
    "N14E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N14E074" / "metric",
    "N12E075": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N12E075" / "metric",
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
    "N19E073": "Northern Western Ghats (Maharashtra)",
    "N10E076": "Southern Western Ghats (Kerala)",
    "N10E077": "Southern Western Ghats (Tamil Nadu / Kerala)",
    "N12E074": "Central Western Ghats (Karnataka / Kerala)",
    "N14E074": "Central Western Ghats (Karnataka)",
    "N12E075": "Central Western Ghats (Karnataka)",
}

SEED = 42
SAMPLES_PER_TILE = 100
MIN_EVENT_DISTANCE_M = 5000.0  # 5,000 metres horizontal geodesic
MIN_SAMPLE_DISTANCE_M = 500.0   # 500 metres horizontal geodesic
MAX_ATTEMPTS_PER_TILE = 50000


def sample_raster_value(dataset, row: int, col: int) -> float:
    """Read a single pixel value and return np.nan if invalid or NoData."""
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
    print("=" * 80)
    print("STAGE 4: WESTERN GHATS REGIONAL BACKGROUND SAMPLING (14 TILES)")
    print("=" * 80)

    # 1. Non-destructive safety check
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output file already exists; refusing to overwrite:\n  {OUTPUT_FILE}"
        )

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Required regional events file not found:\n  {EVENTS_FILE}"
        )

    if not LAND_SHP.exists():
        raise FileNotFoundError(
            f"Required Natural Earth land shapefile not found:\n  {LAND_SHP}"
        )

    # 2. Load and validate regional historical events
    print(f"\nLoading regional historical events: {EVENTS_FILE.relative_to(PROJECT_ROOT)}")
    events_df = pd.read_csv(EVENTS_FILE, low_memory=False)
    print(f"Total events loaded: {len(events_df)}")

    if len(events_df) != 182:
        raise ValueError(f"Expected exactly 182 regional events; found {len(events_df)}")

    if events_df["event_id"].nunique() != 182:
        raise ValueError("Regional event IDs are not unique.")

    event_lons = events_df["longitude"].to_numpy(dtype=float)
    event_lats = events_df["latitude"].to_numpy(dtype=float)

    if not (
        np.all(np.isfinite(event_lons))
        and np.all(np.isfinite(event_lats))
        and np.all(np.abs(event_lons) <= 180)
        and np.all(np.abs(event_lats) <= 90)
    ):
        raise ValueError("Invalid coordinates detected in regional events.")

    # 3. Load and validate Natural Earth land mask
    print(f"\nLoading Natural Earth 10m land polygons: {LAND_SHP.relative_to(PROJECT_ROOT)}")
    land_gdf = gpd.read_file(LAND_SHP)
    print(f"Loaded land features: {len(land_gdf)}, CRS: {land_gdf.crs}")

    if land_gdf.crs != "EPSG:4326":
        print(f"Re-projecting land polygons from {land_gdf.crs} to EPSG:4326...")
        land_gdf = land_gdf.to_crs("EPSG:4326")

    if not land_gdf.is_valid.all():
        print("Fixing invalid geometries in land polygons...")
        land_gdf["geometry"] = land_gdf["geometry"].make_valid()

    # Pre-clip land polygons to Western Ghats regional bounding box to accelerate per-tile operations
    wg_regional_bbox = box(71.0, 7.0, 79.0, 22.0)
    land_regional = land_gdf.clip(wg_regional_bbox)
    print(f"Clipped regional land polygon features: {len(land_regional)}")

    # 4. Initialize geodesic engine and random generator
    geod = Geod(ellps="WGS84")
    rng = np.random.default_rng(SEED)

    # Global tracking of accepted background samples for spatial separation
    global_accepted_lons = []
    global_accepted_lats = []
    all_samples = []
    tile_sampling_summary = {}

    start_time = time.time()

    # 5. Tile-by-tile sampling
    for tile in sorted(TILE_DIRS.keys()):
        tile_dir = TILE_DIRS[tile]
        subregion = TILE_SUBREGIONS[tile]
        print(f"\n--- Sampling Tile: {tile} ({subregion}) ---")
        print(f"  Directory: {tile_dir.relative_to(PROJECT_ROOT)}")

        elev_path = tile_dir / "elevation.tif"
        slope_path = tile_dir / "slope.tif"
        aspect_path = tile_dir / "aspect.tif"

        for p in (elev_path, slope_path, aspect_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing raster file: {p}")

        with (
            rasterio.open(elev_path) as elev_ds,
            rasterio.open(slope_path) as slope_ds,
            rasterio.open(aspect_path) as aspect_ds,
        ):
            # Verify raster alignment
            if not (
                elev_ds.shape == slope_ds.shape == aspect_ds.shape
                and elev_ds.transform == slope_ds.transform == aspect_ds.transform
                and elev_ds.crs == slope_ds.crs == aspect_ds.crs
            ):
                raise ValueError(f"{tile}: terrain rasters are not perfectly aligned.")

            transformer_to_wgs84 = Transformer.from_crs(
                elev_ds.crs, "EPSG:4326", always_xy=True
            )

            # Determine tile bounds in EPSG:4326
            min_lon, min_lat = transformer_to_wgs84.transform(
                elev_ds.bounds.left, elev_ds.bounds.bottom
            )
            max_lon, max_lat = transformer_to_wgs84.transform(
                elev_ds.bounds.right, elev_ds.bounds.top
            )
            tile_clip_box = box(
                min(min_lon, max_lon) - 0.1,
                min(min_lat, max_lat) - 0.1,
                max(min_lon, max_lon) + 0.1,
                max(min_lat, max_lat) + 0.1,
            )

            # Clip and project land polygon to tile CRS for fast rasterization
            land_tile_4326 = land_regional.clip(tile_clip_box)
            land_tile_proj = land_tile_4326.to_crs(elev_ds.crs)

            # Rasterize land mask: True where pixel is on LAND, False where ocean/water
            is_land = geometry_mask(
                land_tile_proj.geometry,
                out_shape=elev_ds.shape,
                transform=elev_ds.transform,
                invert=True,
            )

            land_pixel_count = int(is_land.sum())
            total_pixel_count = is_land.size
            land_fraction_pct = (land_pixel_count / total_pixel_count) * 100.0
            print(
                f"  Land mask active: {land_pixel_count:,} land pixels "
                f"({land_fraction_pct:.2f}% of tile)"
            )

            # Read full elevation raster into memory for fast candidate testing
            elev_array = elev_ds.read(1)
            nodata_val = elev_ds.nodata

            accepted_in_tile = 0
            attempts = 0
            seen_cells = set()
            tile_samples = []

            while (
                accepted_in_tile < SAMPLES_PER_TILE
                and attempts < MAX_ATTEMPTS_PER_TILE
            ):
                batch_size = 1000
                rows = rng.integers(0, elev_ds.height, size=batch_size)
                cols = rng.integers(0, elev_ds.width, size=batch_size)
                attempts += batch_size

                for r, c in zip(rows, cols):
                    cell = (int(r), int(c))
                    if cell in seen_cells:
                        continue
                    seen_cells.add(cell)

                    # 1. Exclude ocean/water pixels using land mask
                    if not is_land[cell[0], cell[1]]:
                        continue

                    # 2. Check valid elevation (not NoData, finite)
                    elev_val = elev_array[cell[0], cell[1]]
                    if elev_val == nodata_val or not np.isfinite(elev_val):
                        continue

                    # 3. Read slope and aspect values
                    slope_val = sample_raster_value(slope_ds, cell[0], cell[1])
                    if not np.isfinite(slope_val):
                        continue

                    aspect_val = sample_raster_value(aspect_ds, cell[0], cell[1])
                    if not np.isfinite(aspect_val):
                        continue

                    # 4. Transform cell center to EPSG:4326
                    x_utm, y_utm = rasterio.transform.xy(
                        elev_ds.transform, cell[0], cell[1], offset="center"
                    )
                    lon, lat = transformer_to_wgs84.transform(x_utm, y_utm)

                    # 5. Calculate horizontal geodesic distance to all 182 historical events
                    _, _, dists_to_events = geod.inv(
                        np.full(len(event_lons), lon),
                        np.full(len(event_lats), lat),
                        event_lons,
                        event_lats,
                    )
                    min_event_dist = float(np.min(dists_to_events))

                    if min_event_dist < MIN_EVENT_DISTANCE_M:
                        continue

                    # 6. Calculate horizontal geodesic separation to all previously accepted background samples
                    if global_accepted_lons:
                        n_bg = len(global_accepted_lons)
                        _, _, dists_to_bgs = geod.inv(
                            np.full(n_bg, lon),
                            np.full(n_bg, lat),
                            np.array(global_accepted_lons, dtype=float),
                            np.array(global_accepted_lats, dtype=float),
                        )
                        min_bg_dist = float(np.min(dists_to_bgs))
                        if min_bg_dist < MIN_SAMPLE_DISTANCE_M:
                            continue

                    # 7. Candidate passed all criteria - calculate aspect features
                    is_flat = aspect_val < 0
                    if is_flat:
                        aspect_sin = 0.0
                        aspect_cos = 0.0
                    else:
                        rad = np.deg2rad(aspect_val)
                        aspect_sin = float(np.sin(rad))
                        aspect_cos = float(np.cos(rad))

                    sample_idx = accepted_in_tile + 1
                    sample_record = {
                        "sample_id": f"{tile}_BG_{sample_idx:04d}",
                        "tile": tile,
                        "subregion": subregion,
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "elevation_m": float(elev_val),
                        "slope_degrees": float(slope_val),
                        "aspect_degrees": float(aspect_val),
                        "aspect_is_flat": bool(is_flat),
                        "aspect_sin": float(aspect_sin),
                        "aspect_cos": float(aspect_cos),
                        "aspect_flat": int(is_flat),
                        "label": "provisional_background",
                        "label_code": 0,
                        "nearest_event_distance_m": float(min_event_dist),
                        "sampling_method": "random_valid_land_cells_geodesic_5km",
                        "random_seed": SEED,
                        "terrain_valid": True,
                        "terrain_status": "valid",
                    }

                    tile_samples.append(sample_record)
                    global_accepted_lons.append(lon)
                    global_accepted_lats.append(lat)
                    accepted_in_tile += 1

                    if accepted_in_tile >= SAMPLES_PER_TILE:
                        break

            all_samples.extend(tile_samples)
            tile_sampling_summary[tile] = {
                "accepted": accepted_in_tile,
                "target": SAMPLES_PER_TILE,
                "attempts": attempts,
                "subregion": subregion,
                "land_fraction_pct": land_fraction_pct,
            }
            print(
                f"  -> Accepted {accepted_in_tile}/{SAMPLES_PER_TILE} samples "
                f"in {attempts:,} attempts."
            )

            if accepted_in_tile < SAMPLES_PER_TILE:
                print(
                    f"  WARNING: Tile {tile} reached only {accepted_in_tile}/{SAMPLES_PER_TILE} "
                    f"samples within {attempts} attempts."
                )

    elapsed_time = time.time() - start_time
    print(f"\nSampling phase completed in {elapsed_time:.2f} seconds.")

    # 6. Post-sampling validation & integrity audit
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKGROUND SAMPLING INTEGRITY AUDIT")
    print("=" * 80)

    bg_df = pd.DataFrame(all_samples)
    total_samples = len(bg_df)
    unique_ids = bg_df["sample_id"].nunique()

    print(f"Total samples collected: {total_samples}")
    print(f"Target sample count:     {14 * SAMPLES_PER_TILE}")
    print(f"Unique sample IDs:       {unique_ids}")

    if total_samples != 1400:
        raise ValueError(f"Expected exactly 1,400 samples; got {total_samples}")

    if unique_ids != 1400:
        raise ValueError(f"Expected 1,400 unique sample IDs; got {unique_ids}")

    # Coordinate uniqueness
    coord_pairs = list(zip(bg_df["longitude"], bg_df["latitude"]))
    if len(set(coord_pairs)) != total_samples:
        raise ValueError("Duplicate coordinates detected in background samples.")
    print("Coordinate uniqueness check: PASSED (1,400 distinct points).")

    # Land mask validation: verify every accepted sample falls within land geometry
    print("\nVerifying all 1,400 samples lie strictly inside Natural Earth land polygons...")
    land_union = land_regional.union_all()
    sample_points = [Point(xy) for xy in coord_pairs]
    in_land_mask = [land_union.contains(pt) for pt in sample_points]
    land_valid_count = sum(in_land_mask)
    print(f"Land mask verification: {land_valid_count}/1,400 points inside land polygon.")

    if land_valid_count != total_samples:
        invalid_pts = bg_df[~np.array(in_land_mask)]
        print(invalid_pts[["sample_id", "tile", "longitude", "latitude"]])
        raise ValueError(f"Found {total_samples - land_valid_count} points outside land mask!")

    # Terrain completeness & range checks
    for col in ["elevation_m", "slope_degrees", "aspect_degrees", "aspect_sin", "aspect_cos"]:
        nan_count = bg_df[col].isna().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in column {col}")

    elev_min, elev_max = bg_df["elevation_m"].min(), bg_df["elevation_m"].max()
    slope_min, slope_max = bg_df["slope_degrees"].min(), bg_df["slope_degrees"].max()
    aspect_min, aspect_max = bg_df["aspect_degrees"].min(), bg_df["aspect_degrees"].max()
    sin_min, sin_max = bg_df["aspect_sin"].min(), bg_df["aspect_sin"].max()
    cos_min, cos_max = bg_df["aspect_cos"].min(), bg_df["aspect_cos"].max()

    print(f"\nElevation range: {elev_min:.1f} m to {elev_max:.1f} m")
    print(f"Slope range:     {slope_min:.2f} deg to {slope_max:.2f} deg")
    print(f"Aspect range:    {aspect_min:.2f} deg to {aspect_max:.2f} deg")
    print(f"Aspect sin range: [{sin_min:.4f}, {sin_max:.4f}]")
    print(f"Aspect cos range: [{cos_min:.4f}, {cos_max:.4f}]")

    if not (slope_min >= 0 and slope_max <= 90):
        raise ValueError(f"Slope out of bounds [0, 90]: {slope_min} to {slope_max}")

    if not (aspect_min >= -1 and aspect_max < 360):
        raise ValueError(f"Aspect out of bounds [-1, 360): {aspect_min} to {aspect_max}")

    assert -1.0 <= sin_min and sin_max <= 1.0, "aspect_sin out of [-1, 1]"
    assert -1.0 <= cos_min and cos_max <= 1.0, "aspect_cos out of [-1, 1]"

    # Distance constraint 1: Every sample >= 5,000 m from all 182 historical events
    print("\nVerifying horizontal geodesic distance constraint (>= 5,000 m to all 182 events)...")
    min_event_dist_found = bg_df["nearest_event_distance_m"].min()
    print(f"Minimum distance to any historical event: {min_event_dist_found:.2f} m")

    if min_event_dist_found < MIN_EVENT_DISTANCE_M:
        violating = bg_df[bg_df["nearest_event_distance_m"] < MIN_EVENT_DISTANCE_M]
        print(violating[["sample_id", "tile", "nearest_event_distance_m"]])
        raise ValueError(
            f"Distance constraint violated: minimum event distance {min_event_dist_found:.2f} m "
            f"is less than required {MIN_EVENT_DISTANCE_M} m"
        )
    print("Event distance constraint: PASSED (all samples >= 5,000 m from all 182 events).")

    # Distance constraint 2: Pairwise separation >= 500 m between all accepted samples
    print("\nVerifying pairwise horizontal geodesic separation (>= 500 m between samples)...")
    sample_lons = bg_df["longitude"].to_numpy(dtype=float)
    sample_lats = bg_df["latitude"].to_numpy(dtype=float)

    min_pairwise_sep = float("inf")
    # Check pairwise separation
    for i in range(len(sample_lons)):
        _, _, p_dists = geod.inv(
            np.full(len(sample_lons) - 1 - i, sample_lons[i]),
            np.full(len(sample_lats) - 1 - i, sample_lats[i]),
            sample_lons[i + 1:],
            sample_lats[i + 1:],
        )
        if len(p_dists) > 0:
            m_sep = float(np.min(p_dists))
            if m_sep < min_pairwise_sep:
                min_pairwise_sep = m_sep
            if m_sep < MIN_SAMPLE_DISTANCE_M:
                raise ValueError(
                    f"Pairwise separation violated: sample {bg_df.iloc[i]['sample_id']} "
                    f"is only {m_sep:.2f} m from another sample!"
                )

    print(f"Minimum pairwise sample separation found: {min_pairwise_sep:.2f} m")
    print("Sample separation constraint: PASSED (all pairs >= 500 m apart).")

    # 7. Write output CSV exclusively
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    bg_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS: Regional background samples written to:\n  {OUTPUT_FILE}")

    # Summary table
    print("\nTile-by-tile background sampling breakdown:")
    print("-" * 75)
    print(f"{'Tile':<10} | {'Subregion':<42} | {'Accepted':<8} | {'Attempts':<8}")
    print("-" * 75)
    for tile in sorted(tile_sampling_summary.keys()):
        info = tile_sampling_summary[tile]
        print(
            f"{tile:<10} | {info['subregion']:<42} | "
            f"{info['accepted']:<8} | {info['attempts']:<8,}"
        )
    print("-" * 75)
    print(f"{'TOTAL':<10} | {'14 Tiles Combined':<42} | {total_samples:<8} |")
    print("-" * 75)


if __name__ == "__main__":
    main()
