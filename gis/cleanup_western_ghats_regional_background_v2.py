"""
Western Ghats Regional Background Sampling V2: Targeted Water-Artifact Cleanup.

Excludes exactly 5 verified water-body / marine artifacts from the original dataset:
1. N14E074_BG_0001 — Linganamakki Reservoir (inland water body)
2. N17E073_BG_0055 — Shastri creek / cove nearshore marine fringe (-15.5 m)
3. N15E073_BG_0097 — Terekhol estuary / Redi beach fringe (-9.0 m)
4. N16E073_BG_0067 — Devgad coastal inlet / tidal mudflat (-4.2 m)
5. N19E072_BG_0020 — Dahanu / Gholvad intertidal beach fringe (-3.0 m)

Preserves:
- All genuine terrestrial negative-elevation agricultural basins (Kuttanad: N09E076_BG_0006, N09E076_BG_0044; Thrissur Kole: N10E076_BG_0069)
- All low-slope coastal plains and valid flat terrestrial terrain
- All other 1,395 validated background samples

Resamples exactly 5 replacement points (1 in each affected tile: N14E074, N17E073, N15E073, N16E073, N19E072):
- Target: Exactly 100 samples per tile across all 14 tiles (1,400 total)
- WGS84 horizontal geodesic distance >= 5,000 m to all 182 historical events (via pyproj.Geod.inv)
- WGS84 horizontal geodesic separation >= 500 m between all background samples (via pyproj.Geod.inv)
- Valid finite terrain values (elevation, slope, aspect)
- Natural Earth land polygon containment (ne_10m_land.shp)
- Explicit verification that replacements do not fall on inland reservoirs or nearshore water
- Deterministic NumPy random seed 42

Output:
- Leaves original data/processed/western_ghats_regional_background_samples.csv UNCHANGED.
- Writes cleaned candidate to data/processed/western_ghats_regional_background_samples_v2.csv.
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

ORIGINAL_BG_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_background_samples.csv"
)

LAND_SHP = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "reference"
    / "naturalearth_land"
    / "ne_10m_land.shp"
)

OUTPUT_V2_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_background_samples_v2.csv"
)

EXCLUDED_ARTIFACT_IDS = {
    "N14E074_BG_0001",  # Linganamakki Reservoir
    "N17E073_BG_0055",  # nearshore marine fringe (-15.5 m)
    "N15E073_BG_0097",  # Terekhol estuary / Redi beach fringe (-9.0 m)
    "N16E073_BG_0067",  # coastal inlet / tidal mudflat (-4.2 m)
    "N19E072_BG_0020",  # intertidal beach fringe (-3.0 m)
}

AFFECTED_TILES = {
    "N14E074": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1" / "N14E074" / "metric",
    "N15E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N15E073" / "metric",
    "N16E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N16E073" / "metric",
    "N17E073": PROJECT_ROOT / "data" / "processed" / "dem" / "western_maharashtra" / "N17E073" / "metric",
    "N19E072": PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats" / "N19E072" / "metric",
}

TILE_SUBREGIONS = {
    "N14E074": "Central Western Ghats (Karnataka)",
    "N15E073": "Western Maharashtra Pilot",
    "N16E073": "Western Maharashtra Pilot",
    "N17E073": "Western Maharashtra Pilot",
    "N19E072": "Northern Western Ghats (Maharashtra)",
}

SEED = 42
MIN_EVENT_DISTANCE_M = 5000.0
MIN_SAMPLE_DISTANCE_M = 500.0


def main():
    print("=" * 80)
    print("STAGE 4: WESTERN GHATS REGIONAL BACKGROUND CLEANUP & RESAMPLING (V2)")
    print("=" * 80)

    # 1. Non-destructive safety checks
    if OUTPUT_V2_FILE.exists():
        raise FileExistsError(
            f"V2 output file already exists; refusing to overwrite:\n  {OUTPUT_V2_FILE}"
        )

    if not ORIGINAL_BG_FILE.exists():
        raise FileNotFoundError(
            f"Original background samples file not found:\n  {ORIGINAL_BG_FILE}"
        )

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Regional events file not found:\n  {EVENTS_FILE}"
        )

    if not LAND_SHP.exists():
        raise FileNotFoundError(
            f"Natural Earth land shapefile not found:\n  {LAND_SHP}"
        )

    # Record original file size and hash/timestamp for read-only verification
    orig_stat_before = ORIGINAL_BG_FILE.stat()
    print(f"Original file verified intact: {ORIGINAL_BG_FILE.name} ({orig_stat_before.st_size} bytes)")

    # 2. Load original background samples
    orig_df = pd.read_csv(ORIGINAL_BG_FILE, low_memory=False)
    print(f"Original background records loaded: {len(orig_df)}")

    if len(orig_df) != 1400:
        raise ValueError(f"Expected 1,400 original background samples; found {len(orig_df)}")

    # 3. Exclude exactly the 5 verified water artifacts
    matched_artifacts = orig_df[orig_df["sample_id"].isin(EXCLUDED_ARTIFACT_IDS)]
    print(f"\nExcluding exactly {len(matched_artifacts)} verified water artifacts:")
    for _, r in matched_artifacts.iterrows():
        print(f"  - {r['sample_id']:<15} | Tile: {r['tile']:<8} | Elev: {r['elevation_m']:6.1f} m | Slope: {r['slope_degrees']:5.2f} deg | ({r['latitude']:.4f}N, {r['longitude']:.4f}E)")

    if len(matched_artifacts) != 5:
        raise ValueError(f"Expected to match exactly 5 artifact IDs; matched {len(matched_artifacts)}")

    retained_df = orig_df[~orig_df["sample_id"].isin(EXCLUDED_ARTIFACT_IDS)].copy()
    print(f"\nRetained valid background samples: {len(retained_df)} (expected 1,395)")

    if len(retained_df) != 1395:
        raise ValueError(f"Expected exactly 1,395 retained samples; got {len(retained_df)}")

    # Verify per-tile retained counts: exactly 99 in the 5 affected tiles, 100 in the other 9
    retained_counts = retained_df["tile"].value_counts().to_dict()
    for tile in sorted(AFFECTED_TILES.keys()):
        if retained_counts.get(tile) != 99:
            raise ValueError(f"Expected 99 retained samples in {tile}; got {retained_counts.get(tile)}")

    # 4. Load events and land mask
    events_df = pd.read_csv(EVENTS_FILE, low_memory=False)
    event_lons = events_df["longitude"].to_numpy(dtype=float)
    event_lats = events_df["latitude"].to_numpy(dtype=float)

    land_gdf = gpd.read_file(LAND_SHP)
    if land_gdf.crs != "EPSG:4326":
        land_gdf = land_gdf.to_crs("EPSG:4326")
    if not land_gdf.is_valid.all():
        land_gdf["geometry"] = land_gdf["geometry"].make_valid()

    wg_regional_bbox = box(71.0, 7.0, 79.0, 22.0)
    land_regional = land_gdf.clip(wg_regional_bbox)

    geod = Geod(ellps="WGS84")
    rng = np.random.default_rng(SEED)

    # Track all accepted coordinates for pairwise distance enforcement
    all_acc_lons = retained_df["longitude"].tolist()
    all_acc_lats = retained_df["latitude"].tolist()

    # 5. Resample exactly 1 replacement for each of the 5 affected tiles
    print("\n" + "=" * 80)
    print("RESAMPLING EXACTLY 5 REPLACEMENT BACKGROUND POINTS")
    print("=" * 80)

    replacements = []
    t0 = time.time()

    for tile in sorted(AFFECTED_TILES.keys()):
        tile_dir = AFFECTED_TILES[tile]
        subregion = TILE_SUBREGIONS[tile]
        elev_path = tile_dir / "elevation.tif"
        slope_path = tile_dir / "slope.tif"
        aspect_path = tile_dir / "aspect.tif"

        # Determine the excluded sample ID to reuse for clean indexing
        matching_excluded = [sid for sid in EXCLUDED_ARTIFACT_IDS if sid.startswith(tile)]
        if len(matching_excluded) != 1:
            raise ValueError(f"Could not determine unique excluded ID for tile {tile}: {matching_excluded}")
        replacement_sample_id = matching_excluded[0]

        print(f"\nResampling replacement for {tile} (Target ID: {replacement_sample_id})...")

        with (
            rasterio.open(elev_path) as elev_ds,
            rasterio.open(slope_path) as slope_ds,
            rasterio.open(aspect_path) as aspect_ds,
        ):
            transformer_to_wgs84 = Transformer.from_crs(
                elev_ds.crs, "EPSG:4326", always_xy=True
            )

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

            land_proj = land_regional.clip(tile_clip_box).to_crs(elev_ds.crs)
            is_land = geometry_mask(
                land_proj.geometry,
                out_shape=elev_ds.shape,
                transform=elev_ds.transform,
                invert=True,
            )

            elev_arr = elev_ds.read(1)
            slope_arr = slope_ds.read(1)
            aspect_arr = aspect_ds.read(1)
            nodata_val = elev_ds.nodata

            accepted_candidate = None
            attempts = 0
            seen_cells = set()

            while accepted_candidate is None and attempts < 100000:
                batch_size = 1000
                rows = rng.integers(0, elev_ds.height, size=batch_size)
                cols = rng.integers(0, elev_ds.width, size=batch_size)
                attempts += batch_size

                for r, c in zip(rows, cols):
                    cell = (int(r), int(c))
                    if cell in seen_cells:
                        continue
                    seen_cells.add(cell)

                    # 1. Must be inside Natural Earth land polygon
                    if not is_land[cell]:
                        continue

                    # 2. Must have valid non-NoData raster values
                    elev_val = elev_arr[cell]
                    if elev_val == nodata_val or not np.isfinite(elev_val):
                        continue

                    slope_val = slope_arr[cell]
                    if slope_val == slope_ds.nodata or not np.isfinite(slope_val):
                        continue

                    aspect_val = aspect_arr[cell]
                    if aspect_val == aspect_ds.nodata or not np.isfinite(aspect_val):
                        continue

                    # 3. Transform cell center to EPSG:4326
                    x_utm, y_utm = rasterio.transform.xy(
                        elev_ds.transform, cell[0], cell[1], offset="center"
                    )
                    lon, lat = transformer_to_wgs84.transform(x_utm, y_utm)

                    # 4. Strict Water-Body and Nearshore Marine Exclusion:
                    # - Exclude flat inland water bodies / reservoirs (e.g. Linganamakki in N14E074: elev ~553m, slope==0)
                    if slope_val == 0.0 and elev_val > 50.0:
                        continue
                    # - Exclude nearshore negative-elevation intertidal marine artifacts (< 0 m along coast)
                    if elev_val < 0.0:
                        continue
                    # - Exclude flat sea-level water plates (elev == 0 m with slope < 0.1 deg)
                    if elev_val == 0.0 and slope_val < 0.1:
                        continue

                    # 5. Geodesic distance to all 182 historical events >= 5,000 m
                    _, _, dists_to_events = geod.inv(
                        np.full(len(event_lons), lon),
                        np.full(len(event_lats), lat),
                        event_lons,
                        event_lats,
                    )
                    min_event_dist = float(np.min(dists_to_events))
                    if min_event_dist < MIN_EVENT_DISTANCE_M:
                        continue

                    # 6. Geodesic separation to all existing background samples >= 500 m
                    n_bg = len(all_acc_lons)
                    _, _, dists_to_bgs = geod.inv(
                        np.full(n_bg, lon),
                        np.full(n_bg, lat),
                        np.array(all_acc_lons, dtype=float),
                        np.array(all_acc_lats, dtype=float),
                    )
                    min_bg_dist = float(np.min(dists_to_bgs))
                    if min_bg_dist < MIN_SAMPLE_DISTANCE_M:
                        continue

                    # 7. Candidate accepted!
                    is_flat = aspect_val < 0
                    if is_flat:
                        aspect_sin = 0.0
                        aspect_cos = 0.0
                    else:
                        rad = np.deg2rad(aspect_val)
                        aspect_sin = float(np.sin(rad))
                        aspect_cos = float(np.cos(rad))

                    accepted_candidate = {
                        "sample_id": replacement_sample_id,
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
                        "sampling_method": "random_valid_land_cells_geodesic_5km_v2_cleaned",
                        "random_seed": SEED,
                        "terrain_valid": True,
                        "terrain_status": "valid",
                    }

                    all_acc_lons.append(lon)
                    all_acc_lats.append(lat)
                    replacements.append(accepted_candidate)

                    print(f"  Accepted replacement in {attempts} attempts:")
                    print(
                        f"    ID: {replacement_sample_id} | Elev: {elev_val:.1f} m | "
                        f"Slope: {slope_val:.2f} deg | Aspect: {aspect_val:.1f} deg | "
                        f"Nearest Event Dist: {min_event_dist:.1f} m | "
                        f"Coords: ({lat:.4f}N, {lon:.4f}E)"
                    )
                    break

            if accepted_candidate is None:
                raise RuntimeError(f"Could not find valid replacement candidate for {tile} in {attempts} attempts.")

    print(f"\nResampling completed in {time.time() - t0:.2f} seconds.")

    # 6. Assemble V2 candidate dataset
    replacement_df = pd.DataFrame(replacements)
    combined_v2_df = pd.concat([retained_df, replacement_df], ignore_index=True)
    combined_v2_df = combined_v2_df.sort_values(by=["tile", "sample_id"]).reset_index(drop=True)

    print(f"\nTotal V2 samples assembled: {len(combined_v2_df)}")

    # 7. Comprehensive Post-Cleanup Verification & Audit
    print("\n" + "=" * 80)
    print("COMPREHENSIVE V2 DATASET INTEGRITY AUDIT")
    print("=" * 80)

    total_samples = len(combined_v2_df)
    unique_ids = combined_v2_df["sample_id"].nunique()
    print(f"Total samples:     {total_samples} (required: 1,400)")
    print(f"Unique sample IDs: {unique_ids} (required: 1,400)")

    if total_samples != 1400:
        raise ValueError(f"Expected exactly 1,400 samples; got {total_samples}")
    if unique_ids != 1400:
        raise ValueError(f"Expected 1,400 unique sample IDs; got {unique_ids}")

    # Per-tile counts
    tile_counts = combined_v2_df["tile"].value_counts().to_dict()
    for tile in sorted(tile_counts.keys()):
        if tile_counts[tile] != 100:
            raise ValueError(f"Tile {tile} has {tile_counts[tile]} samples; expected 100")
    print("Per-tile counts: PASSED (exactly 100 samples in every one of the 14 tiles).")

    # Coordinate uniqueness
    coord_pairs = list(zip(combined_v2_df["longitude"], combined_v2_df["latitude"]))
    if len(set(coord_pairs)) != total_samples:
        raise ValueError("Duplicate coordinates detected in V2 background samples.")
    print("Coordinate uniqueness: PASSED (1,400 distinct coordinates).")

    # Confirmation of exclusions
    remaining_artifacts = combined_v2_df[combined_v2_df["sample_id"].isin(EXCLUDED_ARTIFACT_IDS)]
    # Note: replacement samples reuse the IDs, but their coordinates, elevations, and slopes MUST NOT match the original artifacts
    for _, repl in remaining_artifacts.iterrows():
        orig_match = orig_df[orig_df["sample_id"] == repl["sample_id"]].iloc[0]
        if orig_match["longitude"] == repl["longitude"] and orig_match["latitude"] == repl["latitude"]:
            raise ValueError(f"Artifact {repl['sample_id']} coordinates were not replaced!")
    print("Artifact exclusion: PASSED (all 5 original artifact coordinates successfully replaced).")

    # Confirmation that genuine below-sea-level agricultural basins remain
    kuttanad_1 = combined_v2_df[combined_v2_df["sample_id"] == "N09E076_BG_0006"]
    kuttanad_2 = combined_v2_df[combined_v2_df["sample_id"] == "N09E076_BG_0044"]
    kole_wetlands = combined_v2_df[combined_v2_df["sample_id"] == "N10E076_BG_0069"]

    if kuttanad_1.empty or kuttanad_2.empty or kole_wetlands.empty:
        raise ValueError("Genuine below-sea-level samples in Kuttanad or Kole wetlands were inadvertently removed!")
    print(
        f"Genuine below-sea-level agricultural basins preserved:\n"
        f"  - Kuttanad 1 (N09E076_BG_0006): Elev = {kuttanad_1.iloc[0]['elevation_m']:.2f} m\n"
        f"  - Kuttanad 2 (N09E076_BG_0044): Elev = {kuttanad_2.iloc[0]['elevation_m']:.2f} m\n"
        f"  - Kole Basin (N10E076_BG_0069): Elev = {kole_wetlands.iloc[0]['elevation_m']:.2f} m"
    )

    # Elevation distribution
    below_0 = (combined_v2_df["elevation_m"] < 0).sum()
    exact_0 = (combined_v2_df["elevation_m"] == 0).sum()
    between_0_5 = ((combined_v2_df["elevation_m"] > 0) & (combined_v2_df["elevation_m"] <= 5)).sum()
    above_5 = (combined_v2_df["elevation_m"] > 5).sum()

    print("\nV2 Elevation Distribution (1,400 samples):")
    print(f"  - Below 0 m:        {below_0:4d} ({below_0/total_samples*100:.2f}%) [Genuine agricultural basins & Konkan tidal mudflats]")
    print(f"  - Exactly 0 m:      {exact_0:4d} ({exact_0/total_samples*100:.2f}%) [Coastal plains]")
    print(f"  - Between 0 and 5 m:{between_0_5:4d} ({between_0_5/total_samples*100:.2f}%) [Lowlands]")
    print(f"  - Above 5 m:        {above_5:4d} ({above_5/total_samples*100:.2f}%) [Terrestrial hills, ridges, valleys, plateaus]")

    elev_min = combined_v2_df["elevation_m"].min()
    elev_max = combined_v2_df["elevation_m"].max()
    slope_min = combined_v2_df["slope_degrees"].min()
    slope_max = combined_v2_df["slope_degrees"].max()
    aspect_min = combined_v2_df["aspect_degrees"].min()
    aspect_max = combined_v2_df["aspect_degrees"].max()
    print(f"\nElevation range: {elev_min:.1f} m to {elev_max:.1f} m")
    print(f"Slope range:     {slope_min:.2f} deg to {slope_max:.2f} deg")
    print(f"Aspect range:    {aspect_min:.2f} deg to {aspect_max:.2f} deg")

    # Land mask containment validation
    print("\nVerifying all 1,400 V2 samples lie inside Natural Earth land polygons...")
    land_union = land_regional.union_all()
    sample_points = [Point(xy) for xy in coord_pairs]
    in_land = [land_union.contains(pt) for pt in sample_points]
    if sum(in_land) != total_samples:
        raise ValueError(f"Found {total_samples - sum(in_land)} points outside land mask!")
    print(f"Land mask containment: PASSED (1,400/1,400 points inside land polygon).")

    # Geodesic distance to all 182 historical events
    print("\nVerifying horizontal geodesic distance constraint (>= 5,000 m to all 182 events)...")
    v2_lons = combined_v2_df["longitude"].to_numpy(dtype=float)
    v2_lats = combined_v2_df["latitude"].to_numpy(dtype=float)

    min_ev_dists = []
    for i in range(total_samples):
        _, _, d = geod.inv(
            np.full(len(event_lons), v2_lons[i]),
            np.full(len(event_lats), v2_lats[i]),
            event_lons,
            event_lats,
        )
        min_ev_dists.append(np.min(d))

    min_ev_dist = float(np.min(min_ev_dists))
    print(f"Minimum distance to any historical event: {min_ev_dist:.2f} m")
    if min_ev_dist < MIN_EVENT_DISTANCE_M:
        raise ValueError(f"Event distance constraint violated: min distance {min_ev_dist:.2f} m < {MIN_EVENT_DISTANCE_M} m")
    print("Event distance constraint: PASSED (all 1,400 samples >= 5,000 m from all 182 events).")

    # Geodesic pairwise separation
    print("\nVerifying pairwise horizontal geodesic separation (>= 500 m between samples)...")
    min_pairwise_sep = float("inf")
    for i in range(total_samples - 1):
        _, _, p_dists = geod.inv(
            np.full(total_samples - 1 - i, v2_lons[i]),
            np.full(total_samples - 1 - i, v2_lats[i]),
            v2_lons[i + 1:],
            v2_lats[i + 1:],
        )
        m = float(np.min(p_dists))
        if m < min_pairwise_sep:
            min_pairwise_sep = m
        if m < MIN_SAMPLE_DISTANCE_M:
            raise ValueError(f"Pairwise separation violated at index {i}: {m:.2f} m < {MIN_SAMPLE_DISTANCE_M} m")

    print(f"Minimum pairwise sample separation: {min_pairwise_sep:.2f} m")
    print("Pairwise separation constraint: PASSED (all pairs >= 500 m apart).")

    # 8. Check that original file was not modified
    orig_stat_after = ORIGINAL_BG_FILE.stat()
    if (
        orig_stat_before.st_size != orig_stat_after.st_size
        or orig_stat_before.st_mtime != orig_stat_after.st_mtime
    ):
        raise RuntimeError("Original background samples file was modified! Preserving integrity violated.")
    print("\nOriginal background file preservation check: PASSED (unchanged).")

    # 9. Save V2 dataset exclusively
    OUTPUT_V2_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_v2_df.to_csv(OUTPUT_V2_FILE, index=False)
    print(f"\nSUCCESS: V2 background samples written to:\n  {OUTPUT_V2_FILE}")


if __name__ == "__main__":
    main()
