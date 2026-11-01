from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

SEED = 42
SAMPLES_PER_TILE = 10
CANDIDATES_PER_TILE = 5000
MIN_EVENT_DISTANCE_KM = 5.0
MIN_BACKGROUND_DISTANCE_KM = 0.5
EARTH_RADIUS_KM = 6371.0088

DATASET = Path("data/processed/initial_training_dataset.csv")
DEM_ROOT = Path("data/processed/dem")
OUTPUT = Path("data/processed/regional_background_samples_30oct.csv")
AUDIT = Path("data/processed/regional_background_sampling_audit_30oct.csv")

REGIONS = {
    "Himalayas": "himalayas",
    "Northeast India": "northeast",
    "Western Maharashtra": "western_maharashtra",
}

rng = np.random.default_rng(SEED)


def unit_xyz(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    return np.column_stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ))


def chord_to_km(chord):
    angle = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return EARTH_RADIUS_KM * angle


def make_tree(latitudes, longitudes):
    if len(latitudes) == 0:
        return None
    return cKDTree(unit_xyz(latitudes, longitudes))


def query_distance(tree, latitudes, longitudes):
    if tree is None:
        return np.full(len(latitudes), np.inf)
    chord, _ = tree.query(unit_xyz(latitudes, longitudes), k=1)
    return chord_to_km(chord)


def read_raster_values(dataset, coordinates):
    values = np.ma.asarray(
        list(dataset.sample(coordinates, masked=True))
    )
    return np.asarray(values.filled(np.nan)[:, 0], dtype=float)


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError(
            "Output already exists. Nothing was overwritten:\n"
            f"{OUTPUT}\n{AUDIT}"
        )

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    df = pd.read_csv(DATASET)

    required = {
        "latitude", "longitude", "dem_region",
        "tile", "label"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    events = df[df["label"] == 1].copy()
    backgrounds = df[df["label"] == 0].copy()

    if events.empty:
        raise ValueError("No historical positive events found.")

    all_samples = []
    audit_rows = []

    for region, folder in REGIONS.items():
        print(f"\n========== {region} ==========")

        region_events = events[events["dem_region"] == region]
        region_backgrounds = backgrounds[
            backgrounds["dem_region"] == region
        ]

        if region_events.empty:
            print("No historical events. Region skipped.")
            continue

        event_lat = region_events["latitude"].to_numpy(dtype=float)
        event_lon = region_events["longitude"].to_numpy(dtype=float)

        valid_event = (
            np.isfinite(event_lat) & np.isfinite(event_lon)
            & (np.abs(event_lat) <= 90)
            & (np.abs(event_lon) <= 180)
        )
        event_lat = event_lat[valid_event]
        event_lon = event_lon[valid_event]

        event_tree = make_tree(event_lat, event_lon)

        existing_lat = region_backgrounds["latitude"].to_numpy(dtype=float)
        existing_lon = region_backgrounds["longitude"].to_numpy(dtype=float)

        valid_existing = (
            np.isfinite(existing_lat) & np.isfinite(existing_lon)
        )
        existing_lat = existing_lat[valid_existing]
        existing_lon = existing_lon[valid_existing]

        existing_tree = make_tree(existing_lat, existing_lon)

        tiles = sorted(
            region_events["tile"].dropna().astype(str).unique()
        )

        for tile in tiles:
            base = DEM_ROOT / folder / tile
            metric = base / "metric"
            if not metric.exists():
                metric = base

            paths = {
                name: metric / f"{name}.tif"
                for name in ("elevation", "slope", "aspect")
            }

            if not all(path.exists() for path in paths.values()):
                missing_files = [
                    str(path) for path in paths.values()
                    if not path.exists()
                ]
                print(f"{tile}: SKIPPED, missing {missing_files}")
                audit_rows.append({
                    "dem_region": region,
                    "tile": tile,
                    "accepted_samples": 0,
                    "status": "missing_raster"
                })
                continue

            accepted = []
            candidate_count = 0

            with rasterio.open(paths["elevation"]) as elev, \
                 rasterio.open(paths["slope"]) as slope, \
                 rasterio.open(paths["aspect"]) as aspect:

                aligned = (
                    elev.shape == slope.shape == aspect.shape
                    and elev.transform == slope.transform == aspect.transform
                    and elev.crs == slope.crs == aspect.crs
                )

                if not aligned or elev.crs is None:
                    raise ValueError(
                        f"{region}/{tile}: raster alignment or CRS failure"
                    )

                transformer = Transformer.from_crs(
                    elev.crs, "EPSG:4326", always_xy=True
                )

                rows = rng.integers(
                    0, elev.height, size=CANDIDATES_PER_TILE
                )
                cols = rng.integers(
                    0, elev.width, size=CANDIDATES_PER_TILE
                )

                xs, ys = rasterio.transform.xy(
                    elev.transform, rows, cols, offset="center"
                )
                coordinates = list(zip(xs, ys))

                elevation_values = read_raster_values(elev, coordinates)
                slope_values = read_raster_values(slope, coordinates)
                aspect_values = read_raster_values(aspect, coordinates)

                lon, lat = transformer.transform(xs, ys)
                lat = np.asarray(lat, dtype=float)
                lon = np.asarray(lon, dtype=float)

                valid = (
                    np.isfinite(elevation_values)
                    & np.isfinite(slope_values)
                    & np.isfinite(aspect_values)
                    & np.isfinite(lat)
                    & np.isfinite(lon)
                    & (elevation_values > 0)
                    & (slope_values >= 0)
                    & (slope_values <= 90)
                    & (aspect_values >= -1)
                    & (aspect_values <= 360)
                    & (np.abs(lat) <= 90)
                    & (np.abs(lon) <= 180)
                )

                candidate_indices = np.flatnonzero(valid)
                candidate_count = len(candidate_indices)

                if candidate_count:
                    c_lat = lat[candidate_indices]
                    c_lon = lon[candidate_indices]

                    event_distances = query_distance(
                        event_tree, c_lat, c_lon
                    )
                    existing_distances = query_distance(
                        existing_tree, c_lat, c_lon
                    )

                    eligible = (
                        (event_distances >= MIN_EVENT_DISTANCE_KM)
                        & (
                            existing_distances
                            >= MIN_BACKGROUND_DISTANCE_KM
                        )
                    )

                    eligible_indices = np.flatnonzero(eligible)

                    accepted_lat = []
                    accepted_lon = []

                    for local_index in eligible_indices:
                        idx = candidate_indices[local_index]

                        current_lat = float(lat[idx])
                        current_lon = float(lon[idx])

                        if accepted:
                            new_tree = make_tree(
                                accepted_lat, accepted_lon
                            )
                            spacing = query_distance(
                                new_tree,
                                [current_lat],
                                [current_lon]
                            )[0]

                            if spacing < MIN_BACKGROUND_DISTANCE_KM:
                                continue

                        nearest_km = float(event_distances[local_index])

                        accepted.append({
                            "sample_id": (
                                f"{region.replace(' ', '_')}_"
                                f"{tile}_BG{len(accepted)+1:03d}"
                            ),
                            "latitude": current_lat,
                            "longitude": current_lon,
                            "elevation_m": float(elevation_values[idx]),
                            "slope_degrees": float(slope_values[idx]),
                            "aspect_degrees": float(aspect_values[idx]),
                            "dem_region": region,
                            "tile": tile,
                            "label": 0,
                            "sample_type": "provisional_background",
                            "label_confidence": "low",
                            "nearest_recorded_event_km": nearest_km,
                            "sampling_method": "random_valid_dem_cells",
                            "random_seed": SEED,
                        })

                        accepted_lat.append(current_lat)
                        accepted_lon.append(current_lon)

                        if len(accepted) >= SAMPLES_PER_TILE:
                            break

            all_samples.extend(accepted)

            status = (
                "target_reached"
                if len(accepted) == SAMPLES_PER_TILE
                else "target_not_reached"
            )

            print(
                f"{tile}: candidates={candidate_count}, "
                f"accepted={len(accepted)}/{SAMPLES_PER_TILE}"
            )

            audit_rows.append({
                "dem_region": region,
                "tile": tile,
                "valid_candidate_count": candidate_count,
                "accepted_samples": len(accepted),
                "target_samples": SAMPLES_PER_TILE,
                "status": status
            })

    if not all_samples:
        raise RuntimeError(
            "No samples generated. No output dataset was written."
        )

    result = pd.DataFrame(all_samples)
    audit = pd.DataFrame(audit_rows)

    # Final safety checks
    if result["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected.")

    if result.duplicated(["latitude", "longitude"]).any():
        raise ValueError("Duplicate coordinates detected.")

    if not (result["nearest_recorded_event_km"] >=
            MIN_EVENT_DISTANCE_KM).all():
        raise ValueError("A sample is too close to a recorded event.")

    if not (result["label"] == 0).all():
        raise ValueError("Unexpected labels in generated samples.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    audit.to_csv(AUDIT, index=False)

    print("\n========== SAMPLING SUMMARY ==========")
    print("Samples generated:", len(result))
    print("\nRegion counts:")
    print(result.groupby("dem_region").size().to_string())
    print("\nTile audit:")
    print(audit.to_string(index=False))
    print("\nSaved:", OUTPUT)
    print("Saved:", AUDIT)
    print("\nExisting datasets and models were not modified.")


if __name__ == "__main__":
    main()
