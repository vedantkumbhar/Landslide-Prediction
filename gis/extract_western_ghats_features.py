
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "india_landslide_events.csv"
)

DEM_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_event_features.csv"
)

PILOT_TILES = ["N09E076", "N19E072", "N11E076"]

RASTER_FILES = {
    "elevation_m": "elevation.tif",
    "slope_degrees": "slope.tif",
    "aspect_degrees": "aspect.tif",
}


def tile_from_coordinates(latitude, longitude):
    """Return the one-degree SRTM tile name for a coordinate."""
    lat_band = int(np.floor(latitude))
    lon_band = int(np.floor(longitude))

    ns = "N" if lat_band >= 0 else "S"
    ew = "E" if lon_band >= 0 else "W"

    return f"{ns}{abs(lat_band):02d}{ew}{abs(lon_band):03d}"


def load_tile_rasters(tile):
    """Open and validate the three aligned terrain rasters."""
    tile_dir = DEM_ROOT / tile / "metric"
    paths = {
        feature: tile_dir / filename
        for feature, filename in RASTER_FILES.items()
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{tile}: missing required raster(s):\n"
            + "\n".join(missing)
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
            raise ValueError(
                f"{tile}: terrain rasters are not aligned."
            )

    return datasets


def sample_value(dataset, row, col):
    """Return a valid raster value, or NaN for NoData."""
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
    print("=" * 65)
    print("WESTERN GHATS EVENT TERRAIN FEATURE EXTRACTION")
    print("=" * 65)

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Events file not found: {EVENTS_FILE}")

    events = pd.read_csv(EVENTS_FILE)

    for column in ["latitude", "longitude"]:
        if column not in events.columns:
            raise ValueError(
                f"Required coordinate column missing: {column}"
            )

        events[column] = pd.to_numeric(
            events[column], errors="coerce"
        )

    coordinate_valid = (
        events["latitude"].between(-90, 90)
        & events["longitude"].between(-180, 180)
    )

    events["dem_tile"] = None
    events["terrain_source_tile"] = None
    events["elevation_m"] = np.nan
    events["slope_degrees"] = np.nan
    events["aspect_degrees"] = np.nan
    events["aspect_is_flat"] = False
    events["terrain_valid"] = False
    events["terrain_status"] = "invalid_coordinates"

    raster_cache = {}

    try:
        for tile in PILOT_TILES:
            tile_mask = coordinate_valid & events.apply(
                lambda row: (
                    tile_from_coordinates(
                        row["latitude"], row["longitude"]
                    ) == tile
                ) if pd.notna(row["latitude"])
                    and pd.notna(row["longitude"])
                    else False,
                axis=1,
            )

            indices = events.index[tile_mask].tolist()

            print(f"\n{tile}: {len(indices)} events in geographic tile")

            if not indices:
                continue

            datasets = load_tile_rasters(tile)
            raster_cache[tile] = datasets

            reference = datasets["elevation_m"]

            transformer = Transformer.from_crs(
                "EPSG:4326",
                reference.crs,
                always_xy=True,
            )

            lons = events.loc[indices, "longitude"].to_numpy(dtype=float)
            lats = events.loc[indices, "latitude"].to_numpy(dtype=float)

            xs, ys = transformer.transform(lons, lats)
            rows, cols = rasterio.transform.rowcol(
                reference.transform, xs, ys
            )

            for idx, row, col in zip(indices, rows, cols):
                events.at[idx, "dem_tile"] = tile

                if not (
                    0 <= row < reference.height
                    and 0 <= col < reference.width
                ):
                    events.at[idx, "terrain_status"] = "outside_raster"
                    continue

                values = {
                    feature: sample_value(ds, row, col)
                    for feature, ds in datasets.items()
                }

                for feature, value in values.items():
                    events.at[idx, feature] = value

                events.at[idx, "terrain_source_tile"] = tile

                valid = all(np.isfinite(v) for v in values.values())
                events.at[idx, "terrain_valid"] = valid
                events.at[idx, "aspect_is_flat"] = (
                    np.isfinite(values["aspect_degrees"])
                    and values["aspect_degrees"] == -1
                )

                events.at[idx, "terrain_status"] = (
                    "valid" if valid else "invalid_or_nodata_sample"
                )

            print(
                f"Events with valid terrain in {tile}: "
                f"{int(events.loc[indices, 'terrain_valid'].sum())}"
            )

    finally:
        for datasets in raster_cache.values():
            for ds in datasets.values():
                ds.close()

    # Preserve every original event, including events outside the three tiles.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("EXTRACTION SUMMARY")
    print("=" * 65)
    print(f"Input events: {len(events)}")
    print(f"Output rows: {len(events)}")
    print(f"Valid terrain records: {int(events['terrain_valid'].sum())}")

    print("\nStatus counts:")
    print(events["terrain_status"].value_counts(dropna=False).to_string())

    print("\nRecords by tile:")
    print(events["dem_tile"].value_counts(dropna=False).to_string())

    print("\nTerrain feature summary:")
    print(
        events.loc[events["terrain_valid"], list(RASTER_FILES)]
        .describe()
        .round(3)
        .to_string()
    )

    print(f"\nSaved new dataset:\n{OUTPUT_FILE}")
    print("Existing datasets and DEM rasters were not modified.")


if __name__ == "__main__":
    main()
