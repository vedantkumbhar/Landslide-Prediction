
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_N27E088_events.csv"
)

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N27E088"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_N27E088_event_features.csv"
)


def main():
    print("=" * 65)
    print("NORTHEAST LANDSLIDE TERRAIN FEATURE EXTRACTION")
    print("=" * 65)

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Events file not found: {EVENTS_FILE}")

    events = pd.read_csv(EVENTS_FILE)

    raster_files = {
        "elevation_m": DEM_DIR / "elevation.tif",
        "slope_degrees": DEM_DIR / "slope.tif",
        "aspect_degrees": DEM_DIR / "aspect.tif",
    }

    for path in raster_files.values():
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

    # Use elevation raster as the reference grid.
    with rasterio.open(raster_files["elevation_m"]) as reference:
        transformer = Transformer.from_crs(
            "EPSG:4326",
            reference.crs,
            always_xy=True,
        )

        x, y = transformer.transform(
            events["longitude"].to_numpy(),
            events["latitude"].to_numpy(),
        )

        rows, cols = rasterio.transform.rowcol(
            reference.transform, x, y
        )

        rows = np.asarray(rows)
        cols = np.asarray(cols)

        inside_grid = (
            (rows >= 0)
            & (rows < reference.height)
            & (cols >= 0)
            & (cols < reference.width)
        )

    # Sample all terrain rasters at each event location.
    for feature_name, raster_path in raster_files.items():
        values = np.full(len(events), np.nan, dtype=float)

        with rasterio.open(raster_path) as src:
            if src.crs != reference.crs:
                raise ValueError(
                    f"CRS mismatch in {raster_path.name}"
                )

            for i in np.flatnonzero(inside_grid):
                row = rows[i]
                col = cols[i]

                value = src.read(
                    1,
                    window=((row, row + 1), (col, col + 1)),
                    masked=True,
                )[0, 0]

                if not np.ma.is_masked(value):
                    number = float(value)
                    if np.isfinite(number) and number != src.nodata:
                        values[i] = number

        events[feature_name] = values

    # Aspect -1 represents flat terrain in our rasters.
    events["aspect_is_flat"] = (
        events["aspect_degrees"] == -1
    )

    terrain_columns = [
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
    ]

    events["terrain_valid"] = events[terrain_columns].notna().all(
        axis=1
    )

    # Save extracted features.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_FILE, index=False)

    print(f"\nHistorical events: {len(events)}")
    print(f"Events inside raster grid: {int(inside_grid.sum())}")
    print(
        "Events with all three valid terrain features: "
        f"{int(events['terrain_valid'].sum())}"
    )

    print("\nTERRAIN FEATURE SUMMARY")
    print("-" * 40)
    print(events[terrain_columns].describe().round(3).to_string())

    print("\nMissing terrain values:")
    print(events[terrain_columns].isna().sum().to_string())

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")
    print("\nFeature extraction completed.")


if __name__ == "__main__":
    main()
