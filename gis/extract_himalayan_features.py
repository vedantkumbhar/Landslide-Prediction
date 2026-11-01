
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "india_landslide_events.csv"
)

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "himalayas"
    / "N30E078"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "himalayas_N30E078_event_features.csv"
)


def main():
    raster_paths = {
        "elevation_m": DEM_DIR / "elevation.tif",
        "slope_deg": DEM_DIR / "slope.tif",
        "aspect_deg": DEM_DIR / "aspect.tif",
    }

    required_paths = [EVENTS_PATH, *raster_paths.values()]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(EVENTS_PATH)

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    # Transform geographic coordinates into the DEM's projected CRS.
    with rasterio.open(raster_paths["elevation_m"]) as src:
        target_crs = src.crs

        xs, ys = transform(
            "EPSG:4326",
            target_crs,
            df["longitude"].tolist(),
            df["latitude"].tolist(),
        )

        df["utm_x"] = xs
        df["utm_y"] = ys

        bounds = src.bounds

        inside = (
            df["utm_x"].between(bounds.left, bounds.right)
            & df["utm_y"].between(bounds.bottom, bounds.top)
        )

        matched = df.loc[inside].copy()

    if matched.empty:
        raise ValueError("No historical events fall inside the DEM.")

    coordinates = list(
        zip(matched["utm_x"], matched["utm_y"])
    )

   
    # Sample all three rasters at the same event coordinates.
    for feature_name, raster_path in raster_paths.items():
        with rasterio.open(raster_path) as src:
            if src.crs != target_crs:
                raise ValueError(
                    f"CRS mismatch in {raster_path.name}"
                )

            samples = list(
                src.sample(
                    coordinates,
                    indexes=1,
                    masked=True,
                )
            )

            values = np.ma.vstack(samples).reshape(-1)
            values = values.astype(float)
            values = np.where(
                np.ma.getmaskarray(values),
                np.nan,
                values,
            )

            matched[feature_name] = values


    # Aspect = -1 represents flat terrain in our current raster.
    # Keep this value for now; do not treat it as missing data.
    feature_columns = list(raster_paths.keys())

    matched["terrain_data_valid"] = (
        np.isfinite(matched[feature_columns]).all(axis=1)
        & (matched["elevation_m"] != -9999)
        & (matched["slope_deg"] != -9999)
        & (matched["aspect_deg"] != -9999)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("HIMALAYAN EVENT TERRAIN FEATURE EXTRACTION")
    print("=" * 60)
    print(f"India events with valid coordinates: {len(df)}")
    print(f"Events inside pilot DEM: {len(matched)}")
    print(
        "Events with valid terrain samples:",
        int(matched["terrain_data_valid"].sum()),
    )
    print(
        "Events with incomplete terrain samples:",
        int((~matched["terrain_data_valid"]).sum()),
    )

    print("\nTerrain feature summary:")
    print(matched[feature_columns].describe().round(2))

    print(f"\nSaved feature dataset: {OUTPUT_PATH}")
    print("Original historical dataset was not modified.")


if __name__ == "__main__":
    main()
