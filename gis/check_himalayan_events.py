
from pathlib import Path

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

DEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "himalayas"
    / "N30E078"
    / "elevation.tif"
)


def main():
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(EVENTS_PATH)

    if not DEM_PATH.exists():
        raise FileNotFoundError(DEM_PATH)

    df = pd.read_csv(EVENTS_PATH)

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    with rasterio.open(DEM_PATH) as src:
        x, y = transform(
            "EPSG:4326",
            src.crs,
            df["longitude"].tolist(),
            df["latitude"].tolist(),
        )

        df["utm_x"] = x
        df["utm_y"] = y

        bounds = src.bounds

        inside = (
            df["utm_x"].between(bounds.left, bounds.right)
            & df["utm_y"].between(bounds.bottom, bounds.top)
        )

        matched = df.loc[inside].copy()

        print("=" * 55)
        print("HISTORICAL EVENTS INSIDE HIMALAYAN DEM")
        print("=" * 55)
        print(f"Valid India event coordinates: {len(df):,}")
        print(f"Events inside DEM bounds: {len(matched):,}")

        print("\nEvents by administrative region:")
        if matched.empty:
            print("No events fall inside this raster.")
        else:
            print(
                matched["admin_division_name"]
                .fillna("Unknown")
                .value_counts()
                .to_string()
            )

        print("\nDEM geographic bounds after transformation:")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {bounds}")

    print("\nNo source data was modified.")


if __name__ == "__main__":
    main()
