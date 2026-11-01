
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "N27E088": (
        PROJECT_ROOT
        / "data/processed/dem/northeast/N27E088/elevation.tif"
    ),
    "N26E088": (
        PROJECT_ROOT
        / "data/processed/dem/northeast/N26E088/elevation.tif"
    ),
}


def main():
    print("=" * 65)
    print("NORTHEAST DEM TILE GEOMETRY INSPECTION")
    print("=" * 65)

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing raster: {path}")

        with rasterio.open(path) as src:
            geographic_bounds = transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds,
                densify_pts=41,
            )

            print(f"\n{name}")
            print("-" * 40)
            print(f"CRS: {src.crs}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Transform: {src.transform}")
            print(f"Projected bounds: {src.bounds}")
            print(
                "Approximate geographic bounds "
                "(west, south, east, north):"
            )
            print(
                ", ".join(f"{value:.6f}" for value in geographic_bounds)
            )
            print(f"Pixel size: {src.res}")
            print(f"NoData: {src.nodata}")

    print("\nGeometry inspection completed.")


if __name__ == "__main__":
    main()
