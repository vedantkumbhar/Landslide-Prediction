
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "northeast"
    / "N27E088"
    / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)


def main():
    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"DEM file not found: {DEM_PATH}"
        )

    with rasterio.open(DEM_PATH) as src:
        data = src.read(1, masked=True)
        valid = data.compressed()

        print("=" * 60)
        print("DEM TILE METADATA")
        print("=" * 60)
        print(f"File: {DEM_PATH.name}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Pixel size: {src.res}")
        print(f"NoData value: {src.nodata}")
        print(f"Valid pixels: {valid.size:,}")
        print(
            "NoData pixels:",
            int(data.mask.sum()),
        )

        if valid.size:
            print(f"Minimum elevation: {valid.min():.2f} m")
            print(f"Maximum elevation: {valid.max():.2f} m")
            print(f"Mean elevation: {valid.mean():.2f} m")
            print(
                "Non-finite valid pixels:",
                int((~np.isfinite(valid)).sum()),
            )

        print("\nExpected approximate geographic bounds:")
        print("Longitude: 88°E to 89°E")
        print("Latitude: 27°N to 28°N")

        print("\nOriginal raster was not modified.")


if __name__ == "__main__":
    main()
