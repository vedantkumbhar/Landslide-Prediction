
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DEM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "northeast"
    / "N26E088"
    / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)


def main():
    print("=" * 65)
    print("N26E088 RAW DEM VALIDATION")
    print("=" * 65)

    if not RAW_DEM.exists():
        raise FileNotFoundError(f"DEM file not found: {RAW_DEM}")

    with rasterio.open(RAW_DEM) as src:
        data = src.read(1, masked=True)
        valid = data.compressed()

        print(f"\nFile: {RAW_DEM}")
        print(f"File size: {RAW_DEM.stat().st_size:,} bytes")
        print(f"CRS: {src.crs}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bounds: {src.bounds}")
        print(f"Pixel size: {src.res}")
        print(f"NoData value: {src.nodata}")
        print(f"Total pixels: {src.width * src.height:,}")
        print(f"Valid pixels: {valid.size:,}")
        print(f"NoData pixels: {data.mask.sum():,}")

        if valid.size:
            print(f"Minimum elevation: {valid.min():.3f} m")
            print(f"Maximum elevation: {valid.max():.3f} m")
            print(f"Mean elevation: {valid.mean():.3f} m")
            print(
                "Non-finite valid pixels: "
                f"{int((~np.isfinite(valid)).sum()):,}"
            )

        print("\nCHECKS")
        print("-" * 40)

        crs_ok = src.crs is not None and src.crs.to_epsg() == 4326
        dimensions_ok = src.width > 0 and src.height > 0
        bounds = src.bounds
        bounds_ok = (
            abs(bounds.left - 88.0) < 0.01
            and abs(bounds.right - 89.0) < 0.01
            and abs(bounds.bottom - 26.0) < 0.01
            and abs(bounds.top - 27.0) < 0.01
        )
        data_ok = valid.size > 0 and np.isfinite(valid).all()

        print(f"WGS84 geographic CRS: {crs_ok}")
        print(f"Valid raster dimensions: {dimensions_ok}")
        print(f"Expected N26E088 coverage: {bounds_ok}")
        print(f"Valid finite elevation data: {data_ok}")

        if crs_ok and dimensions_ok and bounds_ok and data_ok:
            print("\nRaw DEM validation: PASS")
        else:
            print("\nRaw DEM validation: CHECK REQUIRED")


if __name__ == "__main__":
    main()
