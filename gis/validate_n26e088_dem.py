
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N26E088"
)

RASTER_NAMES = [
    "elevation.tif",
    "slope.tif",
    "aspect.tif",
]


def main():
    print("=" * 65)
    print("N26E088 PROCESSED DEM VALIDATION")
    print("=" * 65)

    reference = None
    arrays = {}

    for name in RASTER_NAMES:
        path = DEM_DIR / name

        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()

            print(f"\n{name.upper()}")
            print("-" * 40)
            print(f"CRS: {src.crs}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Pixel size: {src.res}")
            print(f"NoData: {src.nodata}")
            print(f"Valid pixels: {valid.size:,}")
            print(f"NoData pixels: {int(data.mask.sum()):,}")

            if valid.size:
                finite = np.isfinite(valid)
                print(f"Non-finite valid values: {int((~finite).sum()):,}")
                print(f"Minimum: {valid.min():.6f}")
                print(f"Maximum: {valid.max():.6f}")
                print(f"Mean: {valid.mean():.6f}")

            current = {
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "transform": src.transform,
            }

            if reference is None:
                reference = current
            else:
                print(
                    "Same CRS:",
                    current["crs"] == reference["crs"],
                )
                print(
                    "Same dimensions:",
                    current["width"] == reference["width"]
                    and current["height"] == reference["height"],
                )
                print(
                    "Same transform:",
                    current["transform"] == reference["transform"],
                )

            arrays[name] = data

    elevation = arrays["elevation.tif"].compressed()
    slope = arrays["slope.tif"].compressed()
    aspect = arrays["aspect.tif"].compressed()

    directional_aspect = aspect[aspect != -1]

    print("\n" + "=" * 65)
    print("TERRAIN RANGE CHECKS")
    print("=" * 65)

    elevation_ok = (
        np.isfinite(elevation).all()
        and (elevation >= -500).all()
        and (elevation <= 9000).all()
    )

    slope_ok = (
        np.isfinite(slope).all()
        and (slope >= 0).all()
        and (slope <= 90).all()
    )

    aspect_ok = (
        np.isfinite(aspect).all()
        and (directional_aspect >= 0).all()
        and (directional_aspect < 360).all()
    )

    print(f"Elevation range valid: {elevation_ok}")
    print(f"Slope between 0 and 90 degrees: {slope_ok}")
    print(f"Aspect follows expected range: {aspect_ok}")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
