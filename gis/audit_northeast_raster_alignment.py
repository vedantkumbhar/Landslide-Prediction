
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DEM_ROOT = ROOT / "data" / "processed" / "dem" / "northeast"

FILES = {
    "N27E088": [
        "elevation.tif",
        "slope.tif",
        "aspect.tif",
    ],
    "N26E088": [
        "elevation.tif",
        "slope.tif",
        "aspect.tif",
    ],
}


def main():
    print("=" * 70)
    print("NORTHEAST DEM RASTER ALIGNMENT AUDIT")
    print("=" * 70)

    for tile, filenames in FILES.items():
        print(f"\n{'=' * 25} {tile} {'=' * 25}")
        reference = None

        for filename in filenames:
            path = DEM_ROOT / tile / filename

            if not path.exists():
                print(f"MISSING: {path}")
                continue

            with rasterio.open(path) as src:
                data = src.read(1, masked=True)
                valid = data.compressed()
                grid = (
                    src.crs,
                    src.width,
                    src.height,
                    src.transform,
                    src.nodata,
                )

                if reference is None:
                    reference = grid
                    aligned = True
                else:
                    aligned = (
                        src.crs == reference[0]
                        and src.width == reference[1]
                        and src.height == reference[2]
                        and src.transform == reference[3]
                    )

                print(f"\nFile: {filename}")
                print(f"CRS: {src.crs}")
                print(f"Dimensions: {src.width} x {src.height}")
                print(f"Pixel size: {src.res}")
                print(f"NoData: {src.nodata}")
                print(f"Valid pixels: {len(valid):,}")
                print(f"Grid matches elevation: {aligned}")

                if len(valid):
                    print(f"Minimum: {float(np.min(valid)):.3f}")
                    print(f"Maximum: {float(np.max(valid)):.3f}")

                    if filename == "slope.tif":
                        print(
                            "Pixels outside 0-90 degrees:",
                            int(np.sum((valid < 0) | (valid > 90))),
                        )

                    elif filename == "aspect.tif":
                        print(
                            "Pixels outside -1 to 360 degrees:",
                            int(np.sum((valid < -1) | (valid >= 360))),
                        )

    print("\nAudit completed. No files were modified.")


if __name__ == "__main__":
    main()

