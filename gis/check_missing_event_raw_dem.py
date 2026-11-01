
from pathlib import Path

import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DEM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "northeast"
    / "N27E088"
    / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)

LATITUDE = 26.9974
LONGITUDE = 88.4314


def main():
    print("=" * 65)
    print("CHECKING ORIGINAL SRTM DEM AT EVENT 6966")
    print("=" * 65)

    if not RAW_DEM.exists():
        raise FileNotFoundError(f"Raw DEM not found: {RAW_DEM}")

    with rasterio.open(RAW_DEM) as src:
        print(f"\nRaw DEM: {RAW_DEM}")
        print(f"CRS: {src.crs}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bounds: {src.bounds}")
        print(f"NoData value: {src.nodata}")

        transformer = Transformer.from_crs(
            "EPSG:4326",
            src.crs,
            always_xy=True,
        )

        x, y = transformer.transform(LONGITUDE, LATITUDE)

        row, col = rasterio.transform.rowcol(
            src.transform, x, y
        )

        print(f"\nEvent coordinates: {LATITUDE}, {LONGITUDE}")
        print(f"Raw raster row: {row}, column: {col}")

        if not (0 <= row < src.height and 0 <= col < src.width):
            print("Event is outside the original raster.")
            return

        value = src.read(
            1,
            window=((row, row + 1), (col, col + 1)),
            masked=True,
        )[0, 0]

        if value is np.ma.masked:
            print("Original elevation: NoData")
        else:
            print(f"Original elevation: {float(value):.3f} m")

        r0 = max(0, row - 1)
        r1 = min(src.height, row + 2)
        c0 = max(0, col - 1)
        c1 = min(src.width, col + 2)

        neighborhood = src.read(
            1,
            window=((r0, r1), (c0, c1)),
            masked=True,
        )

        print("\nOriginal DEM 3 x 3 neighborhood:")
        for raster_row in neighborhood:
            print(
                " ".join(
                    "--" if masked else f"{float(value):.2f}"
                    for value, masked in zip(
                        raster_row.data,
                        np.ma.getmaskarray(raster_row),
                    )
                )
            )

    print("\nRaw DEM check completed.")


if __name__ == "__main__":
    import numpy as np
    main()
