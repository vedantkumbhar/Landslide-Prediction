
from pathlib import Path

import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N27E088"
)

LATITUDE = 26.9974
LONGITUDE = 88.4314
EVENT_ID = 6966


def main():
    print("=" * 65)
    print(f"INSPECTING MISSING TERRAIN VALUES: EVENT {EVENT_ID}")
    print("=" * 65)

    elevation_path = DEM_DIR / "elevation.tif"

    with rasterio.open(elevation_path) as reference:
        transformer = Transformer.from_crs(
            "EPSG:4326",
            reference.crs,
            always_xy=True,
        )

        x, y = transformer.transform(LONGITUDE, LATITUDE)
        row, col = rasterio.transform.rowcol(
            reference.transform, x, y
        )

        print(f"\nCoordinates: {LATITUDE}, {LONGITUDE}")
        print(f"Projected coordinates: X={x:.3f}, Y={y:.3f}")
        print(f"Raster row: {row}, column: {col}")

        for name in ["elevation.tif", "slope.tif", "aspect.tif"]:
            path = DEM_DIR / name

            with rasterio.open(path) as src:
                print(f"\n{name}")
                print("-" * 40)

                if not (
                    0 <= row < src.height
                    and 0 <= col < src.width
                ):
                    print("Event pixel is outside raster dimensions.")
                    continue

                value = src.read(
                    1,
                    window=((row, row + 1), (col, col + 1)),
                    masked=True,
                )[0, 0]

                if value is not None and not (
                    hasattr(value, "mask") and value.mask
                ):
                    print(f"Pixel value: {float(value):.6f}")
                else:
                    print("Pixel value: NoData")

                print(f"Raster NoData value: {src.nodata}")

                # Inspect a 3 x 3 neighborhood around the event.
                r0 = max(0, row - 1)
                r1 = min(src.height, row + 2)
                c0 = max(0, col - 1)
                c1 = min(src.width, col + 2)

                neighborhood = src.read(
                    1,
                    window=((r0, r1), (c0, c1)),
                    masked=True,
                )

                print("3 x 3 neighborhood (masked cells shown as --):")
                for raster_row in neighborhood:
                    print(
                        " ".join(
                            "--" if is_masked else f"{float(v):.2f}"
                            for v, is_masked in zip(
                                raster_row.data,
                                raster_row.mask
                                if hasattr(raster_row, "mask")
                                else [False] * len(raster_row),
                            )
                        )
                    )

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
