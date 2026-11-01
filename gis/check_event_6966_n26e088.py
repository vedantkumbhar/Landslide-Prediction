
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N26E088"
)

LATITUDE = 26.9974
LONGITUDE = 88.4314
EVENT_ID = 6966


def main():
    print("=" * 65)
    print(f"CHECKING EVENT {EVENT_ID} AGAINST N26E088")
    print("=" * 65)

    with rasterio.open(DEM_DIR / "elevation.tif") as reference:
        transformer = Transformer.from_crs(
            "EPSG:4326",
            reference.crs,
            always_xy=True,
        )

        x, y = transformer.transform(LONGITUDE, LATITUDE)
        row, col = rasterio.transform.rowcol(
            reference.transform, x, y
        )

        print(f"\nEvent coordinates: {LATITUDE}, {LONGITUDE}")
        print(f"Projected coordinates: X={x:.3f}, Y={y:.3f}")
        print(f"Raster row: {row}, column: {col}")

        if not (
            0 <= row < reference.height
            and 0 <= col < reference.width
        ):
            print("\nEvent is outside the N26E088 raster.")
            return

    results = {}

    for name in ["elevation.tif", "slope.tif", "aspect.tif"]:
        path = DEM_DIR / name

        with rasterio.open(path) as src:
            value = src.read(
                1,
                window=((row, row + 1), (col, col + 1)),
                masked=True,
            )[0, 0]

            if np.ma.is_masked(value) or not np.isfinite(float(value)):
                results[name] = None
                print(f"{name}: NoData")
            else:
                results[name] = float(value)
                print(f"{name}: {float(value):.6f}")

    print("\nRESULT")
    print("-" * 40)

    if all(value is not None for value in results.values()):
        print("All three terrain features are available.")
        print(f"Elevation: {results['elevation.tif']:.2f} m")
        print(f"Slope: {results['slope.tif']:.2f} degrees")

        aspect = results["aspect.tif"]
        if aspect == -1:
            print("Aspect: Flat terrain (stored as -1)")
        else:
            print(f"Aspect: {aspect:.2f} degrees")
    else:
        print("At least one terrain feature remains unavailable.")
        print("Do not impute values until the raster coverage is investigated.")

    print("\nEvent check completed.")


if __name__ == "__main__":
    main()
