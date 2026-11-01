
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parent.parent

RAW = (
    ROOT / "data" / "raw" / "dem" / "northeast"
    / "N27E088" / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)

PROCESSED = (
    ROOT / "data" / "processed" / "dem" / "northeast"
    / "N27E088" / "elevation.tif"
)


def main():
    print("=" * 65)
    print("VERIFYING N27E088 MAXIMUM ELEVATION")
    print("=" * 65)

    with rasterio.open(PROCESSED) as proc:
        data = proc.read(1, masked=True)
        values = data.compressed()

        if values.size == 0:
            print("No valid processed elevation values.")
            return

        flat_index = int(np.argmax(data.filled(-np.inf)))
        row, col = np.unravel_index(flat_index, data.shape)
        processed_value = float(data[row, col])

        x, y = proc.xy(row, col)
        lon, lat = transform(
            proc.crs, "EPSG:4326", [x], [y]
        )

        print(f"Processed maximum: {processed_value:.3f} m")
        print(f"Pixel row/column: {row}, {col}")
        print(f"Projected coordinates: {x:.2f}, {y:.2f}")
        print(f"Geographic coordinates: {lon[0]:.6f}, {lat[0]:.6f}")

    print("\nChecking corresponding raw DEM location...")

    with rasterio.open(RAW) as raw:
        print("Raw CRS:", raw.crs)
        print("Raw dimensions:", raw.width, "x", raw.height)
        print("Raw bounds:", raw.bounds)
        print("Raw NoData:", raw.nodata)

        raw_x, raw_y = transform(
            "EPSG:4326", raw.crs, [lon[0]], [lat[0]]
        )

        try:
            raw_row, raw_col = raw.index(raw_x[0], raw_y[0])
            print(f"Raw pixel row/column: {raw_row}, {raw_col}")

            if 0 <= raw_row < raw.height and 0 <= raw_col < raw.width:
                raw_value = raw.read(1, window=(
                    (raw_row, raw_row + 1),
                    (raw_col, raw_col + 1),
                ))[0, 0]

                print(f"Raw elevation at location: {float(raw_value):.3f}")
                print(
                    "Interpretation: compare this value with the processed "
                    "elevation, accounting for reprojection and resampling."
                )
            else:
                print("Location falls outside the raw raster.")

        except Exception as exc:
            print("Could not sample raw DEM:", exc)


if __name__ == "__main__":
    main()
