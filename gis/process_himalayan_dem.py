
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DEM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "himalayas"
    / "N30E078"
    / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "himalayas"
    / "N30E078"
)

TARGET_CRS = "EPSG:32644"
NODATA = -9999.0


def main():
    if not INPUT_DEM.exists():
        raise FileNotFoundError(f"Input DEM not found: {INPUT_DEM}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # First pass: determine the projected raster dimensions.
    with rasterio.open(INPUT_DEM) as src:
        if src.crs is None:
            raise ValueError("Input DEM has no CRS.")

        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            tiled=True,
        )

        print("Input dimensions:", src.width, "x", src.height)
        print("Projected dimensions:", width, "x", height)
        print("Projected CRS:", TARGET_CRS)

        # Reproject into a disk-backed GeoTIFF.
        projected_path = OUTPUT_DIR / "elevation.tif"

        with rasterio.open(projected_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )

    print("Saved projected elevation:", projected_path)

    # Use neighbouring pixels and metric pixel spacing to calculate
    # terrain derivatives. Process one row at a time to limit memory.
    with rasterio.open(projected_path) as src:
        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)

        slope_path = OUTPUT_DIR / "slope.tif"
        aspect_path = OUTPUT_DIR / "aspect.tif"

        out_profile = src.profile.copy()
        out_profile.update(
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            tiled=True,
        )

        width = src.width
        height = src.height

        with (
            rasterio.open(slope_path, "w", **out_profile) as slope_dst,
            rasterio.open(aspect_path, "w", **out_profile) as aspect_dst,
        ):
            for row in range(height):
                # Read a 3-row window so the central row has neighbours.
                start = max(0, row - 1)
                end = min(height, row + 2)

                window = rasterio.windows.Window(
                    0, start, width, end - start
                )
                block = src.read(1, window=window, masked=True)
                block = block.filled(np.nan).astype("float32")

                if block.shape[0] < 2:
                    slope_row = np.full(width, NODATA, dtype="float32")
                    aspect_row = np.full(width, NODATA, dtype="float32")
                else:
                    center_idx = row - start

                    # One-sided differences at edges; central differences
                    # for interior rows and columns.
                    center = block[center_idx]
                    above = block[max(0, center_idx - 1)]
                    below = block[min(block.shape[0] - 1, center_idx + 1)]

                    left = np.roll(center, 1)
                    right = np.roll(center, -1)

                    dx = (right - left) / (2 * pixel_width)
                    dy = (above - below) / (2 * pixel_height)

                    dx[0] = (right[0] - center[0]) / pixel_width
                    dx[-1] = (center[-1] - left[-1]) / pixel_width

                    slope_row = np.degrees(
                        np.arctan(np.hypot(dx, dy))
                    ).astype("float32")

                    aspect_row = (
                        np.degrees(np.arctan2(-dx, -dy)) + 360
                    ) % 360

                    flat = np.hypot(dx, dy) < 1e-6
                    aspect_row[flat] = -1

                    invalid = (
                        ~np.isfinite(center)
                        | ~np.isfinite(dx)
                        | ~np.isfinite(dy)
                    )
                    slope_row[invalid] = NODATA
                    aspect_row[invalid] = NODATA

                row_window = rasterio.windows.Window(
                    0, row, width, 1
                )
                slope_dst.write(
                    slope_row.reshape(1, -1),
                    1,
                    window=row_window,
                )
                aspect_dst.write(
                    aspect_row.reshape(1, -1),
                    1,
                    window=row_window,
                )

                if (row + 1) % 500 == 0:
                    print(f"Processed {row + 1}/{height} rows")

    print("\nTerrain processing completed.")
    print("Elevation:", projected_path)
    print("Slope:", slope_path)
    print("Aspect:", aspect_path)


if __name__ == "__main__":
    main()
