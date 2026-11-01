
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/processed/dem/western_ghats")
TILES = ["N09E076", "N19E072", "N11E076"]

for tile in TILES:
    folder = ROOT / tile / "metric"

    print(f"\n{'=' * 60}")
    print(f"Tile: {tile}")
    print("=" * 60)

    with rasterio.open(folder / "elevation.tif") as elev, \
         rasterio.open(folder / "slope.tif") as slope:

        total = 0
        edge = 0
        interior = 0
        elevation_nodata_nearby = 0

        for _, window in elev.block_windows(1):
            e = elev.read(1, window=window, masked=True)
            s = slope.read(1, window=window, masked=True)

            elevation_valid = ~np.ma.getmaskarray(e)
            slope_invalid = np.ma.getmaskarray(s)
            mismatch = elevation_valid & slope_invalid

            total += int(mismatch.sum())

            row_start = int(window.row_off)
            row_end = row_start + int(window.height)
            col_start = int(window.col_off)
            col_end = col_start + int(window.width)

            rows, cols = np.indices(mismatch.shape)
            global_rows = rows + row_start
            global_cols = cols + col_start

            is_edge = (
                (global_rows == 0)
                | (global_rows == elev.height - 1)
                | (global_cols == 0)
                | (global_cols == elev.width - 1)
            )

            edge += int((mismatch & is_edge).sum())
            interior += int((mismatch & ~is_edge).sum())

        print(f"Elevation-valid / slope-invalid pixels: {total:,}")
        print(f"Pixels on outer raster edge: {edge:,}")
        print(f"Pixels inside raster boundary: {interior:,}")

print("\nInspection complete. No files modified.")
