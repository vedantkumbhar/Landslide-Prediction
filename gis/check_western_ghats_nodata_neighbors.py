
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

        mismatch_count = 0
        mismatch_near_nodata = 0
        mismatch_not_near_nodata = 0

        # Read by windows with a one-pixel margin so neighboring
        # elevation cells are included at window boundaries.
        for _, window in elev.block_windows(1):
            r0 = int(window.row_off)
            c0 = int(window.col_off)
            h = int(window.height)
            w = int(window.width)

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(elev.height, r0 + h + 1)
            cc1 = min(elev.width, c0 + w + 1)

            expanded = rasterio.windows.Window(
                cc0, rr0, cc1 - cc0, rr1 - rr0
            )

            e = elev.read(1, window=expanded, masked=True)
            s = slope.read(1, window=window, masked=True)

            er0 = r0 - rr0
            ec0 = c0 - cc0

            center = e[
                er0:er0 + h,
                ec0:ec0 + w
            ]

            center_valid = ~np.ma.getmaskarray(center)
            slope_invalid = np.ma.getmaskarray(s)
            mismatch = center_valid & slope_invalid

            if not mismatch.any():
                continue

            # A center pixel is near NoData if any of its eight
            # neighboring elevation cells is invalid.
            invalid_elevation = np.ma.getmaskarray(e)
            near_nodata = np.zeros((h, w), dtype=bool)

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    source_r0 = er0 + dr
                    source_c0 = ec0 + dc
                    source_r1 = source_r0 + h
                    source_c1 = source_c0 + w

                    # Clip shifts at raster boundaries.
                    for i in range(h):
                        gr = r0 + i
                        nr = gr + dr
                        if nr < 0 or nr >= elev.height:
                            continue

                        for j in range(w):
                            if not mismatch[i, j]:
                                continue

                            gc = c0 + j
                            nc = gc + dc
                            if nc < 0 or nc >= elev.width:
                                continue

                            local_r = nr - rr0
                            local_c = nc - cc0

                            if invalid_elevation[local_r, local_c]:
                                near_nodata[i, j] = True

            mismatch_count += int(mismatch.sum())
            mismatch_near_nodata += int((mismatch & near_nodata).sum())
            mismatch_not_near_nodata += int(
                (mismatch & ~near_nodata).sum()
            )

        print(f"Total mismatches: {mismatch_count:,}")
        print(f"Near elevation NoData: {mismatch_near_nodata:,}")
        print(f"Not near elevation NoData: {mismatch_not_near_nodata:,}")

print("\nCheck complete. No raster files modified.")
