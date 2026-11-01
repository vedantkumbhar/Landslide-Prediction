
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/processed/dem/western_ghats")
TILES = ["N09E076", "N19E072", "N11E076"]


def valid_values(ds, window):
    values = ds.read(1, window=window, masked=True)
    return values.data, ~np.ma.getmaskarray(values)


for tile in TILES:
    folder = ROOT / tile / "metric"

    print(f"\n{'=' * 60}\nTile: {tile}\n{'=' * 60}")

    with rasterio.open(folder / "elevation.tif") as elev, \
         rasterio.open(folder / "slope.tif") as slope:

        total = 0
        edge = 0
        cardinal_invalid = 0
        cardinal_valid = 0
        nonfinite_center = 0

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

            ev, evalid = valid_values(elev, expanded)
            _, svalid = valid_values(slope, window)

            er = r0 - rr0
            ec = c0 - cc0

            center = ev[er:er + h, ec:ec + w]
            center_valid = evalid[er:er + h, ec:ec + w]
            mismatch = center_valid & ~svalid

            if not mismatch.any():
                continue

            total += int(mismatch.sum())
            nonfinite_center += int(
                np.count_nonzero(mismatch & ~np.isfinite(center))
            )

            rows, cols = np.indices((h, w))
            gr = rows + r0
            gc = cols + c0

            on_edge = (
                (gr == 0) | (gr == elev.height - 1)
                | (gc == 0) | (gc == elev.width - 1)
            )
            edge += int(np.count_nonzero(mismatch & on_edge))

            # Check whether all four cardinal neighbors are valid.
            for i, j in zip(*np.where(mismatch)):
                row = r0 + int(i)
                col = c0 + int(j)
                neighbors = []

                for nr, nc in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if 0 <= nr < elev.height and 0 <= nc < elev.width:
                        neighbors.append(
                            bool(evalid[nr - rr0, nc - cc0])
                        )
                    else:
                        neighbors.append(False)

                if all(neighbors):
                    cardinal_valid += 1
                else:
                    cardinal_invalid += 1

        print(f"Total elevation-valid/slope-invalid: {total:,}")
        print(f"On outer raster edge: {edge:,}")
        print(f"At least one cardinal neighbor invalid: {cardinal_invalid:,}")
        print(f"All four cardinal neighbors valid: {cardinal_valid:,}")
        print(f"Non-finite center elevations: {nonfinite_center:,}")

print("\nInspection complete. No raster files modified.")
