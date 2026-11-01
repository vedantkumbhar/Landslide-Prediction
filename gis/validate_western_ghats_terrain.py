
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
DEM_ROOT = ROOT / "data" / "processed" / "dem" / "western_ghats"

TILES = ["N09E076", "N19E072", "N11E076"]
RASTER_NAMES = ["elevation", "slope", "aspect"]


def main():
    for tile in TILES:
        print(f"\n{'=' * 60}")
        print(f"Terrain validation: {tile}")
        print("=" * 60)

        for name in RASTER_NAMES:
            path = DEM_ROOT / tile / "metric" / f"{name}.tif"

            if not path.exists():
                print(f"{name}: MISSING — {path}")
                continue

            with rasterio.open(path) as ds:
                valid_count = 0
                total_count = ds.width * ds.height
                minimum = float("inf")
                maximum = float("-inf")
                nonfinite_count = 0
                valid_outside_expected_range = 0

                for _, window in ds.block_windows(1):
                    values = ds.read(1, window=window, masked=True)
                    valid = values.compressed()

                    if valid.size == 0:
                        continue

                    finite = np.isfinite(valid)
                    nonfinite_count += int((~finite).sum())
                    valid = valid[finite]

                    if valid.size == 0:
                        continue

                    valid_count += int(valid.size)
                    minimum = min(minimum, float(valid.min()))
                    maximum = max(maximum, float(valid.max()))

                    if name == "slope":
                        valid_outside_expected_range += int(
                            ((valid < 0) | (valid > 90)).sum()
                        )
                    elif name == "aspect":
                        # -1 is permitted for flat terrain.
                        valid_outside_expected_range += int(
                            ((valid < -1) | (valid >= 360)).sum()
                        )

                nodata_percent = (
                    100 * (total_count - valid_count) / total_count
                )

                print(f"\n{name}:")
                print(f"  CRS: {ds.crs}")
                print(f"  Dimensions: {ds.width} x {ds.height}")
                print(f"  Pixel size: {ds.res}")
                print(f"  Valid pixels: {valid_count:,} / {total_count:,}")
                print(f"  Nodata percentage: {nodata_percent:.2f}%")

                if valid_count:
                    print(f"  Minimum: {minimum:.3f}")
                    print(f"  Maximum: {maximum:.3f}")
                else:
                    print("  WARNING: No valid pixels found.")

                print(f"  Non-finite valid pixels: {nonfinite_count}")
                print(
                    "  Values outside expected range: "
                    f"{valid_outside_expected_range}"
                )

                if ds.crs is None:
                    print("  WARNING: CRS is missing.")

                if name == "elevation" and valid_count:
                    if minimum < -500 or maximum > 9000:
                        print(
                            "  WARNING: Elevation range needs investigation."
                        )

                if nodata_percent > 20:
                    print(
                        "  WARNING: High nodata percentage; "
                        "inspect raster coverage."
                    )


if __name__ == "__main__":
    main()
