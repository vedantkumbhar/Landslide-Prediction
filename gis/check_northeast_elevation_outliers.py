
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DEM_ROOT = ROOT / "data" / "processed" / "dem" / "northeast"

FILES = [
    ("N27E088", DEM_ROOT / "N27E088" / "elevation.tif"),
    ("N26E088", DEM_ROOT / "N26E088" / "elevation.tif"),
]


def main():
    for tile, path in FILES:
        print("\n" + "=" * 60)
        print(f"ELEVATION OUTLIER CHECK: {tile}")
        print("=" * 60)

        with rasterio.open(path) as src:
            data = src.read(1, masked=True).compressed()
            data = data[np.isfinite(data)]

            if data.size == 0:
                print("No valid elevation values.")
                continue

            for percentile in [50, 90, 95, 99, 99.9, 99.99, 100]:
                print(
                    f"P{percentile:6g}: "
                    f"{np.percentile(data, percentile):.3f} m"
                )

            for threshold in [3000, 4000, 5000, 6000, 8000]:
                count = int(np.sum(data > threshold))
                print(f"Pixels above {threshold:4d} m: {count:,}")

            maximum = float(data.max())
            rows, cols = np.where(
                src.read(1) == maximum
            )

            if len(rows):
                x, y = src.xy(int(rows[0]), int(cols[0]))
                print(f"Maximum elevation: {maximum:.3f} m")
                print(f"Maximum pixel row/col: {rows[0]}, {cols[0]}")
                print(f"Maximum pixel coordinates: {x:.2f}, {y:.2f}")


if __name__ == "__main__":
    main()
