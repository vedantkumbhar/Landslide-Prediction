
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent

RAW = (
    ROOT / "data" / "raw" / "dem" / "northeast"
    / "N27E088" / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)

with rasterio.open(RAW) as src:
    # Geographic coordinates of the maximum found in our previous check
    lon, lat = 88.148380, 27.701030

    row, col = src.index(lon, lat)
    radius = 5

    r0 = max(0, row - radius)
    r1 = min(src.height, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(src.width, col + radius + 1)

    window = src.read(1, window=((r0, r1), (c0, c1)))
    nodata = src.nodata

    valid = window != nodata
    values = window[valid].astype(float)

    print("=" * 60)
    print("RAW DEM PEAK NEIGHBORHOOD")
    print("=" * 60)
    print(f"Center row/column: {row}, {col}")
    print(f"Neighborhood size: {window.shape[0]} x {window.shape[1]}")
    print(f"Valid pixels: {len(values)}")

    if len(values):
        print(f"Minimum: {values.min():.1f} m")
        print(f"Median: {np.median(values):.1f} m")
        print(f"Maximum: {values.max():.1f} m")
        print(f"Pixels above 8000 m: {np.sum(values > 8000)}")

    print("\nElevation neighborhood (metres):")
    print(window)

    print("\nNoData mask (1 = NoData, 0 = valid):")
    print((~valid).astype(int))
