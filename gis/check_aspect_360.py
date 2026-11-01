
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/processed/dem/western_ghats")
TILES = ["N09E076", "N19E072", "N11E076"]

for tile in TILES:
    raster_path = ROOT / tile / "metric" / "aspect.tif"

    if not raster_path.exists():
        print(f"{tile}: aspect raster NOT FOUND")
        continue

    count_360 = 0
    valid_count = 0
    count_minus_1 = 0

    with rasterio.open(raster_path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)
            values = data.compressed()

            valid_count += values.size
            count_360 += int(np.count_nonzero(values == 360.0))
            count_minus_1 += int(np.count_nonzero(values == -1.0))

    print(f"\nTile: {tile}")
    print(f"Valid aspect pixels: {valid_count:,}")
    print(f"Pixels exactly equal to 360 degrees: {count_360:,}")
    print(f"Flat-terrain sentinel pixels (-1): {count_minus_1:,}")
