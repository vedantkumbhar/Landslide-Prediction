
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/processed/dem/western_ghats")
TILES = ["N09E076", "N19E072", "N11E076"]

for tile in TILES:
    folder = ROOT / tile / "metric"
    paths = {
        "elevation": folder / "elevation.tif",
        "slope": folder / "slope.tif",
        "aspect": folder / "aspect.tif",
    }

    print(f"\n{'=' * 60}")
    print(f"Tile: {tile}")
    print("=" * 60)

    if not all(path.exists() for path in paths.values()):
        print("ERROR: A required raster is missing.")
        continue

    counts = {
        "elevation_only": 0,
        "slope_only": 0,
        "aspect_only": 0,
        "elevation_valid_slope_invalid": 0,
        "elevation_valid_aspect_invalid": 0,
        "slope_valid_aspect_invalid": 0,
        "all_three_valid": 0,
        "all_three_invalid": 0,
    }

    with rasterio.open(paths["elevation"]) as elev, \
         rasterio.open(paths["slope"]) as slope, \
         rasterio.open(paths["aspect"]) as aspect:

        for _, window in elev.block_windows(1):
            e = elev.read_masks(1, window=window) > 0
            s = slope.read_masks(1, window=window) > 0
            a = aspect.read_masks(1, window=window) > 0

            counts["elevation_only"] += int(
                np.count_nonzero(e & ~s & ~a)
            )
            counts["slope_only"] += int(
                np.count_nonzero(s & ~e & ~a)
            )
            counts["aspect_only"] += int(
                np.count_nonzero(a & ~e & ~s)
            )
            counts["elevation_valid_slope_invalid"] += int(
                np.count_nonzero(e & ~s)
            )
            counts["elevation_valid_aspect_invalid"] += int(
                np.count_nonzero(e & ~a)
            )
            counts["slope_valid_aspect_invalid"] += int(
                np.count_nonzero(s & ~a)
            )
            counts["all_three_valid"] += int(
                np.count_nonzero(e & s & a)
            )
            counts["all_three_invalid"] += int(
                np.count_nonzero(~e & ~s & ~a)
            )

    for name, count in counts.items():
        print(f"{name}: {count:,}")

print("\nDiagnosis complete. No raster files were modified.")
