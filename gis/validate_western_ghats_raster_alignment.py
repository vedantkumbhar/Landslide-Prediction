
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path("data/processed/dem/western_ghats")
TILES = ["N09E076", "N19E072", "N11E076"]
LAYERS = ["elevation.tif", "slope.tif", "aspect.tif"]

all_passed = True

for tile in TILES:
    print(f"\n{'=' * 55}")
    print(f"Tile: {tile}")
    print("=" * 55)

    paths = [ROOT / tile / "metric" / name for name in LAYERS]

    if not all(path.exists() for path in paths):
        print("FAIL: One or more terrain rasters are missing.")
        all_passed = False
        continue

    with rasterio.open(paths[0]) as elevation, \
         rasterio.open(paths[1]) as slope, \
         rasterio.open(paths[2]) as aspect:

        reference = {
            "crs": elevation.crs,
            "width": elevation.width,
            "height": elevation.height,
            "transform": elevation.transform,
        }

        for name, ds in zip(LAYERS, [elevation, slope, aspect]):
            aligned = (
                ds.crs == reference["crs"]
                and ds.width == reference["width"]
                and ds.height == reference["height"]
                and ds.transform == reference["transform"]
            )

            print(
                f"{name}: CRS={ds.crs}, "
                f"size={ds.width}x{ds.height}, "
                f"aligned={aligned}"
            )

            if not aligned:
                all_passed = False

        if all(
            ds.crs == reference["crs"]
            and ds.width == reference["width"]
            and ds.height == reference["height"]
            and ds.transform == reference["transform"]
            for ds in [elevation, slope, aspect]
        ):
            masks_match = True
            for _, window in elevation.block_windows(1):
                masks = [
                    ds.read_masks(1, window=window) > 0
                    for ds in [elevation, slope, aspect]
                ]
                if not (
                    np.array_equal(masks[0], masks[1])
                    and np.array_equal(masks[0], masks[2])
                ):
                    masks_match = False
                    break

            print(f"Valid-data masks match: {masks_match}")

            if not masks_match:
                all_passed = False

print(f"\nFINAL RESULT: {'PASS' if all_passed else 'CHECK REQUIRED'}")
