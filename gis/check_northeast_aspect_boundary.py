
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DEM_ROOT = ROOT / "data" / "processed" / "dem" / "northeast"

for tile in ["N27E088", "N26E088"]:
    path = DEM_ROOT / tile / "aspect.tif"

    print("\n" + "=" * 60)
    print(f"ASPECT BOUNDARY CHECK: {tile}")
    print("=" * 60)

    with rasterio.open(path) as src:
        data = src.read(1, masked=True).compressed()
        data = data[np.isfinite(data)]

        print("Total valid pixels:", len(data))
        print("Exactly 360 degrees:", int(np.sum(data == 360.0)))
        print("Above or equal to 360:", int(np.sum(data >= 360.0)))
        print("Below 0, excluding flat -1:", int(np.sum((data < 0) & (data != -1))))
        print("Unique values above 359.9:", np.unique(data[data > 359.9])[:20])

print("\nNo files were modified.")
