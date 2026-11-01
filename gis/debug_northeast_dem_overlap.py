
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parent.parent
DEM_ROOT = ROOT / "data" / "processed" / "dem" / "northeast"

FILES = {
    "North elevation": DEM_ROOT / "N27E088" / "elevation.tif",
    "North slope": DEM_ROOT / "N27E088" / "slope.tif",
    "South elevation": DEM_ROOT / "N26E088" / "elevation.tif",
}


def inspect_raster(label, path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)

        print(f"\n{label}")
        print("-" * 45)
        print("CRS:", src.crs)
        print("Dimensions:", src.width, "x", src.height)
        print("NoData value:", src.nodata)
        print("Bounds:", src.bounds)
        print("Valid pixels:", int(data.count()))
        print("Valid percentage:", round(data.count() / data.size * 100, 3), "%")

        if data.count():
            values = data.compressed()
            print("Min:", float(values.min()))
            print("Max:", float(values.max()))


def main():
    for label, path in FILES.items():
        if not path.exists():
            print(f"FILE MISSING: {path}")
            return
        inspect_raster(label, path)

    print("\n" + "=" * 60)
    print("REPROJECTION CHECK")
    print("=" * 60)

    with rasterio.open(FILES["North elevation"]) as north, \
         rasterio.open(FILES["South elevation"]) as south:

        aligned = np.full(
            (north.height, north.width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(south, 1),
            destination=aligned,
            src_transform=south.transform,
            src_crs=south.crs,
            src_nodata=south.nodata,
            dst_transform=north.transform,
            dst_crs=north.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )

        north_data = north.read(1, masked=True).filled(np.nan)

        print("North finite pixels:", int(np.isfinite(north_data).sum()))
        print("Reprojected south finite pixels:", int(np.isfinite(aligned).sum()))
        print(
            "Overlapping finite pixels:",
            int((np.isfinite(north_data) & np.isfinite(aligned)).sum()),
        )


if __name__ == "__main__":
    main()
