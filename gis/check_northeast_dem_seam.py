
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TILES = {
    "N27E088": (
        PROJECT_ROOT / "data/processed/dem/northeast/N27E088/elevation.tif"
    ),
    "N26E088": (
        PROJECT_ROOT / "data/processed/dem/northeast/N26E088/elevation.tif"
    ),
}


def main():
    print("=" * 65)
    print("NORTHEAST DEM SEAM CONSISTENCY CHECK")
    print("=" * 65)

    for name, path in TILES.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} DEM not found: {path}")

    with rasterio.open(TILES["N27E088"]) as north:
        north_data = north.read(1, masked=True).filled(np.nan)
        north_transform = north.transform
        north_crs = north.crs
        north_bounds = north.bounds

    with rasterio.open(TILES["N26E088"]) as south:
        south_data = south.read(1, masked=True).filled(np.nan)
        south_transform = south.transform
        south_crs = south.crs
        south_bounds = south.bounds

    print("\nTILE INFORMATION")
    print("-" * 40)
    print(f"N27E088 CRS: {north_crs}")
    print(f"N27E088 bounds: {north_bounds}")
    print(f"N26E088 CRS: {south_crs}")
    print(f"N26E088 bounds: {south_bounds}")

    if north_crs != south_crs:
        raise ValueError("Tile CRS mismatch. Align the rasters first.")

    # These tiles should be adjacent north-south tiles.
    # Compare a narrow strip near their shared geographic boundary.
    with rasterio.open(TILES["N27E088"]) as north, \
         rasterio.open(TILES["N26E088"]) as south:

        # Reproject the southern raster onto the northern raster's grid.
        south_on_north = np.full(
            (north.height, north.width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=south.read(1),
            destination=south_on_north,
            src_transform=south.transform,
            src_crs=south.crs,
            src_nodata=south.nodata,
            dst_transform=north.transform,
            dst_crs=north.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    overlap = np.isfinite(north_data) & np.isfinite(south_on_north)

    print("\nGRID OVERLAP CHECK")
    print("-" * 40)
    print(f"Overlapping valid pixels: {int(overlap.sum()):,}")

    if overlap.any():
        differences = north_data[overlap] - south_on_north[overlap]

        print("\nElevation differences in overlap:")
        print(f"Mean difference: {differences.mean():.3f} m")
        print(f"Mean absolute difference: {np.abs(differences).mean():.3f} m")
        print(f"Maximum absolute difference: {np.abs(differences).max():.3f} m")
    else:
        print(
            "No overlapping valid pixels were found. "
            "This may be expected for adjacent tiles; "
            "a boundary-strip comparison is needed instead."
        )

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
