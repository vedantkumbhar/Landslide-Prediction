
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NORTH_FILE = (
    PROJECT_ROOT
    / "data/processed/dem/northeast/N27E088/elevation.tif"
)

SOUTH_FILE = (
    PROJECT_ROOT
    / "data/processed/dem/northeast/N26E088/elevation.tif"
)


def main():
    print("=" * 65)
    print("NORTHEAST DEM COMMON-GRID COMPARISON")
    print("=" * 65)

    with rasterio.open(NORTH_FILE) as north, \
         rasterio.open(SOUTH_FILE) as south:

        if north.crs != south.crs:
            raise ValueError("The rasters must use the same CRS.")

        north_data = north.read(1, masked=True).filled(np.nan)

        # Resample the southern DEM to the northern grid.
        south_aligned = np.full(
            (north.height, north.width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(south, 1),
            destination=south_aligned,
            src_transform=south.transform,
            src_crs=south.crs,
            src_nodata=south.nodata,
            dst_transform=north.transform,
            dst_crs=north.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        # Compare only valid pixels in both datasets.
        overlap = (
            np.isfinite(north_data)
            & np.isfinite(south_aligned)
        )

        north_values = north_data[overlap]
        south_values = south_aligned[overlap]

        print(f"\nCommon valid pixels: {north_values.size:,}")

        if north_values.size == 0:
            print("No valid overlap available for comparison.")
            return

        difference = north_values - south_values
        absolute_difference = np.abs(difference)

        print("\nElevation difference statistics")
        print("-" * 40)
        print(f"Mean signed difference: {difference.mean():.3f} m")
        print(f"Mean absolute difference: {absolute_difference.mean():.3f} m")
        print(f"Median absolute difference: {np.median(absolute_difference):.3f} m")
        print(f"95th percentile absolute difference: {np.percentile(absolute_difference, 95):.3f} m")
        print(f"Maximum absolute difference: {absolute_difference.max():.3f} m")

        print("\nInterpretation:")
        print(
            "These are differences between independently processed "
            "and resampled DEMs, not direct measurements of elevation "
            "accuracy. Review the statistics alongside source coverage "
            "and terrain conditions."
        )

    print("\nCommon-grid comparison completed.")


if __name__ == "__main__":
    main()
