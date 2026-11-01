
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEM_ROOT = PROJECT_ROOT / "data/processed/dem/northeast"

NORTH_ELEV = DEM_ROOT / "N27E088/elevation.tif"
NORTH_SLOPE = DEM_ROOT / "N27E088/slope.tif"
SOUTH_ELEV = DEM_ROOT / "N26E088/elevation.tif"


def main():
    print("=" * 65)
    print("ANALYZING NORTHEAST DEM ELEVATION DIFFERENCES")
    print("=" * 65)

    with rasterio.open(NORTH_ELEV) as north, \
         rasterio.open(NORTH_SLOPE) as slope_src, \
         rasterio.open(SOUTH_ELEV) as south:

        north_data = north.read(1, masked=True).filled(np.nan)
        slope_data = slope_src.read(1, masked=True).filled(np.nan)

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

    valid = (
        np.isfinite(north_data)
        & np.isfinite(south_aligned)
        & np.isfinite(slope_data)
    )

    north_values = north_data[valid]
    south_values = south_aligned[valid]
    slopes = slope_data[valid]

    difference = north_values - south_values
    absolute = np.abs(difference)

    print(f"\nValid comparison pixels: {len(difference):,}")

    if len(difference) == 0:
        print("No valid pixels to analyze.")
        return

    print("\nOverall differences")
    print("-" * 40)
    print(f"Mean signed difference: {difference.mean():.3f} m")
    print(f"Mean absolute difference: {absolute.mean():.3f} m")
    print(f"Median absolute difference: {np.median(absolute):.3f} m")
    print(f"95th percentile absolute difference: {np.percentile(absolute, 95):.3f} m")

    # Descriptive summaries by slope class.
    print("\nDifference by northern-tile slope class")
    print("-" * 65)

    bins = [
        (0, 5, "0–5°"),
        (5, 15, "5–15°"),
        (15, 30, "15–30°"),
        (30, 90.01, "30° and above"),
    ]

    for lower, upper, label in bins:
        mask = (slopes >= lower) & (slopes < upper)
        count = int(mask.sum())

        if count:
            print(
                f"{label:16s} n={count:4d} | "
                f"MAE={absolute[mask].mean():7.3f} m | "
                f"P95={np.percentile(absolute[mask], 95):7.3f} m"
            )
        else:
            print(f"{label:16s} n=0")

    print(
        "\nThese are descriptive comparisons, not a formal accuracy "
        "assessment. Both datasets may share source or processing errors."
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()

