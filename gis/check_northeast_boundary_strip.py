
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

STRIP_METRES = 300


def main():
    print("=" * 65)
    print("NORTHEAST DEM BOUNDARY STRIP CHECK")
    print("=" * 65)

    with rasterio.open(NORTH_FILE) as north, \
         rasterio.open(SOUTH_FILE) as south:

        if north.crs != south.crs:
            raise ValueError("The two DEMs use different CRSs.")

        # Find the shared projected boundary.
        boundary_y = (
            north.bounds.bottom + south.bounds.top
        ) / 2

        print(f"\nShared boundary Y: {boundary_y:.3f} m")
        print(f"Strip width on each side: {STRIP_METRES} m")

        north_data = north.read(1, masked=True).filled(np.nan)
        south_data = south.read(1, masked=True).filled(np.nan)

        north_rows = np.where(
            (north.transform.f + np.arange(north.height) * north.transform.e)
            >= boundary_y
        )[0]

        south_rows = np.where(
            (south.transform.f + np.arange(south.height) * south.transform.e)
            <= boundary_y
        )[0]

        if len(north_rows) == 0 or len(south_rows) == 0:
            print("Could not identify boundary rows.")
            return

        # Report elevation summaries near the boundary.
        north_strip_rows = north_rows[
            np.abs(
                north.transform.f
                + north_rows * north.transform.e
                - boundary_y
            ) <= STRIP_METRES
        ]

        south_strip_rows = south_rows[
            np.abs(
                south.transform.f
                + south_rows * south.transform.e
                - boundary_y
            ) <= STRIP_METRES
        ]

        def summarize(label, data, rows):
            strip = data[rows, :]
            valid = strip[np.isfinite(strip)]

            print(f"\n{label}")
            print("-" * 40)
            print(f"Rows examined: {len(rows)}")
            print(f"Valid pixels: {valid.size:,}")

            if valid.size:
                print(f"Minimum elevation: {valid.min():.3f} m")
                print(f"Mean elevation: {valid.mean():.3f} m")
                print(f"Maximum elevation: {valid.max():.3f} m")

        summarize("NORTH TILE", north_data, north_strip_rows)
        summarize("SOUTH TILE", south_data, south_strip_rows)

        # Reproject the southern elevation to the northern grid.
        south_aligned = np.full(
            (north.height, north.width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=south.read(1),
            destination=south_aligned,
            src_transform=south.transform,
            src_crs=south.crs,
            src_nodata=south.nodata,
            dst_transform=north.transform,
            dst_crs=north.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        overlap = (
            np.isfinite(north_data)
            & np.isfinite(south_aligned)
        )

        # Restrict the comparison to pixels near the southern
        # edge of the northern tile.
        row_distance = np.abs(
            north.transform.f
            + np.arange(north.height) * north.transform.e
            - boundary_y
        )

        near_boundary = row_distance[:, None] <= STRIP_METRES
        mask = overlap & near_boundary

        print("\nALIGNED OVERLAP NEAR BOUNDARY")
        print("-" * 40)
        print(f"Compared pixels: {int(mask.sum()):,}")

        if mask.any():
            difference = north_data[mask] - south_aligned[mask]

            print(f"Mean difference: {difference.mean():.3f} m")
            print(
                "Mean absolute difference: "
                f"{np.abs(difference).mean():.3f} m"
            )
            print(
                "Maximum absolute difference: "
                f"{np.abs(difference).max():.3f} m"
            )
        else:
            print("No valid overlapping pixels in the boundary strip.")

    print("\nBoundary strip check completed.")


if __name__ == "__main__":
    main()

