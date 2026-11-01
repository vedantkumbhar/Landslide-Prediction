
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import convolve


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DEM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "northeast"
    / "N26E088"
    / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N26E088"
)

TARGET_CRS = "EPSG:32645"
NODATA = -9999.0


def main():
    print("=" * 65)
    print("PROCESSING N26E088 DEM")
    print("=" * 65)

    if not INPUT_DEM.exists():
        raise FileNotFoundError(f"Input DEM not found: {INPUT_DEM}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elevation_path = OUTPUT_DIR / "elevation.tif"

    # Reproject the elevation raster.
    with rasterio.open(INPUT_DEM) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=30,
        )

        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            nodata=NODATA,
            count=1,
            compress="deflate",
        )

        with rasterio.open(elevation_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )

    print(f"\nElevation saved: {elevation_path}")

    # Calculate slope and aspect from the reprojected elevation.
    # A one-pixel border is excluded because its full neighborhood
    # is not available.
    with rasterio.open(elevation_path) as src:
        elevation = src.read(1, masked=True).astype("float64").filled(np.nan)
        transform = src.transform
        profile = src.profile.copy()

    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)

    valid_center = np.isfinite(elevation)
    valid_neighbors = convolve(
        valid_center.astype("uint8"),
        np.ones((3, 3), dtype="uint8"),
        mode="constant",
        cval=0,
    ) == 9

    # Central differences on a projected grid measured in metres.
    safe_elevation = np.where(valid_center, elevation, 0.0)

    dzdx = (
        safe_elevation[1:-1, 2:]
        - safe_elevation[1:-1, :-2]
    ) / (2 * pixel_x)

    dzdy = (
        safe_elevation[:-2, 1:-1]
        - safe_elevation[2:, 1:-1]
    ) / (2 * pixel_y)

    slope = np.full(elevation.shape, NODATA, dtype="float32")
    aspect = np.full(elevation.shape, NODATA, dtype="float32")

    slope_inner = np.degrees(
        np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    )

    aspect_inner = np.mod(
        np.degrees(np.arctan2(-dzdx, -dzdy)),
        360.0,
    )

    valid_inner = valid_neighbors[1:-1, 1:-1]

    slope[1:-1, 1:-1][valid_inner] = slope_inner[valid_inner]
    aspect[1:-1, 1:-1][valid_inner] = aspect_inner[valid_inner]

    flat = (
        valid_inner
        & (np.sqrt(dzdx**2 + dzdy**2) < 1e-10)
    )
    aspect_inner_output = aspect[1:-1, 1:-1]
    aspect_inner_output[flat] = -1.0

    for name, array in [
        ("slope.tif", slope),
        ("aspect.tif", aspect),
    ]:
        output_path = OUTPUT_DIR / name
        out_profile = profile.copy()
        out_profile.update(
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
        )

        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(array, 1)

        print(f"Saved: {output_path}")

    print("\nN26E088 processing completed.")


if __name__ == "__main__":
    main()
