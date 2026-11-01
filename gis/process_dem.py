
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DEM = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "rasters_SRTMGL1"
    / "output_SRTMGL1.tif"
)

# Separate output folder to preserve our original test results.
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dem" / "metric"

TARGET_CRS = "EPSG:32643"
NODATA = -9999.0


def main():
    if not INPUT_DEM.exists():
        raise FileNotFoundError(f"DEM not found: {INPUT_DEM}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the original DEM and prepare its metric projection.
    with rasterio.open(INPUT_DEM) as src:
        if src.crs is None:
            raise ValueError("Input DEM has no CRS.")

        elevation = src.read(1, masked=True).astype("float32")

        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
        )

        projected = np.full((height, width), NODATA, dtype="float32")

        reproject(
            source=elevation.filled(NODATA),
            destination=projected,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(projected) & (projected != NODATA)

    if valid.sum() == 0:
        raise ValueError("No valid elevation pixels after reprojection.")

    # Save the reprojected elevation raster.
    elevation_path = OUTPUT_DIR / "elevation.tif"
    with rasterio.open(elevation_path, "w", **profile) as dst:
        dst.write(projected, 1)

    # Calculate gradients using pixel spacing in metres.
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    # Avoid invalid values interfering with the gradient calculation.
    work = projected.copy()
    work[~valid] = np.nan

    # This initial test tile has valid coverage throughout.
    if not valid.all():
        raise ValueError(
            "Projected DEM contains NoData pixels. "
            "Handle masked neighbourhoods before calculating derivatives."
        )

    grad_y, grad_x = np.gradient(
        work,
        pixel_height,
        pixel_width,
    )

    slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))

    # Aspect is clockwise from north.
    aspect = (
        np.degrees(np.arctan2(-grad_x, grad_y)) + 360
    ) % 360

    # Flat pixels have no meaningful compass direction.
    aspect[np.hypot(grad_x, grad_y) < 1e-6] = -1

    for name, array in [
        ("slope", slope),
        ("aspect", aspect),
    ]:
        output_path = OUTPUT_DIR / f"{name}.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype("float32"), 1)

        print(f"\n{name.upper()}")
        print(f"Saved: {output_path}")
        print(f"Minimum: {array.min():.2f}")
        print(f"Maximum: {array.max():.2f}")

    print("\nELEVATION")
    print(f"Saved: {elevation_path}")
    print(f"Minimum: {projected[valid].min():.2f} m")
    print(f"Maximum: {projected[valid].max():.2f} m")

    print(f"\nProjected CRS: {TARGET_CRS}")
    print(f"Pixel size: {pixel_width:.2f} x {pixel_height:.2f} metres")
    print(f"Raster dimensions: {width} x {height}")
    print("\nMetric terrain processing completed.")


if __name__ == "__main__":
    main()
