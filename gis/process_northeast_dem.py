
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "northeast"
    / "N27E088"
    / "rasters_SRTMGL1 (1)"
    / "output_SRTMGL1.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N27E088"
)

TARGET_CRS = "EPSG:32645"
NODATA = -9999.0
ROWS_PER_BLOCK = 256


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input DEM not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    elevation_path = OUTPUT_DIR / "elevation.tif"
    slope_path = OUTPUT_DIR / "slope.tif"
    aspect_path = OUTPUT_DIR / "aspect.tif"

    # Step 1: Reproject elevation to a metric CRS.
    with rasterio.open(INPUT_PATH) as src:
        if src.crs is None or src.crs.to_string() != "EPSG:4326":
            raise ValueError(f"Expected EPSG:4326, got {src.crs}")

        dst_transform, dst_width, dst_height = (
            calculate_default_transform(
                src.crs,
                TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
            )
        )

        elevation_profile = {
            "driver": "GTiff",
            "width": dst_width,
            "height": dst_height,
            "count": 1,
            "dtype": "float32",
            "crs": TARGET_CRS,
            "transform": dst_transform,
            "nodata": NODATA,
            "compress": "deflate",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }

        with rasterio.open(
            elevation_path, "w", **elevation_profile
        ) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )

    # Step 2: Calculate slope and aspect block by block.
    with rasterio.open(elevation_path) as elev:
        pixel_width = abs(elev.transform.a)
        pixel_height = abs(elev.transform.e)

        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError("Invalid raster pixel dimensions.")

        profile = elev.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
        )

        with (
            rasterio.open(slope_path, "w", **profile) as slope_dst,
            rasterio.open(aspect_path, "w", **profile) as aspect_dst,
        ):
            for row_start in range(0, elev.height, ROWS_PER_BLOCK):
                row_end = min(
                    row_start + ROWS_PER_BLOCK, elev.height
                )

                # Include one neighbouring row on each side where possible.
                read_start = max(0, row_start - 1)
                read_end = min(elev.height, row_end + 1)

                block = elev.read(
                    1,
                    window=Window(
                        0,
                        read_start,
                        elev.width,
                        read_end - read_start,
                    ),
                )

                valid = np.isfinite(block) & (block != NODATA)

                # Extract the central rows that will be written.
                offset = row_start - read_start
                count = row_end - row_start
                center = block[offset:offset + count]
                center_valid = valid[offset:offset + count]

                # North/south neighbours.
                north = np.full_like(center, np.nan)
                south = np.full_like(center, np.nan)
                north_valid = np.zeros(center.shape, dtype=bool)
                south_valid = np.zeros(center.shape, dtype=bool)

                if row_start > 0:
                    north[:] = block[offset - 1:offset + count - 1]
                    north_valid[:] = valid[offset - 1:offset + count - 1]
                else:
                    # Only the actual first raster row lacks a north neighbour.
                    north[1:] = center[:-1]
                    north_valid[1:] = center_valid[:-1]

                if row_end < elev.height:
                    south[:] = block[offset + 1:offset + count + 1]
                    south_valid[:] = valid[offset + 1:offset + count + 1]
                else:
                    # Only the actual last raster row lacks a south neighbour.
                    south[:-1] = center[1:]
                    south_valid[:-1] = center_valid[1:]

                # East/west neighbours.
                west = np.full_like(center, np.nan)
                east = np.full_like(center, np.nan)
                west_valid = np.zeros(center.shape, dtype=bool)
                east_valid = np.zeros(center.shape, dtype=bool)

                west[:, 1:] = center[:, :-1]
                east[:, :-1] = center[:, 1:]

                west_valid[:, 1:] = center_valid[:, :-1]
                east_valid[:, :-1] = center_valid[:, 1:]

                # Central differences in metres.
                with np.errstate(invalid="ignore", divide="ignore"):
                    dzdx = (east - west) / (2 * pixel_width)
                    dzdy = (north - south) / (2 * pixel_height)

                neighborhood_valid = (
                    center_valid
                    & north_valid
                    & south_valid
                    & west_valid
                    & east_valid
                    & np.isfinite(dzdx)
                    & np.isfinite(dzdy)
                )

                slope = np.degrees(
                    np.arctan(np.hypot(dzdx, dzdy))
                )

                # Aspect convention: clockwise from north.
                aspect = np.mod(
                    np.degrees(np.arctan2(-dzdx, -dzdy)),
                    360.0,
                )

                flat = (
                    neighborhood_valid
                    & np.isclose(dzdx, 0.0)
                    & np.isclose(dzdy, 0.0)
                )
                aspect[flat] = -1.0

                slope_out = np.where(
                    neighborhood_valid, slope, NODATA
                ).astype(np.float32)

                aspect_out = np.where(
                    neighborhood_valid, aspect, NODATA
                ).astype(np.float32)

                window = Window(
                    0,
                    row_start,
                    elev.width,
                    count,
                )

                slope_dst.write(slope_out, 1, window=window)
                aspect_dst.write(aspect_out, 1, window=window)

                print(
                    f"Processed rows: {row_end}/{elev.height}"
                )

    print("\nTerrain processing completed.")
    print(f"Elevation: {elevation_path}")
    print(f"Slope: {slope_path}")
    print(f"Aspect: {aspect_path}")
    print("Original source DEM was not modified.")


if __name__ == "__main__":
    main()
