
from pathlib import Path
import math
import re

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = ROOT / "data" / "raw" / "dem" / "western_ghats"
OUT_ROOT = ROOT / "data" / "processed" / "dem" / "western_ghats"

PILOT_TILES = ("N09E076", "N19E072", "N11E076")

NODATA = -9999.0
HGT_NODATA = -32768


def tile_parts(tile):
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile)

    if not match:
        raise ValueError(f"Invalid tile name: {tile}")

    ns, lat, ew, lon = match.groups()

    latitude = int(lat) * (1 if ns == "N" else -1)
    longitude = int(lon) * (1 if ew == "E" else -1)

    return latitude, longitude


def utm_crs_for_tile(tile):
    latitude, longitude = tile_parts(tile)

    # UTM zone for the approximate centre of the one-degree tile.
    zone = math.floor((longitude + 0.5 + 180) / 6) + 1
    zone = max(1, min(60, zone))

    epsg = (32600 if latitude >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def validate_raw_hgt(path, tile):
    expected_bytes = 3601 * 3601 * 2

    if not path.exists():
        raise FileNotFoundError(f"Missing raw HGT: {path}")

    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{tile}: incorrect HGT size "
            f"{path.stat().st_size}; expected {expected_bytes}"
        )

    with rasterio.open(path) as src:
        if src.driver != "SRTMHGT":
            raise ValueError(f"{tile}: unexpected driver {src.driver}")

        if (src.width, src.height) != (3601, 3601):
            raise ValueError(f"{tile}: unexpected HGT dimensions")

        if src.crs != CRS.from_epsg(4326):
            raise ValueError(f"{tile}: unexpected source CRS {src.crs}")

    return path


def output_is_valid(path, expected_crs=None):
    try:
        with rasterio.open(path) as ds:
            if ds.width < 100 or ds.height < 100:
                return False

            if ds.count != 1 or ds.crs is None:
                return False

            if expected_crs is not None and ds.crs != expected_crs:
                return False

            sample_width = min(128, ds.width)
            sample_height = min(128, ds.height)

            windows = [
                Window(0, 0, sample_width, sample_height),
                Window(
                    ds.width - sample_width,
                    ds.height - sample_height,
                    sample_width,
                    sample_height,
                ),
                Window(
                    ds.width // 2 - sample_width // 2,
                    ds.height // 2 - sample_height // 2,
                    sample_width,
                    sample_height,
                ),
            ]

            for window in windows:
                values = ds.read(1, window=window, masked=False)
                mask = ds.read_masks(1, window=window) > 0

                if mask.any() and not np.isfinite(values[mask]).all():
                    return False

            return True

    except Exception:
        return False


def process_tile(tile):
    raw_path = RAW_ROOT / tile / f"{tile}.hgt"
    validate_raw_hgt(raw_path, tile)

    metric_dir = OUT_ROOT / tile / "metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    elevation_path = metric_dir / "elevation.tif"
    slope_path = metric_dir / "slope.tif"
    aspect_path = metric_dir / "aspect.tif"

    final_paths = (elevation_path, slope_path, aspect_path)

    # Never overwrite any existing final output.
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"{tile}: output files already exist; refusing to overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )

    temp_paths = (
        metric_dir / "elevation.tmp.tif",
        metric_dir / "slope.tmp.tif",
        metric_dir / "aspect.tmp.tif",
    )

    # Avoid overwriting stale temporary files from an interrupted run.
    existing_temps = [path for path in temp_paths if path.exists()]
    if existing_temps:
        raise FileExistsError(
            f"{tile}: temporary files already exist. Inspect them first:\n"
            + "\n".join(str(path) for path in existing_temps)
        )

    target_crs = utm_crs_for_tile(tile)

    print(f"\nProcessing {tile}")
    print("Target CRS:", target_crs)

    try:
        with rasterio.open(raw_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=30,
            )

            profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": 1,
                "dtype": "float32",
                "crs": target_crs,
                "transform": transform,
                "nodata": NODATA,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
            }

            with rasterio.open(temp_paths[0], "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    dst_nodata=NODATA,
                    resampling=Resampling.bilinear,
                )

        # Calculate terrain derivatives one row at a time.
        with rasterio.open(temp_paths[0]) as src:
            dx_size = abs(src.transform.a)
            dy_size = abs(src.transform.e)

            profile = src.profile.copy()
            width = src.width
            height = src.height

            with (
                rasterio.open(temp_paths[1], "w", **profile) as slope_dst,
                rasterio.open(temp_paths[2], "w", **profile) as aspect_dst,
            ):
                for row in range(height):
                    start = max(0, row - 1)
                    end = min(height, row + 2)

                    window = Window(0, start, width, end - start)
                    block = src.read(1, window=window, masked=True)
                    block = block.filled(np.nan).astype("float32")

                    center_idx = row - start
                    center = block[center_idx]

                    # Require both a north/south and an east/west
                    # neighbour for a valid terrain derivative.
                    above = (
                        block[center_idx - 1]
                        if center_idx > 0
                        else np.full(width, np.nan, dtype="float32")
                    )
                    below = (
                        block[center_idx + 1]
                        if center_idx + 1 < block.shape[0]
                        else np.full(width, np.nan, dtype="float32")
                    )

                    left = np.full(width, np.nan, dtype="float32")
                    right = np.full(width, np.nan, dtype="float32")

                    if width > 1:
                        left[1:] = center[:-1]
                        right[:-1] = center[1:]

                    dx = (right - left) / (2 * dx_size)
                    dy = (above - below) / (2 * dy_size)

                    gradient = np.hypot(dx, dy)

                    with np.errstate(invalid="ignore", divide="ignore"):
                        slope_row = np.degrees(
                            np.arctan(gradient)
                        ).astype("float32")

                        aspect_row = (
                            np.degrees(np.arctan2(-dx, -dy)) + 360
                        ) % 360

                        aspect_row = aspect_row.astype("float32")

                    # Aspect is undefined on nearly flat terrain.
                    aspect_row[gradient < 1e-6] = -1

                    invalid = (
                        ~np.isfinite(center)
                        | ~np.isfinite(above)
                        | ~np.isfinite(below)
                        | ~np.isfinite(left)
                        | ~np.isfinite(right)
                        | ~np.isfinite(slope_row)
                        | ~np.isfinite(aspect_row)
                        | (center == NODATA)
                        | (above == NODATA)
                        | (below == NODATA)
                        | (left == NODATA)
                        | (right == NODATA)
                    )

                    slope_row[invalid] = NODATA
                    aspect_row[invalid] = NODATA

                    row_window = Window(0, row, width, 1)

                    slope_dst.write(
                        slope_row.reshape(1, -1),
                        1,
                        window=row_window,
                    )
                    aspect_dst.write(
                        aspect_row.reshape(1, -1),
                        1,
                        window=row_window,
                    )

                    if (row + 1) % 1000 == 0:
                        print(
                            f"  {tile}: processed "
                            f"{row + 1}/{height} rows"
                        )

        # Validate all temporary rasters before publishing any final files.
        for path in temp_paths:
            if not output_is_valid(path, target_crs):
                raise ValueError(f"Raster validation failed: {path}")

        # The final names are checked again immediately before publishing.
        if any(path.exists() for path in final_paths):
            raise FileExistsError(
                f"{tile}: final output appeared during processing; "
                "refusing to overwrite it."
            )

        for temporary, final in zip(temp_paths, final_paths):
            temporary.replace(final)

        print("Successfully processed:", tile)
        print("  Elevation:", elevation_path)
        print("  Slope:", slope_path)
        print("  Aspect:", aspect_path)

    except Exception:
        # Keep final outputs untouched; remove temporary files created
        # during this run so a retry starts cleanly.
        for path in temp_paths:
            if path.exists():
                path.unlink()
        raise


def main():
    print("=" * 65)
    print("WESTERN GHATS DEM TERRAIN PROCESSING")
    print("=" * 65)
    print("Tiles:", ", ".join(PILOT_TILES))
    print("Raw root:", RAW_ROOT)
    print("Output root:", OUT_ROOT)
    print("Existing final rasters will not be overwritten.")

    for tile in PILOT_TILES:
        process_tile(tile)

    print("\nAll pilot tiles processed and validated.")
    print("Next step: inspect terrain statistics and raster coverage.")


if __name__ == "__main__":
    main()

