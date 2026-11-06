"""
Process Tier 1 Western Ghats SRTM DEM tiles into metric terrain rasters.

Tiles:
- N19E073 (Maharashtra: Malshej Ghat, Kasara, Igatpuri)
- N10E076 (Tamil Nadu / Kerala: Valparai, Anamalai, Palakkad)
- N10E077 (Tamil Nadu / Kerala: Munnar, Idukki, Kodaikanal)
- N12E074 (Karnataka: Mangalore, Ullal, Dakshina Kannada)
- N14E074 (Karnataka / Goa: Uttara Kannada, Karwar, Canacona)
- N12E075 (Karnataka: Coorg / Kodagu, Madikeri, Subrahmanya)

Reuses the established project methodology from gis/process_western_ghats_dem.py.
Outputs GeoTIFFs:
- elevation.tif: Reprojected metric elevation (m)
- slope.tif: Terrain slope (degrees, [0, 90])
- aspect.tif: Compass aspect (degrees, [0, 360), -1 for flat terrain, -9999 for NoData)

Target CRS: UTM Zone 43N (EPSG:32643) at 30 m resolution.
Output directory: data/processed/dem/western_ghats_tier1/{TILE}/metric/
Never overwrites existing files or modifies pre-existing tiles.
"""

from pathlib import Path
import math
import os
import re
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "dem" / "western_ghats"
OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "dem" / "western_ghats_tier1"

TIER1_TILES = [
    "N19E073",
    "N10E076",
    "N10E077",
    "N12E074",
    "N14E074",
    "N12E075",
]

NODATA = -9999.0
HGT_NODATA = -32768
TARGET_CRS = CRS.from_epsg(32643)  # UTM Zone 43N


def tile_parts(tile: str):
    """Parse tile coordinates (e.g. N19E073 -> lat 19, lon 73)."""
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile)
    if not match:
        raise ValueError(f"Invalid tile name: {tile}")
    ns, lat, ew, lon = match.groups()
    latitude = int(lat) * (1 if ns == "N" else -1)
    longitude = int(lon) * (1 if ew == "E" else -1)
    return latitude, longitude


def validate_raw_hgt(path: Path, tile: str):
    """Validate that uncompressed HGT exists and is readable."""
    expected_bytes = 3601 * 3601 * 2
    if not path.exists():
        raise FileNotFoundError(f"{tile}: raw HGT missing at {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{tile}: incorrect HGT size {path.stat().st_size:,}; expected {expected_bytes:,}"
        )
    with rasterio.open(path) as src:
        if src.driver != "SRTMHGT":
            raise ValueError(f"{tile}: unexpected driver {src.driver}")
        if (src.width, src.height) != (3601, 3601):
            raise ValueError(f"{tile}: unexpected dimensions {src.width}x{src.height}")
        if src.crs != CRS.from_epsg(4326):
            raise ValueError(f"{tile}: unexpected CRS {src.crs}")
    return path


def output_is_valid(path: Path, expected_crs: CRS = TARGET_CRS) -> bool:
    """Validate that a generated GeoTIFF is readable, non-empty, and has correct CRS."""
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


def process_tile(tile: str) -> dict:
    """Reproject DEM and derive slope and aspect GeoTIFFs for a tile."""
    raw_path = RAW_ROOT / tile / f"{tile}.hgt"
    validate_raw_hgt(raw_path, tile)

    metric_dir = OUT_ROOT / tile / "metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    elevation_path = metric_dir / "elevation.tif"
    slope_path = metric_dir / "slope.tif"
    aspect_path = metric_dir / "aspect.tif"
    final_paths = (elevation_path, slope_path, aspect_path)

    # Check if all final rasters already exist and are valid (never overwrite)
    if all(p.exists() for p in final_paths):
        if all(output_is_valid(p, TARGET_CRS) for p in final_paths):
            print(f"  {tile}: rasters already exist and pass validation (skipping).")
            return {
                "tile": tile,
                "elevation_path": str(elevation_path),
                "slope_path": str(slope_path),
                "aspect_path": str(aspect_path),
                "status": "reused_existing",
            }
        else:
            raise ValueError(
                f"{tile}: rasters exist but fail validation. Manual check required."
            )

    # Refuse overwrite if any partial final raster exists
    existing_finals = [p for p in final_paths if p.exists()]
    if existing_finals:
        raise FileExistsError(
            f"{tile}: partial final output files exist; refusing to overwrite:\n"
            + "\n".join(str(p) for p in existing_finals)
        )

    # Temporary staging paths
    temp_dir = metric_dir / ".tmp_stage"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = (
        temp_dir / "elevation.tmp.tif",
        temp_dir / "slope.tmp.tif",
        temp_dir / "aspect.tmp.tif",
    )

    print(f"\nProcessing {tile}")
    print(f"  Target CRS: {TARGET_CRS} (UTM Zone 43N, 30m resolution)")

    try:
        # Step 1: Reproject raw HGT to metric UTM Zone 43N at 30m resolution
        with rasterio.open(raw_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
                resolution=30.0,
            )

            profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": 1,
                "dtype": "float32",
                "crs": TARGET_CRS,
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
                    dst_crs=TARGET_CRS,
                    dst_nodata=NODATA,
                    resampling=Resampling.bilinear,
                )

        # Step 2: Compute slope and aspect row-by-row
        with rasterio.open(temp_paths[0]) as src:
            dx_size = abs(src.transform.a)  # exactly 30m
            dy_size = abs(src.transform.e)  # exactly 30m

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

                    dx = (right - left) / (2.0 * dx_size)
                    dy = (above - below) / (2.0 * dy_size)

                    gradient = np.hypot(dx, dy)

                    with np.errstate(invalid="ignore", divide="ignore"):
                        slope_row = np.degrees(np.arctan(gradient)).astype("float32")
                        aspect_row = ((np.degrees(np.arctan2(-dx, -dy)) + 360.0) % 360.0).astype("float32")

                    # Aspect is undefined on nearly flat terrain (gradient < 1e-6)
                    aspect_row[gradient < 1e-6] = -1.0

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
                    slope_dst.write(slope_row.reshape(1, -1), 1, window=row_window)
                    aspect_dst.write(aspect_row.reshape(1, -1), 1, window=row_window)

                    if (row + 1) % 1000 == 0:
                        print(f"    {tile}: processed {row + 1}/{height} rows")

        # Step 3: Validate all temporary rasters before publishing
        for path in temp_paths:
            if not output_is_valid(path, TARGET_CRS):
                raise ValueError(f"Raster validation failed: {path}")

        # Step 4: Exclusive publication check
        if any(p.exists() for p in final_paths):
            raise FileExistsError(
                f"{tile}: final output appeared during processing; refusing to overwrite."
            )

        for temp_p, final_p in zip(temp_paths, final_paths):
            temp_p.replace(final_p)

        print(f"  Successfully processed {tile}:")
        print(f"    Elevation: {elevation_path.name} ({elevation_path.stat().st_size:,} bytes)")
        print(f"    Slope:     {slope_path.name} ({slope_path.stat().st_size:,} bytes)")
        print(f"    Aspect:    {aspect_path.name} ({aspect_path.stat().st_size:,} bytes)")

        return {
            "tile": tile,
            "elevation_path": str(elevation_path),
            "slope_path": str(slope_path),
            "aspect_path": str(aspect_path),
            "status": "success",
        }

    finally:
        # Clean up temporary staging files if they still exist
        for p in temp_paths:
            if p.exists():
                p.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()


def main():
    print("=" * 70)
    print("STAGE 3: PROCESS METRIC DEM RASTERS (WESTERN GHATS TIER 1)")
    print("=" * 70)
    print(f"Tiles ({len(TIER1_TILES)}): {', '.join(TIER1_TILES)}")
    print(f"Raw input root:  {RAW_ROOT}")
    print(f"Target output:   {OUT_ROOT}")
    print(f"Target CRS:      EPSG:32643 (UTM Zone 43N)")
    print(f"Resolution:      30.0 m x 30.0 m")
    print("=" * 70)

    results = []
    for idx, tile in enumerate(TIER1_TILES, 1):
        print(f"\n[{idx}/{len(TIER1_TILES)}] Ingesting tile: {tile}")
        res = process_tile(tile)
        results.append(res)

    print("\n" + "=" * 70)
    print("STAGE 3 PROCESSING COMPLETED")
    print("=" * 70)
    for r in results:
        print(f"  {r['tile']:<8} [{r['status']}]")


if __name__ == "__main__":
    main()
