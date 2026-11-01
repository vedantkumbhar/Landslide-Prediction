from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import gzip
import shutil
import math
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/processed/dem/dem_tile_plan.csv"
RAW_ROOT = ROOT / "data/raw/dem"
OUT_ROOT = ROOT / "data/processed/dem"
LOG_PATH = OUT_ROOT / "batch_processing_log.csv"
NODATA = -9999.0
HGT_NODATA = -32768
EXPECTED_BYTES = 3601 * 3601 * 2
BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def tile_parts(tile):
    import re
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile)
    if not match:
        raise ValueError(f"Invalid tile name: {tile}")
    ns, lat, ew, lon = match.groups()
    latitude = int(lat) * (1 if ns == "N" else -1)
    longitude = int(lon) * (1 if ew == "E" else -1)
    return latitude, longitude


def utm_crs_for_tile(tile):
    _, lon = tile_parts(tile)
    zone = math.floor((lon + 0.5 + 180) / 6) + 1
    zone = max(1, min(60, zone))
    _, lat = tile_parts(tile)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def region_folder(region):
    return {
        "Himalayas": "himalayas",
        "Northeast India": "northeast",
        "Western Maharashtra": "western_maharashtra",
    }.get(region, "other")


def download_tile(tile, raw_dir):
    archive = raw_dir / f"{tile}.hgt.gz"
    hgt = raw_dir / f"{tile}.hgt"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if hgt.exists() and hgt.stat().st_size == EXPECTED_BYTES:
        print("Using existing HGT:", hgt)
        return hgt

    if not archive.exists() or archive.stat().st_size == 0:
        lat, _ = tile_parts(tile)
        band = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
        url = f"{BASE_URL}/{band}/{tile}.hgt.gz"
        part = archive.with_suffix(".gz.part")

        print("Downloading:", url)
        request = Request(url, headers={"User-Agent": "LandslidePredictionResearch/1.0"})
        last_error = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=90) as response, open(part, "wb") as dst:
                    shutil.copyfileobj(response, dst)
                with gzip.open(part, "rb") as src:
                    size = 0
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                if size != EXPECTED_BYTES:
                    raise ValueError(f"Unexpected decompressed size: {size}")
                part.replace(archive)
                break
            except Exception as exc:
                last_error = exc
                if part.exists():
                    part.unlink()
                if attempt < 2:
                    print(f"Attempt {attempt + 1} failed: {exc}; retrying...")
                    time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Download failed for {tile}: {last_error}")

    # Extract to a temporary directory using the exact .hgt extension.
    temp_dir = raw_dir / ".tmp_validate"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_hgt = temp_dir / f"{tile}.hgt"
    try:
        with gzip.open(archive, "rb") as src, open(temp_hgt, "wb") as dst:
            shutil.copyfileobj(src, dst)

        if temp_hgt.stat().st_size != EXPECTED_BYTES:
            raise ValueError(f"Bad HGT size for {tile}")

        with rasterio.open(temp_hgt) as ds:
            if ds.driver != "SRTMHGT" or ds.width != 3601 or ds.height != 3601:
                raise ValueError(f"Invalid SRTM grid for {tile}")
            if ds.crs != CRS.from_epsg(4326):
                raise ValueError(f"Unexpected source CRS for {tile}: {ds.crs}")

        temp_hgt.replace(hgt)
    except Exception:
        if temp_hgt.exists():
            temp_hgt.unlink()
        raise
    finally:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    print("Validated raw tile:", hgt)
    return hgt


def output_is_valid(path, expected_crs=None):
    try:
        with rasterio.open(path) as ds:
            if ds.width < 100 or ds.height < 100 or ds.crs is None:
                return False
            if expected_crs and ds.crs != expected_crs:
                return False
            sample = ds.read(1, window=Window(0, 0, min(32, ds.width), min(32, ds.height)))
            return sample.size > 0
    except Exception:
        return False


def process_tile(tile, hgt, out_dir):
    metric_dir = out_dir / "metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    elevation_path = metric_dir / "elevation.tif"
    slope_path = metric_dir / "slope.tif"
    aspect_path = metric_dir / "aspect.tif"
    target_crs = utm_crs_for_tile(tile)

    if all(output_is_valid(p, target_crs) for p in
           (elevation_path, slope_path, aspect_path)):
        print("Skipping already processed tile:", tile)
        return

    temp_elev = metric_dir / "elevation.tmp.tif"
    temp_slope = metric_dir / "slope.tmp.tif"
    temp_aspect = metric_dir / "aspect.tmp.tif"

    for p in (temp_elev, temp_slope, temp_aspect):
        if p.exists():
            p.unlink()

    with rasterio.open(hgt) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=30
        )
        profile = {
            "driver": "GTiff", "width": width, "height": height,
            "count": 1, "dtype": "float32", "crs": target_crs,
            "transform": transform, "nodata": NODATA,
            "compress": "deflate", "tiled": True,
            "blockxsize": 256, "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(temp_elev, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform, dst_crs=target_crs,
                dst_nodata=NODATA, resampling=Resampling.bilinear,
            )

    # Calculate derivatives one row at a time to limit memory use.
    with rasterio.open(temp_elev) as src:
        dx_size = abs(src.transform.a)
        dy_size = abs(src.transform.e)
        profile = src.profile.copy()
        width, height = src.width, src.height

        with (
            rasterio.open(temp_slope, "w", **profile) as slope_dst,
            rasterio.open(temp_aspect, "w", **profile) as aspect_dst,
        ):
            for row in range(height):
                start = max(0, row - 1)
                end = min(height, row + 2)
                window = Window(0, start, width, end - start)
                block = src.read(1, window=window, masked=True)
                block = block.filled(np.nan).astype("float32")
                center_idx = row - start
                center = block[center_idx]

                if block.shape[0] < 2:
                    slope_row = np.full(width, NODATA, dtype="float32")
                    aspect_row = np.full(width, NODATA, dtype="float32")
                else:
                    above = block[max(0, center_idx - 1)]
                    below = block[min(block.shape[0] - 1, center_idx + 1)]
                    left = np.roll(center, 1)
                    right = np.roll(center, -1)

                    dx = (right - left) / (2 * dx_size)
                    dy = (above - below) / (2 * dy_size)
                    dx[0] = (right[0] - center[0]) / dx_size
                    dx[-1] = (center[-1] - left[-1]) / dx_size

                    slope_row = np.degrees(np.arctan(np.hypot(dx, dy))).astype("float32")
                    aspect_row = (np.degrees(np.arctan2(-dx, -dy)) + 360) % 360
                    aspect_row[np.hypot(dx, dy) < 1e-6] = -1

                    invalid = (
                        ~np.isfinite(center) | ~np.isfinite(dx) | ~np.isfinite(dy)
                        | ~np.isfinite(above) | ~np.isfinite(below)
                        | ~np.isfinite(left) | ~np.isfinite(right)
                        | (center == NODATA) | (above == NODATA) | (below == NODATA)
                        | (left == NODATA) | (right == NODATA)
                    )
                    slope_row[invalid] = NODATA
                    aspect_row[invalid] = NODATA

                row_window = Window(0, row, width, 1)
                slope_dst.write(slope_row.reshape(1, -1), 1, window=row_window)
                aspect_dst.write(aspect_row.astype("float32").reshape(1, -1), 1, window=row_window)

                if (row + 1) % 1000 == 0:
                    print(f"  {tile}: processed {row + 1}/{height} rows")

    # Validate all outputs before publishing them.
    for p in (temp_elev, temp_slope, temp_aspect):
        if not output_is_valid(p, target_crs):
            raise ValueError(f"Output validation failed: {p}")

    for temp, final in (
        (temp_elev, elevation_path),
        (temp_slope, slope_path),
        (temp_aspect, aspect_path),
    ):
        temp.replace(final)

    print("Processed successfully:", tile)
    print("  CRS:", target_crs)
    print("  Elevation:", elevation_path)
    print("  Slope:", slope_path)
    print("  Aspect:", aspect_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", help="Comma-separated tile names to process")
    parser.add_argument("--limit", type=int, help="Limit number of missing tiles")
    args = parser.parse_args()

    plan = pd.read_csv(PLAN)
    status_rows = []

    if args.tiles:
        requested = [x.strip().upper() for x in args.tiles.split(",") if x.strip()]
        plan = plan[plan["tile"].isin(requested)]
        missing = set(requested) - set(plan["tile"])
        if missing:
            raise ValueError(f"Tiles not found in plan: {sorted(missing)}")
    else:
        status_path = OUT_ROOT / "dem_download_status.csv"
        status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        if not status.empty and "processed_elevation_found" in status:
            completed = set(status.loc[
                status["processed_elevation_found"].astype(str).str.lower().eq("true"),
                "tile"
            ])
            plan = plan[~plan["tile"].isin(completed)]
        plan = plan.sort_values("event_count", ascending=False)
        if args.limit:
            plan = plan.head(args.limit)

    if plan.empty:
        print("No tiles selected.")
        return

    for _, row in plan.iterrows():
        tile = str(row["tile"])
        region = str(row["region"])
        raw_dir = RAW_ROOT / region_folder(region) / tile
        out_dir = OUT_ROOT / region_folder(region) / tile
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "tile": tile, "region": region,
            "event_count": int(row["event_count"]),
            "started_at": started, "status": "failed", "error": "",
        }
        try:
            print(f"\n{'=' * 60}\nProcessing {tile} | {region}")
            hgt = download_tile(tile, raw_dir)
            process_tile(tile, hgt, out_dir)
            record["status"] = "success"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print("FAILED:", tile, record["error"])
        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status_rows.append(record)

        old = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
        combined = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
        combined.to_csv(LOG_PATH, index=False)

    successes = sum(r["status"] == "success" for r in status_rows)
    print(f"\nBatch complete: {successes}/{len(status_rows)} succeeded.")
    print("Log:", LOG_PATH)
    print("Refresh the coverage audit with: python gis/check_dem_download_status.py")


if __name__ == "__main__":
    main()
