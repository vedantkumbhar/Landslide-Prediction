"""
Tier 1 Western Ghats DEM Tile Downloader and Validator.

Downloads and validates the 6 approved Tier 1 Western Ghats SRTM DEM tiles:
- N19E073 (Maharashtra: Malshej, Kasara, Igatpuri)
- N10E076 (Tamil Nadu / Kerala: Valparai, Anamalai)
- N10E077 (Tamil Nadu / Kerala: Munnar, Idukki, Kodaikanal)
- N12E074 (Karnataka: Mangalore, Ullal, Dakshina Kannada)
- N14E074 (Karnataka / Goa: Uttara Kannada, Karwar, Canacona)
- N12E075 (Karnataka: Coorg / Kodagu, Madikeri, Subrahmanya)

Reuses the established free-data pattern from gis/download_western_ghats_dem_batch.py.
Preserves both compressed (.hgt.gz) and decompressed (.hgt) files.
Validates dimensions, uncompressed file size (25,934,402 bytes), SRTMHGT driver,
EPSG:4326, geographic bounds, and raster readability.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import gzip
import re
import shutil
import ssl
import time

import numpy as np
import rasterio
from rasterio.crs import CRS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "dem" / "western_ghats"

TIER1_TILES = [
    "N19E073",
    "N10E076",
    "N10E077",
    "N12E074",
    "N14E074",
    "N12E075",
]

BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
EXPECTED_BYTES = 3601 * 3601 * 2  # 25,934,402 bytes
MAX_ATTEMPTS = 4
DOWNLOAD_TIMEOUT = 180


def tile_parts(tile: str):
    """Parse tile coordinates (e.g. N19E073 -> lat 19, lon 73)."""
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile)
    if not match:
        raise ValueError(f"Invalid SRTM tile name: {tile}")

    ns, lat, ew, lon = match.groups()
    latitude = int(lat) * (1 if ns == "N" else -1)
    longitude = int(lon) * (1 if ew == "E" else -1)
    return latitude, longitude


def validate_archive(archive_path: Path, tile: str) -> bool:
    """Verify that gzip archive decompresses to exactly EXPECTED_BYTES."""
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        return False

    try:
        decompressed_size = 0
        with gzip.open(archive_path, "rb") as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                decompressed_size += len(chunk)
        return decompressed_size == EXPECTED_BYTES
    except (OSError, EOFError):
        return False


def validate_hgt(path: Path, tile: str):
    """Validate uncompressed HGT file integrity, dimensions, driver, CRS, and bounds."""
    if not path.exists():
        raise FileNotFoundError(f"{tile}: file not found at {path}")

    actual_bytes = path.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            f"{tile}: expected {EXPECTED_BYTES:,} bytes; found {actual_bytes:,}"
        )

    lat, lon = tile_parts(tile)
    expected_left = float(lon)
    expected_bottom = float(lat)
    expected_right = float(lon + 1)
    expected_top = float(lat + 1)

    with rasterio.open(path) as ds:
        if ds.driver != "SRTMHGT":
            raise ValueError(f"{tile}: unexpected driver {ds.driver}; expected SRTMHGT")

        if ds.width != 3601 or ds.height != 3601:
            raise ValueError(f"{tile}: unexpected dimensions {ds.width}x{ds.height}; expected 3601x3601")

        if ds.crs != CRS.from_epsg(4326):
            raise ValueError(f"{tile}: unexpected CRS {ds.crs}; expected EPSG:4326")

        b = ds.bounds
        tol = 1e-4
        if (
            abs(b.left - expected_left) > tol
            or abs(b.bottom - expected_bottom) > tol
            or abs(b.right - expected_right) > tol
            or abs(b.top - expected_top) > tol
        ):
            raise ValueError(
                f"{tile}: bounds mismatch. Found ({b.left}, {b.bottom}, {b.right}, {b.top}); "
                f"expected ({expected_left}, {expected_bottom}, {expected_right}, {expected_top})"
            )

        # Read sample window to confirm readability
        sample = ds.read(1, window=((100, 110), (100, 110)), masked=True)
        if sample.size == 0:
            raise ValueError(f"{tile}: raster sample window is empty")

    return path


def download_archive(tile: str, archive_path: Path, partial_path: Path) -> Path:
    """Download gzip archive from AWS Skadi with retries."""
    latitude, _ = tile_parts(tile)
    band = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}"
    url = f"{BASE_URL}/{band}/{tile}.hgt.gz"

    context = ssl.create_default_context()
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS}: GET {url}")
            request = Request(
                url,
                headers={
                    "User-Agent": "LandslidePredictionResearch/1.0",
                    "Accept-Encoding": "identity",
                },
            )

            with urlopen(request, timeout=DOWNLOAD_TIMEOUT, context=context) as response:
                with open(partial_path, "wb") as dst:
                    shutil.copyfileobj(response, dst)

            if not validate_archive(partial_path, tile):
                raise ValueError(
                    f"{tile}: downloaded archive is incomplete or failed size validation"
                )

            partial_path.replace(archive_path)
            print(f"  Downloaded & validated archive: {archive_path.name} ({archive_path.stat().st_size:,} bytes)")
            return archive_path

        except HTTPError as exc:
            last_error = exc
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(
                    f"HTTP {exc.code} client error for {tile}: {exc}"
                ) from exc

        except (URLError, TimeoutError, OSError, ssl.SSLError, EOFError, ValueError) as exc:
            last_error = exc

        finally:
            if partial_path.exists():
                partial_path.unlink()

        if attempt < MAX_ATTEMPTS:
            delay = min(4 * attempt, 15)
            print(f"  Attempt {attempt} failed ({last_error}). Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"Download failed for {tile} after {MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def download_and_validate_tile(tile: str) -> dict:
    """Download, decompress, and strictly validate one DEM tile."""
    raw_dir = RAW_ROOT / tile
    raw_dir.mkdir(parents=True, exist_ok=True)

    hgt_path = raw_dir / f"{tile}.hgt"
    archive_path = raw_dir / f"{tile}.hgt.gz"
    partial_path = raw_dir / f"{tile}.hgt.gz.part"
    temp_dir = raw_dir / ".tmp_validate"
    temp_path = temp_dir / f"{tile}.hgt"

    result = {
        "tile": tile,
        "source": f"{BASE_URL}/{tile[:3]}/{tile}.hgt.gz",
        "archive_path": str(archive_path),
        "hgt_path": str(hgt_path),
        "archive_size": 0,
        "hgt_size": 0,
        "status": "pending",
        "error": None,
    }

    # 1. Download archive if not already valid
    if archive_path.exists() and validate_archive(archive_path, tile):
        print(f"  Reusing existing valid archive: {archive_path.name}")
    else:
        download_archive(tile, archive_path, partial_path)

    result["archive_size"] = archive_path.stat().st_size

    # 2. Decompress to temporary path and validate
    if hgt_path.exists():
        validate_hgt(hgt_path, tile)
        print(f"  Existing HGT file validated: {hgt_path.name}")
        result["hgt_size"] = hgt_path.stat().st_size
        result["status"] = "success"
        return result

    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Decompressing {archive_path.name} -> {temp_path.name}...")
        with gzip.open(archive_path, "rb") as src:
            with open(temp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # Validate uncompressed HGT
        validate_hgt(temp_path, tile)
        print("  Decompressed HGT passed all integrity & raster checks.")

        temp_path.replace(hgt_path)
        result["hgt_size"] = hgt_path.stat().st_size
        result["status"] = "success"

    finally:
        if temp_path.exists():
            temp_path.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    return result


def main():
    print("=" * 70)
    print("STAGE 2: DOWNLOAD & VALIDATE TIER 1 WESTERN GHATS DEM TILES")
    print("=" * 70)
    print(f"Target tiles ({len(TIER1_TILES)}): {', '.join(TIER1_TILES)}")
    print(f"Target directory: {RAW_ROOT}")
    print(f"Data source: AWS Skadi Public SRTM ({BASE_URL})")
    print(f"Expected uncompressed size per tile: {EXPECTED_BYTES:,} bytes (3601 x 3601)")
    print("=" * 70)

    results = []
    for idx, tile in enumerate(TIER1_TILES, 1):
        print(f"\n[{idx}/{len(TIER1_TILES)}] Processing tile: {tile}")
        try:
            res = download_and_validate_tile(tile)
            results.append(res)
            print(f"  Result: SUCCESS ({tile})")
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  Result: FAILED ({tile}) -> {err_msg}")
            results.append({
                "tile": tile,
                "source": f"{BASE_URL}/{tile[:3]}/{tile}.hgt.gz",
                "archive_path": str(RAW_ROOT / tile / f"{tile}.hgt.gz"),
                "hgt_path": str(RAW_ROOT / tile / f"{tile}.hgt"),
                "archive_size": 0,
                "hgt_size": 0,
                "status": "failed",
                "error": err_msg,
            })
            # Per requirement: Stop if any download fails without fabricating success
            print(f"\nExecution halted due to failure on tile {tile}.")
            break

    print("\n" + "=" * 70)
    print("STAGE 2 EXECUTION SUMMARY")
    print("=" * 70)
    for r in results:
        status_sym = "[OK]" if r["status"] == "success" else "[FAILED]"
        print(f"  {r['tile']:<8} {status_sym} | Archive: {r['archive_size']:,} B | HGT: {r['hgt_size']:,} B | {r['status']}")
        if r["error"]:
            print(f"     Error: {r['error']}")

    all_success = len(results) == len(TIER1_TILES) and all(r["status"] == "success" for r in results)
    if all_success:
        print("\nALL 6 TIER 1 TILES DOWNLOADED AND VALIDATED SUCCESSFULLY.")
    else:
        print("\nSTAGE 2 INCOMPLETE. Some tiles failed.")


if __name__ == "__main__":
    main()
