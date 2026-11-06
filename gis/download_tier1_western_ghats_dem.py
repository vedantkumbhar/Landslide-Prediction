"""
Tier 1 Western Ghats DEM Tile Downloader and Validator.

Downloads and validates the 6 approved Tier 1 Western Ghats SRTM DEM tiles:
- N19E073 (Maharashtra: Malshej, Kasara, Igatpuri)
- N10E076 (Tamil Nadu / Kerala: Valparai, Anamalai)
- N10E077 (Tamil Nadu / Kerala: Munnar, Idukki, Kodaikanal)
- N12E074 (Karnataka: Mangalore, Ullal, Dakshina Kannada)
- N14E074 (Karnataka / Goa: Uttara Kannada, Karwar, Canacona)
- N12E075 (Karnataka: Coorg / Kodagu, Madikeri, Subrahmanya)

Sources: Verified free public SRTMGL1 V003 via NASA Earthdata LP DAAC / AWS Skadi.
Preserves both compressed (.hgt.gz) and decompressed (.hgt) files in:
data/raw/dem/western_ghats/{TILE}/

Strictly validates:
- Dimensions: 3601 x 3601
- File size: 25,934,402 bytes
- Driver: SRTMHGT
- CRS: EPSG:4326
- Geographic bounds matching 1-degree tile coordinates
- Raster window readability
"""

from pathlib import Path
import gzip
import io
import math
import os
import re
import shutil
import time
import zipfile

import numpy as np
import rasterio
from rasterio.crs import CRS
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "dem" / "western_ghats"
ENV_FILE = PROJECT_ROOT / ".env"

TIER1_TILES = [
    "N19E073",
    "N10E076",
    "N10E077",
    "N12E074",
    "N14E074",
    "N12E075",
]

NASA_BASE_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1.003"
EXPECTED_BYTES = 3601 * 3601 * 2  # 25,934,402 bytes


def get_earthdata_token() -> str:
    """Read NASA Earthdata Bearer Token from .env if present."""
    if not ENV_FILE.exists():
        return ""
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("NASA_EARTHDATA_TOKEN="):
                return line.strip().split("=", 1)[1].strip()
    return ""


def tile_parts(tile: str):
    """Parse tile coordinates (e.g. N19E073 -> lat 19, lon 73)."""
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile)
    if not match:
        raise ValueError(f"Invalid SRTM tile name: {tile}")

    ns, lat, ew, lon = match.groups()
    latitude = int(lat) * (1 if ns == "N" else -1)
    longitude = int(lon) * (1 if ew == "E" else -1)
    return latitude, longitude


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
    half_pixel = 0.5 / 3600.0  # SRTM 3601 grid cell registration
    expected_left = float(lon) - half_pixel
    expected_bottom = float(lat) - half_pixel
    expected_right = float(lon + 1) + half_pixel
    expected_top = float(lat + 1) + half_pixel

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
                f"{tile}: bounds mismatch. Found ({b.left:.6f}, {b.bottom:.6f}, {b.right:.6f}, {b.top:.6f}); "
                f"expected ({expected_left:.6f}, {expected_bottom:.6f}, {expected_right:.6f}, {expected_top:.6f})"
            )

        # Read sample window to confirm raster readability
        sample = ds.read(1, window=((100, 110), (100, 110)), masked=True)
        if sample.size == 0:
            raise ValueError(f"{tile}: raster sample window is empty")

    return path


def validate_gz_archive(gz_path: Path, expected_bytes: bytes, tile: str):
    """Decompress .hgt.gz in memory, verify size and byte-for-byte identity against uncompressed bytes."""
    if not gz_path.exists():
        raise FileNotFoundError(f"{tile}: compressed file not found at {gz_path}")
    if gz_path.stat().st_size == 0:
        raise ValueError(f"{tile}: compressed file is empty: {gz_path}")

    try:
        with gzip.open(gz_path, "rb") as f_gz:
            decompressed = f_gz.read()
    except Exception as exc:
        raise ValueError(f"{tile}: failed to decompress {gz_path.name}: {exc}") from exc

    if len(decompressed) != EXPECTED_BYTES:
        raise ValueError(
            f"{tile}: decompressed {gz_path.name} size is {len(decompressed):,} bytes; expected {EXPECTED_BYTES:,}"
        )

    if decompressed != expected_bytes:
        raise ValueError(
            f"{tile}: decompressed {gz_path.name} contents do not match {tile}.hgt byte-for-byte"
        )

    return True


class PublicationCleanupError(RuntimeError):
    """Raised when publication write fails and cleanup of the partial destination file also fails."""

    def __init__(self, tile: str, file_path: Path, write_error: Exception, cleanup_error: Exception):
        self.tile = tile
        self.file_path = file_path
        self.write_error = write_error
        self.cleanup_error = cleanup_error
        super().__init__(
            f"{tile}: write failed on {file_path.name} ({write_error}) "
            f"and cleanup failed ({cleanup_error}). Partial file remains at {file_path}."
        )


def publish_exclusive(temp_file: Path, final_file: Path, tile: str):
    """Publish temp_file to final_file using exclusive creation; never overwrites."""
    if final_file.exists():
        raise FileExistsError(
            f"{tile}: refusal to overwrite existing destination file: {final_file}"
        )

    # Use exclusive creation mode 'xb' (fails atomically at OS level if final_file exists)
    created_by_us = False
    try:
        with open(temp_file, "rb") as f_src, open(final_file, "xb") as f_dst:
            created_by_us = True
            shutil.copyfileobj(f_src, f_dst)
    except FileExistsError as exc:
        raise FileExistsError(
            f"{tile}: destination file {final_file.name} already exists. Overwrite prohibited."
        ) from exc
    except Exception as exc:
        if created_by_us and final_file.exists():
            try:
                final_file.unlink()
            except Exception as cleanup_err:
                raise PublicationCleanupError(
                    tile=tile,
                    file_path=final_file,
                    write_error=exc,
                    cleanup_error=cleanup_err,
                ) from exc
        raise IOError(
            f"{tile}: failed during exclusive write to {final_file.name}: {exc}"
        ) from exc


def download_tile_nasa(tile: str, target_dir: Path, token: str) -> Path:
    """Download and extract SRTMGL1 tile from NASA Earthdata LP DAAC into temporary files."""
    final_hgt = target_dir / f"{tile}.hgt"
    final_gz = target_dir / f"{tile}.hgt.gz"

    if final_hgt.exists() or final_gz.exists():
        raise FileExistsError(
            f"{tile}: cannot initiate download because target file already exists "
            f"(hgt exists={final_hgt.exists()}, gz exists={final_gz.exists()}). Overwriting is prohibited."
        )

    url = f"{NASA_BASE_URL}/{tile}.SRTMGL1.hgt/{tile}.SRTMGL1.hgt.zip"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    temp_dir = target_dir / ".tmp_validate"
    temp_zip = temp_dir / f"{tile}.zip.part"
    temp_hgt = temp_dir / f"{tile}.hgt"
    temp_gz = temp_dir / f"{tile}.hgt.gz"

    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Fetching from NASA Earthdata: {url}")
    t0 = time.time()
    try:
        with requests.get(url, headers=headers, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            with open(temp_zip, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

        print(f"  Archive downloaded ({temp_zip.stat().st_size:,} bytes in {time.time()-t0:.1f}s). Extracting HGT...")

        # Extract HGT from zip
        with zipfile.ZipFile(temp_zip) as zf:
            hgt_names = [n for n in zf.namelist() if n.lower().endswith(".hgt")]
            if not hgt_names:
                raise ValueError(f"{tile}: no .hgt file found in downloaded zip archive.")
            with zf.open(hgt_names[0]) as src, open(temp_hgt, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # Validate extracted temporary HGT
        validate_hgt(temp_hgt, tile)
        hgt_bytes = temp_hgt.read_bytes()
        print(f"  Validated temporary HGT: {temp_hgt.name} ({len(hgt_bytes):,} bytes)")

        # Create temporary compressed .hgt.gz
        print(f"  Compressing to temporary {temp_gz.name}...")
        with open(temp_hgt, "rb") as f_in, gzip.open(temp_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Validate temporary GZ byte-for-byte against temporary HGT
        validate_gz_archive(temp_gz, hgt_bytes, tile)
        print(f"  Validated temporary GZ: byte-for-byte identity confirmed.")

        # Re-verify destination paths do not exist before publishing
        if final_hgt.exists() or final_gz.exists():
            raise FileExistsError(
                f"{tile}: destination file appeared during processing. Overwriting is strictly prohibited."
            )

        # Publish both files exclusively with verified atomic rollback
        published_hgt = False
        published_gz = False
        try:
            publish_exclusive(temp_hgt, final_hgt, tile)
            published_hgt = True

            publish_exclusive(temp_gz, final_gz, tile)
            published_gz = True

            print(f"  Safely published {final_hgt.name} and {final_gz.name} via exclusive creation.")
        except Exception as pub_exc:
            failed_rollbacks = []
            successful_rollbacks = []

            # If write failed and cleanup inside publish_exclusive also failed, track the partial file
            if isinstance(pub_exc, PublicationCleanupError):
                failed_rollbacks.append((str(pub_exc.file_path), str(pub_exc.cleanup_error)))

            # Only rollback files created by this publication attempt; never touch pre-existing files
            if published_hgt and final_hgt.exists():
                try:
                    final_hgt.unlink()
                    successful_rollbacks.append(str(final_hgt))
                    print(f"  Rolled back newly published {final_hgt.name}.")
                except Exception as rb_err:
                    failed_rollbacks.append((str(final_hgt), str(rb_err)))
                    print(f"  CRITICAL: failed to rollback {final_hgt.name}: {rb_err}")

            if published_gz and final_gz.exists():
                try:
                    final_gz.unlink()
                    successful_rollbacks.append(str(final_gz))
                    print(f"  Rolled back newly published {final_gz.name}.")
                except Exception as rb_err:
                    failed_rollbacks.append((str(final_gz), str(rb_err)))
                    print(f"  CRITICAL: failed to rollback {final_gz.name}: {rb_err}")

            if failed_rollbacks:
                err_details = "; ".join(f"{f} ({err})" for f, err in failed_rollbacks)
                raise RuntimeError(
                    f"{tile}: publication failed ({pub_exc}). Rollback INCOMPLETE. "
                    f"Partial files remain on disk: {err_details}. Manual cleanup required."
                ) from pub_exc
            elif successful_rollbacks:
                rb_list = ", ".join(successful_rollbacks)
                raise RuntimeError(
                    f"{tile}: publication failed ({pub_exc}). Successful rollback performed; "
                    f"removed newly published partial files: {rb_list}."
                ) from pub_exc
            else:
                raise RuntimeError(
                    f"{tile}: publication failed ({pub_exc}). No files were published to destination."
                ) from pub_exc

    finally:
        if temp_zip.exists():
            temp_zip.unlink()
        if temp_hgt.exists():
            temp_hgt.unlink()
        if temp_gz.exists():
            temp_gz.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    return final_hgt


def process_tile(tile: str, token: str) -> dict:
    """Validate existing tile files or download new tile with strict non-destructive checks."""
    raw_dir = RAW_ROOT / tile
    raw_dir.mkdir(parents=True, exist_ok=True)

    hgt_path = raw_dir / f"{tile}.hgt"
    gz_path = raw_dir / f"{tile}.hgt.gz"

    result = {
        "tile": tile,
        "source": "NASA Earthdata LP DAAC (SRTMGL1.003)",
        "hgt_path": str(hgt_path),
        "gz_path": str(gz_path),
        "hgt_size": 0,
        "gz_size": 0,
        "status": "pending",
        "error": None,
    }

    # Strict Safety Check: If either file already exists on disk, validate without downloading.
    if hgt_path.exists() or gz_path.exists():
        if not hgt_path.exists():
            raise FileNotFoundError(
                f"{tile}: .hgt.gz exists but .hgt is missing ({hgt_path}). "
                f"Existing files are incomplete and automatic replacement is strictly prohibited."
            )
        if not gz_path.exists():
            raise FileNotFoundError(
                f"{tile}: .hgt exists but .hgt.gz is missing ({gz_path}). "
                f"Existing files are incomplete and automatic replacement is strictly prohibited."
            )

        print(f"  Validating existing files: {hgt_path.name} & {gz_path.name}...")
        validate_hgt(hgt_path, tile)
        hgt_bytes = hgt_path.read_bytes()
        validate_gz_archive(gz_path, hgt_bytes, tile)

        print(f"  Existing files validated successfully: byte-for-byte match confirmed.")
        result["hgt_size"] = hgt_path.stat().st_size
        result["gz_size"] = gz_path.stat().st_size
        result["status"] = "success"
        return result

    # Neither file exists on disk: proceed with safe download using temporary staging
    download_tile_nasa(tile, raw_dir, token)

    # Final post-publication verification
    validate_hgt(hgt_path, tile)
    hgt_bytes = hgt_path.read_bytes()
    validate_gz_archive(gz_path, hgt_bytes, tile)

    result["hgt_size"] = hgt_path.stat().st_size
    result["gz_size"] = gz_path.stat().st_size
    result["status"] = "success"

    return result


def main():
    print("=" * 70)
    print("STAGE 2: DOWNLOAD & VALIDATE TIER 1 WESTERN GHATS DEM TILES")
    print("=" * 70)
    print(f"Target tiles ({len(TIER1_TILES)}): {', '.join(TIER1_TILES)}")
    print(f"Target folder: {RAW_ROOT}")
    print(f"Primary source: NASA Earthdata LP DAAC (SRTMGL1 V003)")
    print(f"Expected uncompressed size: {EXPECTED_BYTES:,} bytes (3601 x 3601)")
    print("=" * 70)

    token = get_earthdata_token()
    if token:
        print("NASA_EARTHDATA_TOKEN detected in .env (authenticated NASA LP DAAC access enabled).")
    else:
        print("WARNING: NASA_EARTHDATA_TOKEN not found in .env; proceeding without bearer auth.")

    results = []
    for idx, tile in enumerate(TIER1_TILES, 1):
        print(f"\n[{idx}/{len(TIER1_TILES)}] Ingesting tile: {tile}")
        try:
            res = process_tile(tile, token)
            results.append(res)
            print(f"  Tile {tile}: SUCCESS")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"  Tile {tile}: FAILED -> {err}")
            results.append({
                "tile": tile,
                "source": "NASA Earthdata LP DAAC",
                "hgt_path": str(RAW_ROOT / tile / f"{tile}.hgt"),
                "gz_path": str(RAW_ROOT / tile / f"{tile}.hgt.gz"),
                "hgt_size": 0,
                "gz_size": 0,
                "status": "failed",
                "error": err,
            })
            print(f"\nExecution stopped on failure per safety rules.")
            break

    print("\n" + "=" * 70)
    print("STAGE 2 FINAL VALIDATION REPORT")
    print("=" * 70)
    for r in results:
        sym = "[OK]" if r["status"] == "success" else "[FAILED]"
        print(f"  {r['tile']:<8} {sym} | HGT: {r['hgt_size']:,} B | GZ: {r['gz_size']:,} B | {r['status']}")
        if r["error"]:
            print(f"     Error: {r['error']}")

    all_done = (len(results) == len(TIER1_TILES)) and all(r["status"] == "success" for r in results)
    if all_done:
        print("\nALL 6 TIER 1 DEM TILES SUCCESSFULLY DOWNLOADED AND VALIDATED.")
    else:
        print("\nSTAGE 2 INCOMPLETE.")


if __name__ == "__main__":
    main()
