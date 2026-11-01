
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import argparse
import gzip
import re
import shutil
import ssl
import time

import pandas as pd
import rasterio
from rasterio.crs import CRS


ROOT = Path(__file__).resolve().parents[1]

PLAN_PATH = (
    ROOT / "data" / "processed" / "dem"
    / "western_ghats_esa_tile_plan_v2.csv"
)

RAW_ROOT = ROOT / "data" / "raw" / "dem" / "western_ghats"
OUT_ROOT = ROOT / "data" / "processed" / "dem" / "western_ghats"
LOG_PATH = OUT_ROOT / "western_ghats_download_log.csv"

BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
EXPECTED_BYTES = 3601 * 3601 * 2

# This pilot script intentionally permits only these three tiles.
ALLOWED_PILOT_TILES = {
    "N09E076",
    "N19E072",
    "N11E076",
}

MAX_ATTEMPTS = 4
DOWNLOAD_TIMEOUT = 180


def tile_parts(tile):
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile)

    if not match:
        raise ValueError(f"Invalid SRTM tile name: {tile}")

    ns, lat, ew, lon = match.groups()

    latitude = int(lat) * (1 if ns == "N" else -1)
    longitude = int(lon) * (1 if ew == "E" else -1)

    return latitude, longitude


def validate_hgt(path, tile):
    """Validate file size and the SRTM raster structure."""
    if not path.exists():
        raise FileNotFoundError(path)

    actual_bytes = path.stat().st_size

    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            f"{tile}: expected {EXPECTED_BYTES} bytes; "
            f"found {actual_bytes}"
        )

    # Rasterio recognizes SRTMHGT from the .hgt extension.
    with rasterio.open(path) as ds:
        if ds.driver != "SRTMHGT":
            raise ValueError(
                f"{tile}: unexpected raster driver {ds.driver}"
            )

        if ds.width != 3601 or ds.height != 3601:
            raise ValueError(
                f"{tile}: unexpected dimensions "
                f"{ds.width} x {ds.height}"
            )

        if ds.crs != CRS.from_epsg(4326):
            raise ValueError(
                f"{tile}: unexpected CRS {ds.crs}"
            )

        # Read a small window to confirm the raster is readable.
        sample = ds.read(
            1,
            window=((100, 110), (100, 110)),
            masked=True,
        )

        if sample.size == 0:
            raise ValueError(f"{tile}: raster sample is empty")

    return path


def validate_archive(archive_path, tile):
    """Check that the gzip archive decompresses to a valid HGT-sized grid."""
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


def download_archive(tile, archive_path, partial_path):
    """Download a gzip archive with retries and safe temporary files."""
    latitude, _ = tile_parts(tile)

    band = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}"
    url = f"{BASE_URL}/{band}/{tile}.hgt.gz"

    # Use the default certificate-verifying SSL context.
    context = ssl.create_default_context()

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Downloading attempt {attempt}/{MAX_ATTEMPTS}: {url}")

            request = Request(
                url,
                headers={
                    "User-Agent": "LandslidePredictionResearch/1.0",
                    "Accept-Encoding": "identity",
                },
            )

            with urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT,
                context=context,
            ) as response:
                with open(partial_path, "wb") as dst:
                    shutil.copyfileobj(response, dst)

            if not validate_archive(partial_path, tile):
                raise ValueError(
                    f"{tile}: downloaded archive is incomplete or invalid"
                )

            # Publish the archive only after validation.
            partial_path.replace(archive_path)

            print("Validated gzip archive:", archive_path)
            return archive_path

        except HTTPError as exc:
            last_error = exc

            # Missing tiles and other client errors usually won't be fixed
            # by retrying immediately.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(
                    f"Server returned HTTP {exc.code} for {tile}: {exc}"
                ) from exc

        except (
            URLError,
            TimeoutError,
            OSError,
            ssl.SSLError,
            EOFError,
            ValueError,
        ) as exc:
            last_error = exc

        finally:
            if partial_path.exists():
                partial_path.unlink()

        if attempt < MAX_ATTEMPTS:
            delay = min(5 * attempt, 20)
            print(
                f"Attempt failed: {last_error}. "
                f"Retrying in {delay} seconds..."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Download failed for {tile} after "
        f"{MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def download_tile(tile):
    """Download and validate one tile without replacing existing raw files."""
    tile_parts(tile)

    raw_dir = RAW_ROOT / tile
    raw_dir.mkdir(parents=True, exist_ok=True)

    hgt_path = raw_dir / f"{tile}.hgt"
    archive_path = raw_dir / f"{tile}.hgt.gz"
    partial_path = raw_dir / f"{tile}.hgt.gz.part"

    temp_dir = raw_dir / ".tmp_validate"
    temp_path = temp_dir / f"{tile}.hgt"

    # Never overwrite an existing HGT file.
    if hgt_path.exists():
        validate_hgt(hgt_path, tile)
        print("Using existing validated HGT:", hgt_path)
        return hgt_path

    # Reuse a valid archive if the previous run downloaded one.
    if archive_path.exists():
        if validate_archive(archive_path, tile):
            print("Reusing validated archive:", archive_path)
        else:
            raise RuntimeError(
                f"Existing archive failed validation: {archive_path}. "
                "Inspect or move it aside manually before retrying."
            )
    else:
        download_archive(tile, archive_path, partial_path)

    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with gzip.open(archive_path, "rb") as src:
            with open(temp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        validate_hgt(temp_path, tile)

        # Never overwrite an unexpected destination file.
        if hgt_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {hgt_path}"
            )

        temp_path.replace(hgt_path)

    finally:
        if temp_path.exists():
            temp_path.unlink()

        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    print("Validated raw tile:", hgt_path)
    return hgt_path


def main():
    parser = argparse.ArgumentParser(
        description="Download a separate Western Ghats SRTM pilot."
    )

    parser.add_argument(
        "--tiles",
        default="N09E076,N19E072,N11E076",
        help="Comma-separated pilot tile names",
    )

    args = parser.parse_args()

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing Western Ghats plan: {PLAN_PATH}"
        )

    requested = [
        item.strip().upper()
        for item in args.tiles.split(",")
        if item.strip()
    ]

    if not requested:
        raise ValueError("No tiles selected.")

    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate tile names were supplied.")

    disallowed = set(requested) - ALLOWED_PILOT_TILES

    if disallowed:
        raise ValueError(
            "This pilot is restricted to "
            f"{sorted(ALLOWED_PILOT_TILES)}. "
            f"Rejected tiles: {sorted(disallowed)}"
        )

    plan = pd.read_csv(PLAN_PATH)
    planned_tiles = set(plan["tile"].astype(str).str.upper())

    missing = set(requested) - planned_tiles

    if missing:
        raise ValueError(
            "Tiles missing from the Western Ghats V2 plan: "
            f"{sorted(missing)}"
        )

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Save results after each tile so an interrupted batch retains its log.
    records = []

    old_log = (
        pd.read_csv(LOG_PATH)
        if LOG_PATH.exists()
        else pd.DataFrame()
    )

    for tile in requested:
        event_row = plan.loc[
            plan["tile"].astype(str).str.upper() == tile
        ].iloc[0]

        record = {
            "tile": tile,
            "historical_event_count": int(
                event_row["historical_event_count"]
            ),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "failed",
            "raw_hgt_path": "",
            "error": "",
        }

        try:
            print(f"\n{'=' * 60}")
            print(f"Western Ghats pilot: {tile}")

            hgt_path = download_tile(tile)

            record["status"] = "success"
            record["raw_hgt_path"] = str(hgt_path)

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print("FAILED:", tile, record["error"])

        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        records.append(record)

        updated_log = pd.concat(
            [old_log, pd.DataFrame([record])],
            ignore_index=True,
        )

        updated_log.to_csv(LOG_PATH, index=False)
        old_log = updated_log

    succeeded = sum(
        record["status"] == "success"
        for record in records
    )

    print(
        f"\nDownload pilot complete: "
        f"{succeeded}/{len(records)} succeeded."
    )
    print("Raw files:", RAW_ROOT)
    print("Log:", LOG_PATH)
    print("Terrain rasters have not been generated.")
    print("Original DEM plans and processed rasters were not modified.")


if __name__ == "__main__":
    main()
