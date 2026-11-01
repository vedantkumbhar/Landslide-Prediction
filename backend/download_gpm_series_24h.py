
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import requests
from dotenv import load_dotenv


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN")

COLLECTION_ID = "C2723758340-GES_DISC"

START_TIME = datetime(
    2026, 11, 1, 0, 0, tzinfo=timezone.utc
)

END_TIME = datetime(
    2026, 11, 2, 0, 0, tzinfo=timezone.utc
)

INTERVAL_MINUTES = 30
EXPECTED_GRANULES = 48

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "satellite"
    / "gpm_imerge_series_24h"
)

CMR_URL = (
    "https://cmr.earthdata.nasa.gov/search/granules.json"
)


# ==================================================
# TIMESTAMP UTILITIES
# ==================================================

def expected_start_times():

    timestamps = []
    current = START_TIME

    while current < END_TIME:

        timestamps.append(current)

        current += timedelta(
            minutes=INTERVAL_MINUTES
        )

    return timestamps


def parse_granule_start(filename):

    match = re.search(
        r"3B-HHR-E\.MS\.MRG\.3IMERG\."
        r"(\d{8})-S(\d{6})",
        filename
    )

    if not match:
        raise ValueError(
            f"Cannot parse granule timestamp: {filename}"
        )

    date_part, time_part = match.groups()

    return datetime.strptime(
        date_part + time_part,
        "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)


def safe_filename(title):

    # Remove NASA collection prefix.
    filename = title.split(":", 1)[-1].strip()

    # Remove Windows-invalid filename characters.
    filename = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        filename
    )

    return filename


# ==================================================
# HDF5 VALIDATION
# ==================================================

def is_valid_hdf5(filepath):

    try:

        with h5py.File(filepath, "r") as hdf:

            if "Grid/precipitation" not in hdf:
                return False

            if "Grid/time_bnds" not in hdf:
                return False

            if hdf["Grid/precipitation"].shape != (
                1, 3600, 1800
            ):
                return False

        return True

    except Exception:
        return False


# ==================================================
# NASA CMR DISCOVERY
# ==================================================

def discover_granules(session):

    print("\nSearching NASA CMR for 24-hour GPM data...")

    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": (
            START_TIME.strftime("%Y-%m-%dT%H:%M:%SZ")
            + ","
            + END_TIME.strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "page_size": 200,
        "sort_key[]": "start_date",
    }

    response = session.get(
        CMR_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    entries = response.json()["feed"]["entry"]

    expected = set(expected_start_times())

    selected = {}

    for entry in entries:

        title = entry.get("title", "")

        if "3IMERG" not in title:
            continue

        try:
            granule_start = parse_granule_start(title)
        except ValueError:
            continue

        # Keep only the exact requested 48 timestamps.
        if granule_start not in expected:
            continue

        download_url = None

        for link in entry.get("links", []):

            href = link.get("href", "")

            if (
                href.endswith(".HDF5")
                and not link.get("inherited", False)
                and not link.get("rel", "").endswith("/metadata")
            ):
                download_url = href
                break

        if download_url:

            selected[granule_start] = {
                "title": title,
                "url": download_url
            }

    missing = sorted(
        expected - set(selected)
    )

    if missing:

        print("\nMissing granules:")

        for timestamp in missing:
            print(timestamp.isoformat())

        raise RuntimeError(
            f"Only {len(selected)}/{EXPECTED_GRANULES} "
            "required granules discovered. "
            "Download cancelled."
        )

    if len(selected) != EXPECTED_GRANULES:

        raise RuntimeError(
            f"Expected {EXPECTED_GRANULES} granules, "
            f"found {len(selected)}."
        )

    ordered = [
        selected[timestamp]
        for timestamp in sorted(selected)
    ]

    print(
        f"Discovered: {len(ordered)}/{EXPECTED_GRANULES}"
    )

    print(
        "First granule:",
        sorted(selected)[0].isoformat()
    )

    print(
        "Last granule:",
        sorted(selected)[-1].isoformat()
    )

    return ordered


# ==================================================
# DOWNLOAD GRANULE
# ==================================================

def download_granule(session, granule):

    filename = safe_filename(
        granule["title"]
    )

    destination = OUTPUT_DIR / filename

    temporary_file = OUTPUT_DIR / (
        filename + ".part"
    )

    # Skip valid files already downloaded.
    if destination.exists():

        if is_valid_hdf5(destination):

            print(
                f"SKIPPED (already valid): {filename}"
            )

            return True

        raise RuntimeError(
            "Existing invalid file found. "
            "Refusing to overwrite:\n"
            f"{destination}"
        )

    # Remove an incomplete temporary download.
    if temporary_file.exists():
        temporary_file.unlink()

    print(f"\nDownloading: {filename}")

    try:

        with session.get(
            granule["url"],
            stream=True,
            timeout=(30, 300)
        ) as response:

            response.raise_for_status()

            with open(
                temporary_file,
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

        if not is_valid_hdf5(temporary_file):

            raise RuntimeError(
                "Downloaded file failed HDF5 validation."
            )

        os.replace(
            temporary_file,
            destination
        )

        size_mb = (
            destination.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"SUCCESS: {size_mb:.2f} MB"
        )

        return True

    except Exception:

        if temporary_file.exists():
            temporary_file.unlink()

        raise


# ==================================================
# FINAL VALIDATION
# ==================================================

def validate_final_downloads():

    files = list(
        OUTPUT_DIR.glob("*.HDF5")
    )

    expected = set(
        expected_start_times()
    )

    actual = {}

    for filepath in files:

        try:

            timestamp = parse_granule_start(
                filepath.name
            )

        except ValueError:
            continue

        actual[timestamp] = filepath

    if set(actual) != expected:

        missing = sorted(
            expected - set(actual)
        )

        unexpected = sorted(
            set(actual) - expected
        )

        print("\nTimestamp validation failed.")

        print("Missing granules:", len(missing))
        print("Unexpected granules:", len(unexpected))

        raise RuntimeError(
            "Final timestamp validation failed."
        )

    for timestamp in sorted(expected):

        filepath = actual[timestamp]

        if not is_valid_hdf5(filepath):

            raise RuntimeError(
                f"HDF5 validation failed: {filepath.name}"
            )

    print("\nAll 48 granules passed HDF5 validation.")
    print("Timestamp continuity: PASSED")
    print("Final validation: PASSED")


# ==================================================
# MAIN
# ==================================================

def main():

    if not EARTHDATA_TOKEN:

        print(
            "ERROR: NASA_EARTHDATA_TOKEN "
            "not found in .env"
        )

        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    session = requests.Session()

    session.headers.update({
        "Authorization": f"Bearer {EARTHDATA_TOKEN}",
        "User-Agent": "Landslide-Prediction-Project/1.0",
    })

    try:

        granules = discover_granules(session)

        print("\nStarting 24-hour GPM download...")

        successful = 0

        for index, granule in enumerate(
            granules,
            start=1
        ):

            print(
                f"\n[{index}/{EXPECTED_GRANULES}]"
            )

            if download_granule(
                session,
                granule
            ):
                successful += 1

            time.sleep(0.5)

        print("\n" + "=" * 55)
        print("24-HOUR GPM DOWNLOAD SUMMARY")
        print("=" * 55)

        print(
            f"Expected granules : {EXPECTED_GRANULES}"
        )

        print(
            f"Successful/skipped: {successful}"
        )

        print(
            f"Output directory  : {OUTPUT_DIR}"
        )

        validate_final_downloads()

    finally:

        session.close()


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR:", error)
        sys.exit(1)
