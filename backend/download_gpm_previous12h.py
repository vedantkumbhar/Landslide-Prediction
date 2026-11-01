
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

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")
TOKEN = os.getenv("NASA_EARTHDATA_TOKEN")

COLLECTION_ID = "C2723758340-GES_DISC"

START = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
END = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)

EXPECTED_COUNT = 24
INTERVAL_MINUTES = 30

OUTPUT_DIR = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_previous12h"
)

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def expected_times():
    result = []
    current = START

    while current < END:
        result.append(current)
        current += timedelta(minutes=INTERVAL_MINUTES)

    return result


def parse_start(filename):
    match = re.search(
        r"3IMERG\.(\d{8})-S(\d{6})",
        filename
    )

    if not match:
        raise ValueError(f"Invalid granule name: {filename}")

    date_part, time_part = match.groups()

    return datetime.strptime(
        date_part + time_part,
        "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)


def safe_filename(title):
    filename = title.split(":", 1)[-1].strip()
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def valid_hdf5(path):
    try:
        with h5py.File(path, "r") as hdf:
            if "Grid/precipitation" not in hdf:
                return False

            if "Grid/time_bnds" not in hdf:
                return False

            if hdf["Grid/precipitation"].shape != (1, 3600, 1800):
                return False

        return True

    except Exception:
        return False


def discover(session):
    print("\nSearching NASA CMR for missing 12-hour data...")

    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": (
            START.strftime("%Y-%m-%dT%H:%M:%SZ")
            + ","
            + END.strftime("%Y-%m-%dT%H:%M:%SZ")
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
    expected = set(expected_times())
    selected = {}

    for entry in entries:
        title = entry.get("title", "")

        if "3IMERG" not in title:
            continue

        try:
            timestamp = parse_start(title)
        except ValueError:
            continue

        if timestamp not in expected:
            continue

        for link in entry.get("links", []):
            href = link.get("href", "")

            if (
                href.endswith(".HDF5")
                and not link.get("inherited", False)
                and not link.get("rel", "").endswith("/metadata")
            ):
                selected[timestamp] = {
                    "title": title,
                    "url": href
                }
                break

    missing = sorted(expected - set(selected))

    if missing:
        print("\nGranules not yet available:")
        for timestamp in missing:
            print(timestamp.isoformat())

        raise RuntimeError(
            f"Only {len(selected)}/24 granules found. "
            "No downloads started."
        )

    ordered = [
        selected[timestamp]
        for timestamp in sorted(selected)
    ]

    print(f"Discovered: {len(ordered)}/24")
    print("First:", sorted(selected)[0])
    print("Last: ", sorted(selected)[-1])

    return ordered


def download(session, granule):
    filename = safe_filename(granule["title"])
    destination = OUTPUT_DIR / filename
    temporary = OUTPUT_DIR / (filename + ".part")

    if destination.exists():
        if valid_hdf5(destination):
            print(f"SKIPPED: {filename}")
            return

        raise RuntimeError(
            f"Invalid existing file. Refusing to overwrite: {destination}"
        )

    if temporary.exists():
        temporary.unlink()

    print(f"\nDownloading: {filename}")

    try:
        with session.get(
            granule["url"],
            stream=True,
            timeout=(30, 300)
        ) as response:
            response.raise_for_status()

            with open(temporary, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        if not valid_hdf5(temporary):
            raise RuntimeError("HDF5 validation failed.")

        os.replace(temporary, destination)

        size = destination.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: {size:.2f} MB")

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def final_validation():
    expected = set(expected_times())
    actual = {}

    for path in OUTPUT_DIR.glob("*.HDF5"):
        timestamp = parse_start(path.name)
        actual[timestamp] = path

    if set(actual) != expected:
        raise RuntimeError(
            f"Final timestamp mismatch: {len(actual)}/24"
        )

    for timestamp in sorted(expected):
        if not valid_hdf5(actual[timestamp]):
            raise RuntimeError(
                f"Invalid HDF5: {actual[timestamp].name}"
            )

    print("\nAll 24 HDF5 files validated.")
    print("Timestamp continuity: PASSED")
    print("Final validation: PASSED")


def main():
    if not TOKEN:
        raise RuntimeError(
            "NASA_EARTHDATA_TOKEN not found in .env"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {TOKEN}",
        "User-Agent": "Landslide-Prediction-Project/1.0"
    })

    try:
        granules = discover(session)

        print("\nStarting missing 12-hour download...")

        for index, granule in enumerate(granules, 1):
            print(f"\n[{index}/24]")
            download(session, granule)
            time.sleep(0.5)

        final_validation()

        print("\nDOWNLOAD COMPLETED")
        print("Output:", OUTPUT_DIR)

    finally:
        session.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)
