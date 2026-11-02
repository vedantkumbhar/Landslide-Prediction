
import os
import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_series_3h"
)

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_ID = "C2723758340-GES_DISC"

# Defaults retained for backward compatibility.
START_TIME = "2026-11-01T08:00:00Z"
END_TIME = "2026-11-01T11:00:00Z"
EXPECTED_GRANULES = 6

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def discover_granules(start_time=None, end_time=None):

    start_time = start_time or START_TIME
    end_time = end_time or END_TIME

    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": f"{start_time},{end_time}",
        "page_size": 100,
        "sort_key": "start_date"
    }

    print("\nSearching NASA CMR...")
    print("Start:", start_time)
    print("End:", end_time)

    response = requests.get(
        CMR_URL,
        params=params,
        timeout=60
    )
    response.raise_for_status()

    entries = response.json().get(
        "feed", {}
    ).get("entry", [])

    granules = []

    for entry in entries:

        data_url = None

        for link in entry.get("links", []):

            if link.get("rel", "").endswith("/data#"):
                data_url = link.get("href")
                break

        if data_url and data_url.lower().endswith(".hdf5"):

            granules.append({
                "title": entry.get("title"),
                "start": entry.get("time_start"),
                "end": entry.get("time_end"),
                "url": data_url
            })

    granules.sort(key=lambda item: item["start"])

    granules = [
        granule for granule in granules
        if start_time <= granule["start"] < end_time
    ]

    print("Granules discovered:", len(granules))

    for granule in granules:
        print(
            granule["start"],
            "->",
            granule["end"],
            granule["title"]
        )

    return granules


def download_granule(granule, token, output_dir=None):

    output_dir = Path(output_dir or OUTPUT_DIR)

    url = granule["url"]

    filename = Path(
        urlparse(url).path
    ).name

    output_file = output_dir / filename

    if output_file.exists():

        with open(output_file, "rb") as file:
            signature = file.read(8)

        if signature == HDF5_SIGNATURE:
            print("\nSKIPPED - Already downloaded:", filename)
            return "skipped"

        raise RuntimeError(
            f"Existing file is invalid. Manual inspection required: {output_file}"
        )

    temporary_file = output_file.with_name(
        output_file.name + ".download"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("\nDownloading:", filename)

    try:

        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, 180)
        ) as response:

            print("HTTP STATUS:", response.status_code)
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            )

            if "text/html" in content_type.lower():
                raise RuntimeError(
                    "NASA returned HTML instead of HDF5 data."
                )

            with open(temporary_file, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

        if not temporary_file.exists():
            raise RuntimeError("Temporary file not created.")

        if temporary_file.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")

        with open(temporary_file, "rb") as file:
            signature = file.read(8)

        if signature != HDF5_SIGNATURE:
            raise RuntimeError("Invalid HDF5 signature.")

        temporary_file.replace(output_file)

        size_mb = output_file.stat().st_size / (1024 * 1024)

        print("SUCCESS:", filename)
        print(f"Size: {size_mb:.2f} MB")

        return "downloaded"

    except Exception:

        if temporary_file.exists():
            temporary_file.unlink()

        raise


def main(
    start_time=None,
    end_time=None,
    output_dir=None,
    expected_granules=None
):

    # Use supplied values or retain the existing defaults.
    start_time = start_time or START_TIME
    end_time = end_time or END_TIME
    output_dir = Path(output_dir or OUTPUT_DIR)
    expected_granules = (
        EXPECTED_GRANULES
        if expected_granules is None
        else expected_granules
    )

    token = os.getenv("NASA_EARTHDATA_TOKEN")

    if not token:
        raise RuntimeError(
            "NASA_EARTHDATA_TOKEN not found in .env"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    granules = discover_granules(
        start_time,
        end_time
    )

    if len(granules) != expected_granules:
        raise RuntimeError(
            f"Expected {expected_granules} granules, "
            f"but discovered {len(granules)}. "
            "No downloads started. Check the CMR results."
        )

    print("\nGPM DOWNLOAD")
    print("=" * 55)
    print("Start:", start_time)
    print("End:", end_time)
    print("Expected granules:", expected_granules)

    downloaded = 0
    skipped = 0

    for granule in granules:

        result = download_granule(
            granule,
            token,
            output_dir
        )

        if result == "downloaded":
            downloaded += 1
        else:
            skipped += 1

    print("\nDOWNLOAD SUMMARY")
    print("=" * 55)
    print("Granules discovered:", len(granules))
    print("New downloads:", downloaded)
    print("Already available:", skipped)
    print("Output directory:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download NASA GPM IMERG granules."
    )

    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-granules", type=int, default=None)

    args = parser.parse_args()

    main(
        start_time=args.start,
        end_time=args.end,
        output_dir=args.output_dir,
        expected_granules=args.expected_granules
    )
