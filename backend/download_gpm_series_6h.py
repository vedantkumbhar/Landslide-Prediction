
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_series_6h"
)

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

COLLECTION_ID = "C2723758340-GES_DISC"

START_TIME = "2026-11-01T05:30:00Z"
END_TIME = "2026-11-01T11:30:00Z"

EXPECTED_GRANULES = 12

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def discover_granules():

    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": f"{START_TIME},{END_TIME}",
        "page_size": 100,
        "sort_key": "start_date"
    }

    print("\nSearching NASA CMR...")

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

            start = entry.get("time_start")
            end = entry.get("time_end")

            if start and end and START_TIME <= start < END_TIME:

                granules.append({
                    "title": entry.get("title"),
                    "start": start,
                    "end": end,
                    "url": data_url
                })

    granules.sort(key=lambda item: item["start"])

    print("Granules discovered:", len(granules))

    for granule in granules:
        print(
            granule["start"],
            "->",
            granule["end"],
            granule["title"]
        )

    return granules


def validate_granules(granules):

    if len(granules) != EXPECTED_GRANULES:
        raise RuntimeError(
            f"Expected {EXPECTED_GRANULES} granules, "
            f"but found {len(granules)}. "
            "No downloads started."
        )

    for i in range(1, len(granules)):

        previous_end = requests.utils.parse_header_links(
            ""
        )

        from datetime import datetime, timedelta

        previous_start = datetime.fromisoformat(
            granules[i - 1]["start"].replace("Z", "+00:00")
        )

        current_start = datetime.fromisoformat(
            granules[i]["start"].replace("Z", "+00:00")
        )

        if current_start - previous_start != timedelta(minutes=30):
            raise RuntimeError(
                "Granules are not consecutive 30-minute observations."
            )

    print("\nGranule count and continuity validation passed.")


def download_granule(granule, token):

    url = granule["url"]

    filename = Path(urlparse(url).path).name
    output_file = OUTPUT_DIR / filename

    if output_file.exists():

        with open(output_file, "rb") as file:
            signature = file.read(8)

        if signature == HDF5_SIGNATURE:
            print("\nSKIPPED - Already downloaded:", filename)
            return "skipped"

        raise RuntimeError(
            f"Existing file is invalid: {output_file}"
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


def main():

    token = os.getenv("NASA_EARTHDATA_TOKEN")

    if not token:
        raise RuntimeError(
            "NASA_EARTHDATA_TOKEN not found in .env"
        )

    granules = discover_granules()

    validate_granules(granules)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nStarting 6-hour GPM download")
    print("=" * 55)

    downloaded = 0
    skipped = 0

    for granule in granules:

        result = download_granule(
            granule,
            token
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
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
