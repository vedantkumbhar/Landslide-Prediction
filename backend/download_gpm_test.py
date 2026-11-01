
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "data" / "raw" / "satellite" / "gpm_imerge_test"
)

OUTPUT_FILE = OUTPUT_DIR / (
    "3B-HHR-E.MS.MRG.3IMERG."
    "20261101-S103000-E105959.0630.V07C.HDF5"
)

URL = (
    "https://data.gesdisc.earthdata.nasa.gov/data/"
    "GPM_L3/GPM_3IMERGHHE.07/2026/305/"
    "3B-HHR-E.MS.MRG.3IMERG."
    "20261101-S103000-E105959.0630.V07C.HDF5"
)


def main():

    token = os.getenv("NASA_EARTHDATA_TOKEN")

    if not token:
        raise RuntimeError("NASA Earthdata token not found.")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"File already exists. Refusing to overwrite: {OUTPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_suffix(".download")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("Connecting to NASA GPM...")
    print("Downloading first satellite granule...")

    try:
        with requests.get(
            URL,
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
                    "NASA returned HTML instead of satellite data."
                )

            with open(temporary_file, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        if not temporary_file.exists() or temporary_file.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")

        # Verify the HDF5 file signature.
        with open(temporary_file, "rb") as file:
            signature = file.read(8)

        if signature != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError(
                "Downloaded file does not have the expected HDF5 signature."
            )

        temporary_file.replace(OUTPUT_FILE)

        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

        print("\nDOWNLOAD SUCCESSFUL")
        print("Saved:", OUTPUT_FILE)
        print(f"File size: {size_mb:.2f} MB")
        print("HDF5 signature: VALID")

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise


if __name__ == "__main__":
    main()
