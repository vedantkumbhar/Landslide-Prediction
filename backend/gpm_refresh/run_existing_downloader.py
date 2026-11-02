
import argparse
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.gpm_refresh.refresh_config import create_run_directories


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXISTING_DOWNLOADER = (
    PROJECT_ROOT / "backend" / "download_gpm_series_3h.py"
)

UTC = timezone.utc


def parse_utc(value):
    try:
        return datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=UTC)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Use YYYY-MM-DDTHH:MM:SSZ"
        )


def parse_granule_time(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(UTC)


def load_downloader():

    spec = importlib.util.spec_from_file_location(
        "gpm_downloader_3h",
        EXISTING_DOWNLOADER
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load GPM downloader.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def validate_granules(granules, start, end):

    valid_granules = []

    for granule in granules:

        granule_start = parse_granule_time(
            granule["start"]
        )

        if start <= granule_start < end:
            valid_granules.append(granule)

    valid_granules.sort(
        key=lambda item: parse_granule_time(item["start"])
    )

    expected = [
        start + timedelta(minutes=30 * i)
        for i in range(6)
    ]

    actual = [
        parse_granule_time(item["start"])
        for item in valid_granules
    ]

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))

    print("\n========== TEMPORAL VALIDATION ==========")
    print("Requested interval:", start, "to", end)
    print("Expected granules:", len(expected))
    print("Actual granules:", len(actual))

    if len(actual) != 6 or missing or unexpected:
        raise RuntimeError(
            "Temporal validation failed. "
            f"Missing: {missing}; Unexpected: {unexpected}. "
            "No downloads started."
        )

    print("\nVALIDATION PASSED")
    print("Exactly six expected granules confirmed.")

    return valid_granules


def main():

    parser = argparse.ArgumentParser(
        description="Validated NASA GPM 3-hour refresh."
    )

    parser.add_argument("--start", required=True, type=parse_utc)
    parser.add_argument("--end", required=True, type=parse_utc)
    parser.add_argument("--run-id", default=None)

    args = parser.parse_args()

    if (args.end - args.start).total_seconds() != 10800:
        parser.error("Interval must be exactly three hours.")

    if (
        args.start.minute not in (0, 30)
        or args.end.minute not in (0, 30)
    ):
        parser.error("Times must align with 30-minute intervals.")

    if args.start.second != 0 or args.end.second != 0:
        parser.error("Seconds must be zero.")

    downloader = load_downloader()

    start_text = (
        args.start - timedelta(seconds=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    end_text = (
        args.end + timedelta(seconds=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("\nGPM 3-HOUR REFRESH")
    print("=" * 50)

    # Discover granules using slightly expanded boundaries.
    discovered = downloader.discover_granules(
        start_text,
        end_text
    )

    # Validate exact requested interval before downloading.
    valid_granules = validate_granules(
        discovered,
        args.start,
        args.end
    )

    # Create isolated run directory after validation.
    directories = create_run_directories(args.run_id)

    output_dir = directories["downloads"] / "3h"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nOutput directory:", output_dir)

    # Download only the already-validated granules.
    token = downloader.os.getenv("NASA_EARTHDATA_TOKEN")

    if not token:
        raise RuntimeError(
            "NASA_EARTHDATA_TOKEN not found in .env"
        )

    downloaded = 0
    skipped = 0

    for granule in valid_granules:

        result = downloader.download_granule(
            granule,
            token,
            output_dir
        )

        if result == "downloaded":
            downloaded += 1
        else:
            skipped += 1

    print("\nDOWNLOAD SUMMARY")
    print("=" * 50)
    print("Validated granules:", len(valid_granules))
    print("New downloads:", downloaded)
    print("Already available:", skipped)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
