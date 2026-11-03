import argparse
import importlib.util
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.gpm_refresh.refresh_config import (
    create_run_directories,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXISTING_DOWNLOADER = (
    PROJECT_ROOT / "backend" / "download_gpm_series_3h.py"
)

UTC = timezone.utc
GRANULE_INTERVAL_MINUTES = 30


# ============================================================
# TIME UTILITIES
# ============================================================

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

    timestamp = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    return timestamp.astimezone(UTC)


def format_utc(timestamp):

    return timestamp.astimezone(UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


# ============================================================
# LOAD EXISTING DOWNLOADER
# ============================================================

def load_downloader():

    spec = importlib.util.spec_from_file_location(
        "gpm_downloader_3h",
        EXISTING_DOWNLOADER
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load GPM downloader."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# GRANULE VALIDATION
# ============================================================

def validate_granules(granules, start, end):

    duration_seconds = (
        end - start
    ).total_seconds()

    if duration_seconds <= 0:
        raise ValueError(
            "End time must be later than start time."
        )

    interval_seconds = (
        GRANULE_INTERVAL_MINUTES * 60
    )

    if duration_seconds % interval_seconds != 0:
        raise ValueError(
            "Requested duration must be divisible "
            "by 30 minutes."
        )

    expected_count = int(
        duration_seconds / interval_seconds
    )

    expected_times = [
        start + timedelta(
            minutes=GRANULE_INTERVAL_MINUTES * i
        )
        for i in range(expected_count)
    ]

    valid_granules = []

    for granule in granules:

        granule_start = parse_granule_time(
            granule["start"]
        )

        if start <= granule_start < end:
            valid_granules.append(granule)

    valid_granules.sort(
        key=lambda item: parse_granule_time(
            item["start"]
        )
    )

    actual_times = [
        parse_granule_time(item["start"])
        for item in valid_granules
    ]

    missing = sorted(
        set(expected_times) - set(actual_times)
    )

    unexpected = sorted(
        set(actual_times) - set(expected_times)
    )

    duplicates = len(actual_times) != len(
        set(actual_times)
    )

    print("\n========== TEMPORAL VALIDATION ==========")
    print("Requested interval:", start, "to", end)
    print("Expected granules:", expected_count)
    print("Actual granules:", len(actual_times))

    if missing:
        print("Missing granules:", missing)

    if unexpected:
        print("Unexpected granules:", unexpected)

    if duplicates:
        print("Duplicate granule timestamps detected.")

    if (
        len(actual_times) != expected_count
        or missing
        or unexpected
        or duplicates
    ):
        raise RuntimeError(
            "Temporal validation failed. "
            "No downloads started."
        )

    print("\nVALIDATION PASSED")
    print(
        f"Exactly {expected_count} expected granules confirmed."
    )

    return valid_granules


# ============================================================
# MAIN DOWNLOAD WORKFLOW
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validated NASA GPM IMERG rainfall downloader "
            "for configurable time windows."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        type=parse_utc
    )

    parser.add_argument(
        "--end",
        required=True,
        type=parse_utc
    )

    parser.add_argument(
        "--run-id",
        default=None
    )

    args = parser.parse_args()

    start = args.start
    end = args.end

    duration_seconds = (
        end - start
    ).total_seconds()

    if duration_seconds <= 0:
        parser.error(
            "End time must be later than start time."
        )

    interval_seconds = (
        GRANULE_INTERVAL_MINUTES * 60
    )

    if duration_seconds % interval_seconds != 0:
        parser.error(
            "Duration must be divisible by 30 minutes."
        )

    if (
        start.minute not in (0, 30)
        or end.minute not in (0, 30)
        or start.second != 0
        or end.second != 0
    ):
        parser.error(
            "Start and end times must align "
            "with 30-minute intervals."
        )

    expected_granules = int(
        duration_seconds / interval_seconds
    )

    hours = duration_seconds / 3600

    if hours.is_integer():
        window_name = f"{int(hours)}h"
    else:
        window_name = f"{hours:g}h"

    print("\nNASA GPM RAINFALL REFRESH")
    print("=" * 55)
    print("Window:", window_name)
    print("Start:", format_utc(start))
    print("End:", format_utc(end))
    print("Expected granules:", expected_granules)

    # --------------------------------------------------------
    # LOAD DOWNLOADER
    # --------------------------------------------------------

    downloader = load_downloader()

    # --------------------------------------------------------
    # DISCOVER GRANULES
    # --------------------------------------------------------

    discovery_start = format_utc(
        start - timedelta(seconds=1)
    )

    discovery_end = format_utc(
        end + timedelta(seconds=1)
    )

    print("\nDiscovering GPM granules...")

    discovered = downloader.discover_granules(
        discovery_start,
        discovery_end
    )

    # --------------------------------------------------------
    # VALIDATE BEFORE DOWNLOADING
    # --------------------------------------------------------

    valid_granules = validate_granules(
        discovered,
        start,
        end
    )

    # --------------------------------------------------------
    # CREATE RUN DIRECTORIES
    # --------------------------------------------------------

    directories = create_run_directories(
        args.run_id
    )

    output_dir = (
        directories["downloads"] / window_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nOutput directory:", output_dir)

    # --------------------------------------------------------
    # CHECK EARTHDATA TOKEN
    # --------------------------------------------------------

    token = os.getenv(
        "NASA_EARTHDATA_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "NASA_EARTHDATA_TOKEN not found in .env"
        )

    # --------------------------------------------------------
    # DOWNLOAD VALIDATED GRANULES
    # --------------------------------------------------------

    downloaded = 0
    skipped = 0

    for index, granule in enumerate(
        valid_granules, start=1
    ):

        print(
            f"\nProcessing granule {index}/{expected_granules}"
        )

        result = downloader.download_granule(
            granule,
            token,
            output_dir
        )

        if result == "downloaded":
            downloaded += 1

        elif result == "skipped":
            skipped += 1

        else:
            raise RuntimeError(
                f"Unexpected download result: {result}"
            )

    # --------------------------------------------------------
    # DOWNLOAD SUMMARY
    # --------------------------------------------------------

    print("\n========== DOWNLOAD SUMMARY ==========")

    print("Window:", window_name)
    print("Validated granules:", len(valid_granules))
    print("New downloads:", downloaded)
    print("Already available:", skipped)
    print("Output directory:", output_dir)

    print("\nGPM DOWNLOAD COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()