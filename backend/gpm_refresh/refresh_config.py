
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests

# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAINFALL_DIR = PROCESSED_DIR / "rainfall"
GPM_RAW_DIR = RAW_DATA_DIR / "rainfall" / "gpm"


# ============================================================
# REFRESH CONFIGURATION
# ============================================================

UTC = timezone.utc

REGIONAL_DATASET = (
    PROCESSED_DIR / "regional_training_dataset_30oct.csv"
)

INITIAL_PREDICTIONS = (
    RAINFALL_DIR / "regional_gpm_risk_predictions_01nov2026.csv"
)

MAX_MATCH_DISTANCE_KM = 8.0

GPM_INTERVAL_MINUTES = 30
GRANULES_PER_HOUR = 2

EXPECTED_GRANULE_SHAPE = (1, 3600, 1800)

GPM_PRODUCT = "NASA GPM IMERG V07C"

WINDOW_DURATIONS = (3, 6, 12, 24)

# NASA CMR configuration
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

COLLECTION_ID = "C2723758340-GES_DISC"

CMR_PAGE_SIZE = 100
CMR_TIMEOUT_SECONDS = 60

# Search backwards for at most seven days.
MAX_ENDPOINT_SEARCH_DAYS = 7

# ============================================================
# RUN MANAGEMENT
# ============================================================

def get_run_id():
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def get_run_directory(run_id=None):

    if run_id is None:
        run_id = get_run_id()

    return RAINFALL_DIR / "refresh_runs" / run_id


def create_run_directories(run_id=None):

    run_dir = get_run_directory(run_id)

    directories = {
        "root": run_dir,
        "downloads": run_dir / "downloads",
        "extracted": run_dir / "extracted",
        "predictions": run_dir / "predictions",
        "maps": run_dir / "maps",
        "logs": run_dir / "logs",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


# ============================================================
# UTC TIME UTILITIES
# ============================================================

def normalize_utc(timestamp):

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)

    return timestamp.astimezone(UTC)


def format_gpm_timestamp(timestamp):

    timestamp = normalize_utc(timestamp)

    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_granule_count(hours):

    if hours <= 0:
        raise ValueError("Hours must be positive.")

    return hours * GRANULES_PER_HOUR


def align_to_3hour_boundary(reference_time):

    reference_time = normalize_utc(reference_time)

    interval_seconds = 3 * 3600

    timestamp = int(reference_time.timestamp())

    end_timestamp = (
        timestamp // interval_seconds
    ) * interval_seconds

    return datetime.fromtimestamp(end_timestamp, UTC)


# ============================================================
# NASA CMR GRANULE DISCOVERY
# ============================================================

def discover_cmr_granules(start_time, end_time):
    """
    Discover NASA GPM IMERG Early V07 granules
    within a requested UTC interval.

    CMR can return granules overlapping query boundaries.
    Therefore, exact start timestamps are filtered locally.
    """

    start_time = normalize_utc(start_time)
    end_time = normalize_utc(end_time)

    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": (
            f"{format_gpm_timestamp(start_time)},"
            f"{format_gpm_timestamp(end_time)}"
        ),
        "page_size": CMR_PAGE_SIZE,
        "sort_key": "start_date",
    }

    response = requests.get(
        CMR_URL,
        params=params,
        timeout=CMR_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    data = response.json()

    entries = data.get("feed", {}).get("entry", [])

    # Keep only granules starting inside the requested interval.
    valid_entries = []

    for entry in entries:

        raw_start = entry.get("time_start")

        if not raw_start:
            continue

        granule_start = datetime.fromisoformat(
            raw_start.replace("Z", "+00:00")
        ).astimezone(UTC)

        if start_time <= granule_start < end_time:
            valid_entries.append(entry)

    return valid_entries


# ============================================================
# COMPLETE 24-HOUR AVAILABILITY VALIDATION
# ============================================================

def validate_complete_24h_window(start_time, end_time):
    """
    Confirm that all 48 expected 30-minute granules
    are available for a complete 24-hour window.

    Missing, duplicate or irregular timestamps cause failure.
    """

    start_time = normalize_utc(start_time)
    end_time = normalize_utc(end_time)

    expected_count = get_granule_count(24)

    expected_timestamps = [
        start_time + timedelta(minutes=GPM_INTERVAL_MINUTES * index)
        for index in range(expected_count)
    ]

    entries = discover_cmr_granules(
        start_time,
        end_time,
    )

    actual_timestamps = []

    for entry in entries:

        raw_start = entry.get("time_start")

        if not raw_start:
            continue

        timestamp = datetime.fromisoformat(
            raw_start.replace("Z", "+00:00")
        ).astimezone(UTC)

        actual_timestamps.append(timestamp)

    actual_timestamps.sort()

    # Reject duplicate timestamps.
    if len(actual_timestamps) != len(set(actual_timestamps)):
        return False, "Duplicate granule timestamps detected."

    # Require exactly 48 granules.
    if len(actual_timestamps) != expected_count:
        return (
            False,
            f"Expected {expected_count} granules, "
            f"found {len(actual_timestamps)}."
        )

    # Require exact expected timestamps.
    if actual_timestamps != expected_timestamps:

        missing = sorted(
            set(expected_timestamps) - set(actual_timestamps)
        )

        unexpected = sorted(
            set(actual_timestamps) - set(expected_timestamps)
        )

        return (
            False,
            f"Timestamp mismatch. "
            f"Missing: {len(missing)}, "
            f"Unexpected: {len(unexpected)}."
        )

    return True, "All 48 granules are available."


# ============================================================
# LATEST AVAILABLE COMMON UTC ENDPOINT
# ============================================================

def get_latest_completed_end_time(reference_time=None):
    """
    Find the latest 3-hour-aligned endpoint for which
    the entire preceding 24-hour GPM observation window
    has all 48 expected granules.

    Searches backwards in 3-hour increments.

    Fails safely if no complete window is found.
    """

    if reference_time is None:
        reference_time = datetime.now(UTC)

    reference_time = normalize_utc(reference_time)

    candidate_end = align_to_3hour_boundary(reference_time)

    max_search_hours = MAX_ENDPOINT_SEARCH_DAYS * 24

    candidates_to_check = max_search_hours // 3

    print("\nNASA GPM AVAILABILITY CHECK")
    print("=" * 65)

    print("Collection:", COLLECTION_ID)
    print("Initial endpoint:", format_gpm_timestamp(candidate_end))

    for attempt in range(candidates_to_check + 1):

        candidate_start = candidate_end - timedelta(hours=24)

        print("\nChecking candidate endpoint:")
        print("Start:", format_gpm_timestamp(candidate_start))
        print("End:  ", format_gpm_timestamp(candidate_end))

        is_complete, message = validate_complete_24h_window(
            candidate_start,
            candidate_end,
        )

        print("Result:", message)

        if is_complete:

            print("\nCOMPLETE 24-HOUR WINDOW FOUND")
            print("Selected endpoint:", format_gpm_timestamp(candidate_end))

            return candidate_end

        candidate_end -= timedelta(hours=3)

    raise RuntimeError(
        "No complete 24-hour GPM window found within "
        f"{MAX_ENDPOINT_SEARCH_DAYS} days. "
        "No rainfall refresh should proceed."
    )


# ============================================================
# COMMON UTC RAINFALL WINDOWS
# ============================================================

def get_common_rainfall_windows(reference_time=None):
    """
    Calculate synchronized 3h, 6h, 12h and 24h windows.

    All windows end at the latest verified common endpoint.

    The endpoint is selected only after confirming that
    the entire preceding 24-hour period has 48 granules.
    """

    end_time = get_latest_completed_end_time(reference_time)

    windows = {}

    print("\nSELECTED COMMON RAINFALL WINDOWS")
    print("=" * 65)

    for hours in WINDOW_DURATIONS:

        start_time = end_time - timedelta(hours=hours)

        windows[hours] = {
            "start": start_time,
            "end": end_time,
            "expected_granules": get_granule_count(hours),
        }

        print(
            f"{hours}-hour: "
            f"{format_gpm_timestamp(start_time)} "
            f"to "
            f"{format_gpm_timestamp(end_time)} "
            f"| Expected granules: {get_granule_count(hours)}"
        )

    return windows
