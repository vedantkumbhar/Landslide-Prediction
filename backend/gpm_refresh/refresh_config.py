
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
# COMMON UTC TIME WINDOW
# ============================================================

def get_latest_completed_end_time(reference_time=None):
    """
    Find one common completed UTC timestamp.

    The timestamp is aligned to a completed 3-hour boundary.
    All rainfall windows will use this same end time.
    """

    if reference_time is None:
        reference_time = datetime.now(UTC)

    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)
    else:
        reference_time = reference_time.astimezone(UTC)

    interval_seconds = 3 * 3600

    timestamp = int(reference_time.timestamp())

    end_timestamp = (
        timestamp // interval_seconds
    ) * interval_seconds

    return datetime.fromtimestamp(end_timestamp, UTC)


def get_common_rainfall_windows(reference_time=None):
    """
    Calculate synchronized 3h, 6h, 12h and 24h windows.

    Every window ends at exactly the same UTC timestamp.
    """

    end_time = get_latest_completed_end_time(reference_time)

    windows = {}

    for hours in WINDOW_DURATIONS:

        start_time = end_time - timedelta(hours=hours)

        windows[hours] = {
            "start": start_time,
            "end": end_time,
            "expected_granules": get_granule_count(hours)
        }

    return windows


# ============================================================
# TIME UTILITIES
# ============================================================

def format_gpm_timestamp(timestamp):

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    else:
        timestamp = timestamp.astimezone(UTC)

    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_granule_count(hours):

    if hours <= 0:
        raise ValueError("Hours must be positive.")

    return hours * GRANULES_PER_HOUR
