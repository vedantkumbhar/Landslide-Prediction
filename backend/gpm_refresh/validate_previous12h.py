
from pathlib import Path
from datetime import datetime, timedelta, timezone

import h5py


# Previous 12-hour GPM window
RUN_ID = "20261102_previous12h"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall"
    / "refresh_runs"
    / RUN_ID
    / "downloads"
    / "12h"
)

TIME_ORIGIN = datetime(1980, 1, 6, tzinfo=timezone.utc)

EXPECTED_START = datetime(
    2026, 11, 1, 3, 0, tzinfo=timezone.utc
)

EXPECTED_COUNT = 24
INTERVAL_SECONDS = 1800
EXPECTED_SHAPE = (1, 3600, 1800)


def main():
    files = sorted(INPUT_DIR.glob("*.HDF5"))

    print("\n========== PREVIOUS 12H GPM VALIDATION ==========")
    print("Input directory:", INPUT_DIR)
    print("Files found:", len(files))
    print("Expected files:", EXPECTED_COUNT)

    if len(files) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} files, found {len(files)}."
        )

    previous_end = None

    for index, file_path in enumerate(files):
        print(f"\nChecking {index + 1}/{EXPECTED_COUNT}")
        print("File:", file_path.name)

        with h5py.File(file_path, "r") as hdf:

            if "Grid/time_bnds" not in hdf:
                raise RuntimeError("Missing Grid/time_bnds")

            bounds = hdf["Grid/time_bnds"][:]

            if bounds.shape != (1, 2):
                raise RuntimeError(
                    f"Unexpected time bounds shape: {bounds.shape}"
                )

            start_seconds, end_seconds = bounds[0]

            actual_start = TIME_ORIGIN + timedelta(
                seconds=int(start_seconds)
            )

            actual_end = TIME_ORIGIN + timedelta(
                seconds=int(end_seconds)
            )

            if actual_end - actual_start != timedelta(
                seconds=INTERVAL_SECONDS
            ):
                raise RuntimeError("Incorrect granule duration.")

            expected_start = EXPECTED_START + timedelta(
                seconds=index * INTERVAL_SECONDS
            )

            if actual_start != expected_start:
                raise RuntimeError(
                    f"Unexpected start time: {actual_start}; "
                    f"expected {expected_start}"
                )

            if previous_end is not None and actual_start != previous_end:
                raise RuntimeError(
                    f"Time gap or overlap detected before {file_path.name}"
                )

            precipitation = hdf["Grid/precipitation"]

            if precipitation.shape != EXPECTED_SHAPE:
                raise RuntimeError(
                    f"Unexpected precipitation shape: "
                    f"{precipitation.shape}"
                )

            units = precipitation.attrs.get("units", b"")

            if isinstance(units, bytes):
                units = units.decode("utf-8")

            if units != "mm/hr":
                raise RuntimeError(
                    f"Unexpected precipitation units: {units}"
                )

            print("Start UTC:", actual_start)
            print("End UTC:", actual_end)
            print("Duration: 30 minutes")
            print("Precipitation shape:", precipitation.shape)
            print("Units:", units)
            print("STATUS: PASS")

            previous_end = actual_end

    expected_end = EXPECTED_START + timedelta(
        seconds=EXPECTED_COUNT * INTERVAL_SECONDS
    )

    print("\n========== FINAL RESULT ==========")
    print("Validated granules:", EXPECTED_COUNT)
    print("Coverage start:", EXPECTED_START)
    print("Coverage end:", expected_end)
    print("Temporal gaps: 0")
    print("Temporal overlaps: 0")
    print("Invalid granules: 0")
    print("VALIDATION SUCCESSFUL")


if __name__ == "__main__":
    main()
