from pathlib import Path
import argparse
import h5py
import pandas as pd
import numpy as np

from backend.gpm_refresh.refresh_config import (
    get_run_directory,
    GPM_INTERVAL_MINUTES,
    EXPECTED_GRANULE_SHAPE,
)


def validate(run_id, duration_hours, expected_start):

    granule_dir = (
        get_run_directory(run_id)
        / "downloads"
        / f"{duration_hours}h"
    )

    expected_count = duration_hours * 60 // GPM_INTERVAL_MINUTES
    expected_shape = EXPECTED_GRANULE_SHAPE

    files = sorted(granule_dir.glob("*.HDF5"))

    print("\n========== GPM HDF5 VALIDATION ==========")
    print("Run ID:", run_id)
    print("Duration:", duration_hours, "hours")
    print("Directory:", granule_dir)
    print("Files found:", len(files))
    print("Expected files:", expected_count)

    if len(files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} granules, found {len(files)}"
        )

    expected_start = pd.Timestamp(expected_start)

    if expected_start.tzinfo is None:
        expected_start = expected_start.tz_localize("UTC")
    else:
        expected_start = expected_start.tz_convert("UTC")

    previous_end = None

    for index, path in enumerate(files):

        print(f"\nChecking {index + 1}/{expected_count}: {path.name}")

        with h5py.File(path, "r") as hdf:

            if "Grid/time_bnds" not in hdf:
                raise ValueError(f"Missing time bounds: {path.name}")

            if "Grid/time" not in hdf:
                raise ValueError(f"Missing time metadata: {path.name}")

            bounds = np.asarray(hdf["Grid/time_bnds"])

            if bounds.shape != (1, 2):
                raise ValueError(
                    f"Invalid time bounds shape: {path.name}"
                )

            units = hdf["Grid/time"].attrs.get("units")

            if isinstance(units, bytes):
                units = units.decode("utf-8")

            if not units or "since" not in units:
                raise ValueError(
                    f"Invalid time units in {path.name}"
                )

            origin = units.split("since", 1)[1].strip()
            base = pd.Timestamp(origin)

            if base.tzinfo is None:
                base = base.tz_localize("UTC")
            else:
                base = base.tz_convert("UTC")

            actual_start = (
                base + pd.to_timedelta(float(bounds[0, 0]), unit="s")
            )

            actual_end = (
                base + pd.to_timedelta(float(bounds[0, 1]), unit="s")
            )

            if actual_end - actual_start != pd.Timedelta(
                minutes=GPM_INTERVAL_MINUTES
            ):
                raise ValueError(
                    f"Incorrect granule duration: {path.name}"
                )

            expected_time = expected_start + pd.Timedelta(
                minutes=index * GPM_INTERVAL_MINUTES
            )

            if actual_start != expected_time:
                raise ValueError(
                    f"Unexpected start in {path.name}: "
                    f"{actual_start}; expected {expected_time}"
                )

            if previous_end is not None and actual_start != previous_end:
                raise ValueError(
                    f"Time gap or overlap: {path.name}"
                )

            precipitation = hdf["Grid/precipitation"]

            if precipitation.shape != expected_shape:
                raise ValueError(
                    f"Invalid precipitation shape: {path.name}"
                )

            units = precipitation.attrs.get("units", b"")

            if isinstance(units, bytes):
                units = units.decode("utf-8")

            if units.strip() != "mm/hr":
                raise ValueError(
                    f"Unexpected precipitation units: {units}"
                )

            print("Start:", actual_start)
            print("End:", actual_end)
            print("Shape:", precipitation.shape)
            print("Units:", units)
            print("STATUS: PASS")

            previous_end = actual_end

    expected_end = expected_start + pd.Timedelta(
        hours=duration_hours
    )

    if previous_end != expected_end:
        raise ValueError("Final observation boundary mismatch")

    print("\n========== VALIDATION SUCCESSFUL ==========")
    print("Validated granules:", expected_count)
    print("Coverage:", expected_start, "to", expected_end)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--hours",
        type=int,
        required=True,
        choices=[3, 6, 12, 24],
    )
    parser.add_argument("--start", required=True)

    args = parser.parse_args()

    validate(
        run_id=args.run_id,
        duration_hours=args.hours,
        expected_start=args.start,
    )


if __name__ == "__main__":
    main()