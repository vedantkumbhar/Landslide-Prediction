
from pathlib import Path
from datetime import datetime, timedelta, timezone
import h5py
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "satellite"

SERIES = {
    "3H": [RAW / "gpm_imerge_series_3h"],
    "6H": [RAW / "gpm_imerge_series_6h"],
    "12H": [RAW / "gpm_imerge_series_12h"],
    "24H": [
        RAW / "gpm_imerge_previous12h",
        RAW / "gpm_imerge_series_12h",
    ],
}

EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

def read_bounds(path):
    with h5py.File(path, "r") as f:
        if "Grid/time_bnds" not in f:
            raise KeyError("Grid/time_bnds not found")

        dataset = f["Grid/time_bnds"]
        values = np.asarray(dataset[...]).astype(float).ravel()

        units = decode_attr(
            dataset.attrs.get(
                "units",
                "seconds since 1980-01-06 00:00:00 UTC"
            )
        )

        if "seconds since 1980-01-06" not in units:
            raise ValueError(f"Unexpected time units: {units}")

        if len(values) < 2:
            raise ValueError("Invalid time_bnds dimensions")

        start = EPOCH + timedelta(seconds=float(values[0]))
        end = EPOCH + timedelta(seconds=float(values[1]))

        return start, end

print("\nGPM HDF5 TIME BOUNDS VALIDATION")
print("=" * 70)

for series_name, directories in SERIES.items():
    files = []

    for directory in directories:
        if not directory.exists():
            print(f"\nWARNING: Missing directory: {directory}")
            continue

        for pattern in ("*.HDF5", "*.h5", "*.hdf5"):
            files.extend(directory.glob(pattern))

    files = sorted(set(files))

    print(f"\n{series_name} SERIES")
    print("-" * 70)
    print("Files found:", len(files))

    if not files:
        continue

    intervals = []

    for path in files:
        try:
            start, end = read_bounds(path)
            intervals.append((start, end, path.name))
        except Exception as exc:
            print(f"ERROR: {path.name}: {exc}")

    intervals.sort(key=lambda item: item[0])

    if not intervals:
        continue

    print("\nFirst granule:")
    print("File:", intervals[0][2])
    print("Start:", intervals[0][0].isoformat())
    print("End:", intervals[0][1].isoformat())

    print("\nLast granule:")
    print("File:", intervals[-1][2])
    print("Start:", intervals[-1][0].isoformat())
    print("End:", intervals[-1][1].isoformat())

    gaps = []
    overlaps = []
    invalid = []

    for i, (start, end, filename) in enumerate(intervals):
        if end <= start:
            invalid.append(filename)

        if i == 0:
            continue

        previous_end = intervals[i - 1][1]

        if start > previous_end:
            gaps.append((previous_end, start))

        elif start < previous_end:
            overlaps.append((start, previous_end))

    print("\nInterval checks:")
    print("Invalid intervals:", len(invalid))
    print("Gaps:", len(gaps))
    print("Overlaps:", len(overlaps))

    if gaps:
        print("First gaps:")
        for gap_start, gap_end in gaps[:5]:
            print(gap_start.isoformat(), "to", gap_end.isoformat())

    if overlaps:
        print("First overlaps:")
        for overlap_start, overlap_end in overlaps[:5]:
            print(
                overlap_start.isoformat(),
                "to",
                overlap_end.isoformat()
            )

    print("\nActual complete coverage:")
    print("Start:", intervals[0][0].isoformat())
    print("End:", intervals[-1][1].isoformat())

print("\nHDF5 TIME BOUNDS AUDIT COMPLETE")
print("Original satellite files were not modified.")
