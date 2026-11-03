from pathlib import Path
from datetime import timedelta, datetime, timezone
import argparse
import subprocess
import sys

from backend.gpm_refresh.refresh_config import (
    PROJECT_ROOT,
    get_run_directory,
    create_run_directories,
    get_common_rainfall_windows,
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_INPUT = (
    PROJECT_ROOT
    / "data/processed/rainfall/refresh_runs/20261102_152742/predictions/"
    / "regional_gpm_risk_predictions_02nov2026_with_3h.csv"
)

EXPECTED_RECORDS = 2009
EXPECTED_INTERVALS_24H = 48


# =========================================================
# HELPERS
# =========================================================

def format_utc(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_command(module_name, *arguments):
    return [
        sys.executable,
        "-m",
        module_name,
        *[str(argument) for argument in arguments],
    ]


# =========================================================
# BUILD REFRESH PLAN
# =========================================================

def build_refresh_plan(run_id, windows):

    previous_12h_start = windows[12]["start"] - timedelta(hours=12)
    previous_12h_end = windows[12]["start"]

    latest_12h_start = windows[12]["start"]
    latest_12h_end = windows[12]["end"]

    previous_run_id = f"{run_id}_previous12h"
    latest_run_id = f"{run_id}_latest12h"

    run_dir = get_run_directory(run_id)

    paths = {
        "previous_12h_csv": (
            run_dir / "extracted" / "previous12h.csv"
        ),
        "latest_12h_csv": (
            run_dir / "extracted" / "latest12h.csv"
        ),
        "combined_24h_csv": (
            run_dir / "extracted" / "combined_24h.csv"
        ),
        "integrated_csv": (
            run_dir
            / "predictions"
            / "regional_gpm_risk_predictions_integrated.csv"
        ),
        "map": (
            run_dir
            / "maps"
            / "regional_landslide_risk_map.html"
        ),
    }

    commands = []

    # -----------------------------------------------------
    # 1–4: DOWNLOAD GRANULES
    # -----------------------------------------------------

    download_windows = [
        (
            f"{run_id}_3h",
            windows[3]["start"],
            windows[3]["end"],
        ),
        (
            f"{run_id}_6h",
            windows[6]["start"],
            windows[6]["end"],
        ),
        (
            previous_run_id,
            previous_12h_start,
            previous_12h_end,
        ),
        (
            latest_run_id,
            latest_12h_start,
            latest_12h_end,
        ),
    ]

    for sub_run_id, start, end in download_windows:
        commands.append(
            build_command(
                "backend.gpm_refresh.run_existing_downloader",
                "--start",
                format_utc(start),
                "--end",
                format_utc(end),
                "--run-id",
                sub_run_id,
            )
        )

    # -----------------------------------------------------
    # 5–8: VALIDATE HDF5 GRANULES
    # -----------------------------------------------------

    validation_windows = [
        (f"{run_id}_3h", 3, windows[3]["start"]),
        (f"{run_id}_6h", 6, windows[6]["start"]),
        (previous_run_id, 12, previous_12h_start),
        (latest_run_id, 12, latest_12h_start),
    ]

    for sub_run_id, hours, start in validation_windows:
        commands.append(
            build_command(
                "backend.gpm_refresh.validate_gpm_hdf5",
                "--run-id",
                sub_run_id,
                "--hours",
                hours,
                "--start",
                format_utc(start),
            )
        )

    # -----------------------------------------------------
    # 9: EXTRACT 3-HOUR RAINFALL
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.extract_refresh_3h",
            "--run-id",
            f"{run_id}_3h",
            "--start",
            format_utc(windows[3]["start"]),
            "--input-csv",
            BASE_INPUT,
        )
    )

    # -----------------------------------------------------
    # 10: EXTRACT 6-HOUR RAINFALL
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.extract_refresh_6h",
            "--run-id",
            f"{run_id}_6h",
            "--start",
            format_utc(windows[6]["start"]),
            "--input-csv",
            BASE_INPUT,
        )
    )

    # -----------------------------------------------------
    # 11: EXTRACT PREVIOUS 12 HOURS
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.extract_refresh_12h",
            "--run-id",
            previous_run_id,
            "--start-time",
            format_utc(previous_12h_start),
            "--input-csv",
            BASE_INPUT,
            "--output-csv",
            paths["previous_12h_csv"],
        )
    )

    # -----------------------------------------------------
    # 12: EXTRACT LATEST 12 HOURS
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.extract_refresh_12h",
            "--run-id",
            latest_run_id,
            "--start-time",
            format_utc(latest_12h_start),
            "--input-csv",
            BASE_INPUT,
            "--output-csv",
            paths["latest_12h_csv"],
        )
    )

    # -----------------------------------------------------
    # 13: COMBINE TWO 12-HOUR WINDOWS
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.combine_refresh_24h",
            "--previous-run-id",
            previous_run_id,
            "--latest-run-id",
            latest_run_id,
            "--previous-csv",
            paths["previous_12h_csv"],
            "--latest-csv",
            paths["latest_12h_csv"],
            "--output-csv",
            paths["combined_24h_csv"],
            "--expected-records",
            EXPECTED_RECORDS,
            "--expected-intervals",
            EXPECTED_INTERVALS_24H,
        )
    )

    # -----------------------------------------------------
    # 14: VALIDATE 24-HOUR DATA
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.validate_refresh_24h",
            "--run-id",
            run_id,
            "--input-csv",
            paths["combined_24h_csv"],
            "--expected-rows",
            EXPECTED_RECORDS,
            "--expected-intervals",
            EXPECTED_INTERVALS_24H,
            "--expected-start",
            format_utc(previous_12h_start),
            "--expected-end",
            format_utc(latest_12h_end),
        )
    )

    # -----------------------------------------------------
    # 15: INTEGRATE RAINFALL WITH PREDICTIONS
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.gpm_refresh.integrate_rainfall_predictions",
            "--run-id",
            run_id,
            "--base-csv",
            BASE_INPUT,
            "--input-3h",
            get_run_directory(f"{run_id}_3h")
            / "extracted"
            / "gpm_regional_rainfall_3h.csv",
            "--input-6h",
            get_run_directory(f"{run_id}_6h")
            / "extracted"
            / "gpm_regional_rainfall_6h.csv",
            "--input-12h",
            paths["latest_12h_csv"],
            "--input-24h",
            paths["combined_24h_csv"],
            "--output-csv",
            paths["integrated_csv"],
        )
    )

    # -----------------------------------------------------
    # 16: GENERATE UPDATED GIS MAP
    # -----------------------------------------------------

    commands.append(
        build_command(
            "backend.create_regional_risk_map",
            "--input-csv",
            paths["integrated_csv"],
            "--output-map",
            paths["map"],
        )
    )

    return commands, paths


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Complete NASA GPM rainfall refresh pipeline."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional unique run ID.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the pipeline. Default mode is dry-run.",
    )

    args = parser.parse_args()

    run_id = (
        args.run_id
        if args.run_id
        else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )

    print("=" * 70)
    print("LANDSLIDE PREDICTION - FULL GPM REFRESH")
    print("=" * 70)

    print(f"Run ID: {run_id}")
    print(f"Base prediction CSV: {BASE_INPUT}")

    if not BASE_INPUT.exists():
        raise FileNotFoundError(
            f"Base prediction CSV not found: {BASE_INPUT}"
        )

    # Check run directory collisions.
    related_run_ids = [
        run_id,
        f"{run_id}_3h",
        f"{run_id}_6h",
        f"{run_id}_previous12h",
        f"{run_id}_latest12h",
    ]

    existing_directories = [
        str(get_run_directory(item))
        for item in related_run_ids
        if get_run_directory(item).exists()
    ]

    if existing_directories:
        raise FileExistsError(
            "One or more run directories already exist:\n"
            + "\n".join(existing_directories)
            + "\nChoose a new run ID."
        )

    # Calculate all observation windows once.
    windows = get_common_rainfall_windows()

    commands, paths = build_refresh_plan(run_id, windows)

    previous_12h_start = windows[12]["start"] - timedelta(hours=12)
    previous_12h_end = windows[12]["start"]

    print("\nCOMMON UTC OBSERVATION WINDOWS")
    print("-" * 70)

    print(
        "3-hour:",
        format_utc(windows[3]["start"]),
        "to",
        format_utc(windows[3]["end"]),
    )

    print(
        "6-hour:",
        format_utc(windows[6]["start"]),
        "to",
        format_utc(windows[6]["end"]),
    )

    print(
        "Previous 12-hour:",
        format_utc(previous_12h_start),
        "to",
        format_utc(previous_12h_end),
    )

    print(
        "Latest 12-hour:",
        format_utc(windows[12]["start"]),
        "to",
        format_utc(windows[12]["end"]),
    )

    print("\nEXECUTION PLAN")
    print("-" * 70)

    for index, command in enumerate(commands, start=1):
        print(f"\n{index:02d}. {command!r}")

    print("\n" + "=" * 70)

    # -----------------------------------------------------
    # DRY-RUN MODE
    # -----------------------------------------------------

    if not args.execute:
        print("DRY RUN COMPLETE.")
        print("No folders, downloads, or processing steps were created.")
        print("Review the command arguments before execution.")

        print("\nTo execute, use:")
        print(
            f"python -m backend.gpm_refresh.run_full_refresh "
            f"--run-id {run_id} --execute"
        )
        return

    # -----------------------------------------------------
    # EXECUTION MODE
    # -----------------------------------------------------

    output_paths = [
        paths["previous_12h_csv"],
        paths["latest_12h_csv"],
        paths["combined_24h_csv"],
        paths["integrated_csv"],
        paths["map"],
    ]

    existing_outputs = [
        str(path)
        for path in output_paths
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Output files already exist:\n"
            + "\n".join(existing_outputs)
        )

    directories = create_run_directories(run_id)
    log_path = directories["logs"] / "full_refresh.log"

    print("\nEXECUTING FULL REFRESH")
    print(f"Log file: {log_path}")
    print("=" * 70)

    with open(log_path, "w", encoding="utf-8") as log_file:

        for index, command in enumerate(commands, start=1):

            display = subprocess.list2cmdline(command)

            print(
                f"\n[{index:02d}/{len(commands):02d}] {display}"
            )

            log_file.write(f"\n{'=' * 70}\n")
            log_file.write(f"STEP {index:02d}\n")
            log_file.write(display + "\n")
            log_file.flush()

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.stdout:
                print(result.stdout)
                log_file.write(result.stdout + "\n")

            if result.stderr:
                print(result.stderr)
                log_file.write(result.stderr + "\n")

            log_file.flush()

            if result.returncode != 0:
                message = (
                    f"\nFAILED at step {index:02d}.\n"
                    f"Return code: {result.returncode}\n"
                    f"See log: {log_path}"
                )

                print(message)
                log_file.write(message + "\n")

                raise SystemExit(result.returncode)

    print("\n" + "=" * 70)
    print("FULL GPM REFRESH COMPLETED SUCCESSFULLY.")
    print(f"Integrated predictions: {paths['integrated_csv']}")
    print(f"Updated GIS map: {paths['map']}")
    print(f"Execution log: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()