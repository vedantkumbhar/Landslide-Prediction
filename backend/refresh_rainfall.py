
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ROOT / "backend" / "fetch_rainfall.py",
    ROOT / "backend" / "build_rainfall_features.py",
]


def run_script(script: Path) -> None:
    print(f"\nRunning: {script.name}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"FAILED: {script.name}")

        if result.stderr:
            print(result.stderr)

        raise SystemExit(result.returncode)

    if result.stderr:
        print("Warnings or additional messages:")
        print(result.stderr)

    print(f"Completed: {script.name}")


def refresh_rainfall() -> None:
    for script in SCRIPTS:
        if not script.is_file():
            print(f"Required script not found: {script}")
            raise SystemExit(1)

        run_script(script)

    print("\nRainfall refresh and feature rebuilding completed successfully.")


if __name__ == "__main__":
    refresh_rainfall()
