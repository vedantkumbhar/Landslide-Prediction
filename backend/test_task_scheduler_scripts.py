"""Tests for Stage 4 Windows Task Scheduler PowerShell setup scripts."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTER_SCRIPT = PROJECT_ROOT / "scripts" / "register_landslide_refresh_task.ps1"
UNREGISTER_SCRIPT = PROJECT_ROOT / "scripts" / "unregister_landslide_refresh_task.ps1"
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_landslide_refresh_task.ps1"


def _run_ps(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


class TestTaskSchedulerScripts(unittest.TestCase):
    """Validate Stage 4 setup, check, and safe unregister PowerShell scripts."""

    def test_scripts_exist(self) -> None:
        self.assertTrue(REGISTER_SCRIPT.is_file())
        self.assertTrue(UNREGISTER_SCRIPT.is_file())
        self.assertTrue(CHECK_SCRIPT.is_file())

    def test_register_dry_run_json_real_project(self) -> None:
        proc = _run_ps(REGISTER_SCRIPT, "-DryRun", "-Json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        payload = json.loads(proc.stdout)

        self.assertEqual(payload["task_name"], "LandslidePrediction_GPM_Refresh")
        self.assertTrue(payload["dry_run"])
        self.assertEqual(Path(payload["project_root"]), PROJECT_ROOT)
        self.assertEqual(
            Path(payload["python_executable"]),
            PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        )
        self.assertEqual(payload["entrypoint_module"], "backend.run_scheduled_refresh")
        self.assertEqual(
            payload["task_arguments"],
            "-m backend.run_scheduled_refresh --max-latency-hours 12.0 --workers 4",
        )
        self.assertEqual(Path(payload["working_directory"]), PROJECT_ROOT)
        self.assertEqual(
            Path(payload["scheduler_log"]),
            PROJECT_ROOT / "outputs" / "refresh_reports" / "scheduler.log",
        )
        self.assertEqual(payload["max_latency_hours"], 12.0)
        self.assertTrue(payload["unchanged_window_skip"])
        self.assertEqual(payload["repetition_interval"], "01:00:00")
        self.assertIn("PT1H", payload["recurrence"])
        self.assertTrue(payload["start_when_available"])
        self.assertEqual(payload["multiple_instances_policy"], "IgnoreNew")
        self.assertFalse(payload["wake_to_run"])
        self.assertFalse(payload["run_while_shut_down"])
        self.assertTrue(payload["checks"]["project_root_ok"])
        self.assertTrue(payload["checks"]["python_executable_ok"])
        self.assertTrue(payload["checks"]["entrypoint_script_ok"])
        self.assertTrue(payload["checks"]["cli_help_check_ok"])
        self.assertTrue(payload["checks"]["freshness_guard_ok"])

    def test_register_rejects_loosened_freshness_guard(self) -> None:
        proc = _run_ps(REGISTER_SCRIPT, "-DryRun", "-MaxLatencyHours", "24.0")
        self.assertNotEqual(proc.returncode, 0)
        combined = (proc.stdout + "\n" + proc.stderr).lower()
        self.assertIn("12", combined)

    def test_register_rejects_invalid_workers(self) -> None:
        proc = _run_ps(REGISTER_SCRIPT, "-DryRun", "-Workers", "0")
        self.assertNotEqual(proc.returncode, 0)

    def test_register_rejects_wildcard_task_name(self) -> None:
        proc = _run_ps(REGISTER_SCRIPT, "-DryRun", "-TaskName", "Landslide*")
        self.assertNotEqual(proc.returncode, 0)

    def test_register_rejects_missing_python_in_temp_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = _run_ps(REGISTER_SCRIPT, "-DryRun", "-ProjectRoot", tmpdir)
            self.assertNotEqual(proc.returncode, 0)
            combined = (proc.stdout + "\n" + proc.stderr).lower()
            self.assertIn("python", combined)

    def test_unregister_dry_run_rejects_wildcards_and_paths(self) -> None:
        for bad_name in ("Landslide*", "Task?", r"Folder\Task", "Folder/Task"):
            proc = _run_ps(UNREGISTER_SCRIPT, "-DryRun", "-TaskName", bad_name)
            self.assertNotEqual(proc.returncode, 0, msg=f"Should reject {bad_name}")

    def test_unregister_dry_run_preserves_data_and_logs(self) -> None:
        proc = _run_ps(UNREGISTER_SCRIPT, "-DryRun", "-Json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["task_name"], "LandslidePrediction_GPM_Refresh")
        self.assertTrue(payload["project_data_touched"] is False)
        self.assertTrue(payload["logs_or_backups_kept"] is True)

    def test_check_script_real_project_passes_without_require_registered(self) -> None:
        proc = _run_ps(CHECK_SCRIPT, "-Json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["checks"]["project_root_ok"])
        self.assertTrue(payload["checks"]["python_executable_ok"])
        self.assertTrue(payload["checks"]["entrypoint_script_ok"])
        self.assertTrue(payload["checks"]["cli_help_check_ok"])

    def test_check_script_fails_for_missing_project_files_or_require_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = _run_ps(CHECK_SCRIPT, "-ProjectRoot", tmpdir, "-Json")
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["checks"]["python_executable_ok"])

        proc_req = _run_ps(
            CHECK_SCRIPT,
            "-TaskName",
            "LandslidePrediction_GPM_Refresh_NonExistent_Test",
            "-RequireRegistered",
        )
        self.assertNotEqual(proc_req.returncode, 0)


if __name__ == "__main__":
    unittest.main()
