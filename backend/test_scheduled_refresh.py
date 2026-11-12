"""Tests for Stage 2 intelligent scheduled GPM refresh entrypoint."""

from __future__ import annotations

import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from backend.refresh_safety import SKIPPED_RUNS_LOG_FILENAME
from backend.run_scheduled_refresh import (
    DEFAULT_SCHEDULER_LOG_FILE,
    SCHEDULER_LOG_BACKUP_COUNT,
    SCHEDULER_LOG_MAX_BYTES,
    _close_scheduler_logger,
    build_scheduler_logger,
    evaluate_unchanged_window_skip,
    main as scheduler_cli_main,
    run_scheduled_refresh,
)
from backend.run_standard_refresh import (
    DEFAULT_REPORT_DIR,
    REGION_ORDER,
    run_standard_refresh,
)


OBS_START_PREV = "2026-11-11T03:00:00Z"
OBS_END_PREV = "2026-11-12T03:00:00Z"
OBS_START_NEW = "2026-11-11T06:00:00Z"
OBS_END_NEW = "2026-11-12T06:00:00Z"
EXEC_NOW = datetime(2026, 11, 12, 8, 30, 0, tzinfo=timezone.utc)


def _make_discovery(
    obs_start: str = OBS_START_PREV,
    obs_end: str = OBS_END_PREV,
    granules: int = 48,
    missing: int = 0,
    latency_hours: float = 5.5,
    is_fresh: bool = True,
    freshness_status: str = "FRESH",
    freshness_reason: str = "OK",
) -> dict:
    return {
        "resolved_end_dt": datetime.fromisoformat(obs_end.replace("Z", "+00:00")),
        "observation_start_utc": obs_start,
        "observation_end_utc": obs_end,
        "discovery_source": "NASA_CMR_DRY_RUN",
        "resolver_status": "DRY_RUN_CMR_DISCOVERED" if granules == 48 else "INCOMPLETE_DATA",
        "latency_hours": latency_hours,
        "max_latency_hours": 12.0,
        "is_fresh": is_fresh,
        "freshness_status": freshness_status,
        "freshness_label": freshness_status,
        "freshness_reason": freshness_reason,
        "granules_discovered_cmr": granules,
        "granules_downloaded": 0,
        "granules_cached": granules,
        "granules_validated": granules,
        "expected_granules": 48,
        "missing_intervals_count": missing,
        "is_24h_complete": (granules == 48 and missing == 0),
        "telemetry": {"granules_validated": granules},
    }


def _make_published_report(
    status: str = "SUCCESS",
    obs_start: str = OBS_START_PREV,
    obs_end: str = OBS_END_PREV,
    region_overrides: dict | None = None,
    skipped_regions: list | None = None,
    run_id: str = "std_refresh_20261112T080206Z",
) -> dict:
    regions_payload = {}
    for rkey in REGION_ORDER:
        entry = {
            "display_name": rkey,
            "status": "SUCCESS",
            "validation_passed": True,
            "observation_start_utc": obs_start,
            "observation_end_utc": obs_end,
        }
        if region_overrides and rkey in region_overrides:
            entry.update(region_overrides[rkey])
        regions_payload[rkey] = entry

    return {
        "schema_version": "standard_refresh_report_v1",
        "run_id": run_id,
        "execution_timestamp_utc": "2026-11-12T08:02:06Z",
        "status": status,
        "observation_window": {
            "observation_start_utc": obs_start,
            "observation_end_utc": obs_end,
            "window_hours": 24,
            "discovery_source": "NASA_CMR_ONLINE",
        },
        "validation_and_test_results": {
            "preflight_status": "PASSED",
            "skipped_regions": list(skipped_regions or []),
            "errors": [],
            "warnings": [],
        },
        "regions": regions_payload,
    }


class TestScheduledRefreshStage2(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_sched_refresh_"))
        self.report_dir = self.tmp_dir / "refresh_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = self.tmp_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.tx_root = self.tmp_dir / "tx_backups"
        self.lock_path = self.report_dir / ".standard_refresh.lock"
        self.log_path = self.report_dir / "scheduler.log"

        self.region_specs = self._build_temp_specs(obs_end=OBS_END_PREV)
        self._write_latest_report(_make_published_report())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_temp_specs(self, obs_end: str = OBS_END_PREV) -> dict:
        specs = {}
        for rkey in REGION_ORDER:
            rdir = self.outputs_dir / rkey
            rdir.mkdir(parents=True, exist_ok=True)
            risk_csv = rdir / f"{rkey}_risk.csv"
            risk_meta = rdir / f"{rkey}_risk_metadata.json"
            map_meta = rdir / f"{rkey}_map_metadata.json"
            map_html = rdir / f"{rkey}_map.html"
            gj_file = rdir / f"{rkey}_regions.geojson"

            risk_csv.write_text("grid_id,prob\n1,0.5\n", encoding="utf-8")
            risk_meta.write_text(
                json.dumps({"gpm_observation_metadata": {"observation_end_utc": obs_end}}),
                encoding="utf-8",
            )
            map_meta.write_text(
                json.dumps({"gpm_observation_metadata": {"observation_end_utc": obs_end}}),
                encoding="utf-8",
            )
            map_html.write_text("<html><body>map</body></html>", encoding="utf-8")
            gj_file.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

            specs[rkey] = {
                "display_name": rkey,
                "required_inputs": [],
                "required_models": [],
                "tracked_outputs": [risk_csv, risk_meta, map_meta, map_html, gj_file],
                "risk_metadata_json": risk_meta,
                "map_metadata_json": map_meta,
            }
        return specs

    def _write_latest_report(self, payload: dict) -> Path:
        path = self.report_dir / "latest_refresh_report.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _passing_preflight(**kwargs) -> dict:
        return {"status": "PASSED", "checks_total": 1, "checks_passed": 1, "errors": []}

    def _spawn_lock_holder(self, lock_path: Path, run_id: str = "active_std_run") -> tuple[subprocess.Popen, int]:
        script = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "from backend.refresh_safety import RefreshProcessLock\n"
            "lock = RefreshProcessLock(Path(sys.argv[1]), run_id=sys.argv[2])\n"
            "if not lock.acquire():\n"
            "    sys.exit(3)\n"
            "print(f'LOCKED {os.getpid()}', flush=True)\n"
            "sys.stdin.read()\n"
            "lock.release()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script, str(lock_path), run_id],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        ready = proc.stdout.readline().strip()
        if not ready.startswith("LOCKED "):
            stderr = proc.stderr.read() if proc.stderr else ""
            proc.kill()
            proc.wait()
            self.fail(f"Child lock holder failed: ready={ready!r}, stderr={stderr!r}")
        return proc, int(ready.split()[1])

    def test_01_unchanged_successful_window_skips_cleanly(self) -> None:
        runner_calls = []

        def fake_runner(**kwargs):
            runner_calls.append(kwargs)
            return _make_published_report(run_id="unexpected_call")

        prev_report_raw = (self.report_dir / "latest_refresh_report.json").read_text(encoding="utf-8")
        res = run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            tx_backup_root=self.tx_root,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
            refresh_runner=fake_runner,
        )

        self.assertEqual(res["status"], "SKIPPED_UNCHANGED")
        self.assertEqual(runner_calls, [])
        self.assertEqual(
            (self.report_dir / "latest_refresh_report.json").read_text(encoding="utf-8"),
            prev_report_raw,
        )
        log_text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("outcome=SKIPPED_UNCHANGED", log_text)
        self.assertIn(res["run_id"], log_text)
        self.assertIn(OBS_END_PREV, log_text)

    def test_02_new_observation_window_triggers_processing(self) -> None:
        runner_calls = []

        def fake_runner(**kwargs):
            runner_calls.append(kwargs)
            return _make_published_report(
                status="SUCCESS",
                obs_start=OBS_START_NEW,
                obs_end=OBS_END_NEW,
                run_id="std_refresh_new_win",
            )

        res = run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            tx_backup_root=self.tx_root,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(OBS_START_NEW, OBS_END_NEW, latency_hours=2.5),
            refresh_runner=fake_runner,
        )

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(runner_calls), 1)
        log_text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("action=TRIGGER_REFRESH", log_text)
        self.assertIn("outcome=SUCCESS", log_text)
        self.assertIn("std_refresh_new_win", log_text)

    def test_03_force_flag_triggers_processing_even_when_unchanged(self) -> None:
        runner_calls = []

        def fake_runner(**kwargs):
            runner_calls.append(kwargs)
            return _make_published_report(run_id="std_refresh_forced")

        with (
            patch("backend.run_scheduled_refresh._build_region_specs", return_value=self.region_specs),
            patch("backend.run_scheduled_refresh.check_preflight", side_effect=self._passing_preflight),
            patch(
                "backend.run_scheduled_refresh.discover_and_retrieve_gpm",
                side_effect=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
            ),
            patch("backend.run_scheduled_refresh.run_standard_refresh", side_effect=fake_runner),
        ):
            exit_code = scheduler_cli_main([
                "--force",
                "--report-dir",
                str(self.report_dir),
                "--log-path",
                str(self.log_path),
                "--lock-path",
                str(self.lock_path),
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(runner_calls), 1)
        log_text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("force=True", log_text)
        self.assertIn("outcome=SUCCESS", log_text)

    def test_04_partial_or_failed_previous_status_never_skips(self) -> None:
        for prev_status in ("PARTIAL", "FAILED", "DRY_RUN"):
            self._write_latest_report(_make_published_report(status=prev_status))
            runner_calls = []
            res = run_scheduled_refresh(
                report_dir=self.report_dir,
                log_path=self.log_path,
                lock_path=self.lock_path,
                now_utc=EXEC_NOW,
                region_specs=self.region_specs,
                preflight_fn=self._passing_preflight,
                discovery_fn=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
                refresh_runner=lambda **kw: (runner_calls.append(kw), _make_published_report())[1],
            )
            self.assertEqual(res["status"], "SUCCESS", msg=f"status={prev_status}")
            self.assertEqual(len(runner_calls), 1, msg=f"Must not skip after status={prev_status}")

    def test_05_missing_or_empty_outputs_never_skip(self) -> None:
        # Case A: missing tracked output file
        missing_target = self.region_specs["western_ghats"]["tracked_outputs"][0]
        missing_target.unlink()
        runner_calls_a = []
        run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
            refresh_runner=lambda **kw: (runner_calls_a.append(kw), _make_published_report())[1],
        )
        self.assertEqual(len(runner_calls_a), 1)

        # Case B: empty (0-byte) tracked output file
        missing_target.write_text("", encoding="utf-8")
        runner_calls_b = []
        run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
            refresh_runner=lambda **kw: (runner_calls_b.append(kw), _make_published_report())[1],
        )
        self.assertEqual(len(runner_calls_b), 1)

        # Case C: on-disk metadata disagrees with discovered observation_end_utc
        self.region_specs = self._build_temp_specs(obs_end=OBS_END_PREV)
        drifted_meta = self.region_specs["northeast"]["risk_metadata_json"]
        drifted_meta.write_text(
            json.dumps({"gpm_observation_metadata": {"observation_end_utc": "2026-11-11T12:00:00Z"}}),
            encoding="utf-8",
        )
        can_skip, reason, _ = evaluate_unchanged_window_skip(
            report_dir=self.report_dir,
            discovery=_make_discovery(OBS_START_PREV, OBS_END_PREV),
            region_specs=self.region_specs,
        )
        self.assertFalse(can_skip)
        self.assertIn("On-disk metadata", reason)

    def test_06_inconsistent_regional_timestamps_or_validation_never_skip(self) -> None:
        # Top-level report claims SUCCESS for OBS_END_PREV, but western_ghats has an older window
        self._write_latest_report(
            _make_published_report(
                status="SUCCESS",
                obs_start=OBS_START_PREV,
                obs_end=OBS_END_PREV,
                region_overrides={
                    "western_ghats": {
                        "observation_start_utc": "2026-11-10T12:00:00Z",
                        "observation_end_utc": "2026-11-11T12:00:00Z",
                    }
                },
            )
        )
        runner_calls = []
        run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
            refresh_runner=lambda **kw: (runner_calls.append(kw), _make_published_report())[1],
        )
        self.assertEqual(len(runner_calls), 1)

        # Region with validation_passed=False also must not skip
        self._write_latest_report(
            _make_published_report(
                status="SUCCESS",
                region_overrides={"himalayas": {"validation_passed": False}},
            )
        )
        can_skip, reason, _ = evaluate_unchanged_window_skip(
            report_dir=self.report_dir,
            discovery=_make_discovery(OBS_START_PREV, OBS_END_PREV),
            region_specs=self.region_specs,
        )
        self.assertFalse(can_skip)
        self.assertIn("himalayas", reason)

    def test_07_stale_incomplete_or_unavailable_cmr_rejected_preserving_outputs(self) -> None:
        sentinel_output = self.region_specs["himalayas"]["tracked_outputs"][0]
        sentinel_bytes = sentinel_output.read_text(encoding="utf-8")
        runner_calls = []

        # (a) Stale (> 12h) observation rejected
        res_stale = run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(
                obs_start="2026-11-10T12:00:00Z",
                obs_end="2026-11-11T12:00:00Z",
                granules=48,
                missing=0,
                latency_hours=20.5,
                is_fresh=False,
                freshness_status="STALE_DATA_UNASSESSED",
                freshness_reason="Observation age 20.500h exceeds 12.0h",
            ),
            refresh_runner=lambda **kw: (runner_calls.append(kw), _make_published_report())[1],
        )
        self.assertEqual(res_stale["status"], "FAILED")
        self.assertEqual(res_stale["phase"], "freshness_completeness_gate")
        self.assertEqual(runner_calls, [])
        self.assertEqual(sentinel_output.read_text(encoding="utf-8"), sentinel_bytes)

        # (b) Incomplete (< 48 granules) rejected
        res_incomplete = run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(
                granules=47,
                missing=1,
                is_fresh=False,
                freshness_status="INCOMPLETE_DATA",
                freshness_reason="47/48 granules available",
            ),
            refresh_runner=lambda **kw: (runner_calls.append(kw), _make_published_report())[1],
        )
        self.assertEqual(res_incomplete["status"], "FAILED")
        self.assertEqual(runner_calls, [])

        # (c) CMR unavailable exception rejected and logged
        def failing_discovery(**kw):
            raise ConnectionError("NASA CMR endpoint timeout")

        res_cmr_err = run_scheduled_refresh(
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=failing_discovery,
            refresh_runner=lambda **kw: (runner_calls.append(kw), _make_published_report())[1],
        )
        self.assertEqual(res_cmr_err["status"], "FAILED")
        self.assertEqual(res_cmr_err["phase"], "discovery")
        self.assertEqual(runner_calls, [])

        # (d) Weakening 12h freshness guard is rejected
        with self.assertRaises(ValueError):
            run_scheduled_refresh(
                report_dir=self.report_dir,
                log_path=self.log_path,
                lock_path=self.lock_path,
                max_latency_hours=24.0,
                region_specs=self.region_specs,
            )

        log_text = self.log_path.read_text(encoding="utf-8")
        self.assertIn("outcome=FAILED", log_text)
        self.assertIn("NASA CMR endpoint timeout", log_text)
        self.assertIn("20.5", log_text)

    def test_08_persistent_rotating_utc_logging_configuration(self) -> None:
        self.assertEqual(DEFAULT_SCHEDULER_LOG_FILE, DEFAULT_REPORT_DIR / "scheduler.log")
        self.assertEqual(SCHEDULER_LOG_MAX_BYTES, 5 * 1024 * 1024)
        self.assertEqual(SCHEDULER_LOG_BACKUP_COUNT, 5)

        logger, handler = build_scheduler_logger(self.log_path, run_id="log_check_run")
        try:
            self.assertIsInstance(handler, RotatingFileHandler)
            self.assertEqual(handler.maxBytes, 5 * 1024 * 1024)
            self.assertEqual(handler.backupCount, 5)
            logger.info("run_id=log_check_run outcome=SUCCESS")
            try:
                raise RuntimeError("boom_trace_check")
            except RuntimeError as exc:
                logger.exception("run_id=log_check_run outcome=FAILED error=%s", exc)
        finally:
            _close_scheduler_logger(logger, handler)

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any("Z | INFO | run_id=log_check_run outcome=SUCCESS" in ln for ln in lines))
        self.assertTrue(any("Z | ERROR | run_id=log_check_run outcome=FAILED error=boom_trace_check" in ln for ln in lines))
        self.assertTrue(any("RuntimeError: boom_trace_check" in ln for ln in lines))

    def test_09_shared_process_lock_enforced(self) -> None:
        discovery_calls = []
        runner_calls = []

        # 1. Contention from external process holding shared lock stops before discovery
        proc, child_pid = self._spawn_lock_holder(self.lock_path, run_id="external_std_run")
        try:
            res = run_scheduled_refresh(
                report_dir=self.report_dir,
                log_path=self.log_path,
                lock_path=self.lock_path,
                now_utc=EXEC_NOW,
                region_specs=self.region_specs,
                preflight_fn=self._passing_preflight,
                discovery_fn=lambda **kw: (discovery_calls.append(kw), _make_discovery())[1],
                refresh_runner=lambda **kw: (runner_calls.append(kw), _make_published_report())[1],
            )
            self.assertEqual(res["status"], "SKIPPED_LOCKED")
            self.assertEqual(res["lock_holder"]["pid"], child_pid)
            self.assertEqual(discovery_calls, [])
            self.assertEqual(runner_calls, [])

            cli_exit = scheduler_cli_main([
                "--report-dir",
                str(self.report_dir),
                "--log-path",
                str(self.log_path),
                "--lock-path",
                str(self.lock_path),
            ])
            self.assertEqual(cli_exit, 0)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    stream.close()
            proc.wait(timeout=10)

        skip_entries = [
            json.loads(line)
            for line in (self.report_dir / SKIPPED_RUNS_LOG_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(skip_entries), 2)
        self.assertIn("outcome=SKIPPED_LOCKED", self.log_path.read_text(encoding="utf-8"))

        # 2. Verify integration against real run_standard_refresh lock lifecycle after release
        def failing_offline_runner(**kwargs):
            return run_standard_refresh(
                regions=["himalayas"],
                dry_run=True,
                offline=True,
                verify_model_load=False,
                report_dir=self.report_dir,
                tx_backup_root=self.tx_root,
                lock_path=self.lock_path,
                now_utc=EXEC_NOW + timedelta(seconds=1),
            )

        res_after = run_scheduled_refresh(
            regions=["himalayas"],
            force=True,
            offline=True,
            verify_model_load=False,
            report_dir=self.report_dir,
            log_path=self.log_path,
            lock_path=self.lock_path,
            tx_backup_root=self.tx_root,
            now_utc=EXEC_NOW + timedelta(seconds=1),
            region_specs=self.region_specs,
            preflight_fn=self._passing_preflight,
            discovery_fn=lambda **kw: _make_discovery(OBS_START_PREV, OBS_END_PREV),
            refresh_runner=failing_offline_runner,
        )
        self.assertIn(res_after.get("status"), ("DRY_RUN", "FAILED", "SUCCESS"))


if __name__ == "__main__":
    unittest.main()
