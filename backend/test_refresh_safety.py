"""Targeted tests for Stage 1 safe refresh execution (process locking and crash recovery)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from backend.refresh_safety import (
    SKIPPED_RUNS_LOG_FILENAME,
    TX_JOURNAL_FILENAME,
    TX_SCHEMA_VERSION,
    RefreshProcessLock,
    RegionTransaction,
    recover_interrupted_transactions,
    sha256_file,
)
from backend.run_standard_refresh import (
    main as refresh_cli_main,
    run_standard_refresh,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestRefreshSafetyStage1(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_refresh_safety_"))
        self.report_dir = self.tmp_dir / "refresh_reports"
        self.tx_root = self.tmp_dir / "tx_backups"
        self.targets_dir = self.tmp_dir / "region_outputs"
        self.targets_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.report_dir / ".standard_refresh.lock"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _spawn_lock_holder(
        self,
        lock_path: Path,
        run_id: str = "child_holder_run",
        abrupt: bool = False,
    ) -> tuple[subprocess.Popen, int]:
        if abrupt:
            script = (
                "import os, sys, time\n"
                "from pathlib import Path\n"
                "from backend.refresh_safety import RefreshProcessLock\n"
                "lock = RefreshProcessLock(Path(sys.argv[1]), run_id=sys.argv[2])\n"
                "if not lock.acquire():\n"
                "    sys.exit(3)\n"
                "print(f'LOCKED {os.getpid()}', flush=True)\n"
                "while True:\n"
                "    time.sleep(1)\n"
            )
        else:
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
            self.fail(f"Child lock holder failed to acquire lock: ready={ready!r}, stderr={stderr!r}")
        child_pid = int(ready.split()[1])
        return proc, child_pid

    def test_01_lock_acquisition_and_release(self) -> None:
        lock_a = RefreshProcessLock(self.lock_path, run_id="run_a")
        lock_b = RefreshProcessLock(self.lock_path, run_id="run_b")

        self.assertTrue(lock_a.acquire())
        self.assertTrue(lock_a.acquired)
        self.assertTrue(self.lock_path.exists())

        self.assertFalse(lock_b.acquire())
        self.assertIsNotNone(lock_b.holder_info)
        self.assertEqual(lock_b.holder_info.get("pid"), os.getpid())
        self.assertEqual(lock_b.holder_info.get("run_id"), "run_a")

        lock_a.release()
        self.assertFalse(lock_a.acquired)

        self.assertTrue(lock_b.acquire())
        lock_b.release()

    def test_02_concurrent_execution_rejected_without_touching_data_or_reports(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        latest_json = self.report_dir / "latest_refresh_report.json"
        latest_md = self.report_dir / "latest_refresh_report.md"
        sentinel_latest = {"run_id": "previous_valid_run", "status": "SUCCESS"}
        latest_json.write_text(json.dumps(sentinel_latest), encoding="utf-8")
        latest_md.write_text("# Previous Valid Report\n", encoding="utf-8")

        data_file = self.targets_dir / "protected_output.json"
        data_file.write_text('{"value": 42}', encoding="utf-8")

        proc, child_pid = self._spawn_lock_holder(self.lock_path, run_id="active_refresh_01")
        try:
            res = run_standard_refresh(
                regions=["himalayas"],
                dry_run=True,
                report_dir=self.report_dir,
                tx_backup_root=self.tx_root,
                lock_path=self.lock_path,
                now_utc=datetime(2026, 11, 12, 9, 0, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(res["status"], "SKIPPED_LOCKED")
            self.assertEqual(res["lock_holder"]["pid"], child_pid)
            self.assertEqual(res["lock_holder"]["run_id"], "active_refresh_01")

            exit_code = refresh_cli_main([
                "--regions",
                "himalayas",
                "--dry-run",
                "--report-dir",
                str(self.report_dir),
                "--lock-path",
                str(self.lock_path),
            ])
            self.assertEqual(exit_code, 0)
        finally:
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.wait(timeout=10)

        self.assertEqual(json.loads(latest_json.read_text(encoding="utf-8")), sentinel_latest)
        self.assertEqual(latest_md.read_text(encoding="utf-8"), "# Previous Valid Report\n")
        self.assertEqual(data_file.read_text(encoding="utf-8"), '{"value": 42}')

        skip_log = self.report_dir / SKIPPED_RUNS_LOG_FILENAME
        self.assertTrue(skip_log.exists())
        lines = [json.loads(line) for line in skip_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(item["status"] == "SKIPPED_LOCKED" for item in lines))
        self.assertTrue(all(item["lock_holder"]["pid"] == child_pid for item in lines))

        next_lock = RefreshProcessLock(self.lock_path, run_id="after_child_exit")
        self.assertTrue(next_lock.acquire())
        next_lock.release()

    def test_03_lock_released_after_exceptions(self) -> None:
        with self.assertRaises(RuntimeError):
            with RefreshProcessLock(self.lock_path, run_id="failing_ctx"):
                raise RuntimeError("simulated exception inside locked section")

        lock_reacquire = RefreshProcessLock(self.lock_path, run_id="post_exception")
        self.assertTrue(lock_reacquire.acquire())
        lock_reacquire.release()

        # Also test exception path through run_standard_refresh
        with self.assertRaises(ValueError):
            run_standard_refresh(
                regions=["invalid_region_name"],
                report_dir=self.report_dir,
                tx_backup_root=self.tx_root,
                lock_path=self.lock_path,
            )

        post_cli_lock = RefreshProcessLock(self.lock_path, run_id="post_run_exception")
        self.assertTrue(post_cli_lock.acquire())
        post_cli_lock.release()

    def test_04_lock_auto_released_on_process_crash(self) -> None:
        proc, child_pid = self._spawn_lock_holder(
            self.lock_path, run_id="crashing_holder", abrupt=True
        )
        try:
            os.kill(child_pid, signal.SIGTERM)
            proc.wait(timeout=10)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    stream.close()

        recovered_lock = RefreshProcessLock(self.lock_path, run_id="post_crash_runner")
        self.assertTrue(recovered_lock.acquire(), "OS must release msvcrt lock automatically when holder dies")
        recovered_lock.release()

    def test_05_interrupted_transaction_recovery_and_conservative_temp_cleanup(self) -> None:
        existing_file = self.targets_dir / "existing_risk.geojson"
        new_orphan_file = self.targets_dir / "new_orphan_map.html"
        existing_file.write_text('{"status": "validated_v1"}', encoding="utf-8")
        expected_hash = sha256_file(existing_file)

        backup_dir = self.tx_root / "std_refresh_crashed" / "western_ghats"
        tx = RegionTransaction(
            region_key="western_ghats",
            tracked_paths=[existing_file, new_orphan_file],
            backup_root=backup_dir,
            run_id="std_refresh_crashed",
        )
        # Simulate crash mid-transaction: enter captures verified snapshots & writes IN_PROGRESS,
        # never calling __exit__.
        tx.__enter__()
        crashed_pid = 999991

        journal_path = backup_dir / TX_JOURNAL_FILENAME
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["pid"] = crashed_pid
        _write_json(journal_path, journal)

        # Mutate targets on disk and leave both owned staging temps and unowned .part/.tmp files
        existing_file.write_text('{"status": "corrupt_partial"}', encoding="utf-8")
        new_orphan_file.write_text("<html>half written</html>", encoding="utf-8")

        owned_tmp = self.targets_dir / f".{existing_file.name}.tmp.{crashed_pid}"
        owned_rollback = self.targets_dir / f"{existing_file.name}.rollback_{crashed_pid}"
        unowned_part = self.targets_dir / "3B-DAY.GPM.IMERG.HDF5.part"
        unowned_tmp = self.targets_dir / "user_notes.tmp"
        owned_tmp.write_text("staged", encoding="utf-8")
        owned_rollback.write_text("rollback_staged", encoding="utf-8")
        unowned_part.write_bytes(b"partial_download_keep_me")
        unowned_tmp.write_text("keep_unowned_tmp", encoding="utf-8")

        summary = recover_interrupted_transactions(self.tx_root)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(len(summary["recovered_transactions"]), 1)
        self.assertEqual(summary["recovered_transactions"][0]["region_key"], "western_ghats")
        self.assertEqual(summary["preserved_uncertain"], [])

        # Restored state + hash verified
        self.assertEqual(existing_file.read_text(encoding="utf-8"), '{"status": "validated_v1"}')
        self.assertEqual(sha256_file(existing_file), expected_hash)
        self.assertFalse(new_orphan_file.exists())

        # Owned temp files removed; unowned .part/.tmp files conservatively kept
        self.assertFalse(owned_tmp.exists())
        self.assertFalse(owned_rollback.exists())
        self.assertTrue(unowned_part.exists())
        self.assertTrue(unowned_tmp.exists())

        # Backup directory cleaned only after verified restore
        self.assertFalse(backup_dir.exists())

    def test_06_completed_transaction_never_restored(self) -> None:
        target = self.targets_dir / "committed_output.json"
        target.write_text('{"version": 1}', encoding="utf-8")
        old_hash = sha256_file(target)

        backup_dir = self.tx_root / "std_refresh_done" / "himalayas"
        backup_dir.mkdir(parents=True, exist_ok=True)
        snap = backup_dir / "00_committed_output.json"
        snap.write_text('{"version": 1}', encoding="utf-8")

        # Simulate newer committed state on disk + COMMITTED journal left before dir cleanup
        target.write_text('{"version": 2_committed}', encoding="utf-8")
        _write_json(
            backup_dir / TX_JOURNAL_FILENAME,
            {
                "schema_version": TX_SCHEMA_VERSION,
                "run_id": "std_refresh_done",
                "region_key": "himalayas",
                "pid": 12345,
                "state": "COMMITTED",
                "snapshots_verified": True,
                "manifest": [
                    {
                        "target_path": str(target),
                        "existed_before": True,
                        "snapshot_file": "00_committed_output.json",
                        "sha256": old_hash,
                    }
                ],
            },
        )

        summary = recover_interrupted_transactions(self.tx_root)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["recovered_transactions"], [])
        self.assertEqual(len(summary["cleaned_completed_backups"]), 1)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"version": 2_committed}')
        self.assertFalse(backup_dir.exists())

    def test_07_preserve_backups_when_recovery_uncertain(self) -> None:
        # Prepare 5 uncertain backup scenarios + matching targets that must NOT be overwritten
        cases = {}

        # (a) Missing journal
        t_no_journal = self.targets_dir / "t_no_journal.txt"
        t_no_journal.write_text("CURRENT_NO_JOURNAL", encoding="utf-8")
        d_no_journal = self.tx_root / "run_a" / "himalayas"
        d_no_journal.mkdir(parents=True, exist_ok=True)
        (d_no_journal / "00_t_no_journal.txt").write_text("OLD_BACKUP", encoding="utf-8")
        cases["no_journal"] = (t_no_journal, d_no_journal, "CURRENT_NO_JOURNAL")

        # (b) Corrupt journal JSON
        t_corrupt = self.targets_dir / "t_corrupt.txt"
        t_corrupt.write_text("CURRENT_CORRUPT", encoding="utf-8")
        d_corrupt = self.tx_root / "run_b" / "northeast"
        d_corrupt.mkdir(parents=True, exist_ok=True)
        (d_corrupt / "00_t_corrupt.txt").write_text("OLD_BACKUP", encoding="utf-8")
        (d_corrupt / TX_JOURNAL_FILENAME).write_text("{not valid json", encoding="utf-8")
        cases["corrupt"] = (t_corrupt, d_corrupt, "CURRENT_CORRUPT")

        # (c) Checksum mismatch on IN_PROGRESS snapshot
        t_mismatch = self.targets_dir / "t_mismatch.txt"
        t_mismatch.write_text("CURRENT_MISMATCH", encoding="utf-8")
        d_mismatch = self.tx_root / "run_c" / "western_ghats"
        d_mismatch.mkdir(parents=True, exist_ok=True)
        (d_mismatch / "00_t_mismatch.txt").write_text("TAMPERED_SNAPSHOT", encoding="utf-8")
        _write_json(
            d_mismatch / TX_JOURNAL_FILENAME,
            {
                "schema_version": TX_SCHEMA_VERSION,
                "run_id": "run_c",
                "region_key": "western_ghats",
                "pid": 22222,
                "state": "IN_PROGRESS",
                "snapshots_verified": True,
                "manifest": [
                    {
                        "target_path": str(t_mismatch),
                        "existed_before": True,
                        "snapshot_file": "00_t_mismatch.txt",
                        "sha256": "0" * 64,
                    }
                ],
            },
        )
        cases["mismatch"] = (t_mismatch, d_mismatch, "CURRENT_MISMATCH")

        # (d) PREPARING state (crash before snapshot capture finished)
        t_preparing = self.targets_dir / "t_preparing.txt"
        t_preparing.write_text("CURRENT_PREPARING", encoding="utf-8")
        d_preparing = self.tx_root / "run_d" / "himalayas"
        d_preparing.mkdir(parents=True, exist_ok=True)
        (d_preparing / "00_t_preparing.txt").write_text("OLD_PREPARING", encoding="utf-8")
        _write_json(
            d_preparing / TX_JOURNAL_FILENAME,
            {
                "schema_version": TX_SCHEMA_VERSION,
                "run_id": "run_d",
                "region_key": "himalayas",
                "pid": 33333,
                "state": "PREPARING",
                "snapshots_verified": False,
                "manifest": [
                    {
                        "target_path": str(t_preparing),
                        "existed_before": True,
                        "snapshot_file": "00_t_preparing.txt",
                        "sha256": sha256_file(d_preparing / "00_t_preparing.txt"),
                    }
                ],
            },
        )
        cases["preparing"] = (t_preparing, d_preparing, "CURRENT_PREPARING")

        # (e) ROLLBACK_FAILED state
        t_rb_failed = self.targets_dir / "t_rb_failed.txt"
        t_rb_failed.write_text("CURRENT_RB_FAILED", encoding="utf-8")
        d_rb_failed = self.tx_root / "run_e" / "northeast"
        d_rb_failed.mkdir(parents=True, exist_ok=True)
        (d_rb_failed / "00_t_rb_failed.txt").write_text("OLD_RB_FAILED", encoding="utf-8")
        _write_json(
            d_rb_failed / TX_JOURNAL_FILENAME,
            {
                "schema_version": TX_SCHEMA_VERSION,
                "run_id": "run_e",
                "region_key": "northeast",
                "pid": 44444,
                "state": "ROLLBACK_FAILED",
                "snapshots_verified": True,
                "manifest": [
                    {
                        "target_path": str(t_rb_failed),
                        "existed_before": True,
                        "snapshot_file": "00_t_rb_failed.txt",
                        "sha256": sha256_file(d_rb_failed / "00_t_rb_failed.txt"),
                    }
                ],
            },
        )
        cases["rb_failed"] = (t_rb_failed, d_rb_failed, "CURRENT_RB_FAILED")

        summary = recover_interrupted_transactions(self.tx_root)
        self.assertEqual(summary["recovered_transactions"], [])
        self.assertEqual(summary["cleaned_completed_backups"], [])
        self.assertGreaterEqual(len(summary["preserved_uncertain"]), 5)

        for label, (target, backup_dir, expected_text) in cases.items():
            self.assertEqual(target.read_text(encoding="utf-8"), expected_text, msg=f"{label} target altered")
            self.assertTrue(backup_dir.exists(), msg=f"{label} backup dir was deleted")

    def test_08_run_standard_refresh_triggers_crash_recovery_on_startup(self) -> None:
        # Leave an interrupted IN_PROGRESS transaction under tx_root and run dry_run refresh
        target = self.targets_dir / "recovered_by_startup.json"
        target.write_text('{"clean": true}', encoding="utf-8")
        backup_dir = self.tx_root / "std_refresh_aborted" / "himalayas"
        tx = RegionTransaction(
            region_key="himalayas",
            tracked_paths=[target],
            backup_root=backup_dir,
            run_id="std_refresh_aborted",
        )
        tx.__enter__()
        target.write_text('{"corrupt_partial": true}', encoding="utf-8")

        report = run_standard_refresh(
            regions=["himalayas"],
            dry_run=True,
            offline=True,
            verify_model_load=False,
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
        )
        # Even if offline dry-run succeeds or fails discovery, startup recovery already ran under lock
        self.assertEqual(target.read_text(encoding="utf-8"), '{"clean": true}')
        self.assertFalse(backup_dir.exists())
        recovery = report["validation_and_test_results"]["transaction_recovery"]
        self.assertTrue(recovery["scanned"])
        self.assertEqual(len(recovery["recovered_transactions"]), 1)


if __name__ == "__main__":
    unittest.main()
