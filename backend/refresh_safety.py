"""
Process-wide Locking and Transaction-Aware Crash Recovery for Standard Refresh.

Stage 1 Safe Refresh Execution utilities:
- Non-blocking cross-process OS file lock (msvcrt on Windows, fcntl fallback on POSIX)
  shared identically by manual CLI runs and future scheduled refresh triggers.
- Checksummed transaction journal (tx_journal.json) tracking PREPARING -> IN_PROGRESS ->
  COMMITTED / ROLLING_BACK -> ROLLED_BACK / ROLLBACK_FAILED state transitions.
- Conservative startup crash recovery that restores only positively identified interrupted
  transactions, never restores completed transactions, and preserves backups whenever
  recovery or ownership is uncertain.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - POSIX fallback
    import fcntl

from backend.gis_refresh_utils import atomic_write_json


TX_JOURNAL_FILENAME = "tx_journal.json"
TX_SCHEMA_VERSION = "refresh_tx_journal_v1"
DEFAULT_LOCK_FILENAME = ".standard_refresh.lock"
SKIPPED_RUNS_LOG_FILENAME = "skipped_runs.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


_LOCK_META_WIDTH = 4096
_LOCK_BYTE_OFFSET = 8192


def _lock_fd_nonblocking(fd: int) -> None:
    """Acquire non-blocking exclusive OS lock on sentinel byte offset 8192."""
    if os.name == "nt":
        os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - POSIX fallback
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    """Release OS lock cleanly."""
    if os.name == "nt":
        os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - POSIX fallback
        fcntl.flock(fd, fcntl.LOCK_UN)


def _write_lock_metadata(fd: int, payload: Dict[str, Any]) -> None:
    """Write fixed-width JSON metadata in bytes 0..4095 outside locked byte 8192."""
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    if len(encoded) >= _LOCK_META_WIDTH:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) >= _LOCK_META_WIDTH:
        encoded = encoded[: _LOCK_META_WIDTH - 1]
    padded = encoded + b" " * ((_LOCK_META_WIDTH - 1) - len(encoded)) + b"\n"
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, padded)
    try:
        os.fsync(fd)
    except OSError:
        pass


def _read_lock_metadata_fd(fd: int) -> Optional[Dict[str, Any]]:
    """Read lock holder metadata directly from bytes 0..4095 without unbuffered readahead past 4096."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, _LOCK_META_WIDTH)
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class RefreshProcessLock:
    """
    Cross-process non-blocking exclusive OS file lock for Standard Refresh runs.

    Uses Windows msvcrt.locking (or POSIX fcntl.flock) so the OS automatically
    releases the lock if the owning process terminates unexpectedly or crashes.
    """

    def __init__(self, lock_path: Path, run_id: Optional[str] = None):
        self.lock_path = Path(lock_path)
        self.run_id = run_id or f"pid_{os.getpid()}"
        self._fd: Optional[int] = None
        self.acquired: bool = False
        self.holder_info: Optional[Dict[str, Any]] = None

    def acquire(self) -> bool:
        """Attempt non-blocking exclusive lock acquisition. Returns False if held elsewhere."""
        if self.acquired and self._fd is not None:
            return True

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        fd = os.open(str(self.lock_path), flags, 0o666)
        try:
            _lock_fd_nonblocking(fd)
        except OSError:
            self.holder_info = _read_lock_metadata_fd(fd)
            try:
                os.close(fd)
            except OSError:
                pass
            self.acquired = False
            self._fd = None
            return False

        try:
            _write_lock_metadata(
                fd,
                {
                    "state": "LOCKED",
                    "pid": os.getpid(),
                    "run_id": self.run_id,
                    "acquired_utc": _utc_now_iso(),
                },
            )
        except Exception:
            try:
                _unlock_fd(fd)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            raise

        self._fd = fd
        self.acquired = True
        self.holder_info = None
        return True

    def release(self) -> None:
        """Release the OS lock and record RELEASED state while leaving lockfile inode intact."""
        fd = self._fd
        if fd is None:
            self.acquired = False
            return

        self._fd = None
        self.acquired = False
        try:
            try:
                _write_lock_metadata(
                    fd,
                    {
                        "state": "RELEASED",
                        "pid": os.getpid(),
                        "run_id": self.run_id,
                        "released_utc": _utc_now_iso(),
                    },
                )
            except OSError:
                pass
            try:
                _unlock_fd(fd)
            except OSError:
                pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "RefreshProcessLock":
        if not self.acquire():
            raise BlockingIOError(f"Standard Refresh lock already held: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False


def record_skipped_execution(
    report_dir: Path,
    run_id: str,
    execution_dt: datetime,
    lock_path: Path,
    lock_holder: Optional[Dict[str, Any]],
) -> Path:
    """
    Append an audit entry when a refresh execution is skipped due to an active lock.
    Never overwrites latest_refresh_report.json or latest_refresh_report.md.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / SKIPPED_RUNS_LOG_FILENAME
    entry = {
        "schema_version": "standard_refresh_skip_v1",
        "run_id": run_id,
        "execution_timestamp_utc": execution_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "SKIPPED_LOCKED",
        "reason": "Another Standard Refresh process is currently active.",
        "pid": os.getpid(),
        "lock_path": str(lock_path),
        "lock_holder": lock_holder or {},
    }
    line = (json.dumps(entry) + "\n").encode("utf-8")
    fd = os.open(str(log_path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o666)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return log_path


def _journal_path(backup_root: Path) -> Path:
    return backup_root / TX_JOURNAL_FILENAME


def _read_journal(backup_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read and validate tx_journal.json schema; returns (journal, error_reason)."""
    jpath = _journal_path(backup_root)
    if not jpath.exists() or not jpath.is_file():
        return None, "missing_journal"
    try:
        data = json.loads(jpath.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"corrupt_journal:{exc}"
    if not isinstance(data, dict):
        return None, "invalid_journal_type"
    if data.get("schema_version") != TX_SCHEMA_VERSION:
        return None, f"unsupported_schema:{data.get('schema_version')!r}"
    if not isinstance(data.get("manifest"), list):
        return None, "invalid_manifest"
    state = data.get("state")
    if not isinstance(state, str) or not state:
        return None, "missing_state"
    return data, None


def _verify_manifest_snapshots(backup_root: Path, manifest: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Verify every backed-up file exists and matches its recorded SHA-256 digest."""
    problems: List[str] = []
    for item in manifest:
        if not isinstance(item, dict) or "target_path" not in item or "existed_before" not in item:
            problems.append(f"malformed_manifest_entry:{item!r}")
            continue
        if item["existed_before"]:
            snap_name = item.get("snapshot_file")
            expected_hash = item.get("sha256")
            if not snap_name or not expected_hash:
                problems.append(f"missing_snapshot_metadata:{item.get('target_path')}")
                continue
            snap_path = backup_root / str(snap_name)
            if not snap_path.exists() or not snap_path.is_file():
                problems.append(f"missing_snapshot_file:{snap_path}")
                continue
            actual_hash = sha256_file(snap_path)
            if actual_hash != expected_hash:
                problems.append(f"snapshot_checksum_mismatch:{snap_path.name}")
    return len(problems) == 0, problems


def _cleanup_owned_temp_files(manifest: List[Dict[str, Any]], owner_pid: Optional[int]) -> List[str]:
    """
    Remove orphaned temporary staging files ONLY when ownership by the interrupted
    transaction's PID and tracked target path is positively established.
    """
    cleaned: List[str] = []
    if not isinstance(owner_pid, int) or owner_pid <= 0:
        return cleaned
    for item in manifest:
        target = Path(item["target_path"])
        candidates = [
            target.with_name(f"{target.name}.rollback_{owner_pid}"),
            target.with_name(f".{target.name}.rollback_{owner_pid}"),
            target.with_name(f".{target.name}.tmp.{owner_pid}"),
            target.with_name(f"{target.name}.tmp_{owner_pid}"),
        ]
        for cand in candidates:
            try:
                if cand.exists() and cand.is_file():
                    cand.unlink()
                    cleaned.append(str(cand))
            except OSError:
                pass
    return cleaned


def _restore_and_verify_transaction(
    backup_root: Path,
    journal: Dict[str, Any],
    cleanup_after_verify: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """
    Restore tracked targets from verified snapshots, verify restored state, and clean up
    the backup directory ONLY when full recovery verification succeeds.

    Returns (verified_ok, errors, cleaned_temp_files).
    """
    jpath = _journal_path(backup_root)
    manifest: List[Dict[str, Any]] = list(journal["manifest"])
    errors: List[str] = []

    if journal.get("snapshots_verified") is not True:
        return False, ["snapshots_not_verified"], []

    snaps_ok, snap_problems = _verify_manifest_snapshots(backup_root, manifest)
    if not snaps_ok:
        journal["state"] = "ROLLBACK_FAILED"
        journal["rollback_error"] = "; ".join(snap_problems)
        atomic_write_json(jpath, journal)
        return False, snap_problems, []

    journal["state"] = "ROLLING_BACK"
    journal["rollback_started_utc"] = _utc_now_iso()
    atomic_write_json(jpath, journal)

    for item in manifest:
        target = Path(item["target_path"])
        try:
            if item["existed_before"]:
                snap_path = backup_root / str(item["snapshot_file"])
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp_restore = target.with_name(f"{target.name}.rollback_{os.getpid()}")
                shutil.copy2(snap_path, tmp_restore)
                os.replace(tmp_restore, target)
            elif target.exists():
                target.unlink()
        except Exception as exc:
            errors.append(f"restore_failed:{target}:{exc}")

    # Verify restored targets match original hashes or absence
    for item in manifest:
        target = Path(item["target_path"])
        if item["existed_before"]:
            if not target.exists():
                errors.append(f"verify_missing_after_restore:{target}")
            elif sha256_file(target) != item["sha256"]:
                errors.append(f"verify_checksum_mismatch:{target}")
        else:
            if target.exists():
                errors.append(f"verify_orphan_still_exists:{target}")

    cleaned_temps = _cleanup_owned_temp_files(manifest, journal.get("pid"))

    if errors:
        journal["state"] = "ROLLBACK_FAILED"
        journal["rollback_error"] = "; ".join(errors)
        atomic_write_json(jpath, journal)
        return False, errors, cleaned_temps

    journal["state"] = "ROLLED_BACK"
    journal["rolled_back_utc"] = _utc_now_iso()
    atomic_write_json(jpath, journal)

    if cleanup_after_verify:
        shutil.rmtree(backup_root, ignore_errors=False)

    return True, [], cleaned_temps


class RegionTransaction:
    """
    Journaled transactional backup and rollback guard for a region's mutable outputs.

    Captures SHA-256-verified snapshots and writes tx_journal.json BEFORE mutating any
    output file. On normal exit marks state=COMMITTED before cleanup. On exception or
    later crash recovery restores pre-existing files atomically and removes only verified
    backups or newly created orphan outputs.
    """

    def __init__(
        self,
        region_key: str,
        tracked_paths: Sequence[Path],
        backup_root: Path,
        run_id: Optional[str] = None,
    ):
        self.region_key = region_key
        self.tracked_paths = list(dict.fromkeys(Path(p) for p in tracked_paths))
        self.backup_root = Path(backup_root)
        self.run_id = run_id or self.backup_root.parent.name
        self._journal: Optional[Dict[str, Any]] = None
        self.rolled_back: bool = False
        self.rollback_errors: List[str] = []

    def __enter__(self) -> "RegionTransaction":
        self.backup_root.mkdir(parents=True, exist_ok=True)
        jpath = _journal_path(self.backup_root)
        self._journal = {
            "schema_version": TX_SCHEMA_VERSION,
            "run_id": self.run_id,
            "region_key": self.region_key,
            "pid": os.getpid(),
            "started_utc": _utc_now_iso(),
            "state": "PREPARING",
            "snapshots_verified": False,
            "manifest": [],
        }
        atomic_write_json(jpath, self._journal)

        manifest: List[Dict[str, Any]] = []
        for idx, target in enumerate(self.tracked_paths):
            if target.exists() and target.is_file():
                src_hash = sha256_file(target)
                snap_name = f"{idx:02d}_{target.name}"
                snap_path = self.backup_root / snap_name
                shutil.copy2(target, snap_path)
                if sha256_file(snap_path) != src_hash:
                    raise RuntimeError(f"Backup verification failed for {target} -> {snap_path}")
                manifest.append({
                    "target_path": str(target),
                    "existed_before": True,
                    "snapshot_file": snap_name,
                    "size_bytes": target.stat().st_size,
                    "sha256": src_hash,
                })
            else:
                manifest.append({
                    "target_path": str(target),
                    "existed_before": False,
                    "snapshot_file": None,
                    "size_bytes": None,
                    "sha256": None,
                })

        self._journal["manifest"] = manifest
        self._journal["snapshots_verified"] = True
        self._journal["state"] = "IN_PROGRESS"
        atomic_write_json(jpath, self._journal)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            if self._journal and self._journal.get("state") == "IN_PROGRESS":
                self.rollback()
            elif self._journal:
                self._journal["state"] = "ABORTED_BEFORE_EXECUTION"
                atomic_write_json(_journal_path(self.backup_root), self._journal)
                self.cleanup()
        else:
            if self._journal is not None:
                self._journal["state"] = "COMMITTED"
                self._journal["completed_utc"] = _utc_now_iso()
                atomic_write_json(_journal_path(self.backup_root), self._journal)
            self.cleanup()
        return False

    def cleanup(self) -> None:
        """Remove temporary transaction snapshots only when completion or rollback is verified."""
        try:
            if self.backup_root.exists():
                jpath = _journal_path(self.backup_root)
                if jpath.exists():
                    journal, err = _read_journal(self.backup_root)
                    if err or not journal or journal.get("state") not in (
                        "COMMITTED",
                        "ROLLED_BACK",
                        "ABORTED_BEFORE_EXECUTION",
                    ):
                        return
                shutil.rmtree(self.backup_root, ignore_errors=True)
        except OSError:
            pass

    def rollback(self) -> None:
        """Restore pre-existing files atomically, verify checksums, and keep backups if uncertain."""
        if self._journal is None:
            journal, err = _read_journal(self.backup_root)
            if err or journal is None:
                self.rollback_errors.append(f"Cannot rollback without valid journal: {err}")
                return
            self._journal = journal

        ok, errs, _ = _restore_and_verify_transaction(
            backup_root=self.backup_root,
            journal=self._journal,
            cleanup_after_verify=True,
        )
        self.rolled_back = ok
        self.rollback_errors.extend(errs)
        for e in errs:
            print(f"   [ROLLBACK WARNING] {e}")


def _discover_tx_dirs(tx_root_parent: Path) -> Tuple[List[Path], List[Path]]:
    """
    Collect candidate transaction directories under tx_root_parent.
    Supports both <root>/<run_id>/<region_key> and direct <root>/<tx_dir>.
    Returns (tx_dirs_to_inspect, unjournaled_parent_dirs).
    """
    tx_dirs: List[Path] = []
    uncertain_dirs: List[Path] = []
    if not tx_root_parent.exists() or not tx_root_parent.is_dir():
        return tx_dirs, uncertain_dirs

    for child in sorted(tx_root_parent.iterdir()):
        if not child.is_dir():
            uncertain_dirs.append(child)
            continue
        if _journal_path(child).exists():
            tx_dirs.append(child)
            continue
        subdirs = [p for p in sorted(child.iterdir()) if p.is_dir()]
        loose_files = [p for p in sorted(child.iterdir()) if p.is_file()]
        if subdirs:
            tx_dirs.extend(subdirs)
        if loose_files:
            uncertain_dirs.append(child)
        elif not subdirs:
            # Empty run directory left over after cleanup
            uncertain_dirs.append(child)
    return tx_dirs, uncertain_dirs


def recover_interrupted_transactions(tx_root_parent: Path) -> Dict[str, Any]:
    """
    Inspect orphaned transaction backup directories while holding the process lock:
    - IN_PROGRESS / ROLLING_BACK with verified journal & snapshot checksums -> restore & verify, then clean.
    - COMMITTED / ROLLED_BACK -> never restore; clean leftover verified backup dir cleanly.
    - PREPARING / unknown / corrupt / missing journal / checksum mismatch -> preserve untouched.
    """
    summary: Dict[str, Any] = {
        "scanned": False,
        "recovered_transactions": [],
        "cleaned_completed_backups": [],
        "cleaned_temp_files": [],
        "preserved_uncertain": [],
        "errors": [],
    }
    tx_root_parent = Path(tx_root_parent)
    if not tx_root_parent.exists():
        return summary

    summary["scanned"] = True
    tx_dirs, loose_uncertain = _discover_tx_dirs(tx_root_parent)

    for item in loose_uncertain:
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
            except OSError:
                pass
        else:
            summary["preserved_uncertain"].append({"path": str(item), "reason": "unjournaled_or_non_dir"})

    for tx_dir in tx_dirs:
        journal, err = _read_journal(tx_dir)
        if err or journal is None:
            summary["preserved_uncertain"].append({"path": str(tx_dir), "reason": err or "invalid_journal"})
            continue

        state = journal.get("state")
        if state in ("COMMITTED", "ROLLED_BACK", "ABORTED_BEFORE_EXECUTION"):
            # Completion or rollback already positively verified — never restore
            try:
                shutil.rmtree(tx_dir, ignore_errors=False)
                summary["cleaned_completed_backups"].append({"path": str(tx_dir), "state": state})
            except OSError as exc:
                summary["errors"].append(f"Could not clean verified backup dir {tx_dir}: {exc}")
            continue

        if state in ("IN_PROGRESS", "ROLLING_BACK"):
            ok, errs, cleaned_temps = _restore_and_verify_transaction(
                backup_root=tx_dir,
                journal=journal,
                cleanup_after_verify=True,
            )
            summary["cleaned_temp_files"].extend(cleaned_temps)
            if ok:
                summary["recovered_transactions"].append({
                    "path": str(tx_dir),
                    "run_id": journal.get("run_id"),
                    "region_key": journal.get("region_key"),
                    "files_count": len(journal.get("manifest", [])),
                })
            else:
                summary["preserved_uncertain"].append({
                    "path": str(tx_dir),
                    "reason": f"recovery_verification_failed:{';'.join(errs)}",
                })
                summary["errors"].extend(errs)
            continue

        # PREPARING, ROLLBACK_FAILED, or unknown state -> uncertain, preserve backups untouched
        summary["preserved_uncertain"].append({"path": str(tx_dir), "reason": f"uncertain_state:{state}"})

    # Clean only now-empty run directories; leave non-empty uncertain dirs intact
    for run_dir in sorted(tx_root_parent.iterdir()):
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            try:
                run_dir.rmdir()
            except OSError:
                pass

    return summary
