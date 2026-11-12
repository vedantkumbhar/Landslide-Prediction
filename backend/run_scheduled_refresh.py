"""
Scheduled entrypoint for intelligent NASA GPM IMERG V07C Standard Refresh.

Performs Step 1 preflight and read-only CMR/local discovery under the shared
Standard Refresh process lock, enforces 48/48 granule completeness and the
12-hour freshness guard, skips regional processing only when the previous
published refresh succeeded identically across all three regions and all
required outputs exist non-empty, and otherwise delegates to
``backend.run_standard_refresh.run_standard_refresh``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.gis_refresh_utils import parse_iso8601_utc
from backend.refresh_safety import (
    DEFAULT_LOCK_FILENAME,
    RefreshProcessLock,
    record_skipped_execution,
)
from backend.run_standard_refresh import (
    DEFAULT_MAX_LATENCY_HOURS,
    DEFAULT_REPORT_DIR,
    EXPECTED_GRANULES_24H,
    REGION_ORDER,
    _build_region_specs,
    check_preflight,
    discover_and_retrieve_gpm,
    normalize_regions,
    run_standard_refresh,
)

SCHEDULER_LOG_FILENAME = "scheduler.log"
SCHEDULER_LOG_MAX_BYTES = 5 * 1024 * 1024
SCHEDULER_LOG_BACKUP_COUNT = 5
DEFAULT_SCHEDULER_LOG_FILE = DEFAULT_REPORT_DIR / SCHEDULER_LOG_FILENAME


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _UTCFormatter(logging.Formatter):
    """Formatter emitting UTC timestamps independent of host local timezone."""

    converter = time.gmtime


def build_scheduler_logger(
    log_path: Path,
    run_id: str,
) -> Tuple[logging.Logger, RotatingFileHandler]:
    """
    Create a per-run logger wired to a 5 MB x 5-backup RotatingFileHandler in UTC.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"landslide.scheduler.{run_id}.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=SCHEDULER_LOG_MAX_BYTES,
        backupCount=SCHEDULER_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        _UTCFormatter(
            fmt="%(asctime)sZ | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger, handler


def _close_scheduler_logger(
    logger: logging.Logger,
    handler: RotatingFileHandler,
) -> None:
    """Flush and release file handles so Windows processes never retain open log locks."""
    try:
        handler.flush()
    except Exception:
        pass
    try:
        logger.removeHandler(handler)
    finally:
        handler.close()


def _validate_discovery_gate(
    discovery: Optional[Dict[str, Any]],
    max_latency_hours: float,
) -> Tuple[bool, str]:
    """
    Verify 48/48 granule completeness, 24h window sanity, and 12-hour freshness guard.
    """
    if not isinstance(discovery, dict):
        return False, "Discovery returned empty or invalid payload."

    obs_start_dt = parse_iso8601_utc(discovery.get("observation_start_utc"))
    obs_end_dt = parse_iso8601_utc(discovery.get("observation_end_utc"))
    if obs_start_dt is None or obs_end_dt is None:
        return False, "Discovered observation window lacks explicit UTC timestamps."
    if obs_end_dt - obs_start_dt != timedelta(hours=24):
        return False, (
            f"Discovered window is not 24 hours: "
            f"{discovery.get('observation_start_utc')} -> {discovery.get('observation_end_utc')}."
        )

    avail = int(discovery.get("granules_validated") or 0)
    missing = int(discovery.get("missing_intervals_count") or 0)
    if not discovery.get("is_24h_complete") or avail != EXPECTED_GRANULES_24H or missing != 0:
        return False, (
            f"Incomplete GPM 24h block: granules={avail}/{EXPECTED_GRANULES_24H}, "
            f"missing_intervals={missing}, status={discovery.get('freshness_status')}."
        )

    latency_val = discovery.get("latency_hours")
    try:
        latency_num = float(latency_val) if latency_val is not None else float("nan")
    except (TypeError, ValueError):
        latency_num = float("nan")

    if (
        not discovery.get("is_fresh")
        or discovery.get("freshness_status") != "FRESH"
        or not math.isfinite(latency_num)
        or latency_num < 0.0
        or latency_num > max_latency_hours
    ):
        return False, (
            f"Freshness guard rejected data: status={discovery.get('freshness_status')}, "
            f"latency={latency_val}h (limit {max_latency_hours}h), "
            f"reason={discovery.get('freshness_reason')}."
        )

    return True, "FRESH_COMPLETE"


def _validate_required_region_outputs(
    region_specs: Dict[str, Dict[str, Any]],
    expected_obs_end_dt: datetime,
) -> Tuple[bool, str]:
    """
    Ensure every region's tracked outputs exist and are non-empty, and verify any
    on-disk observation metadata agrees with the expected window end timestamp.
    """
    for rkey in REGION_ORDER:
        spec = region_specs.get(rkey)
        if not isinstance(spec, dict):
            return False, f"Missing region specification for '{rkey}'."

        tracked = spec.get("tracked_outputs") or []
        if not tracked:
            return False, f"No tracked_outputs configured for region '{rkey}'."

        checked: set = set()
        for raw_path in tracked:
            path = Path(raw_path)
            if path in checked:
                continue
            checked.add(path)
            if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                return False, f"Required output for '{rkey}' is missing or empty: {path}"

        for meta_key in ("risk_metadata_json", "map_metadata_json"):
            meta_raw = spec.get(meta_key)
            if not meta_raw:
                continue
            meta_path = Path(meta_raw)
            if not meta_path.exists() or not meta_path.is_file():
                continue
            try:
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(meta_payload, dict):
                gpm_meta = meta_payload.get("gpm_observation_metadata")
                if isinstance(gpm_meta, dict) and gpm_meta.get("observation_end_utc"):
                    disk_end_dt = parse_iso8601_utc(gpm_meta.get("observation_end_utc"))
                    if disk_end_dt is None or disk_end_dt != expected_obs_end_dt:
                        return False, (
                            f"On-disk metadata {meta_path.name} for '{rkey}' has "
                            f"observation_end_utc={gpm_meta.get('observation_end_utc')!r}, "
                            f"expected {_utc_iso(expected_obs_end_dt)!r}."
                        )

    return True, "All required outputs across all three regions exist and are non-empty."


def evaluate_unchanged_window_skip(
    report_dir: Path,
    discovery: Optional[Dict[str, Any]],
    region_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    max_latency_hours: float = DEFAULT_MAX_LATENCY_HOURS,
    force: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Determine whether regional processing can be safely skipped because the
    discovered GPM window is already cleanly published.

    Skips ONLY when ALL conditions hold:
    - ``force`` is False.
    - Discovery satisfies 48/48 completeness and the 12-hour freshness guard.
    - ``latest_refresh_report.json`` exists, parses cleanly, and has status == 'SUCCESS'.
    - Top-level ``observation_window`` matches the discovered UTC window.
    - All three regions ('himalayas', 'northeast', 'western_ghats') report
      status == 'SUCCESS', validation_passed == True, and identical UTC windows.
    - All required outputs across all three regions exist and are non-empty.
    """
    if force:
        return False, "Force refresh requested (--force).", {}

    gate_ok, gate_reason = _validate_discovery_gate(discovery, max_latency_hours=max_latency_hours)
    if not gate_ok or discovery is None:
        return False, gate_reason, {}

    disc_start_dt = parse_iso8601_utc(discovery.get("observation_start_utc"))
    disc_end_dt = parse_iso8601_utc(discovery.get("observation_end_utc"))
    assert disc_start_dt is not None and disc_end_dt is not None

    latest_path = Path(report_dir) / "latest_refresh_report.json"
    if not latest_path.exists() or not latest_path.is_file() or latest_path.stat().st_size <= 0:
        return False, "No existing latest_refresh_report.json found.", {}

    try:
        prev_report = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Unable to parse latest_refresh_report.json: {exc}", {}

    if not isinstance(prev_report, dict):
        return False, "latest_refresh_report.json root is not a JSON object.", {}

    prev_status = prev_report.get("status")
    if prev_status != "SUCCESS":
        return False, f"Previous refresh status is {prev_status!r} (requires 'SUCCESS').", {}

    prev_win = prev_report.get("observation_window")
    if not isinstance(prev_win, dict):
        return False, "Previous report missing observation_window metadata.", {}

    prev_start_dt = parse_iso8601_utc(prev_win.get("observation_start_utc"))
    prev_end_dt = parse_iso8601_utc(prev_win.get("observation_end_utc"))
    if prev_start_dt is None or prev_end_dt is None:
        return False, "Previous report observation_window lacks explicit UTC timestamps.", {}

    if prev_start_dt != disc_start_dt or prev_end_dt != disc_end_dt:
        return False, (
            f"New observation window discovered ({_utc_iso(disc_start_dt)} -> {_utc_iso(disc_end_dt)} "
            f"vs published {_utc_iso(prev_start_dt)} -> {_utc_iso(prev_end_dt)})."
        ), {}

    val_section = prev_report.get("validation_and_test_results")
    if isinstance(val_section, dict) and val_section.get("skipped_regions"):
        return False, f"Previous run skipped regions: {val_section.get('skipped_regions')}.", {}

    prev_regions = prev_report.get("regions")
    if not isinstance(prev_regions, dict):
        return False, "Previous report missing 'regions' section.", {}

    for rkey in REGION_ORDER:
        rdata = prev_regions.get(rkey)
        if not isinstance(rdata, dict):
            return False, f"Previous report missing region '{rkey}'.", {}
        if rdata.get("status") != "SUCCESS" or rdata.get("validation_passed") is not True:
            return False, (
                f"Previous report region '{rkey}' not fully validated "
                f"(status={rdata.get('status')!r}, validation_passed={rdata.get('validation_passed')!r})."
            ), {}
        reg_start_dt = parse_iso8601_utc(rdata.get("observation_start_utc"))
        reg_end_dt = parse_iso8601_utc(rdata.get("observation_end_utc"))
        if reg_start_dt is None or reg_end_dt is None:
            return False, f"Region '{rkey}' report lacks explicit UTC observation window.", {}
        if reg_start_dt != disc_start_dt or reg_end_dt != disc_end_dt:
            return False, (
                f"Inconsistent regional observation window for '{rkey}': "
                f"{rdata.get('observation_start_utc')} -> {rdata.get('observation_end_utc')} "
                f"vs discovered {_utc_iso(disc_start_dt)} -> {_utc_iso(disc_end_dt)}."
            ), {}

    specs = region_specs if region_specs is not None else _build_region_specs()
    outputs_ok, outputs_reason = _validate_required_region_outputs(specs, disc_end_dt)
    if not outputs_ok:
        return False, outputs_reason, {}

    return True, (
        f"Unchanged observation window {_utc_iso(disc_start_dt)} -> {_utc_iso(disc_end_dt)} "
        f"already published by run_id={prev_report.get('run_id')} with valid outputs across all 3 regions."
    ), {
        "previous_run_id": prev_report.get("run_id"),
        "observation_start_utc": _utc_iso(disc_start_dt),
        "observation_end_utc": _utc_iso(disc_end_dt),
    }


def run_scheduled_refresh(
    regions: Optional[Sequence[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    offline: bool = False,
    target_end_time: Optional[datetime] = None,
    max_latency_hours: float = DEFAULT_MAX_LATENCY_HOURS,
    workers: int = 4,
    skip_tests: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
    log_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
    now_utc: Optional[datetime] = None,
    env_file: Optional[Path] = PROJECT_ROOT / ".env",
    verify_model_load: bool = True,
    tx_backup_root: Optional[Path] = None,
    region_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    preflight_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    discovery_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    refresh_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    region_executors: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
    region_validators: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
    test_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Execute an intelligent scheduled GPM refresh check and run full processing only
    when needed.

    Preserves Stage 1 locking and crash-recovery semantics by checking the shared
    OS process lock first, performing preflight + discovery under the lock, and
    delegating to ``run_standard_refresh`` when processing is required.
    """
    if max_latency_hours <= 0 or max_latency_hours > DEFAULT_MAX_LATENCY_HOURS:
        raise ValueError(
            f"max_latency_hours must be within (0, {DEFAULT_MAX_LATENCY_HOURS}]; "
            f"got {max_latency_hours} (freshness guard cannot be weakened)."
        )

    exec_now = now_utc or datetime.now(timezone.utc)
    run_id = f"sched_refresh_{exec_now.strftime('%Y%m%dT%H%M%SZ')}"
    selected_regions = normalize_regions(regions)
    report_dir = Path(report_dir)
    resolved_log_path = Path(log_path) if log_path is not None else (report_dir / SCHEDULER_LOG_FILENAME)
    resolved_lock_path = Path(lock_path) if lock_path is not None else (report_dir / DEFAULT_LOCK_FILENAME)
    specs = region_specs if region_specs is not None else _build_region_specs()

    logger, handler = build_scheduler_logger(resolved_log_path, run_id=run_id)
    pre_lock = RefreshProcessLock(resolved_lock_path, run_id=run_id)
    try:
        if not pre_lock.acquire():
            skipped_log = record_skipped_execution(
                report_dir=report_dir,
                run_id=run_id,
                execution_dt=exec_now,
                lock_path=resolved_lock_path,
                lock_holder=pre_lock.holder_info,
            )
            logger.warning(
                "run_id=%s outcome=SKIPPED_LOCKED lock_path=%s holder=%s skipped_log=%s",
                run_id,
                resolved_lock_path,
                json.dumps(pre_lock.holder_info or {}),
                skipped_log,
            )
            return {
                "schema_version": "scheduled_refresh_result_v1",
                "run_id": run_id,
                "execution_timestamp_utc": _utc_iso(exec_now),
                "status": "SKIPPED_LOCKED",
                "reason": "Another refresh process already holds the shared lock.",
                "lock_path": str(resolved_lock_path),
                "lock_holder": pre_lock.holder_info,
                "skipped_runs_log": str(skipped_log),
                "log_path": str(resolved_log_path),
            }

        try:
            preflight_call = preflight_fn or check_preflight
            preflight = preflight_call(
                regions=selected_regions,
                region_specs=specs,
                offline=offline,
                report_dir=report_dir,
                env_file=env_file,
                verify_model_load=verify_model_load,
            )
            if preflight.get("status") != "PASSED":
                errs = list(preflight.get("errors") or ["Preflight failed."])
                logger.error(
                    "run_id=%s outcome=FAILED phase=preflight errors=%s",
                    run_id,
                    json.dumps(errs),
                )
                return {
                    "schema_version": "scheduled_refresh_result_v1",
                    "run_id": run_id,
                    "execution_timestamp_utc": _utc_iso(exec_now),
                    "status": "FAILED",
                    "phase": "preflight",
                    "preflight": preflight,
                    "errors": errs,
                    "log_path": str(resolved_log_path),
                }

            discovery_call = discovery_fn or discover_and_retrieve_gpm
            try:
                discovery = discovery_call(
                    target_end_time=target_end_time,
                    offline=offline,
                    workers=workers,
                    now_utc=exec_now,
                    max_latency_hours=max_latency_hours,
                    dry_run=True,
                )
            except Exception as exc:
                logger.exception(
                    "run_id=%s outcome=FAILED phase=discovery error=%s",
                    run_id,
                    exc,
                )
                return {
                    "schema_version": "scheduled_refresh_result_v1",
                    "run_id": run_id,
                    "execution_timestamp_utc": _utc_iso(exec_now),
                    "status": "FAILED",
                    "phase": "discovery",
                    "errors": [f"GPM discovery failed: {exc}"],
                    "log_path": str(resolved_log_path),
                }

            gate_ok, gate_reason = _validate_discovery_gate(
                discovery,
                max_latency_hours=max_latency_hours,
            )
            if not gate_ok:
                logger.error(
                    "run_id=%s outcome=FAILED phase=freshness_completeness_gate reason=%s",
                    run_id,
                    gate_reason,
                )
                return {
                    "schema_version": "scheduled_refresh_result_v1",
                    "run_id": run_id,
                    "execution_timestamp_utc": _utc_iso(exec_now),
                    "status": "FAILED",
                    "phase": "freshness_completeness_gate",
                    "discovery": discovery,
                    "errors": [gate_reason],
                    "log_path": str(resolved_log_path),
                }

            should_skip, decision_reason, skip_details = evaluate_unchanged_window_skip(
                report_dir=report_dir,
                discovery=discovery,
                region_specs=specs,
                max_latency_hours=max_latency_hours,
                force=force,
            )
            if should_skip:
                logger.info(
                    "run_id=%s outcome=SKIPPED_UNCHANGED obs_start=%s obs_end=%s latency_hours=%s reason=%s",
                    run_id,
                    discovery.get("observation_start_utc"),
                    discovery.get("observation_end_utc"),
                    discovery.get("latency_hours"),
                    decision_reason,
                )
                return {
                    "schema_version": "scheduled_refresh_result_v1",
                    "run_id": run_id,
                    "execution_timestamp_utc": _utc_iso(exec_now),
                    "status": "SKIPPED_UNCHANGED",
                    "reason": decision_reason,
                    "observation_window": {
                        "observation_start_utc": discovery.get("observation_start_utc"),
                        "observation_end_utc": discovery.get("observation_end_utc"),
                    },
                    "previous_run_id": skip_details.get("previous_run_id"),
                    "discovery": discovery,
                    "log_path": str(resolved_log_path),
                }

            if dry_run:
                logger.info(
                    "run_id=%s outcome=DRY_RUN action=WOULD_REFRESH obs_start=%s obs_end=%s reason=%s",
                    run_id,
                    discovery.get("observation_start_utc"),
                    discovery.get("observation_end_utc"),
                    decision_reason,
                )
                return {
                    "schema_version": "scheduled_refresh_result_v1",
                    "run_id": run_id,
                    "execution_timestamp_utc": _utc_iso(exec_now),
                    "status": "DRY_RUN",
                    "reason": decision_reason,
                    "observation_window": {
                        "observation_start_utc": discovery.get("observation_start_utc"),
                        "observation_end_utc": discovery.get("observation_end_utc"),
                    },
                    "discovery": discovery,
                    "log_path": str(resolved_log_path),
                }

            logger.info(
                "run_id=%s action=TRIGGER_REFRESH force=%s obs_start=%s obs_end=%s latency_hours=%s reason=%s",
                run_id,
                force,
                discovery.get("observation_start_utc"),
                discovery.get("observation_end_utc"),
                discovery.get("latency_hours"),
                decision_reason,
            )
        finally:
            pre_lock.release()

        runner = refresh_runner or run_standard_refresh
        runner_kwargs: Dict[str, Any] = {
            "regions": selected_regions,
            "dry_run": False,
            "offline": offline,
            "target_end_time": target_end_time,
            "max_latency_hours": max_latency_hours,
            "workers": workers,
            "skip_tests": skip_tests,
            "report_dir": report_dir,
            "now_utc": exec_now,
            "env_file": env_file,
            "verify_model_load": verify_model_load,
            "tx_backup_root": tx_backup_root,
            "lock_path": resolved_lock_path,
        }
        if region_executors is not None:
            runner_kwargs["region_executors"] = region_executors
        if region_validators is not None:
            runner_kwargs["region_validators"] = region_validators
        if test_runner is not None:
            runner_kwargs["test_runner"] = test_runner

        try:
            report = runner(**runner_kwargs)
        except Exception as exc:
            logger.exception(
                "run_id=%s outcome=FAILED phase=execution error=%s",
                run_id,
                exc,
            )
            return {
                "schema_version": "scheduled_refresh_result_v1",
                "run_id": run_id,
                "execution_timestamp_utc": _utc_iso(exec_now),
                "status": "FAILED",
                "phase": "execution",
                "errors": [f"Standard Refresh raised an unhandled exception: {exc}"],
                "log_path": str(resolved_log_path),
            }

        status = str(report.get("status", "UNKNOWN"))
        child_run_id = report.get("run_id")
        val_info = report.get("validation_and_test_results") or {}
        err_list: List[str] = list(val_info.get("errors") or [])
        if status in ("SUCCESS", "DRY_RUN"):
            logger.info(
                "run_id=%s child_run_id=%s outcome=%s obs_end=%s",
                run_id,
                child_run_id,
                status,
                (report.get("observation_window") or {}).get("observation_end_utc"),
            )
        elif status == "SKIPPED_LOCKED":
            logger.warning(
                "run_id=%s child_run_id=%s outcome=SKIPPED_LOCKED holder=%s",
                run_id,
                child_run_id,
                json.dumps(report.get("lock_holder") or {}),
            )
        else:
            logger.error(
                "run_id=%s child_run_id=%s outcome=%s errors=%s",
                run_id,
                child_run_id,
                status,
                json.dumps(err_list[:5]),
            )
        return report
    finally:
        _close_scheduler_logger(logger, handler)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scheduled NASA GPM IMERG V07C refresh wrapper with unchanged-window skip detection."
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(REGION_ORDER),
        help="Regions to refresh when triggered (default: all 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run full refresh even when the discovered GPM observation window matches latest SUCCESS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight, discovery, and unchanged-window evaluation without running Steps 4-9.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local HDF5 archive only without CMR discovery/downloads.",
    )
    parser.add_argument(
        "--target-end-utc",
        type=str,
        default=None,
        help="Explicit UTC observation end (YYYY-MM-DDTHH:MM:SSZ).",
    )
    parser.add_argument(
        "--max-latency-hours",
        type=float,
        default=DEFAULT_MAX_LATENCY_HOURS,
        help=f"Maximum allowed observation age in hours (must be <= {DEFAULT_MAX_LATENCY_HOURS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent GPM download workers (default: 4).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Forward --skip-tests to Standard Refresh if triggered.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for run reports, lock file, and scheduler.log.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional override path for scheduler rotating log file.",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="Optional override path for the shared Standard Refresh process lock.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    target_dt = None
    if args.target_end_utc:
        target_dt = parse_iso8601_utc(args.target_end_utc)
        if target_dt is None:
            print(
                "ERROR: --target-end-utc must be explicit UTC ISO-8601 (e.g. 2026-11-11T12:00:00Z)",
                file=sys.stderr,
            )
            return 2

    try:
        result = run_scheduled_refresh(
            regions=args.regions,
            force=args.force,
            dry_run=args.dry_run,
            offline=args.offline,
            target_end_time=target_dt,
            max_latency_hours=args.max_latency_hours,
            workers=args.workers,
            skip_tests=args.skip_tests,
            report_dir=args.report_dir,
            log_path=args.log_path,
            lock_path=args.lock_path,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0 if result.get("status") in ("SUCCESS", "DRY_RUN", "SKIPPED_LOCKED", "SKIPPED_UNCHANGED") else 1


if __name__ == "__main__":
    sys.exit(main())
