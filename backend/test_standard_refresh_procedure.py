
"""
Unit and integration tests for the Unified Standard Refresh Run Procedure (Steps 1-11).
Operates strictly in isolated temporary directories and mocked discovery/execution hooks
without modifying any production datasets or triggering live downloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.gis_refresh_utils import (
    evaluate_freshness_condition,
    validate_dynamic_metadata_thresholds,
)
from backend.export_western_ghats_dynamic_risk_geojson import (
    validate_stage6d_metadata_thresholds as validate_wg_thresholds,
)
from backend.export_northeast_dynamic_risk_geojson import (
    validate_stage74_metadata_thresholds as validate_ne_thresholds,
)
from backend.run_standard_refresh import (
    RegionTransaction,
    main as refresh_cli_main,
    normalize_regions,
    run_standard_refresh,
)


def _mock_valid_region_metrics(
    region_key: str,
    spec: dict,
    expected_obs_end_utc: str,
    **_kwargs,
) -> dict:
    """Build valid mock Step 10 region metrics for isolated orchestrator tests."""
    n = spec["expected_locations"]
    low = n // 2
    mod = int(n * 0.30)
    high = int(n * 0.15)
    vhigh = n - low - mod - high

    return {
        "validation_passed": True,
        "errors": [],
        "warnings": [],
        "total_locations": n,
        "unique_tiles": spec["expected_tiles"],
        "assessed_count": n,
        "unassessed_count": 0,
        "category_counts": {
            "Low": low,
            "Moderate": mod,
            "High": high,
            "Very High": vhigh,
            "Unassessed": 0,
        },
        "relative_cutoffs": {
            "P50_low_to_moderate": 22.5,
            "P80_moderate_to_high": 48.0,
            "P95_high_to_very_high": 74.0,
        },
        "observation_start_utc": "2026-11-10T12:00:00Z",
        "observation_end_utc": expected_obs_end_utc,
        "latency_hours": 4.0,
        "windows_5_monotonic": True,
        "window_stats": {
            w: {
                "present": True,
                "finite_count": n,
                "missing_count": 0,
                "min_mm": 0.0,
                "mean_mm": 3.5,
                "max_mm": 25.0,
            }
            for w in ("1h", "3h", "6h", "12h", "24h")
        },
        "gis_paths": {
            "html_map": str(spec["map_html"]),
            "html_size_bytes": 450_000,
            "map_metadata_json": str(spec["map_metadata_json"]),
            "map_validation_report_json": str(spec["map_report_json"]),
            "risk_csv": str(spec["risk_csv"]),
            "risk_metadata_json": str(spec["risk_metadata_json"]),
            "geojson_outputs": {
                str(spec["risk_csv"].with_suffix(".geojson")): {
                    "feature_count": n,
                    "geometry_types": ["Point"],
                    "size_bytes": 120_000,
                }
            },
            "synced_outputs": [str(p) for p in spec["synced_outputs"]],
        },
    }


class TestStandardRefreshProcedure(unittest.TestCase):
    """Verify operational safeguards, freshness gates, rollbacks, dynamic cutoffs, and reporting."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="std_refresh_test_"))
        self.report_dir = self.tmp_dir / "reports"
        self.tx_dir = self.tmp_dir / "tx_backups"

        self.now_utc = datetime(
            2026, 11, 11, 16, 0, 0, tzinfo=timezone.utc
        )
        self.obs_end_dt = datetime(
            2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc
        )
        self.obs_end_str = "2026-11-11T12:00:00Z"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_preflight_missing_token_fails_safely(self):
        """Step 1 must fail safely when NASA_EARTHDATA_TOKEN is missing in online mode."""
        empty_env = self.tmp_dir / "missing.env"

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "backend.run_standard_refresh.discover_and_retrieve_gpm"
            ) as mock_disc:
                report = run_standard_refresh(
                    regions=["himalayas"],
                    offline=False,
                    skip_tests=True,
                    report_dir=self.report_dir,
                    now_utc=self.now_utc,
                    env_file=empty_env,
                    verify_model_load=False,
                )

        mock_disc.assert_not_called()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(
            report["validation_and_test_results"]["preflight_status"],
            "FAILED",
        )
        self.assertTrue(
            any(
                "NASA_EARTHDATA_TOKEN" in err
                for err in report["validation_and_test_results"]["errors"]
            )
        )
        self.assertTrue(
            (self.report_dir / "latest_refresh_report.json").exists()
        )

    def test_02_preflight_missing_input_dataset_fails_safely(self):
        """Step 1 must reject missing input datasets before discovery or execution."""
        fake_specs = {
            "himalayas": {
                "display_name": "Himalayas",
                "stages": "8.3-8.5",
                "expected_locations": 10,
                "expected_tiles": 1,
                "required_inputs": [
                    (self.tmp_dir / "nonexistent_grid.csv", ["grid_id"])
                ],
                "required_models": [],
                "tracked_outputs": [],
                "synced_outputs": [],
                "test_modules": [],
            }
        }

        with patch(
            "backend.run_standard_refresh._build_region_specs",
            return_value=fake_specs,
        ):
            report = run_standard_refresh(
                regions=["himalayas"],
                offline=True,
                skip_tests=True,
                report_dir=self.report_dir,
                now_utc=self.now_utc,
                env_file=None,
            )

        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(
            report["validation_and_test_results"]["preflight_status"],
            "FAILED",
        )
        self.assertTrue(
            any(
                "Missing or empty input dataset" in e
                for e in report["validation_and_test_results"]["errors"]
            )
        )

    def test_03_all_freshness_conditions_and_incomplete_granules(self):
        """Verify freshness conditions and incomplete granule protection."""
        ref_now = datetime(
            2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc
        )

        exact12 = evaluate_freshness_condition(
            obs_end_utc_str="2026-11-11T00:00:00Z",
            is_24h_complete=True,
            available_granules=48,
            missing_intervals=0,
            max_latency_hours=12.0,
            now_utc=ref_now,
        )
        self.assertTrue(exact12["is_fresh"])
        self.assertEqual(exact12["status_code"], "FRESH")

        stale = evaluate_freshness_condition(
            obs_end_utc_str="2026-11-10T23:59:59Z",
            is_24h_complete=True,
            available_granules=48,
            missing_intervals=0,
            max_latency_hours=12.0,
            now_utc=ref_now,
        )
        self.assertFalse(stale["is_fresh"])
        self.assertEqual(
            stale["status_code"],
            "STALE_DATA_UNASSESSED",
        )

        missing = evaluate_freshness_condition(
            obs_end_utc_str=None,
            is_24h_complete=True,
            available_granules=48,
            missing_intervals=0,
            max_latency_hours=12.0,
            now_utc=ref_now,
        )
        self.assertFalse(missing["is_fresh"])
        self.assertEqual(
            missing["status_code"],
            "UNASSESSED_MISSING_TIMESTAMP",
        )

        invalid = evaluate_freshness_condition(
            obs_end_utc_str="2026-11-11T10:00:00",
            is_24h_complete=True,
            available_granules=48,
            missing_intervals=0,
            max_latency_hours=12.0,
            now_utc=ref_now,
        )
        self.assertFalse(invalid["is_fresh"])
        self.assertEqual(
            invalid["status_code"],
            "UNASSESSED_INVALID_TIMESTAMP",
        )

        future = evaluate_freshness_condition(
            obs_end_utc_str="2026-11-11T13:00:00Z",
            is_24h_complete=True,
            available_granules=48,
            missing_intervals=0,
            max_latency_hours=12.0,
            now_utc=ref_now,
        )
        self.assertFalse(future["is_fresh"])
        self.assertEqual(
            future["status_code"],
            "UNASSESSED_FUTURE_TIMESTAMP",
        )

        incomplete = evaluate_freshness_condition(
            obs_end_utc_str="2026-11-11T10:00:00Z",
            is_24h_complete=False,
            available_granules=46,
            missing_intervals=2,
            max_latency_hours=12.0,
            now_utc=ref_now,
        )
        self.assertFalse(incomplete["is_fresh"])
        self.assertEqual(
            incomplete["status_code"],
            "INCOMPLETE_DATA",
        )

    def test_04_stale_discovery_halts_safely_without_running_regions(self):
        """Stale discovery must halt before regional execution."""
        stale_discovery = {
            "resolved_end_dt": self.obs_end_dt - timedelta(hours=20),
            "observation_start_utc": "2026-11-09T16:00:00Z",
            "observation_end_utc": "2026-11-10T16:00:00Z",
            "discovery_source": "LOCAL_ARCHIVE_FALLBACK",
            "resolver_status": "STALE_DATA",
            "latency_hours": 24.0,
            "max_latency_hours": 12.0,
            "is_fresh": False,
            "freshness_status": "STALE_DATA_UNASSESSED",
            "freshness_reason": "Observation age 24.000h exceeds 12.0h",
            "granules_discovered_cmr": 0,
            "granules_downloaded": 0,
            "granules_cached": 48,
            "granules_validated": 48,
            "expected_granules": 48,
            "missing_intervals_count": 0,
            "is_24h_complete": True,
            "telemetry": {},
        }

        exec_called = []

        with patch(
            "backend.run_standard_refresh.check_preflight",
            return_value={
                "status": "PASSED",
                "checks_passed": 5,
                "checks_total": 5,
                "errors": [],
                "checks": [],
            },
        ):
            with patch(
                "backend.run_standard_refresh.discover_and_retrieve_gpm",
                return_value=stale_discovery,
            ):
                report = run_standard_refresh(
                    regions=[
                        "himalayas",
                        "northeast",
                        "western_ghats",
                    ],
                    offline=True,
                    skip_tests=True,
                    report_dir=self.report_dir,
                    now_utc=self.now_utc,
                    tx_backup_root=self.tx_dir,
                    region_executors={
                        "himalayas": lambda: exec_called.append("him"),
                        "northeast": lambda: exec_called.append("ne"),
                        "western_ghats": lambda: exec_called.append("wg"),
                    },
                )

        self.assertEqual(exec_called, [])
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(
            report["updated_locations_summary"]["total_updated_locations"],
            0,
        )

    def test_05_transaction_rollback_restores_prior_outputs_and_cleans_orphans(self):
        """RegionTransaction restores prior outputs and removes new partial files."""
        orig_csv = self.tmp_dir / "existing_risk.csv"
        orig_html = self.tmp_dir / "existing_map.html"
        orphan_gj = self.tmp_dir / "partial_orphan.geojson"

        orig_csv.write_text(
            "grid_id,risk\n1,42.5\n",
            encoding="utf-8",
        )
        orig_html.write_text(
            "<html>valid prior map</html>",
            encoding="utf-8",
        )

        tx = RegionTransaction(
            region_key="test_region",
            tracked_paths=[orig_csv, orig_html, orphan_gj],
            backup_root=self.tx_dir / "tx1",
        )

        with self.assertRaises(RuntimeError):
            with tx:
                orig_csv.write_text(
                    "CORRUPTED_PARTIAL",
                    encoding="utf-8",
                )
                orig_html.unlink()
                orphan_gj.write_text(
                    '{"type": "FeatureCollection"}',
                    encoding="utf-8",
                )
                raise RuntimeError("Simulated mid-pipeline crash")

        self.assertTrue(tx.rolled_back)
        self.assertEqual(
            orig_csv.read_text(encoding="utf-8"),
            "grid_id,risk\n1,42.5\n",
        )
        self.assertEqual(
            orig_html.read_text(encoding="utf-8"),
            "<html>valid prior map</html>",
        )
        self.assertFalse(orphan_gj.exists())

    def test_06_partial_status_and_rollback_on_one_region_failure(self):
        """A regional failure rolls back that region and reports PARTIAL."""
        fresh_discovery = {
            "resolved_end_dt": self.obs_end_dt,
            "observation_start_utc": "2026-11-10T12:00:00Z",
            "observation_end_utc": self.obs_end_str,
            "discovery_source": "NASA_CMR_ONLINE",
            "resolver_status": "SUCCESS",
            "latency_hours": 4.0,
            "max_latency_hours": 12.0,
            "is_fresh": True,
            "freshness_status": "FRESH",
            "freshness_reason": "Fresh",
            "granules_discovered_cmr": 60,
            "granules_downloaded": 5,
            "granules_cached": 43,
            "granules_validated": 48,
            "expected_granules": 48,
            "missing_intervals_count": 0,
            "is_24h_complete": True,
            "telemetry": {},
        }

        wg_prior_html = self.tmp_dir / "wg_prior.html"
        wg_prior_html.write_text(
            "VALID_PRIOR_WG_HTML",
            encoding="utf-8",
        )

        from backend.run_standard_refresh import _build_region_specs

        patched_specs = _build_region_specs()
        patched_specs["himalayas"]["tracked_outputs"] = []
        patched_specs["northeast"]["tracked_outputs"] = []
        patched_specs["western_ghats"]["tracked_outputs"] = [
            wg_prior_html
        ]

        def failing_wg():
            wg_prior_html.write_text(
                "CORRUPTED_WG_RUN",
                encoding="utf-8",
            )
            raise RuntimeError("Simulated WG integration failure")

        with patch(
            "backend.run_standard_refresh._build_region_specs",
            return_value=patched_specs,
        ):
            with patch(
                "backend.run_standard_refresh.check_preflight",
                return_value={
                    "status": "PASSED",
                    "checks_passed": 10,
                    "checks_total": 10,
                    "errors": [],
                    "checks": [],
                },
            ):
                with patch(
                    "backend.run_standard_refresh.discover_and_retrieve_gpm",
                    return_value=fresh_discovery,
                ):
                    report = run_standard_refresh(
                        regions=[
                            "himalayas",
                            "northeast",
                            "western_ghats",
                        ],
                        offline=False,
                        skip_tests=True,
                        report_dir=self.report_dir,
                        now_utc=self.now_utc,
                        tx_backup_root=self.tx_dir,
                        region_executors={
                            "himalayas": lambda: {"status": "SUCCESS"},
                            "northeast": lambda: {"status": "SUCCESS"},
                            "western_ghats": failing_wg,
                        },
                        region_validators={
                            "himalayas": _mock_valid_region_metrics,
                            "northeast": _mock_valid_region_metrics,
                        },
                    )

        self.assertEqual(report["status"], "PARTIAL")
        self.assertEqual(
            report["regions"]["himalayas"]["status"],
            "SUCCESS",
        )
        self.assertEqual(
            report["regions"]["northeast"]["status"],
            "SUCCESS",
        )
        self.assertEqual(
            report["regions"]["western_ghats"]["status"],
            "FAILED",
        )
        self.assertTrue(
            report["regions"]["western_ghats"]["rollback_performed"]
        )
        self.assertEqual(
            wg_prior_html.read_text(encoding="utf-8"),
            "VALID_PRIOR_WG_HTML",
        )
        self.assertEqual(
            report["updated_locations_summary"]["total_updated_locations"],
            12052 + 14550,
        )

    def test_07_shifted_dynamic_cutoffs_propagate_without_hardcoded_locks(self):
        """Shifted P50/P80/P95 cutoffs validate across threshold consumers."""
        shifted_meta = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": 19.25,
                    "P80_moderate_to_high": 44.80,
                    "P95_high_to_very_high": 78.15,
                }
            }
        }

        c_shared = validate_dynamic_metadata_thresholds(shifted_meta)
        c_wg = validate_wg_thresholds(shifted_meta)
        c_ne = validate_ne_thresholds(shifted_meta)

        for res in (c_shared, c_wg, c_ne):
            self.assertAlmostEqual(
                res["P50_low_to_moderate"],
                19.25,
            )
            self.assertAlmostEqual(
                res["P80_moderate_to_high"],
                44.80,
            )
            self.assertAlmostEqual(
                res["P95_high_to_very_high"],
                78.15,
            )

    def test_08_full_success_12_item_json_and_markdown_report_completeness(self):
        """Full mocked SUCCESS run must produce valid 12-item JSON and Markdown reports."""
        fresh_discovery = {
            "resolved_end_dt": self.obs_end_dt,
            "observation_start_utc": "2026-11-10T12:00:00Z",
            "observation_end_utc": self.obs_end_str,
            "discovery_source": "NASA_CMR_ONLINE",
            "resolver_status": "SUCCESS",
            "latency_hours": 4.0,
            "max_latency_hours": 12.0,
            "is_fresh": True,
            "freshness_status": "FRESH",
            "freshness_reason": "Fresh",
            "granules_discovered_cmr": 54,
            "granules_downloaded": 48,
            "granules_cached": 0,
            "granules_validated": 48,
            "expected_granules": 48,
            "missing_intervals_count": 0,
            "is_24h_complete": True,
            "telemetry": {},
        }

        mock_test_res = {
            "executed": True,
            "all_passed": True,
            "suites_total": 10,
            "suites_passed": 10,
            "total_elapsed_seconds": 12.3,
            "suites": [
                {
                    "module": "backend.mock_suite",
                    "passed": True,
                    "returncode": 0,
                    "elapsed_seconds": 12.3,
                    "summary": "OK",
                }
            ],
        }

        def mock_unified_map_generator(**kwargs):
            """Return a complete schema-compatible map result for Test 08."""
            return {
                "html_size_bytes": 1024,
                "html_size_mb": 0.001,
                "observation_start_utc": "2026-11-10T12:00:00Z",
                "observation_end_utc": self.obs_end_str,
                "generation_timestamp_utc": (
                    self.now_utc.isoformat().replace("+00:00", "Z")
                ),
                "latency_hours": 4.0,
                "validation_passed": True,
                "counts": {
                    "total_cells": 31421,
                    "total_events": 1179,
                    "total_background": 2306,
                },
            }

        from backend.run_standard_refresh import _build_region_specs

        isolated_specs = _build_region_specs()
        for rk in isolated_specs:
            isolated_specs[rk]["tracked_outputs"] = []

        with patch(
            "backend.run_standard_refresh.check_preflight",
            return_value={
                "status": "PASSED",
                "checks_passed": 12,
                "checks_total": 12,
                "errors": [],
                "checks": [],
            },
        ):
            with patch(
                "backend.run_standard_refresh.discover_and_retrieve_gpm",
                return_value=fresh_discovery,
            ):
                with patch(
                    "backend.run_standard_refresh._build_region_specs",
                    return_value=isolated_specs,
                ):
                    report = run_standard_refresh(
                        regions=[
                            "himalayas",
                            "northeast",
                            "western_ghats",
                        ],
                        offline=False,
                        skip_tests=False,
                        report_dir=self.report_dir,
                        now_utc=self.now_utc,
                        tx_backup_root=self.tx_dir,
                        region_executors={
                            r: (lambda: {"status": "SUCCESS"})
                            for r in (
                                "himalayas",
                                "northeast",
                                "western_ghats",
                            )
                        },
                        region_validators={
                            r: _mock_valid_region_metrics
                            for r in (
                                "himalayas",
                                "northeast",
                                "western_ghats",
                            )
                        },
                        test_runner=lambda regions, specs: mock_test_res,
                        unified_map_generator=mock_unified_map_generator,
                        unified_map_path=(
                            self.tmp_dir / "mock_india_unified_risk_map.html"
                        ),
                    )

        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(
            report["updated_locations_summary"]["total_updated_locations"],
            12052 + 14550 + 4819,
        )
        self.assertEqual(
            report["assessment_coverage"]["total_assessed_locations"],
            12052 + 14550 + 4819,
        )
        self.assertEqual(
            report["assessment_coverage"]["total_unassessed_locations"],
            0,
        )
        self.assertEqual(
            report["rainfall_window_status"]["status"],
            "VERIFIED_MONOTONIC",
        )

        required_top_keys = [
            "run_id",
            "execution_timestamp_utc",
            "status",
            "observation_window",
            "data_latency",
            "granule_inventory",
            "rainfall_window_status",
            "updated_locations_summary",
            "assessment_coverage",
            "category_distribution_and_thresholds",
            "gis_outputs",
            "validation_and_test_results",
        ]

        for key in required_top_keys:
            self.assertIn(
                key,
                report,
                f"Missing 12-item report section: {key}",
            )

        latest_json = self.report_dir / "latest_refresh_report.json"
        latest_md = self.report_dir / "latest_refresh_report.md"

        self.assertTrue(latest_json.exists())
        self.assertTrue(latest_md.exists())

        md_content = latest_md.read_text(encoding="utf-8")

        for needle in (
            report["run_id"],
            self.obs_end_str,
            "31,421",
            "VERIFIED_MONOTONIC",
            "Himalayas",
            "Northeast India",
            "Western Ghats",
        ):
            self.assertIn(needle, md_content)

    def test_09_cli_region_normalization_and_naive_target_rejection(self):
        """Verify region normalization and rejection of naive target timestamps."""
        self.assertEqual(
            normalize_regions(["western_ghats", "himalayas"]),
            ["himalayas", "western_ghats"],
        )
        self.assertEqual(
            normalize_regions(["all"]),
            ["himalayas", "northeast", "western_ghats"],
        )

        with self.assertRaises(ValueError):
            normalize_regions(["unknown_region"])

        rc = refresh_cli_main(
            [
                "--target-end-utc",
                "2026-11-11T12:00:00",
                "--report-dir",
                str(self.report_dir),
            ]
        )
        self.assertEqual(rc, 2)

    def test_10_real_project_preflight_passes_all_regions(self):
        """Real project inputs, terrain CSVs, and frozen RF models pass Step 1 pre-flight."""
        from backend.run_standard_refresh import (
            _build_region_specs,
            check_preflight,
        )

        res = check_preflight(
            regions=[
                "himalayas",
                "northeast",
                "western_ghats",
            ],
            region_specs=_build_region_specs(),
            offline=False,
            report_dir=self.report_dir,
            verify_model_load=True,
        )

        self.assertEqual(
            res["status"],
            "PASSED",
            msg=f"Pre-flight errors: {res.get('errors')}",
        )
        self.assertEqual(
            res["checks_passed"],
            res["checks_total"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
