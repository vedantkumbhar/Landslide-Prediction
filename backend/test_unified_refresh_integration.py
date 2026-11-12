"""
Targeted tests for Unified India Landslide Dynamic Risk Map refresh pipeline integration (Stage 2).

Verifies:
1. Full 3-region successful refresh generates the unified map.
2. Partial-region refresh skips unified generation and preserves existing unified map.
3. Regional failure skips unified generation and preserves existing unified map.
4. Unit-test failure skips unified generation.
5. Mismatched timestamps skip unified generation.
6. Unified generation failure preserves previous map and sets run status to PARTIAL.
7. Refresh report records unified map status accurately in JSON and Markdown.
8. Scheduled refresh does not skip when unified map is missing.
9. Scheduled refresh does not skip when unified map timestamp is outdated.
10. Existing regional outputs remain unaffected by unified map generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from backend.create_india_unified_risk_map import (
    DEFAULT_OUTPUT_HTML as PROD_UNIFIED_HTML,
    EXPECTED_TOTAL_BACKGROUND,
    EXPECTED_TOTAL_CELLS,
    EXPECTED_TOTAL_EVENTS,
)
from backend.run_scheduled_refresh import (
    evaluate_unchanged_window_skip,
    run_scheduled_refresh,
)
from backend.run_standard_refresh import (
    REGION_ORDER,
    build_run_summary_report,
    render_run_summary_markdown,
    run_standard_refresh,
)

OBS_START = "2026-11-11T11:00:00Z"
OBS_END = "2026-11-12T11:00:00Z"
OBS_END_DT = datetime(2026, 11, 12, 11, 0, 0, tzinfo=timezone.utc)
EXEC_NOW = datetime(2026, 11, 12, 12, 30, 0, tzinfo=timezone.utc)


def _make_discovery(
    obs_start: str = OBS_START,
    obs_end: str = OBS_END,
    latency_hours: float = 1.5,
    is_fresh: bool = True,
    is_24h_complete: bool = True,
) -> dict:
    return {
        "resolved_end_dt": datetime.fromisoformat(obs_end.replace("Z", "+00:00")),
        "observation_start_utc": obs_start,
        "observation_end_utc": obs_end,
        "discovery_source": "NASA_CMR_MOCK",
        "resolver_status": "DRY_RUN_CMR_DISCOVERED",
        "latency_hours": latency_hours,
        "max_latency_hours": 12.0,
        "is_fresh": is_fresh,
        "freshness_status": "FRESH" if is_fresh else "STALE",
        "freshness_label": "FRESH" if is_fresh else "STALE",
        "freshness_reason": "OK",
        "granules_discovered_cmr": 48,
        "granules_downloaded": 0,
        "granules_cached": 48,
        "granules_validated": 48,
        "expected_granules": 48,
        "missing_intervals_count": 0,
        "is_24h_complete": is_24h_complete,
        "telemetry": {"granules_validated": 48},
    }


def _mock_unified_generator(
    output_html_path: Path,
    max_latency_hours: float = 12.0,
    now_utc: datetime | None = None,
    overwrite: bool = True,
) -> dict:
    target = Path(output_html_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"<html><head><title>Unified India Map</title></head><body>"
        f"<span>Observation Window: {OBS_START} to {OBS_END}</span>"
        f"<script>function computeFreshness() {{}} function updateFreshnessStatus() {{}} setInterval(updateFreshnessStatus, 60000);</script>"
        f"</body></html>"
    )
    target.write_text(body, encoding="utf-8")
    raw_bytes = target.stat().st_size
    return {
        "status": "SUCCESS",
        "validation_passed": True,
        "output_html": str(target),
        "html_size_bytes": raw_bytes,
        "html_size_mb": round(raw_bytes / (1024 * 1024), 2),
        "observation_start_utc": OBS_START,
        "observation_end_utc": OBS_END,
        "generation_timestamp_utc": "2026-11-12T12:00:00Z",
        "latency_hours": 1.0,
        "counts": {
            "total_cells": EXPECTED_TOTAL_CELLS,
            "total_events": EXPECTED_TOTAL_EVENTS,
            "total_background": EXPECTED_TOTAL_BACKGROUND,
        },
    }


class TestUnifiedRefreshIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_unified_refresh_"))
        self.report_dir = self.tmp_dir / "refresh_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = self.tmp_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.tx_root = self.tmp_dir / "tx_backups"
        self.lock_path = self.report_dir / ".standard_refresh.lock"
        self.unified_html_path = self.outputs_dir / "india_unified_dynamic_risk_map.html"

        # Setup mock regional files and specs
        self.region_specs = self._setup_regional_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _setup_regional_outputs(self, obs_end: str = OBS_END) -> dict:
        specs = {}
        for rkey in REGION_ORDER:
            rdir = self.outputs_dir / rkey
            rdir.mkdir(parents=True, exist_ok=True)
            risk_csv = rdir / f"{rkey}_risk.csv"
            risk_meta = rdir / f"{rkey}_risk_metadata.json"
            map_meta = rdir / f"{rkey}_map_metadata.json"
            map_html = rdir / f"{rkey}_map.html"
            gj_file = rdir / f"{rkey}_regions.geojson"

            risk_csv.write_text("location_id,prob,risk_category\n1,0.5,Moderate\n", encoding="utf-8")
            meta_content = json.dumps({
                "gpm_observation_metadata": {
                    "observation_start_utc": OBS_START,
                    "observation_end_utc": obs_end,
                }
            })
            risk_meta.write_text(meta_content, encoding="utf-8")
            map_meta.write_text(meta_content, encoding="utf-8")
            map_html.write_text("<html><body>regional map</body></html>", encoding="utf-8")
            gj_file.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

            specs[rkey] = {
                "display_name": rkey,
                "stages": "Stage 4",
                "required_inputs": [],
                "required_models": [],
                "tracked_outputs": [risk_csv, risk_meta, map_meta, map_html, gj_file],
                "risk_csv": risk_csv,
                "risk_metadata_json": risk_meta,
                "map_metadata_json": map_meta,
                "map_html": map_html,
                "geojson_regions": gj_file,
            }
        return specs

    def _build_mock_executors(self):
        execs = {}
        for rkey in REGION_ORDER:
            def _make_runner(rk):
                def _runner():
                    pass
                return _runner
            execs[rkey] = _make_runner(rkey)
        return execs

    def _build_mock_validators(self, fail_region: str | None = None, mismatched_region: str | None = None):
        vals = {}
        for rkey in REGION_ORDER:
            def _make_val(rk):
                def _val(region_key, spec, expected_obs_end_utc, max_latency_hours):
                    if rk == fail_region:
                        return {"validation_passed": False, "errors": ["Validation failed."]}
                    obs_end = "2026-11-10T11:00:00Z" if rk == mismatched_region else expected_obs_end_utc
                    return {
                        "validation_passed": True,
                        "errors": [],
                        "warnings": [],
                        "observation_start_utc": OBS_START,
                        "observation_end_utc": obs_end,
                        "total_locations": 1000,
                        "assessed_count": 900,
                        "unassessed_count": 100,
                        "category_counts": {"Low": 500, "Moderate": 300, "High": 80, "Very High": 20, "Unassessed": 100},
                        "gis_paths": {
                            "html_map": str(spec.get("map_html", "")),
                            "html_size_bytes": 100,
                        },
                    }
                return _val
            vals[rkey] = _make_val(rkey)
        return vals

    def _build_mock_test_runner(self, pass_tests: bool = True):
        def _runner(succeeded_regions, specs):
            return {
                "executed": True,
                "all_passed": pass_tests,
                "suites_passed": len(succeeded_regions) if pass_tests else 0,
                "suites_total": len(succeeded_regions),
                "suites": [
                    {
                        "module": f"backend.test_{r}",
                        "passed": pass_tests,
                        "elapsed_seconds": 0.05,
                        "summary": "OK" if pass_tests else "FAILED",
                    }
                    for r in succeeded_regions
                ],
            }
        return _runner

    # 1. Full 3-region refresh generates unified map
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_1_full_refresh_generates_unified_map(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        gen_mock = MagicMock(side_effect=_mock_unified_generator)

        report = run_standard_refresh(
            regions=["himalayas", "northeast", "western_ghats"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(),
            test_runner=self._build_mock_test_runner(pass_tests=True),
            unified_map_generator=gen_mock,
            unified_map_path=self.unified_html_path,
        )

        self.assertEqual(report["status"], "SUCCESS")
        gen_mock.assert_called_once()
        umap_sec = report.get("unified_map")
        self.assertIsNotNone(umap_sec)
        self.assertEqual(umap_sec["status"], "SUCCESS")
        self.assertTrue(umap_sec["validation_passed"])
        self.assertTrue(self.unified_html_path.exists())
        self.assertGreater(self.unified_html_path.stat().st_size, 0)
        self.assertEqual(umap_sec["counts"]["total_cells"], EXPECTED_TOTAL_CELLS)

    # 2. Partial-region refresh skips unified generation
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_2_partial_refresh_skips_unified_generation(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        # Place existing unified map before run
        self.unified_html_path.write_text("PREVIOUS_UNIFIED_MAP", encoding="utf-8")
        gen_mock = MagicMock(side_effect=_mock_unified_generator)

        report = run_standard_refresh(
            regions=["himalayas"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(),
            test_runner=self._build_mock_test_runner(pass_tests=True),
            unified_map_generator=gen_mock,
            unified_map_path=self.unified_html_path,
        )

        gen_mock.assert_not_called()
        umap_sec = report.get("unified_map")
        self.assertEqual(umap_sec["status"], "SKIPPED_PARTIAL_REGIONS")
        self.assertEqual(self.unified_html_path.read_text(encoding="utf-8"), "PREVIOUS_UNIFIED_MAP")

    # 3. Regional failure skips unified generation
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_3_regional_failure_skips_unified_generation(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        self.unified_html_path.write_text("PREVIOUS_UNIFIED_MAP", encoding="utf-8")
        gen_mock = MagicMock(side_effect=_mock_unified_generator)

        report = run_standard_refresh(
            regions=["himalayas", "northeast", "western_ghats"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(fail_region="northeast"),
            test_runner=self._build_mock_test_runner(pass_tests=True),
            unified_map_generator=gen_mock,
            unified_map_path=self.unified_html_path,
        )

        gen_mock.assert_not_called()
        self.assertIn(report["status"], ("PARTIAL", "FAILED"))
        umap_sec = report.get("unified_map")
        self.assertEqual(umap_sec["status"], "SKIPPED_REGIONAL_FAILURE")
        self.assertEqual(self.unified_html_path.read_text(encoding="utf-8"), "PREVIOUS_UNIFIED_MAP")

    # 4. Unit-test failure skips unified generation
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_4_unit_test_failure_skips_unified_generation(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        self.unified_html_path.write_text("PREVIOUS_UNIFIED_MAP", encoding="utf-8")
        gen_mock = MagicMock(side_effect=_mock_unified_generator)

        report = run_standard_refresh(
            regions=["himalayas", "northeast", "western_ghats"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(),
            test_runner=self._build_mock_test_runner(pass_tests=False),
            unified_map_generator=gen_mock,
            unified_map_path=self.unified_html_path,
        )

        gen_mock.assert_not_called()
        self.assertEqual(report["status"], "PARTIAL")
        umap_sec = report.get("unified_map")
        self.assertEqual(umap_sec["status"], "SKIPPED_TESTS_FAILED")
        self.assertEqual(self.unified_html_path.read_text(encoding="utf-8"), "PREVIOUS_UNIFIED_MAP")

    # 5. Mismatched timestamps skip generation
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_5_mismatched_timestamps_skip_unified_generation(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        self.unified_html_path.write_text("PREVIOUS_UNIFIED_MAP", encoding="utf-8")
        gen_mock = MagicMock(side_effect=_mock_unified_generator)

        report = run_standard_refresh(
            regions=["himalayas", "northeast", "western_ghats"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(mismatched_region="himalayas"),
            test_runner=self._build_mock_test_runner(pass_tests=True),
            unified_map_generator=gen_mock,
            unified_map_path=self.unified_html_path,
        )

        gen_mock.assert_not_called()
        umap_sec = report.get("unified_map")
        self.assertEqual(umap_sec["status"], "SKIPPED_TIMESTAMP_MISMATCH")
        self.assertEqual(self.unified_html_path.read_text(encoding="utf-8"), "PREVIOUS_UNIFIED_MAP")

    # 6. Unified generation failure preserves previous map & sets PARTIAL status
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_6_unified_generation_failure_preserves_previous_map_and_sets_partial(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        self.unified_html_path.write_text("PREVIOUS_UNIFIED_MAP", encoding="utf-8")

        def _failing_gen(*args, **kwargs):
            raise IOError("Simulated disk error during unified map render")

        report = run_standard_refresh(
            regions=["himalayas", "northeast", "western_ghats"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(),
            test_runner=self._build_mock_test_runner(pass_tests=True),
            unified_map_generator=_failing_gen,
            unified_map_path=self.unified_html_path,
        )

        # Final status MUST be PARTIAL (regional succeeded, unified failed)
        self.assertEqual(report["status"], "PARTIAL")
        umap_sec = report.get("unified_map")
        self.assertIsNotNone(umap_sec)
        self.assertEqual(umap_sec["status"], "FAILED")
        self.assertFalse(umap_sec["validation_passed"])
        self.assertIn("Simulated disk error", umap_sec["error"])

        # Prior unified map preserved
        self.assertEqual(self.unified_html_path.read_text(encoding="utf-8"), "PREVIOUS_UNIFIED_MAP")

        # Regional outputs preserved
        for rkey in REGION_ORDER:
            self.assertEqual(report["regions"][rkey]["status"], "SUCCESS")

    # 7. Refresh report records unified map accurately in JSON and Markdown
    def test_7_refresh_report_records_unified_map_json_and_markdown(self):
        umap_data = {
            "status": "SUCCESS",
            "validation_passed": True,
            "output_html": str(self.unified_html_path),
            "html_size_bytes": 6323894,
            "html_size_mb": 6.03,
            "observation_start_utc": OBS_START,
            "observation_end_utc": OBS_END,
            "generation_timestamp_utc": "2026-11-12T12:00:00Z",
            "latency_hours": 1.5,
            "counts": {
                "total_cells": 31421,
                "total_events": 1179,
                "total_background": 2306,
            },
        }

        report = build_run_summary_report(
            run_id="test_run_123",
            execution_dt=EXEC_NOW,
            status="SUCCESS",
            preflight={"status": "PASSED", "checks_passed": 5, "checks_total": 5},
            discovery=_make_discovery(),
            region_results={
                r: {"status": "SUCCESS", "validation_passed": True, "total_locations": 1000}
                for r in REGION_ORDER
            },
            skipped_regions=[],
            unit_tests={"executed": True, "all_passed": True, "suites": []},
            errors=[],
            warnings=[],
            unified_map=umap_data,
        )

        # JSON verification
        self.assertEqual(report["unified_map"]["status"], "SUCCESS")
        self.assertEqual(report["unified_map"]["counts"]["total_cells"], 31421)

        # Markdown verification
        md = render_run_summary_markdown(report)
        self.assertIn("## Unified India Landslide Dynamic Risk Map", md)
        self.assertIn("- **Status**: `SUCCESS`", md)
        self.assertIn("`6,323,894` bytes", md)
        self.assertIn("Total Cells=`31,421`", md)
        self.assertIn("Events=`1,179`", md)
        self.assertIn("Background=`2,306`", md)

    # 8. Scheduled refresh does not skip when unified map is missing
    def test_8_scheduled_refresh_does_not_skip_when_unified_map_missing(self):
        pub_report = {
            "schema_version": "standard_refresh_report_v1",
            "run_id": "prev_run_001",
            "status": "SUCCESS",
            "observation_window": {
                "observation_start_utc": OBS_START,
                "observation_end_utc": OBS_END,
            },
            "validation_and_test_results": {"skipped_regions": [], "errors": []},
            "regions": {
                r: {
                    "status": "SUCCESS",
                    "validation_passed": True,
                    "observation_start_utc": OBS_START,
                    "observation_end_utc": OBS_END,
                }
                for r in REGION_ORDER
            },
            "unified_map": {
                "status": "SUCCESS",
                "observation_end_utc": OBS_END,
            },
        }
        (self.report_dir / "latest_refresh_report.json").write_text(
            json.dumps(pub_report, indent=2), encoding="utf-8"
        )

        # Ensure unified map file does NOT exist
        if self.unified_html_path.exists():
            self.unified_html_path.unlink()

        skip, reason, _ = evaluate_unchanged_window_skip(
            report_dir=self.report_dir,
            discovery=_make_discovery(),
            region_specs=self.region_specs,
            unified_map_path=self.unified_html_path,
        )

        self.assertFalse(skip)
        self.assertIn("Unified India map is missing or empty", reason)

    # 9. Scheduled refresh does not skip when unified map timestamp is outdated
    def test_9_scheduled_refresh_does_not_skip_when_timestamp_outdated(self):
        pub_report = {
            "schema_version": "standard_refresh_report_v1",
            "run_id": "prev_run_001",
            "status": "SUCCESS",
            "observation_window": {
                "observation_start_utc": OBS_START,
                "observation_end_utc": OBS_END,
            },
            "validation_and_test_results": {"skipped_regions": [], "errors": []},
            "regions": {
                r: {
                    "status": "SUCCESS",
                    "validation_passed": True,
                    "observation_start_utc": OBS_START,
                    "observation_end_utc": OBS_END,
                }
                for r in REGION_ORDER
            },
            "unified_map": {
                "status": "SUCCESS",
                "observation_end_utc": OBS_END,
            },
        }
        (self.report_dir / "latest_refresh_report.json").write_text(
            json.dumps(pub_report, indent=2), encoding="utf-8"
        )

        # Write unified map with OLD timestamp
        old_body = (
            "<html><body>Observation: 2026-11-10T11:00:00Z "
            "computeFreshness updateFreshnessStatus 60000</body></html>"
        )
        self.unified_html_path.write_text(old_body, encoding="utf-8")

        skip, reason, _ = evaluate_unchanged_window_skip(
            report_dir=self.report_dir,
            discovery=_make_discovery(),
            region_specs=self.region_specs,
            unified_map_path=self.unified_html_path,
        )

        self.assertFalse(skip)
        self.assertIn("observation timestamp does not match", reason)

    # 10. Existing regional outputs remain unaffected
    @patch("backend.run_standard_refresh.discover_and_retrieve_gpm")
    @patch("backend.run_standard_refresh.check_preflight")
    def test_10_existing_regional_outputs_remain_unaffected(self, mock_pre, mock_disc):
        mock_pre.return_value = {"status": "PASSED", "checks_passed": 5, "checks_total": 5, "errors": [], "warnings": []}
        mock_disc.return_value = _make_discovery()

        # Record initial contents of regional tracked files
        initial_contents = {}
        for rkey, spec in self.region_specs.items():
            for p in spec["tracked_outputs"]:
                initial_contents[str(p)] = Path(p).read_bytes()

        run_standard_refresh(
            regions=["himalayas", "northeast", "western_ghats"],
            report_dir=self.report_dir,
            tx_backup_root=self.tx_root,
            lock_path=self.lock_path,
            now_utc=EXEC_NOW,
            region_specs=self.region_specs,
            region_executors=self._build_mock_executors(),
            region_validators=self._build_mock_validators(),
            test_runner=self._build_mock_test_runner(pass_tests=True),
            unified_map_generator=_mock_unified_generator,
            unified_map_path=self.unified_html_path,
        )

        # Verify all regional outputs are byte-identical
        for p_str, expected_bytes in initial_contents.items():
            actual_bytes = Path(p_str).read_bytes()
            self.assertEqual(
                actual_bytes,
                expected_bytes,
                f"Regional output {p_str} was altered during refresh run!",
            )


if __name__ == "__main__":
    unittest.main()
