"""Targeted unit and integration tests for Stage 1 Unified India Landslide Risk Map."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.create_india_unified_risk_map import (
    DEFAULT_HIM_MAP_META,
    DEFAULT_HIM_RISK_META,
    DEFAULT_NE_MAP_META,
    DEFAULT_NE_RISK_META,
    DEFAULT_OUTPUT_HTML,
    DEFAULT_WG_MAP_META,
    DEFAULT_WG_RISK_META,
    EXPECTED_COUNTS,
    EXPECTED_TOTAL_BACKGROUND,
    EXPECTED_TOTAL_CELLS,
    EXPECTED_TOTAL_EVENTS,
    generate_india_unified_risk_map,
)


def _extract_js_json_assignment(html_text: str, var_name: str):
    pattern = rf"var {re.escape(var_name)}\s*=\s*(\{{.*?\}}|\[.*?\]);"
    m = re.search(pattern, html_text, flags=re.DOTALL)
    if not m:
        raise AssertionError(f"Could not find embedded JS variable '{var_name}' in HTML.")
    return json.loads(m.group(1))


class TestIndiaUnifiedRiskMap(unittest.TestCase):
    """Verify data completeness, scientific separation, safety gates, and atomic output."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_unified_india_"))
        # Deterministic timestamp 6 hours after 2026-11-12T11:00:00Z (well within 12h)
        self.fresh_now = datetime(2026, 11, 12, 17, 0, 0, tzinfo=timezone.utc)
        # Deterministic timestamp 18 hours after 2026-11-12T11:00:00Z (exceeds 12h limit)
        self.stale_now = datetime(2026, 11, 13, 5, 0, 0, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_production_html_exists_under_25mb_with_all_layers_and_counts(self) -> None:
        """Generated unified HTML must exist, stay < 25 MB, and embed exact synchronized counts."""
        self.assertTrue(
            DEFAULT_OUTPUT_HTML.is_file(),
            f"Missing unified HTML map at {DEFAULT_OUTPUT_HTML}",
        )
        size_bytes = DEFAULT_OUTPUT_HTML.stat().st_size
        self.assertGreater(size_bytes, 100_000)
        self.assertLess(size_bytes, 25 * 1024 * 1024)

        html_text = DEFAULT_OUTPUT_HTML.read_text(encoding="utf-8")

        # 1. All three regions included
        for label in ("Indian Himalayas", "Northeast India", "Western Ghats (Limited Pilot)"):
            self.assertIn(label, html_text)

        # 2. Embedded cell, event, and background counts match exact expectations
        cells_data = _extract_js_json_assignment(html_text, "CELLS_DATA")
        events_data = _extract_js_json_assignment(html_text, "EVENTS_DATA")
        bg_data = _extract_js_json_assignment(html_text, "BG_DATA")
        cutoffs = _extract_js_json_assignment(html_text, "REGION_CUTOFFS")

        self.assertEqual(set(cells_data.keys()), {"himalayas", "northeast", "western_ghats"})
        self.assertEqual(set(events_data.keys()), {"himalayas", "northeast", "western_ghats"})
        self.assertEqual(set(bg_data.keys()), {"himalayas", "northeast", "western_ghats"})

        for rkey, exp in EXPECTED_COUNTS.items():
            self.assertEqual(len(cells_data[rkey]), exp["cells"])
            self.assertEqual(len(events_data[rkey]), exp["events"])
            self.assertEqual(len(bg_data[rkey]), exp["background"])

        self.assertEqual(sum(len(v) for v in cells_data.values()), EXPECTED_TOTAL_CELLS)
        self.assertEqual(sum(len(v) for v in events_data.values()), EXPECTED_TOTAL_EVENTS)
        self.assertEqual(sum(len(v) for v in bg_data.values()), EXPECTED_TOTAL_BACKGROUND)

        # 3. Separate per-region thresholds remain distinct (never collapsed to a single national scale)
        self.assertNotEqual(
            cutoffs["himalayas"]["P50_low_to_moderate"],
            cutoffs["northeast"]["P50_low_to_moderate"],
        )
        self.assertNotEqual(
            cutoffs["himalayas"]["P50_low_to_moderate"],
            cutoffs["western_ghats"]["P50_low_to_moderate"],
        )
        self.assertIn("Do not compare numerical scores across regions", html_text)

        # 7. Western Ghats limited-pilot disclosure & experimental disclaimer present
        self.assertIn("14 validated DEM tiles", html_text)
        self.assertIn("does not imply complete Western Ghats coverage", html_text)
        self.assertIn("EXPERIMENTAL PROVISIONAL SCREENING ONLY", html_text)

        # Links to all three existing regional HTML maps
        self.assertIn("himalayas_dynamic_risk_map.html", html_text)
        self.assertIn("northeast_india_dynamic_risk_map.html", html_text)
        self.assertIn("western_ghats_dynamic_risk_map.html", html_text)

        # 8. Browser 60s freshness guard & UNASSESSED stale handling present
        for token in (
            "computeFreshness",
            "updateFreshnessStatus",
            "60000",
            "STALE_DATA_UNASSESSED",
            "UNASSESSED (Stale Data)",
            "2026-11-11T11:00:00Z",
            "2026-11-12T11:00:00Z",
        ):
            self.assertIn(token, html_text)

        # Reliable basemaps without OpenStreetMap 403 tile endpoint
        self.assertIn("World_Topo_Map", html_text)
        self.assertNotIn("tile.openstreetmap.org", html_text)

        # Lazy / toggleable layer mechanism and markerPane SVG isolation present
        self.assertIn("ensureLayerPopulated", html_text)
        self.assertIn("toggleUnifiedLayer", html_text)
        self.assertIn("markerPane", html_text)

    def test_02_isolated_generation_succeeds_without_touching_production_output(self) -> None:
        """Generating into a temporary path succeeds cleanly with exact return metadata."""
        out_html = self.tmp_dir / "unified_test.html"
        res = generate_india_unified_risk_map(
            output_html_path=out_html,
            now_utc=self.fresh_now,
            overwrite=False,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(out_html.is_file())
        self.assertEqual(res["counts"]["total_cells"], EXPECTED_TOTAL_CELLS)
        self.assertEqual(res["counts"]["total_events"], EXPECTED_TOTAL_EVENTS)
        self.assertEqual(res["counts"]["total_background"], EXPECTED_TOTAL_BACKGROUND)
        self.assertLess(res["html_size_bytes"], 25 * 1024 * 1024)

    def test_03_mismatched_regional_timestamps_rejected(self) -> None:
        """Differing observation_end_utc between regions must be rejected before generating."""
        bad_ne_risk = json.loads(DEFAULT_NE_RISK_META.read_text(encoding="utf-8"))
        bad_ne_risk["gpm_observation_metadata"]["observation_end_utc"] = "2026-11-12T10:00:00Z"
        bad_ne_map = json.loads(DEFAULT_NE_MAP_META.read_text(encoding="utf-8"))
        bad_ne_map["gpm_observation_metadata"]["observation_end_utc"] = "2026-11-12T10:00:00Z"

        bad_risk_path = self.tmp_dir / "ne_risk_bad.json"
        bad_map_path = self.tmp_dir / "ne_map_bad.json"
        bad_risk_path.write_text(json.dumps(bad_ne_risk), encoding="utf-8")
        bad_map_path.write_text(json.dumps(bad_ne_map), encoding="utf-8")

        out_html = self.tmp_dir / "should_not_exist.html"
        with self.assertRaises(ValueError) as ctx:
            generate_india_unified_risk_map(
                ne_risk_metadata_path=bad_risk_path,
                ne_map_metadata_path=bad_map_path,
                output_html_path=out_html,
                now_utc=self.fresh_now,
            )
        self.assertIn("mismatch", str(ctx.exception).lower())
        self.assertFalse(out_html.exists())

    def test_04_stale_data_rejected_and_existing_output_preserved(self) -> None:
        """Stale data (>12h latency) must fail closed and preserve any prior HTML intact."""
        existing_html = self.tmp_dir / "existing_unified.html"
        sentinel = "<!-- SENTINEL_VALID_UNIFIED_MAP -->"
        existing_html.write_text(sentinel, encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            generate_india_unified_risk_map(
                output_html_path=existing_html,
                now_utc=self.stale_now,
                overwrite=True,
            )
        self.assertIn("STALE_DATA_UNASSESSED", str(ctx.exception))
        self.assertEqual(existing_html.read_text(encoding="utf-8"), sentinel)

    def test_05_missing_input_files_rejected_cleanly(self) -> None:
        """Missing metadata, GeoJSON, or event/background CSV inputs raise FileNotFoundError."""
        out_html = self.tmp_dir / "missing_test.html"

        missing_wg_events = self.tmp_dir / "nonexistent_wg_events.csv"
        with self.assertRaises(FileNotFoundError):
            generate_india_unified_risk_map(
                wg_events_csv_path=missing_wg_events,
                output_html_path=out_html,
                now_utc=self.fresh_now,
            )

        missing_him_cells = self.tmp_dir / "nonexistent_him_cells.geojson"
        with self.assertRaises(FileNotFoundError):
            generate_india_unified_risk_map(
                him_cells_geojson_path=missing_him_cells,
                output_html_path=out_html,
                now_utc=self.fresh_now,
            )

        missing_ne_csv = self.tmp_dir / "nonexistent_training.csv"
        with self.assertRaises(FileNotFoundError):
            generate_india_unified_risk_map(
                training_csv_path=missing_ne_csv,
                output_html_path=out_html,
                now_utc=self.fresh_now,
            )

        missing_meta = self.tmp_dir / "nonexistent_meta.json"
        with self.assertRaises(FileNotFoundError):
            generate_india_unified_risk_map(
                him_map_metadata_path=missing_meta,
                output_html_path=out_html,
                now_utc=self.fresh_now,
            )
        self.assertFalse(out_html.exists())

    def test_06_existing_output_preserved_on_atomic_write_failure(self) -> None:
        """If Folium rendering or replacement raises midway, existing HTML and backup safety hold."""
        target_html = self.tmp_dir / "unified_target.html"
        sentinel = "<!-- PREVIOUS_GOOD_UNIFIED_HTML -->"
        target_html.write_text(sentinel, encoding="utf-8")

        def _failing_atomic_save(target: Path, writer_fn):
            tmp = target.with_name(f".{target.name}.tmp.testfail")
            try:
                tmp.write_text("partial broken", encoding="utf-8")
                raise RuntimeError("Simulated write failure mid-generation")
            finally:
                tmp.unlink(missing_ok=True)

        with patch("backend.create_india_unified_risk_map.atomic_save_map_html", side_effect=_failing_atomic_save):
            with self.assertRaises(RuntimeError):
                generate_india_unified_risk_map(
                    output_html_path=target_html,
                    now_utc=self.fresh_now,
                    overwrite=True,
                )

        self.assertEqual(target_html.read_text(encoding="utf-8"), sentinel)
        leftover_tmps = list(self.tmp_dir.glob("*.tmp*"))
        self.assertEqual(leftover_tmps, [])


if __name__ == "__main__":
    unittest.main()
