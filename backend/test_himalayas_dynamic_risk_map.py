"""
Automated unit tests for Stage 8.5: Himalayas Regional Polygon-Based Dynamic GIS Risk Map,
GeoJSON Exports, and Runtime Freshness Guard.

Verifies:
1. Fresh observation timestamp (dynamic browser freshness evaluation).
2. Exactly 12-hour age boundary condition (<= 12.0h evaluates to FRESH).
3. Age above 12 hours (evaluates to STALE_DATA_UNASSESSED).
4. Missing timestamp handling (evaluates to UNASSESSED_MISSING_TIMESTAMP).
5. Invalid timestamp handling (evaluates to UNASSESSED_INVALID_TIMESTAMP).
6. Future timestamp handling (evaluates to UNASSESSED_FUTURE_TIMESTAMP).
7. Periodic freshness update (rechecks every 60 seconds / 60,000 ms via setInterval).
8. Marker style changes when stale (dynamic hazard markers turn grey #95a5a6).
9. Tooltips and popups change to UNASSESSED when stale.
10. Freshness restoration only when observation timestamp genuinely supports it.
11. All 12,052 dynamic grid points remain present with zero duplicate grid IDs in point GeoJSON.
12. Historical events (556) and background samples (429) remain separate and unchanged.
13. GeoJSON feature count remains 13,037 with strict RFC 7946 [lon, lat] ordering.
14. Shared popup template and compact HTML architecture remain intact (<= 10 MB).
15. Stage 8.4 dynamic risk dataset integrity and SHA-256 hash preservation.
16. Overwrite protection against accidental data destruction and boundary preservation.
17. Boundary condition at exactly 12.000000 hours (FRESH).
18. Boundary condition at 12.000001 hours (STALE_DATA_UNASSESSED).
19. Regional polygon dataset cell coverage (all 12,052 source cells represented exactly once).
20. Cell-level polygon dataset completeness, uniqueness, and geometry validity.
21. Regional polygon geometries valid, non-empty, and free of self-intersections.
22. Risk category assignment matches Stage 8.4 underlying cells exactly.
23. Regional polygon summary statistics (min, mean, max risk, terrain/rain components) match underlying cells.
24. RFC 7946 WGS84 polygon coordinate ordering [lon, lat].
25. All active Stage 8.4 risk categories represented in regional polygons with approved palette.
26. Output file sizes and browser performance optimization.
27. Validation report and metadata integrity confirming 100% cell coverage and valid geometries.
"""

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unittest

import numpy as np
import pandas as pd
import shapely
import shapely.validation
from shapely.geometry import shape

from backend.create_himalayas_risk_map import (
    DEFAULT_INPUT_CSV,
    DEFAULT_INPUT_METADATA,
    DEFAULT_TRAINING_DATASET,
    DEFAULT_OUTPUT_HTML,
    DEFAULT_OUTPUT_GEOJSON,
    DEFAULT_OUTPUT_REGIONS_GEOJSON,
    DEFAULT_OUTPUT_CELLS_GEOJSON,
    DEFAULT_OUTPUT_METADATA,
    DEFAULT_VALIDATION_REPORT,
    EXPECTED_LOCATIONS_COUNT,
    EXPECTED_TILES_COUNT,
    EXPECTED_HISTORICAL_EVENTS_COUNT,
    EXPECTED_BACKGROUND_SAMPLES_COUNT,
    LAT_BOUNDS,
    LON_BOUNDS,
    MAX_DATA_LATENCY_HOURS,
    APPROVED_P50,
    APPROVED_P80,
    APPROVED_P95,
    CATEGORY_COLORS,
    DISCLAIMER_TEXT,
    validate_stage84_metadata_thresholds,
    verify_himalayas_map_freshness_gate,
    generate_himalayas_dynamic_risk_map,
    export_himalayas_dynamic_risk_geojson,
    safe_html,
    format_score_val,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestHimalayasDynamicRiskMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_csv = DEFAULT_INPUT_CSV
        cls.metadata_json = DEFAULT_INPUT_METADATA
        cls.training_csv = DEFAULT_TRAINING_DATASET
        cls.output_html = DEFAULT_OUTPUT_HTML
        cls.output_geojson = DEFAULT_OUTPUT_GEOJSON
        cls.output_regions_geojson = DEFAULT_OUTPUT_REGIONS_GEOJSON
        cls.output_cells_geojson = DEFAULT_OUTPUT_CELLS_GEOJSON
        cls.output_metadata = DEFAULT_OUTPUT_METADATA
        cls.validation_report = DEFAULT_VALIDATION_REPORT

        # Expected SHA-256 hash of Stage 8.4 dynamic risk CSV
        cls.approved_stage84_csv_hash = "27B6CAC8BA78B2504BAE793F2A165A83039B6E6D13D9B91D66D0D83788EA11CE"

    @classmethod
    def _execute_node_test(cls, snippet: str) -> dict:
        """Execute a test snippet against the embedded client-side map JavaScript using Node."""
        with open(cls.output_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        match = re.search(r'<script>\s*\(function\(\)\s*\{[\s\S]*?\}\)\(\);\s*</script>', html_content)
        if not match:
            raise ValueError("Embedded client script block not found in HTML map")

        script_code = match.group(0).replace('<script>', '').replace('</script>', '')
        script_code = re.sub(r'var lg(Risk|Terrain|Rain) = [a-zA-Z0-9_]+;', r'var lg\1 = feature_group_mock;', script_code)

        runner_js = f"""
global.document = {{
    readyState: 'complete',
    getElementById: function(id) {{
        if (!this.elements[id]) {{
            this.elements[id] = {{ style: {{}}, textContent: '' }};
        }}
        return this.elements[id];
    }},
    elements: {{}},
    addEventListener: function() {{}}
}};
global.window = global;
global.lastIntervalMs = 0;
global.setInterval = function(fn, ms) {{ global.lastIntervalMs = ms; }};
global.feature_group_mock = {{ get_name: () => 'fg_mock' }};
var feature_group_mock = global.feature_group_mock;

global.L = {{
    circleMarker: function(latlng, opts) {{
        return {{
            latlng: latlng,
            opts: opts,
            style: Object.assign({{}}, opts),
            tooltip: null,
            popupFn: null,
            setStyle: function(s) {{ Object.assign(this.style, s); }},
            bindTooltip: function(t) {{ this.tooltip = t; }},
            setTooltipContent: function(t) {{ this.tooltip = t; }},
            bindPopup: function(p) {{ this.popupFn = p; }},
            addTo: function() {{ return this; }}
        }};
    }}
}};

{script_code}

var __output = (function() {{
    {snippet}
}})();
console.log(JSON.stringify(__output));
"""
        import tempfile
        js_file = Path(tempfile.gettempdir()) / f"_test_node_eval_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.js"
        with open(js_file, "w", encoding="utf-8") as jf:
            jf.write(runner_js)

        try:
            res = subprocess.run(["node", str(js_file)], capture_output=True, text=True, check=True)
            return json.loads(res.stdout.strip())
        finally:
            if js_file.exists():
                try:
                    js_file.unlink()
                except Exception:
                    pass

    # =========================================================================
    # DYNAMIC RUNTIME FRESHNESS TESTS
    # =========================================================================

    def test_01_fresh_observation_timestamp(self):
        """Verify observation age <= 12 hours evaluates dynamically to FRESH."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting(new Date(Date.now() - 4 * 3600000).toISOString());
            return window.computeFreshness();
        """)
        self.assertFalse(res["isStale"])
        self.assertEqual(res["statusStr"], "FRESH")
        self.assertAlmostEqual(res["latencyHours"], 4.0, delta=0.1)

    def test_02_exactly_12_hour_boundary_condition(self):
        """Verify observation age at the 12-hour boundary evaluates to FRESH (<= 12.0h)."""
        res = self._execute_node_test("""
            // 11.99 hours ago (strictly <= 12.0h)
            window.setObservationEndTimeForTesting(new Date(Date.now() - 11.99 * 3600000).toISOString());
            return window.computeFreshness();
        """)
        self.assertFalse(res["isStale"])
        self.assertEqual(res["statusStr"], "FRESH")
        self.assertLessEqual(res["latencyHours"], 12.0)

    def test_03_age_above_12_hours_stale(self):
        """Verify observation age > 12 hours evaluates dynamically to STALE_DATA_UNASSESSED."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting(new Date(Date.now() - 15 * 3600000).toISOString());
            return window.computeFreshness();
        """)
        self.assertTrue(res["isStale"])
        self.assertEqual(res["statusStr"], "STALE_DATA_UNASSESSED")
        self.assertGreater(res["latencyHours"], 12.0)

    def test_04_missing_timestamp_unassessed(self):
        """Verify missing observation timestamp evaluates dynamically to UNASSESSED_MISSING_TIMESTAMP."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting("");
            return window.computeFreshness();
        """)
        self.assertTrue(res["isStale"])
        self.assertEqual(res["statusStr"], "UNASSESSED_MISSING_TIMESTAMP")
        self.assertTrue(res["latencyHours"] is None or math.isnan(res["latencyHours"]))

    def test_05_invalid_timestamp_unassessed(self):
        """Verify invalid or unparseable observation timestamp evaluates to UNASSESSED_INVALID_TIMESTAMP."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting("INVALID_TIMESTAMP_STRING");
            return window.computeFreshness();
        """)
        self.assertTrue(res["isStale"])
        self.assertEqual(res["statusStr"], "UNASSESSED_INVALID_TIMESTAMP")
        self.assertTrue(res["latencyHours"] is None or math.isnan(res["latencyHours"]))

    def test_06_future_timestamp_unassessed(self):
        """Verify observation timestamp in the future evaluates dynamically to UNASSESSED_FUTURE_TIMESTAMP."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting(new Date(Date.now() + 3600000).toISOString());
            return window.computeFreshness();
        """)
        self.assertTrue(res["isStale"])
        self.assertEqual(res["statusStr"], "UNASSESSED_FUTURE_TIMESTAMP")
        self.assertLess(res["latencyHours"], 0.0)

    def test_07_periodic_freshness_update_interval(self):
        """Verify map rechecks freshness every 60 seconds (60,000 ms) via setInterval."""
        res = self._execute_node_test("""
            return { intervalMs: global.lastIntervalMs };
        """)
        self.assertEqual(res["intervalMs"], 60000)

    def test_08_marker_style_changes_when_stale(self):
        """Verify dynamic hazard and rainfall markers dynamically change to grey #95a5a6 when stale."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting(new Date(Date.now() - 16 * 3600000).toISOString());
            var m0 = window.riskMarkers[0];
            var r0 = window.rainMarkers[0];
            return {
                isStale: window.getIsStale(),
                riskColor: m0.marker.style.fillColor,
                rainColor: r0.marker.style.fillColor,
                badgeText: document.getElementById('dashboard-freshness-badge').textContent,
                badgeBg: document.getElementById('dashboard-freshness-badge').style.background
            };
        """)
        self.assertTrue(res["isStale"])
        self.assertEqual(res["riskColor"], "#95a5a6")
        self.assertEqual(res["rainColor"], "#95a5a6")
        self.assertEqual(res["badgeText"], "STALE / UNASSESSED")
        self.assertIn(res["badgeBg"], ["#e74c3c", "rgb(231, 76, 60)"])

    def test_09_tooltips_and_popups_change_to_unassessed_when_stale(self):
        """Verify marker tooltips and popups change to UNASSESSED when stale."""
        res = self._execute_node_test("""
            window.setObservationEndTimeForTesting(new Date(Date.now() - 16 * 3600000).toISOString());
            var m0 = window.riskMarkers[0];
            var r0 = window.rainMarkers[0];
            var popupHtml = window.renderPopupHtml(m0.data);
            return {
                riskTooltip: m0.marker.tooltip,
                rainTooltip: r0.marker.tooltip,
                popupContainsUnassessed: popupHtml.includes("UNASSESSED"),
                popupContainsStaleStatus: popupHtml.includes("STALE_DATA_UNASSESSED"),
                popupContainsStaleRisk: popupHtml.includes("Unassessed (Stale)")
            };
        """)
        self.assertIn("UNASSESSED Risk (Stale Rainfall Data)", res["riskTooltip"])
        self.assertIn("Rain Contrib: UNASSESSED (Stale GPM Rainfall)", res["rainTooltip"])
        self.assertTrue(res["popupContainsUnassessed"])
        self.assertTrue(res["popupContainsStaleStatus"])
        self.assertTrue(res["popupContainsStaleRisk"])

    def test_10_freshness_restoration_only_when_genuinely_supported(self):
        """Verify dynamic markers and popups restore active hazard categories only when observation is fresh."""
        res = self._execute_node_test("""
            // 1. Force stale
            window.setObservationEndTimeForTesting(new Date(Date.now() - 20 * 3600000).toISOString());
            var staleColor = window.riskMarkers[0].marker.style.fillColor;
            // 2. Restore fresh observation timestamp (2 hours ago)
            window.setObservationEndTimeForTesting(new Date(Date.now() - 2 * 3600000).toISOString());
            var restoredColor = window.riskMarkers[0].marker.style.fillColor;
            var origApprovedColor = window.riskMarkers[0].data[22];
            var restoredTooltip = window.riskMarkers[0].marker.tooltip;
            var restoredPopup = window.renderPopupHtml(window.riskMarkers[0].data);
            return {
                staleColor: staleColor,
                restoredColor: restoredColor,
                origApprovedColor: origApprovedColor,
                restoredTooltip: restoredTooltip,
                isStale: window.getIsStale(),
                badgeText: document.getElementById('dashboard-freshness-badge').textContent,
                popupContainsRisk: restoredPopup.includes("Dynamic Risk Index")
            };
        """)
        self.assertEqual(res["staleColor"], "#95a5a6")
        self.assertEqual(res["restoredColor"], res["origApprovedColor"])
        self.assertFalse(res["isStale"])
        self.assertEqual(res["badgeText"], "FRESH (OPERATIONAL)")
        self.assertIn("Risk", res["restoredTooltip"])
        self.assertTrue(res["popupContainsRisk"])

    # =========================================================================
    # PRODUCTION INTEGRITY & DATASET TESTS
    # =========================================================================

    def test_11_all_12052_dynamic_grid_points_present_no_duplicates(self):
        """Verify all 12,052 Stage 8.4 grid locations are represented with zero duplicates."""
        with open(self.output_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        grid_feats = [
            f for f in data["features"]
            if f["properties"].get("feature_type") == "dynamic_grid_prediction"
        ]
        self.assertEqual(len(grid_feats), EXPECTED_LOCATIONS_COUNT)

        grid_ids = [f["properties"]["grid_id"] for f in grid_feats]
        self.assertEqual(len(grid_ids), len(set(grid_ids)))

    def test_12_historical_events_and_background_samples_separated(self):
        """Verify historical events (556) and background samples (429) remain separated from grid."""
        with open(self.output_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        event_feats = [
            f for f in data["features"]
            if f["properties"].get("feature_type") == "historical_landslide_event"
        ]
        bg_feats = [
            f for f in data["features"]
            if f["properties"].get("feature_type") == "provisional_background_sample"
        ]
        self.assertEqual(len(event_feats), EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual(len(bg_feats), EXPECTED_BACKGROUND_SAMPLES_COUNT)

    def test_13_geojson_total_features_count_and_lon_lat_ordering(self):
        """Verify GeoJSON feature count is 13,037 with RFC 7946 coordinates ordered [lon, lat]."""
        with open(self.output_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["features"]), 13037)
        for feat in data["features"]:
            coords = feat["geometry"]["coordinates"]
            self.assertEqual(len(coords), 2)
            lon, lat = coords
            self.assertTrue(LON_BOUNDS[0] <= lon <= LON_BOUNDS[1])
            self.assertTrue(LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1])

    def test_14_shared_popup_architecture_and_compact_html_size(self):
        """Verify shared popup template used and HTML size optimized below 10 MB."""
        size_bytes = self.output_html.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        self.assertLess(size_mb, 10.0)
        self.assertGreater(size_mb, 2.0)

        with open(self.output_html, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("gridRecords", content)
        self.assertIn("renderPopupHtml", content)
        self.assertIn("populateLayers", content)
        self.assertIn("updateFreshnessStatus", content)
        self.assertIn("computeFreshness", content)

    def test_15_stage84_csv_hash_preservation(self):
        """Verify Stage 8.4 dynamic risk CSV has not been modified and retains exact approved hash."""
        import hashlib
        with open(self.input_csv, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(current_hash, self.approved_stage84_csv_hash)

    def test_16_overwrite_and_protection_boundaries(self):
        """Verify overwrite protection raises FileExistsError and prior stage outputs remain intact."""
        with self.assertRaises(FileExistsError):
            generate_himalayas_dynamic_risk_map(
                input_csv_path=self.input_csv,
                input_metadata_path=self.metadata_json,
                training_dataset_path=self.training_csv,
                output_html_path=self.output_html,
                output_geojson_path=self.output_geojson,
                output_metadata_path=self.output_metadata,
                validation_report_path=self.validation_report,
                overwrite=False,
            )

        root = PROJECT_ROOT
        protected_paths = [
            root / "outputs" / "western_ghats_dynamic_risk_map.html",
            root / "data" / "processed" / "western_ghats_dynamic_risk" / "western_ghats_dynamic_hazard_risk.csv",
            root / "outputs" / "northeast_india_dynamic_risk_map.html",
            root / "data" / "processed" / "northeast_dynamic_risk_map" / "northeast_india_dynamic_hazard_risk.geojson",
            root / "data" / "processed" / "northeast_dynamic_risk" / "northeast_india_5km_dynamic_hazard_risk.csv",
            root / "data" / "processed" / "himalayas_spatial_grid" / "himalayas_5km_spatial_grid.csv",
            root / "data" / "processed" / "himalayas_terrain_predictions" / "himalayas_5km_terrain_susceptibility.csv",
            root / "data" / "processed" / "himalayas_gpm_rainfall" / "himalayas_5km_rainfall_terrain_integrated.csv",
            root / "data" / "processed" / "himalayas_dynamic_risk" / "himalayas_5km_dynamic_hazard_risk.csv",
        ]
        for p in protected_paths:
            self.assertTrue(p.exists(), f"Protected file missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Protected file empty: {p}")

    def test_17_boundary_exact_12_hours_zero_fresh(self):
        """Verify elapsed age of exactly 12.000000 hours evaluates to FRESH."""
        res = self._execute_node_test("""
            var originalNow = Date.now;
            var fixedNow = 1700000000000;
            Date.now = function() { return fixedNow; };
            // Exactly 12.000000 hours = 43,200,000 ms
            var obsTime = fixedNow - 43200000;
            window.setObservationEndTimeForTesting(new Date(obsTime).toISOString());
            var out = window.computeFreshness();
            Date.now = originalNow;
            return out;
        """)
        self.assertFalse(res["isStale"])
        self.assertEqual(res["statusStr"], "FRESH")
        self.assertAlmostEqual(res["latencyHours"], 12.0, places=6)

    def test_18_boundary_12_hours_plus_epsilon_stale(self):
        """Verify elapsed age of 12.000001 hours evaluates to STALE_DATA_UNASSESSED."""
        res = self._execute_node_test("""
            var originalNow = Date.now;
            var fixedNow = 1700000000000;
            Date.now = function() { return fixedNow; };
            // 12.000001 hours = 43,200,004 ms
            var obsTime = fixedNow - 43200004;
            window.setObservationEndTimeForTesting(new Date(obsTime).toISOString());
            var out = window.computeFreshness();
            Date.now = originalNow;
            return out;
        """)
        self.assertTrue(res["isStale"])
        self.assertEqual(res["statusStr"], "STALE_DATA_UNASSESSED")
        self.assertGreater(res["latencyHours"], 12.0)
        self.assertAlmostEqual(res["latencyHours"], 12.000001, places=5)

    # =========================================================================
    # REGIONAL POLYGON ENHANCEMENT TESTS (SECTION 8)
    # =========================================================================

    def test_19_all_source_grid_cells_represented_in_regional_polygons(self):
        """Verify all 12,052 source grid cells are represented exactly once in regional polygons."""
        self.assertTrue(self.output_regions_geojson.exists(), f"Missing {self.output_regions_geojson}")
        with open(self.output_regions_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data["features"]
        self.assertGreater(len(features), 0, "No regional polygon features found")

        total_cell_count = sum(f["properties"]["cell_count"] for f in features)
        self.assertEqual(
            total_cell_count,
            EXPECTED_LOCATIONS_COUNT,
            f"Expected {EXPECTED_LOCATIONS_COUNT} cells, got {total_cell_count}",
        )

        all_grid_ids = []
        for feat in features:
            gids = feat["properties"].get("grid_ids", [])
            self.assertEqual(len(gids), feat["properties"]["cell_count"])
            all_grid_ids.extend(gids)

        self.assertEqual(len(all_grid_ids), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(len(set(all_grid_ids)), EXPECTED_LOCATIONS_COUNT)

        df_src = pd.read_csv(self.input_csv)
        self.assertEqual(set(all_grid_ids), set(df_src["grid_id"].tolist()))

    def test_20_cell_level_polygon_geojson_completeness_and_uniqueness(self):
        """Verify cell-level polygon dataset has 12,052 unique, non-empty, valid 5 km cells."""
        self.assertTrue(self.output_cells_geojson.exists(), f"Missing {self.output_cells_geojson}")
        with open(self.output_cells_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data["features"]
        self.assertEqual(len(features), EXPECTED_LOCATIONS_COUNT)

        cell_ids = [f["properties"]["grid_id"] for f in features]
        self.assertEqual(len(set(cell_ids)), EXPECTED_LOCATIONS_COUNT)

        for f in features:
            geom = shape(f["geometry"])
            self.assertTrue(geom.is_valid, f"Cell polygon {f['properties']['grid_id']} invalid")
            self.assertFalse(geom.is_empty, f"Cell polygon {f['properties']['grid_id']} empty")
            self.assertEqual(geom.geom_type, "Polygon")

    def test_21_regional_polygon_geometries_valid_and_non_empty(self):
        """Verify all regional hazard polygons are valid, non-empty geometries without self-intersections."""
        with open(self.output_regions_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        for f in data["features"]:
            geom = shape(f["geometry"])
            self.assertTrue(geom.is_valid, f"Region {f['properties']['region_id']} geometry invalid: {shapely.validation.explain_validity(geom)}")
            self.assertFalse(geom.is_empty, f"Region {f['properties']['region_id']} geometry is empty")
            self.assertIn(geom.geom_type, ["Polygon", "MultiPolygon"])

    def test_22_risk_category_assignment_matches_stage84_exactly(self):
        """Verify risk category assignment for each region matches Stage 8.4 underlying cells exactly."""
        df_src = pd.read_csv(self.input_csv).set_index("grid_id")

        with open(self.output_regions_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feat in data["features"]:
            reg_cat = feat["properties"]["primary_screening_category"]
            gids = feat["properties"]["grid_ids"]
            subset = df_src.loc[gids]
            cats = subset["primary_screening_category"].unique()
            self.assertEqual(len(cats), 1, f"Region contains mixed categories: {cats}")
            self.assertEqual(cats[0], reg_cat)

    def test_23_polygon_summary_statistics_match_underlying_cells(self):
        """Verify polygon summary statistics (min, mean, max risk, terrain/rain components) match underlying cells."""
        df_src = pd.read_csv(self.input_csv).set_index("grid_id")

        with open(self.output_regions_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feat in data["features"]:
            p = feat["properties"]
            gids = p["grid_ids"]
            subset = df_src.loc[gids]

            expected_min = round(float(subset["dynamic_risk_index"].min()), 2)
            expected_mean = round(float(subset["dynamic_risk_index"].mean()), 2)
            expected_max = round(float(subset["dynamic_risk_index"].max()), 2)
            expected_tcontrib = round(float(subset["terrain_contribution"].mean()), 2)
            expected_rcontrib = round(float(subset["rainfall_contribution"].mean()), 2)
            expected_tiles = sorted(subset["tile"].unique().tolist())

            self.assertAlmostEqual(p["dynamic_risk_min"], expected_min, places=2)
            self.assertAlmostEqual(p["dynamic_risk_mean"], expected_mean, places=2)
            self.assertAlmostEqual(p["dynamic_risk_max"], expected_max, places=2)
            self.assertAlmostEqual(p["terrain_contrib_mean"], expected_tcontrib, places=2)
            self.assertAlmostEqual(p["rainfall_contrib_mean"], expected_rcontrib, places=2)
            self.assertEqual(p["tiles_represented"], expected_tiles)

    def test_24_rfc7946_wgs84_polygon_coordinate_ordering(self):
        """Verify RFC 7946 coordinate ordering [lon, lat] within Himalayas bounds."""
        with open(self.output_regions_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feat in data["features"]:
            geom = feat["geometry"]
            coords_list = geom["coordinates"]
            rings = coords_list if geom["type"] == "Polygon" else [ring for poly in coords_list for ring in poly]
            for ring in rings:
                for pt in ring:
                    self.assertEqual(len(pt), 2)
                    lon, lat = pt
                    self.assertTrue(
                        LON_BOUNDS[0] - 0.5 <= lon <= LON_BOUNDS[1] + 0.5,
                        f"Longitude {lon} out of bounds {LON_BOUNDS}",
                    )
                    self.assertTrue(
                        LAT_BOUNDS[0] - 0.5 <= lat <= LAT_BOUNDS[1] + 0.5,
                        f"Latitude {lat} out of bounds {LAT_BOUNDS}",
                    )

    def test_25_all_risk_categories_represented_in_regional_polygons(self):
        """Verify all active Stage 8.4 relative categories are represented with approved colors."""
        with open(self.output_regions_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        cats_found = set(f["properties"]["primary_screening_category"] for f in data["features"])
        for cat in ["Low", "Moderate", "High", "Very High"]:
            self.assertIn(cat, cats_found, f"Category {cat} missing from regional polygons")

        for f in data["features"]:
            cat = f["properties"]["primary_screening_category"]
            color = f["properties"]["color"]
            self.assertEqual(color, CATEGORY_COLORS[cat])

    def test_26_file_sizes_and_browser_performance(self):
        """Verify output sizes remain compact, standalone, and optimized for browser rendering."""
        html_size = self.output_html.stat().st_size / (1024 * 1024)
        regions_size = self.output_regions_geojson.stat().st_size / (1024 * 1024)
        cells_size = self.output_cells_geojson.stat().st_size / (1024 * 1024)

        self.assertLess(html_size, 10.0, f"HTML size {html_size:.2f} MB exceeds 10 MB")
        self.assertGreater(html_size, 2.0)

        self.assertLess(regions_size, 8.0, f"Regional GeoJSON size {regions_size:.2f} MB exceeds 8 MB")
        self.assertGreater(regions_size, 1.0)

        self.assertLess(cells_size, 25.0, f"Cells GeoJSON size {cells_size:.2f} MB exceeds 25 MB")
        self.assertGreater(cells_size, 5.0)

    def test_27_validation_report_and_metadata_integrity(self):
        """Verify validation report and metadata confirm 100% cell coverage and geometry validity."""
        with open(self.validation_report, "r", encoding="utf-8") as f:
            val = json.load(f)
        self.assertEqual(val["status"], "SUCCESS")
        self.assertEqual(val["grid_locations_count"], EXPECTED_LOCATIONS_COUNT)
        self.assertTrue(val["validation_checks"]["all_grid_cells_in_regional_polygons"])
        self.assertTrue(val["validation_checks"]["all_polygon_geometries_valid"])

        with open(self.output_metadata, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["total_grid_locations"], EXPECTED_LOCATIONS_COUNT)
        self.assertTrue(meta["regional_polygons"]["all_geometries_valid"])
        self.assertEqual(meta["regional_polygons"]["total_cells_covered"], EXPECTED_LOCATIONS_COUNT)


if __name__ == "__main__":
    unittest.main()
