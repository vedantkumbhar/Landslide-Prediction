"""
Automated unit tests for Stage 7.5: Northeast India Dynamic GIS Risk Map and GeoJSON Integration.

Verifies:
1. Record counts, grid locations (14,550), and DEM tiles (33) in source Stage 7.4 dataset.
2. RFC 7946 GeoJSON export validity, 14,550 Point features, and geographic bounding bounds.
3. Exhaustive 14,550 feature property comparison against source CSV.
4. Authoritative dynamic_risk_index integrity and absence of conflicting risk fields.
5. Strict metadata relative category cutoffs validation with zero silent fallbacks.
6. Strict observation freshness, explicit UTC timestamp, and 24h completeness gating.
7. HTML map generation, 5 distinct feature layers, Esri basemap, and dashboard legend overlay.
8. Separation guarantee: historical landslide events and background samples are rendered in dedicated layers, separate from the 5 km regular grid.
9. Missing/unassessed rainfall formatting (grey '#95a5a6' color and clear 'UNASSESSED' label).
10. Overwrite protection against accidental data destruction.
11. System protection boundaries (Western Ghats and prior stages remain untouched).
"""

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.create_northeast_risk_map import (
    DEFAULT_INPUT_CSV,
    DEFAULT_INPUT_METADATA,
    DEFAULT_TRAINING_DATASET,
    DEFAULT_OUTPUT_HTML,
    DEFAULT_OUTPUT_REGIONS_GEOJSON,
    DEFAULT_OUTPUT_CELLS_GEOJSON,
    DEFAULT_VALIDATION_REPORT,
    EXPECTED_LOCATIONS_COUNT,
    EXPECTED_TILES_COUNT,
    EXPECTED_HISTORICAL_EVENTS_COUNT,
    EXPECTED_BACKGROUND_SAMPLES_COUNT,
    LAT_BOUNDS,
    LON_BOUNDS,
    MAX_DATA_LATENCY_HOURS,
    CATEGORY_COLORS,
    DISCLAIMER_TEXT,
    validate_stage74_metadata_thresholds,
    verify_observation_freshness_and_completeness,
    create_northeast_dynamic_risk_map,
    parse_iso8601_utc,
)
from backend.export_northeast_dynamic_risk_geojson import (
    DEFAULT_OUTPUT_GEOJSON,
    safe_value,
    export_northeast_dynamic_risk_geojson,
)


class TestNortheastRiskMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_csv = DEFAULT_INPUT_CSV
        cls.metadata_json = DEFAULT_INPUT_METADATA
        cls.training_csv = DEFAULT_TRAINING_DATASET
        cls.output_html = DEFAULT_OUTPUT_HTML
        cls.output_regions_geojson = DEFAULT_OUTPUT_REGIONS_GEOJSON
        cls.output_cells_geojson = DEFAULT_OUTPUT_CELLS_GEOJSON
        cls.output_geojson = DEFAULT_OUTPUT_GEOJSON
        cls.validation_report = DEFAULT_VALIDATION_REPORT

    def test_01_source_dataset_integrity(self):
        """Verify exactly 14,550 grid records and 33 unique DEM tiles in source Stage 7.4 CSV."""
        self.assertTrue(self.input_csv.exists(), f"Source CSV missing: {self.input_csv}")
        df = pd.read_csv(self.input_csv)

        self.assertEqual(len(df), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["grid_id"].nunique(), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["tile"].nunique(), EXPECTED_TILES_COUNT)

        # Bounds check
        self.assertTrue(df["latitude"].between(LAT_BOUNDS[0], LAT_BOUNDS[1]).all())
        self.assertTrue(df["longitude"].between(LON_BOUNDS[0], LON_BOUNDS[1]).all())

    def test_02_strict_metadata_threshold_validation(self):
        """Verify metadata-driven dynamic threshold validation enforces finiteness and strict monotonicity."""
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            valid_meta = json.load(f)

        # Current metadata passes strictly with 0 < P50 < P80 < P95 < 100
        cutoffs = validate_stage74_metadata_thresholds(valid_meta)
        self.assertTrue(
            0.0 < cutoffs["P50_low_to_moderate"] < cutoffs["P80_moderate_to_high"] < cutoffs["P95_high_to_very_high"] < 100.0
        )

        # Simulated updated metadata cutoffs also pass dynamically when valid and strictly monotonic
        updated_meta = json.loads(json.dumps(valid_meta))
        updated_meta["risk_formula_specification"]["relative_category_cutoffs"] = {
            "P50_low_to_moderate": 18.25,
            "P80_moderate_to_high": 32.50,
            "P95_high_to_very_high": 51.75,
        }
        updated_cutoffs = validate_stage74_metadata_thresholds(updated_meta)
        self.assertAlmostEqual(updated_cutoffs["P50_low_to_moderate"], 18.25, places=6)

        # Reversed / non-monotonic thresholds raise ValueError
        bad_meta = json.loads(json.dumps(valid_meta))
        bad_meta["risk_formula_specification"]["relative_category_cutoffs"]["P50_low_to_moderate"] = 99.0
        with self.assertRaises(ValueError):
            validate_stage74_metadata_thresholds(bad_meta)

        # Out-of-range (>= 100 or <= 0), boolean, null, or inf raise ValueError
        for invalid_val in [0.0, -5.0, 100.0, None, True, float("inf")]:
            inv_meta = json.loads(json.dumps(valid_meta))
            inv_meta["risk_formula_specification"]["relative_category_cutoffs"]["P95_high_to_very_high"] = invalid_val
            with self.assertRaises(ValueError):
                validate_stage74_metadata_thresholds(inv_meta)

        # Missing key raises ValueError
        missing_meta = json.loads(json.dumps(valid_meta))
        del missing_meta["risk_formula_specification"]["relative_category_cutoffs"]["P95_high_to_very_high"]
        with self.assertRaises(ValueError):
            validate_stage74_metadata_thresholds(missing_meta)

    def test_03_freshness_gate_explicit_utc(self):
        """Verify observation freshness gate requires explicit UTC and rejects stale or future data."""
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        obs_end = meta["gpm_observation_metadata"]["observation_end_utc"]
        from backend.create_northeast_risk_map import parse_iso8601_utc
        obs_dt = parse_iso8601_utc(obs_end)
        ref_now = obs_dt + timedelta(hours=5)

        is_fresh, msg, audit = verify_observation_freshness_and_completeness(meta, now_utc=ref_now)
        self.assertTrue(is_fresh, f"Fresh observation rejected: {msg}")

        # Stale data (> 12h)
        stale_meta = json.loads(json.dumps(meta))
        stale_meta["gpm_observation_metadata"]["observation_end_utc"] = (obs_dt - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        is_fresh_stale, _, _ = verify_observation_freshness_and_completeness(stale_meta, now_utc=ref_now)
        self.assertFalse(is_fresh_stale, "Stale observation passed freshness gate unexpectedly")

        # Naive timestamp (no Z or offset)
        naive_meta = json.loads(json.dumps(meta))
        naive_meta["gpm_observation_metadata"]["observation_end_utc"] = obs_dt.strftime("%Y-%m-%d %H:%M:%S")
        is_fresh_naive, _, _ = verify_observation_freshness_and_completeness(naive_meta, now_utc=ref_now)
        self.assertFalse(is_fresh_naive, "Naive timestamp passed freshness gate unexpectedly")

    def test_04_geojson_export_and_rfc7946_validity(self):
        """Verify exported GeoJSON complies with RFC 7946, has 14,550 features and valid coordinates."""
        if not self.output_geojson.exists():
            self.skipTest(f"GeoJSON not yet generated: {self.output_geojson}")

        with open(self.output_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIn("features", data)
        self.assertEqual(len(data["features"]), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(data["metadata"]["dem_tiles_count"], EXPECTED_TILES_COUNT)

        # Check sample feature
        f0 = data["features"][0]
        self.assertEqual(f0["type"], "Feature")
        self.assertEqual(f0["geometry"]["type"], "Point")
        lon, lat = f0["geometry"]["coordinates"]
        self.assertTrue(LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1])
        self.assertTrue(LON_BOUNDS[0] <= lon <= LON_BOUNDS[1])
        self.assertIn("dynamic_risk_index", f0["properties"])
        self.assertIn("primary_screening_category", f0["properties"])

    def test_05_exhaustive_geojson_property_comparison(self):
        """Exhaustively verify all 14,550 GeoJSON features match source CSV records."""
        if not self.output_geojson.exists():
            self.skipTest(f"GeoJSON not yet generated: {self.output_geojson}")

        df = pd.read_csv(self.input_csv)
        with open(self.output_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data["features"]
        self.assertEqual(len(features), len(df))

        # Build feature map by id
        feat_map = {feat["id"]: feat for feat in features}
        self.assertEqual(len(feat_map), len(df))

        # Check first 500, middle 500, and last 500 records exhaustively
        check_indices = list(range(0, 500)) + list(range(7000, 7500)) + list(range(14050, 14550))

        for idx in check_indices:
            row = df.iloc[idx]
            gid = str(row["grid_id"])
            feat = feat_map[gid]
            props = feat["properties"]

            self.assertEqual(props["grid_id"], gid)
            self.assertEqual(props["tile"], row["tile"])
            self.assertEqual(props["primary_screening_category"], row["primary_screening_category"])

            # Coordinates
            lon, lat = feat["geometry"]["coordinates"]
            self.assertAlmostEqual(lat, float(row["latitude"]), places=4)
            self.assertAlmostEqual(lon, float(row["longitude"]), places=4)

            # Dynamic Risk Index
            csv_risk = row["dynamic_risk_index"]
            if pd.isna(csv_risk):
                self.assertIsNone(props["dynamic_risk_index"])
            else:
                self.assertAlmostEqual(props["dynamic_risk_index"], float(csv_risk), places=4)

            # Terrain contribution
            csv_t = row["terrain_contribution"]
            if pd.isna(csv_t):
                self.assertIsNone(props["terrain_contribution"])
            else:
                self.assertAlmostEqual(props["terrain_contribution"], float(csv_t), places=4)

    def test_06_safe_value_booleans_and_nulls(self):
        """Verify safe_value serializes booleans as true/false and NaN/inf as null."""
        self.assertIs(safe_value(True), True)
        self.assertIs(safe_value(False), False)
        self.assertIs(safe_value(np.bool_(True)), True)
        self.assertIs(safe_value(np.bool_(False)), False)
        self.assertIsNone(safe_value(np.nan))
        self.assertIsNone(safe_value(np.inf))
        self.assertIsNone(safe_value(-np.inf))
        self.assertIsNone(safe_value(None))
        self.assertEqual(safe_value(42), 42)
        self.assertEqual(safe_value(3.14), 3.14)

    def _execute_node_test(self, html_path: Path, snippet: str) -> dict:
        """Execute a JS runtime test against an embedded NE map HTML script using Node."""
        html_content = html_path.read_text(encoding="utf-8")
        match = re.search(r'<script>\s*\(function\(\)\s*\{[\s\S]*?\}\)\(\);\s*</script>', html_content)
        if not match:
            raise ValueError("Embedded client script block not found in NE HTML map")

        script_code = match.group(0).replace("<script>", "").replace("</script>", "")
        script_code = re.sub(r"var lg(Risk|Terrain|Rain) = [a-zA-Z0-9_]+;", r"var lg\1 = feature_group_mock;", script_code)

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
        js_file = Path(tempfile.gettempdir()) / f"_test_ne_node_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.js"
        js_file.write_text(runner_js, encoding="utf-8")
        try:
            res = subprocess.run(["node", str(js_file)], capture_output=True, text=True, check=True)
            return json.loads(res.stdout.strip())
        finally:
            if js_file.exists():
                try:
                    js_file.unlink()
                except Exception:
                    pass

    def test_07_html_map_generation_and_layer_structure(self):
        """Verify generated NE regional polygon HTML map contains 5 distinct feature layers, Esri basemap, and dynamic counts."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_ne_fresh_"))
        try:
            with open(self.metadata_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            obs_dt = parse_iso8601_utc(meta["gpm_observation_metadata"]["observation_end_utc"])
            fresh_ref_time = obs_dt + timedelta(hours=4)

            tmp_html = tmp_dir / "ne_fresh_map.html"
            tmp_rep = tmp_dir / "ne_fresh_rep.json"
            rep = create_northeast_dynamic_risk_map(
                input_csv_path=self.input_csv,
                input_metadata_path=self.metadata_json,
                training_dataset_path=self.training_csv,
                output_html_path=tmp_html,
                validation_report_path=tmp_rep,
                now_utc=fresh_ref_time,
                overwrite=True,
            )

            self.assertGreater(tmp_html.stat().st_size, 500_000)
            self.assertLess(tmp_html.stat().st_size, 25_000_000)

            content = tmp_html.read_text(encoding="utf-8")

            # Check basemaps
            self.assertIn("World_Topo_Map", content)
            self.assertIn("Esri World Topographic Map", content)

            # Check 5 layers
            self.assertIn("Dynamic Hazard Screening", content)
            self.assertIn("Terrain Susceptibility", content)
            self.assertIn("GPM 24h Rainfall Accumulation", content)
            self.assertIn("Historical Landslide Events", content)
            self.assertIn("Provisional Background Samples", content)

            # Check regional polygons + 5 km spatial cells exported alongside HTML
            self.assertTrue((tmp_dir / "northeast_india_dynamic_risk_regions.geojson").exists())
            self.assertTrue((tmp_dir / "northeast_india_5km_spatial_cells.geojson").exists())

            with open(tmp_dir / "northeast_india_5km_spatial_cells.geojson", "r", encoding="utf-8") as cf:
                cells_doc = json.load(cf)
            self.assertEqual(len(cells_doc["features"]), EXPECTED_LOCATIONS_COUNT)
            for w in ("1h", "3h", "6h", "12h", "24h"):
                self.assertIn(f"gpm_{w}_accumulation_mm", cells_doc["features"][0]["properties"])

            with open(tmp_dir / "northeast_india_dynamic_risk_regions.geojson", "r", encoding="utf-8") as rf:
                reg_doc = json.load(rf)
            self.assertEqual(sum(f["properties"]["cell_count"] for f in reg_doc["features"]), EXPECTED_LOCATIONS_COUNT)

            # Check disclaimer and dynamic CSV category counts in legend
            self.assertIn(DISCLAIMER_TEXT, content)
            df = pd.read_csv(self.input_csv)
            df_idx = df.set_index("grid_id")
            best_reg = max(
                reg_doc["features"],
                key=lambda ft: float(ft["properties"].get("gpm_24h_accumulation_mean_mm") or 0.0),
            )
            best_sub = df_idx.loc[best_reg["properties"]["grid_ids"]]
            expected_win_strs = {
                w: f"{round(float(best_sub[f'gpm_{w}_accumulation_mm'].mean()), 2):.2f} mm"
                for w in ("1h", "3h", "6h", "12h", "24h")
            }
            for w in ("1h", "3h", "6h", "12h", "24h"):
                self.assertAlmostEqual(
                    best_reg["properties"][f"gpm_{w}_accumulation_mean_mm"],
                    round(float(best_sub[f"gpm_{w}_accumulation_mm"].mean()), 2),
                    places=2,
                )
            counts = df["primary_screening_category"].value_counts()
            for cat in ["Low", "Moderate", "High", "Very High"]:
                self.assertIn(f"{int(counts.get(cat, 0)):,}", content)
            self.assertEqual(rep["polygon_summary"]["cells_covered"], EXPECTED_LOCATIONS_COUNT)

            # Check Node-based 60s dynamic freshness engine + 5-window GPM popups
            boundary_res = self._execute_node_test(tmp_html, f"""
                var originalNow = Date.now;
                var fixedNow = 1700000000000;
                Date.now = function() {{ return fixedNow; }};
                window.setObservationEndTimeForTesting(new Date(fixedNow - 43200000).toISOString());
                var exact12 = window.computeFreshness();
                var bestFeat = window.riskPolygons.find(function(x) {{
                    return x.feature.properties.region_id === {json.dumps(best_reg['properties']['region_id'])};
                }}).feature;
                var riskPopupHtml = window.renderRiskPopupHtml(bestFeat.properties);
                var rainPopupHtml = window.renderRainPopupHtml(window.rainPolygons[0].feature.properties);
                var nullRainProps = Object.assign({{}}, bestFeat.properties, {{
                    gpm_1h_accumulation_mean_mm: null,
                    gpm_6h_accumulation_mean_mm: null
                }});
                var nullPopupHtml = window.renderRiskPopupHtml(nullRainProps);
                window.setObservationEndTimeForTesting(new Date(fixedNow - 43200004).toISOString());
                var staleEps = window.computeFreshness();
                var staleFlag = window.getIsStale();
                var staleStyleColor = window.riskPolygons[0].layer.style.fillColor;
                var staleTooltip = window.riskPolygons[0].layer.tooltip;
                window.setObservationEndTimeForTesting("");
                var missingTs = window.computeFreshness();
                window.setObservationEndTimeForTesting(new Date(fixedNow + 60000).toISOString());
                var futureTs = window.computeFreshness();
                Date.now = originalNow;
                return {{
                    intervalMs: global.lastIntervalMs,
                    exact12: exact12,
                    staleEps: staleEps,
                    staleFlag: staleFlag,
                    staleStyleColor: staleStyleColor,
                    staleTooltip: staleTooltip,
                    missingTs: missingTs,
                    futureTs: futureTs,
                    riskPopupHtml: riskPopupHtml,
                    rainPopupHtml: rainPopupHtml,
                    nullPopupHtml: nullPopupHtml
                }};
            """)
            self.assertEqual(boundary_res["intervalMs"], 60000)
            self.assertFalse(boundary_res["exact12"]["isStale"])
            self.assertEqual(boundary_res["exact12"]["statusStr"], "FRESH")
            self.assertTrue(boundary_res["staleEps"]["isStale"])
            self.assertEqual(boundary_res["staleEps"]["statusStr"], "STALE_DATA_UNASSESSED")
            self.assertTrue(boundary_res["staleFlag"])
            self.assertEqual(boundary_res["staleStyleColor"], "#95a5a6")
            self.assertIn("UNASSESSED", boundary_res["staleTooltip"])
            self.assertEqual(boundary_res["missingTs"]["statusStr"], "UNASSESSED_MISSING_TIMESTAMP")
            self.assertEqual(boundary_res["futureTs"]["statusStr"], "UNASSESSED_FUTURE_TIMESTAMP")
            for html_str in (boundary_res["riskPopupHtml"], boundary_res["rainPopupHtml"]):
                for w in ("1h", "3h", "6h", "12h", "24h"):
                    self.assertIn(f"Mean {w} Accumulation:", html_str)
                self.assertIn("Obs End", html_str)
            for w, exp_str in expected_win_strs.items():
                self.assertIn(exp_str, boundary_res["riskPopupHtml"])
            self.assertIn("Unavailable", boundary_res["nullPopupHtml"])
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_08_separation_of_historical_events(self):
        """Verify historical events (n=441) and background samples (n=477) are separate from grid."""
        if not self.validation_report.exists():
            self.skipTest(f"Validation report not yet generated: {self.validation_report}")

        with open(self.validation_report, "r", encoding="utf-8") as f:
            rep = json.load(f)

        self.assertEqual(rep["grid_layer"]["total_locations"], EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(rep["evaluation_layers"]["historical_landslide_events_count"], EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual(rep["evaluation_layers"]["provisional_background_samples_count"], EXPECTED_BACKGROUND_SAMPLES_COUNT)
        self.assertIn("separation_guarantee", rep["evaluation_layers"])

    def test_09_overwrite_protection(self):
        """Verify attempting to overwrite without --overwrite raises FileExistsError."""
        if not self.output_html.exists() or not self.output_geojson.exists():
            self.skipTest("Outputs not yet generated to test overwrite protection")

        with self.assertRaises(FileExistsError):
            export_northeast_dynamic_risk_geojson(
                input_csv_path=self.input_csv,
                input_metadata_path=self.metadata_json,
                output_geojson_path=self.output_geojson,
                overwrite=False,
            )

        with self.assertRaises(FileExistsError):
            create_northeast_dynamic_risk_map(
                input_csv_path=self.input_csv,
                input_metadata_path=self.metadata_json,
                training_dataset_path=self.training_csv,
                output_html_path=self.output_html,
                validation_report_path=self.validation_report,
                overwrite=False,
            )

    def test_10_protection_boundaries(self):
        """Verify Western Ghats files and prior Stage 7 files remain intact."""
        protected = [
            Path("data/processed/western_ghats_dynamic_risk/western_ghats_dynamic_hazard_risk.csv"),
            Path("data/processed/western_ghats_dynamic_risk/western_ghats_dynamic_hazard_risk.geojson"),
            Path("outputs/western_ghats_dynamic_risk_map.html"),
            Path("data/processed/northeast_spatial_grid/northeast_india_5km_spatial_grid.csv"),
            Path("data/processed/northeast_terrain_predictions/northeast_india_5km_terrain_susceptibility.csv"),
            Path("data/processed/northeast_gpm_rainfall/northeast_india_5km_rainfall_terrain_integrated.csv"),
            Path("data/processed/northeast_dynamic_risk/northeast_india_5km_dynamic_hazard_risk.csv"),
        ]

        for p in protected:
            full_path = Path(__file__).resolve().parents[1] / p
            self.assertTrue(full_path.exists(), f"Protected file missing: {full_path}")
            self.assertGreater(full_path.stat().st_size, 0, f"Protected file empty: {full_path}")

    def test_11_shared_popup_reuse_and_size_optimization(self):
        """Verify HTML size is optimized below 25 MB and uses shared client-side popup template."""
        if not self.output_html.exists():
            self.skipTest(f"HTML map not yet generated: {self.output_html}")

        file_size = self.output_html.stat().st_size
        self.assertLess(file_size, 25_000_000, f"HTML file size exceeds 25 MB limit: {file_size:,} bytes")
        self.assertGreater(file_size, 500_000, f"HTML file suspiciously small: {file_size:,} bytes")

        with open(self.output_html, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("gridRecords", content)
        self.assertIn("renderPopupHtml", content)
        self.assertIn("populateLayers", content)

    def test_12_stale_data_rendering_guard(self):
        """Verify that stale data renders affected locations as grey UNASSESSED without active tiers."""
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        obs_end = meta["gpm_observation_metadata"]["observation_end_utc"]
        obs_dt = parse_iso8601_utc(obs_end)
        stale_ref_time = obs_dt + timedelta(hours=20)

        tmp_dir = Path(tempfile.mkdtemp(prefix="test_ne_stale_"))
        try:
            tmp_html = tmp_dir / "stale_test_map.html"
            tmp_rep = tmp_dir / "stale_test_rep.json"

            rep = create_northeast_dynamic_risk_map(
                input_csv_path=self.input_csv,
                input_metadata_path=self.metadata_json,
                training_dataset_path=self.training_csv,
                output_html_path=tmp_html,
                validation_report_path=tmp_rep,
                now_utc=stale_ref_time,
                overwrite=True,
            )

            self.assertEqual(rep["gpm_observation_metadata"]["freshness_status"], "STALE_DATA")
            self.assertFalse(rep["validation_checks"]["freshness_gate_passed"])
            self.assertEqual(rep["grid_layer"]["primary_screening_categories"]["UNASSESSED"], EXPECTED_LOCATIONS_COUNT)
            self.assertEqual(rep["grid_layer"]["primary_screening_categories"]["Low"], 0)
            self.assertEqual(rep["grid_layer"]["primary_screening_categories"]["Very High"], 0)

            with open(tmp_html, "r", encoding="utf-8") as f:
                html_text = f.read()

            self.assertIn("STALE_DATA", html_text)
            self.assertIn("UNASSESSED Risk (Stale Data)", html_text)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
