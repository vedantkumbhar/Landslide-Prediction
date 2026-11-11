"""
Tests for Western Ghats Limited Pilot — 5 km Regional Polygon Dynamic GIS Map & GeoJSONs.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import pandas as pd
from shapely.geometry import shape

from backend.create_western_ghats_polygon_risk_map import (
    CATEGORY_COLORS,
    DEFAULT_INPUT_CSV,
    DEFAULT_INPUT_METADATA,
    DEFAULT_OUTPUT_CELLS_GEOJSON,
    DEFAULT_OUTPUT_GEOJSON,
    DEFAULT_OUTPUT_HTML,
    DEFAULT_OUTPUT_METADATA,
    DEFAULT_OUTPUT_REGIONS_GEOJSON,
    DEFAULT_VALIDATION_REPORT,
    EXPECTED_BACKGROUND_SAMPLES_COUNT,
    EXPECTED_HISTORICAL_EVENTS_COUNT,
    EXPECTED_LOCATIONS_COUNT,
    EXPECTED_TILES_COUNT,
    LAT_BOUNDS,
    LON_BOUNDS,
    generate_western_ghats_polygon_risk_map,
)
from backend.gis_refresh_utils import validate_dynamic_metadata_thresholds


class TestWesternGhatsPolygonRiskMap(unittest.TestCase):
    @classmethod
    def _execute_node_test(cls, snippet: str) -> dict:
        with open(DEFAULT_OUTPUT_HTML, "r", encoding="utf-8") as f:
            html_content = f.read()

        after_head = html_content.split("</head>", 1)[1]
        scripts = re.findall(r"<script[^>]*>([\s\S]*?)</script>", after_head)
        if len(scripts) < 2:
            raise ValueError(f"Expected >=2 inline scripts after </head>, found {len(scripts)}")

        client_script = scripts[0]
        folium_script = scripts[-1]

        runner_js = f"""
var _domCallbacks = [];
var _loadCallbacks = [];
global.document = {{
    readyState: 'loading',
    getElementById: function(id) {{
        if (!this.elements[id]) {{
            this.elements[id] = {{ style: {{}}, textContent: '' }};
        }}
        return this.elements[id];
    }},
    elements: {{}},
    addEventListener: function(evt, fn) {{
        if (evt === 'DOMContentLoaded') _domCallbacks.push(fn);
    }}
}};
global.window = global;
global.window.addEventListener = function(evt, fn) {{
    if (evt === 'load') _loadCallbacks.push(fn);
}};
global.lastIntervalMs = 0;
global.setInterval = function(fn, ms) {{ global.lastIntervalMs = ms; }};

var _mapInstances = [];
var _tileLayers = [];
var _featureGroups = [];
var _layerControls = [];

function makeFeatureGroup() {{
    var fg = {{
        children: [],
        addedToMap: false,
        frontCount: 0,
        addTo: function(m) {{
            if (m && typeof m.addLayer === 'function') m.addLayer(this);
            this.addedToMap = true;
            return this;
        }},
        addLayer: function(lyr) {{ this.children.push(lyr); return this; }},
        bringToFront: function() {{ this.frontCount += 1; return this; }}
    }};
    _featureGroups.push(fg);
    return fg;
}}

function makeVectorLayer(feat, latlngs) {{
    return {{
        feature: feat || null,
        latlngs: latlngs || null,
        style: {{}},
        tooltip: null,
        popupFn: null,
        addedToGroup: null,
        events: {{}},
        setStyle: function(s) {{ Object.assign(this.style, s); }},
        bindTooltip: function(t) {{ this.tooltip = t; }},
        setTooltipContent: function(t) {{ this.tooltip = t; }},
        bindPopup: function(p) {{ this.popupFn = p; }},
        on: function(ev, fn) {{ this.events[ev] = fn; }},
        addTo: function(fg) {{
            this.addedToGroup = fg;
            if (fg && typeof fg.addLayer === 'function') fg.addLayer(this);
            return this;
        }}
    }};
}}

global.L = {{
    CRS: {{ EPSG3857: 'EPSG:3857' }},
    map: function(id, opts) {{
        var m = {{
            id: id,
            opts: opts || {{}},
            layers: [],
            fitBoundsCalls: [],
            overlayListeners: [],
            addLayer: function(l) {{ if (this.layers.indexOf(l) === -1) this.layers.push(l); return this; }},
            hasLayer: function(l) {{ return this.layers.indexOf(l) !== -1; }},
            fitBounds: function(b, optsB) {{ this.fitBoundsCalls.push({{ bounds: b, opts: optsB }}); return this; }},
            on: function(ev, fn) {{ if (ev === 'overlayadd') this.overlayListeners.push(fn); }},
            triggerOverlayAdd: function() {{
                for (var i = 0; i < this.overlayListeners.length; i++) this.overlayListeners[i]();
            }}
        }};
        _mapInstances.push(m);
        return m;
    }},
    control: {{
        scale: function() {{ return {{ addTo: function() {{ return this; }} }}; }},
        layers: function(baseLayers, overlays, opts) {{
            var ctrl = {{
                baseLayers: baseLayers || {{}},
                overlays: overlays || {{}},
                opts: opts || {{}},
                addTo: function() {{ return this; }}
            }};
            _layerControls.push(ctrl);
            return ctrl;
        }}
    }},
    tileLayer: function(url, opts) {{
        var tl = {{
            url: url,
            opts: opts || {{}},
            addedToMap: false,
            addTo: function(m) {{
                this.addedToMap = true;
                if (m && typeof m.addLayer === 'function') m.addLayer(this);
                return this;
            }}
        }};
        _tileLayers.push(tl);
        return tl;
    }},
    featureGroup: function(opts) {{
        return makeFeatureGroup();
    }},
    geoJSON: function(fc, opts) {{
        var subLayers = [];
        var feats = (fc && fc.features) ? fc.features : [];
        for (var i = 0; i < feats.length; i++) {{
            var lyr = makeVectorLayer(feats[i], null);
            if (opts && typeof opts.style === 'function') {{
                lyr.setStyle(opts.style(feats[i]));
            }}
            if (opts && typeof opts.onEachFeature === 'function') {{
                opts.onEachFeature(feats[i], lyr);
            }}
            subLayers.push(lyr);
        }}
        return {{
            layers: subLayers,
            addTo: function(fg) {{
                for (var j = 0; j < subLayers.length; j++) {{
                    subLayers[j].addTo(fg);
                }}
                return this;
            }}
        }};
    }},
    polygon: function(latlngs, opts) {{
        var lyr = makeVectorLayer(null, latlngs);
        lyr.setStyle(opts || {{}});
        return lyr;
    }},
    circleMarker: function(latlng, opts) {{
        var lyr = makeVectorLayer(null, [latlng]);
        lyr.latlng = latlng;
        lyr.setStyle(opts || {{}});
        return lyr;
    }}
}};

// 1. Evaluate embedded client script BEFORE Folium bottom script (real browser parse order)
{client_script}

// 2. Evaluate Folium bottom script defining map, tiles, featureGroups, fitBounds, layerControl
{folium_script}

// 3. Fire DOMContentLoaded / load lifecycle
global.document.readyState = 'complete';
for (var _i = 0; _i < _domCallbacks.length; _i++) {{ _domCallbacks[_i](); }}
for (var _j = 0; _j < _loadCallbacks.length; _j++) {{ _loadCallbacks[_j](); }}

var __output = (function() {{
    {snippet}
}})();
console.log(JSON.stringify(__output));
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as jf:
            jf.write(runner_js)
            js_path = Path(jf.name)
        try:
            res = subprocess.run(["node", str(js_path)], capture_output=True, text=True, check=True)
            return json.loads(res.stdout.strip())
        finally:
            if js_path.exists():
                js_path.unlink()

    def test_01_fresh_observation_and_interval(self) -> None:
        res = self._execute_node_test(
            """
            window.setObservationEndTimeForTesting(new Date(Date.now() - 3 * 3600000).toISOString());
            return { f: window.computeFreshness(), interval: global.lastIntervalMs };
            """
        )
        self.assertFalse(res["f"]["isStale"])
        self.assertEqual(res["f"]["statusStr"], "FRESH")
        self.assertEqual(res["interval"], 60000)

    def test_02_exact_12h_and_over_12h_boundaries(self) -> None:
        # 12.000000 hours exact -> FRESH
        res_exact = self._execute_node_test(
            """
            var obsMs = Date.parse('2026-11-11T00:00:00Z');
            Date.now = function() { return obsMs + 12 * 3600000; };
            window.setObservationEndTimeForTesting('2026-11-11T00:00:00Z');
            return {
                f: window.computeFreshness(),
                color: window.riskPolygons[0].layer.style.fillColor,
                tip: window.riskPolygons[0].layer.tooltip
            };
            """
        )
        self.assertFalse(res_exact["f"]["isStale"])
        self.assertEqual(res_exact["f"]["statusStr"], "FRESH")
        self.assertEqual(res_exact["f"]["latencyHours"], 12.0)

        # 12.000001 hours (+4 ms over 12h) -> STALE_DATA_UNASSESSED, polygons turn grey #95a5a6
        res_over = self._execute_node_test(
            """
            var obsMs = Date.parse('2026-11-11T00:00:00Z');
            Date.now = function() { return obsMs + 12 * 3600000 + 4; };
            window.setObservationEndTimeForTesting('2026-11-11T00:00:00Z');
            return {
                f: window.computeFreshness(),
                riskColor: window.riskPolygons[0].layer.style.fillColor,
                rainColor: window.rainPolygons[0].layer.style.fillColor,
                riskTip: window.riskPolygons[0].layer.tooltip
            };
            """
        )
        self.assertTrue(res_over["f"]["isStale"])
        self.assertEqual(res_over["f"]["statusStr"], "STALE_DATA_UNASSESSED")
        self.assertEqual(res_over["riskColor"], "#95a5a6")
        self.assertEqual(res_over["rainColor"], "#95a5a6")
        self.assertIn("UNASSESSED", res_over["riskTip"])

    def test_03_missing_invalid_future_timestamps(self) -> None:
        res = self._execute_node_test(
            """
            window.setObservationEndTimeForTesting('');
            var rEmpty = window.computeFreshness();
            window.setObservationEndTimeForTesting('2026-11-11T12:00:00+05:30');
            var rNonUtc = window.computeFreshness();
            window.setObservationEndTimeForTesting('not-a-timestamp');
            var rBad = window.computeFreshness();
            window.setObservationEndTimeForTesting(new Date(Date.now() + 2 * 3600000).toISOString());
            var rFuture = window.computeFreshness();
            return { rEmpty: rEmpty, rNonUtc: rNonUtc, rBad: rBad, rFuture: rFuture };
            """
        )
        self.assertEqual(res["rEmpty"]["statusStr"], "UNASSESSED_MISSING_TIMESTAMP")
        self.assertEqual(res["rNonUtc"]["statusStr"], "UNASSESSED_INVALID_TIMESTAMP")
        self.assertEqual(res["rBad"]["statusStr"], "UNASSESSED_INVALID_TIMESTAMP")
        self.assertEqual(res["rFuture"]["statusStr"], "UNASSESSED_FUTURE_TIMESTAMP")
        self.assertTrue(all(v["isStale"] for v in res.values()))

    def test_04_regional_polygons_cells_coverage_and_validity(self) -> None:
        with open(DEFAULT_OUTPUT_REGIONS_GEOJSON, "r", encoding="utf-8") as f:
            regions_doc = json.load(f)
        with open(DEFAULT_OUTPUT_CELLS_GEOJSON, "r", encoding="utf-8") as f:
            cells_doc = json.load(f)

        self.assertEqual(len(cells_doc["features"]), EXPECTED_LOCATIONS_COUNT)
        self.assertIn("Limited", regions_doc["metadata"]["disclaimer"])
        self.assertIn("Limited", cells_doc["metadata"]["disclaimer"])

        covered_ids = []
        cats_found = set()
        for feat in regions_doc["features"]:
            geom = shape(feat["geometry"])
            self.assertTrue(geom.is_valid and not geom.is_empty)
            props = feat["properties"]
            covered_ids.extend(props["grid_ids"])
            cat = props["primary_screening_category"]
            cats_found.add(cat)
            self.assertEqual(props["color"], CATEGORY_COLORS[cat])
            minx, miny, maxx, maxy = geom.bounds
            self.assertTrue(LON_BOUNDS[0] - 0.5 <= minx and maxx <= LON_BOUNDS[1] + 0.5)
            self.assertTrue(LAT_BOUNDS[0] - 0.5 <= miny and maxy <= LAT_BOUNDS[1] + 0.5)

        self.assertEqual(len(covered_ids), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(len(set(covered_ids)), EXPECTED_LOCATIONS_COUNT)
        for cat in ("Low", "Moderate", "High", "Very High"):
            self.assertIn(cat, cats_found)

    def test_05_regional_aggregates_match_source_csv_and_5_windows(self) -> None:
        df_src = pd.read_csv(DEFAULT_INPUT_CSV).set_index("grid_id")
        with open(DEFAULT_OUTPUT_REGIONS_GEOJSON, "r", encoding="utf-8") as f:
            regions_doc = json.load(f)

        for feat in regions_doc["features"]:
            p = feat["properties"]
            subset = df_src.loc[p["grid_ids"]]
            self.assertAlmostEqual(p["dynamic_risk_min"], round(float(subset["dynamic_risk_index"].min()), 2), places=2)
            self.assertAlmostEqual(p["dynamic_risk_mean"], round(float(subset["dynamic_risk_index"].mean()), 2), places=2)
            self.assertAlmostEqual(p["dynamic_risk_max"], round(float(subset["dynamic_risk_index"].max()), 2), places=2)
            self.assertAlmostEqual(p["terrain_contrib_mean"], round(float(subset["terrain_contribution"].mean()), 2), places=2)
            self.assertAlmostEqual(p["rainfall_contrib_mean"], round(float(subset["rainfall_contribution"].mean()), 2), places=2)
            for w in ("1h", "3h", "6h", "12h", "24h"):
                self.assertAlmostEqual(
                    p[f"gpm_{w}_accumulation_mean_mm"],
                    round(float(subset[f"gpm_{w}_accumulation_mm"].mean()), 2),
                    places=2,
                )

    def test_06_point_geojson_separated_layers(self) -> None:
        with open(DEFAULT_OUTPUT_GEOJSON, "r", encoding="utf-8") as f:
            pts = json.load(f)
        meta = pts["metadata"]
        self.assertEqual(meta["validated_dem_tiles_count"], EXPECTED_TILES_COUNT)
        self.assertEqual(meta["total_grid_locations"], EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(meta["historical_events_count"], EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual(meta["background_samples_count"], EXPECTED_BACKGROUND_SAMPLES_COUNT)
        self.assertIn("Limited", meta["disclaimer"])
        self.assertIn("does not imply complete Western Ghats coverage", meta["disclaimer"])

        by_type = {}
        for feat in pts["features"]:
            ft = feat["properties"]["feature_type"]
            by_type[ft] = by_type.get(ft, 0) + 1
        self.assertEqual(by_type["dynamic_grid_prediction"], EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(by_type["historical_landslide_event"], EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual(by_type["provisional_background_sample"], EXPECTED_BACKGROUND_SAMPLES_COUNT)
        self.assertEqual(len(pts["features"]), EXPECTED_LOCATIONS_COUNT + EXPECTED_HISTORICAL_EVENTS_COUNT + EXPECTED_BACKGROUND_SAMPLES_COUNT)

    def test_07_popups_disclosure_and_reports(self) -> None:
        with open(DEFAULT_OUTPUT_REGIONS_GEOJSON, "r", encoding="utf-8") as f:
            regions_doc = json.load(f)
        sample_feat = max(
            regions_doc["features"],
            key=lambda x: x["properties"].get("gpm_24h_accumulation_mean_mm") or 0.0,
        )
        props_json = json.dumps(sample_feat["properties"])
        res = self._execute_node_test(
            f"""
            var p = {props_json};
            return {{
                riskHtml: window.renderRiskPopupHtml(p),
                rainHtml: window.renderRainPopupHtml(p),
                terHtml: window.renderTerrainPopupHtml(p)
            }};
            """
        )
        for key in ("riskHtml", "rainHtml"):
            html_out = res[key]
            for w in ("1h", "3h", "6h", "12h", "24h"):
                self.assertIn(f"Mean {w} Accumulation:", html_out)
            self.assertIn("14 validated", html_out)
        self.assertIn("does not imply complete western ghats coverage", res["riskHtml"].lower())
        self.assertIn("14 validated", res["terHtml"])
        self.assertTrue(regions_doc["features"][0]["properties"]["region_id"].startswith("WG_HAZARD_REG_"))

        html_text = DEFAULT_OUTPUT_HTML.read_text(encoding="utf-8")
        self.assertIn("14 Validated DEM Tiles", html_text)
        self.assertIn("4,819 Cells", html_text)
        self.assertIn("182 Historical Events", html_text)
        self.assertIn("1,400 Background Samples", html_text)
        self.assertIn("does not imply complete western ghats coverage", html_text.lower())

        html_mb = DEFAULT_OUTPUT_HTML.stat().st_size / (1024 * 1024)
        self.assertLess(html_mb, 15.0)

        with open(DEFAULT_OUTPUT_METADATA, "r", encoding="utf-8") as f:
            map_meta = json.load(f)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            val_rep = json.load(f)
        self.assertEqual(map_meta["dem_tiles_count"], 14)
        self.assertEqual(map_meta["total_grid_locations"], EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(map_meta["historical_events_count"], EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual(map_meta["background_samples_count"], EXPECTED_BACKGROUND_SAMPLES_COUNT)
        self.assertTrue(all(val_rep["validation_checks"].values()))

    def test_08_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing = tmp / "map.html"
            existing.write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_western_ghats_polygon_risk_map(
                    input_csv_path=DEFAULT_INPUT_CSV,
                    input_metadata_path=DEFAULT_INPUT_METADATA,
                    output_html_path=existing,
                    output_geojson_path=tmp / "pts.geojson",
                    output_regions_geojson_path=tmp / "reg.geojson",
                    output_cells_geojson_path=tmp / "cells.geojson",
                    output_metadata_path=tmp / "meta.json",
                    validation_report_path=tmp / "rep.json",
                    overwrite=False,
                )

    def test_09_browser_dom_script_order_basemap_bounds_and_all_layers(self) -> None:
        raw_bytes = DEFAULT_OUTPUT_HTML.read_bytes()
        self.assertNotIn(b"\r\n", raw_bytes)
        html_text = raw_bytes.decode("utf-8")
        trailing_ws_lines = [
            idx
            for idx, line in enumerate(html_text.splitlines(), 1)
            if line != line.rstrip(" \t")
        ]
        self.assertEqual(trailing_ws_lines, [], f"Found trailing whitespace at HTML lines {trailing_ws_lines[:10]}")
        self.assertNotIn("cartodb", html_text.lower())
        self.assertNotIn("cartocdn", html_text.lower())

        with open(DEFAULT_OUTPUT_METADATA, "r", encoding="utf-8") as f:
            map_meta = json.load(f)

        res = self._execute_node_test(
            """
            var mapObj = _mapInstances[0];
            var ctrl = _layerControls[0];
            var activeTiles = _tileLayers.filter(function(t) { return t.addedToMap; }).map(function(t) { return t.url; });
            var overlaySummary = {};
            Object.keys(ctrl.overlays).forEach(function(label) {
                var fg = ctrl.overlays[label];
                overlaySummary[label] = {
                    count: fg.children.length,
                    addedToMap: fg.addedToMap
                };
            });
            mapObj.triggerOverlayAdd();
            var frontCalls = _featureGroups.map(function(fg) { return fg.frontCount; });
            var cellSample = window.cellPolygons.length ? {
                rings: window.cellPolygons[0].layer.latlngs.length,
                fillColor: window.cellPolygons[0].layer.style.fillColor,
                popupHtml: window.renderCellPopupHtml(window.cellPolygons[0].row)
            } : null;
            return {
                preferCanvas: !!mapObj.opts.preferCanvas,
                fitBoundsCount: mapObj.fitBoundsCalls.length,
                fitBounds: mapObj.fitBoundsCalls.length ? mapObj.fitBoundsCalls[0].bounds : null,
                activeTiles: activeTiles,
                baseLayerKeys: Object.keys(ctrl.baseLayers),
                overlaySummary: overlaySummary,
                riskPolygonsCount: window.riskPolygons.length,
                terrainPolygonsCount: window.terrainPolygons.length,
                rainPolygonsCount: window.rainPolygons.length,
                cellPolygonsCount: window.cellPolygons.length,
                frontCalls: frontCalls,
                cellSample: cellSample
            };
            """
        )

        self.assertTrue(res["preferCanvas"])
        self.assertEqual(res["fitBoundsCount"], 1)
        sw, ne = res["fitBounds"]
        self.assertTrue(8.5 <= sw[0] <= 9.5 and 72.0 <= sw[1] <= 73.0)
        self.assertTrue(19.5 <= ne[0] <= 20.5 and 77.5 <= ne[1] <= 78.5)

        self.assertEqual(len(res["activeTiles"]), 1)
        self.assertIn("server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/", res["activeTiles"][0])
        self.assertEqual(res["baseLayerKeys"], ["Esri Satellite Imagery", "OpenStreetMap Standard"])
        self.assertEqual(map_meta["basemap_default"], "Esri Satellite Imagery")

        expected_risk_count = map_meta["regional_polygons"]["dynamic_hazard_screening_regions"]
        expected_terrain_count = map_meta["regional_polygons"]["terrain_susceptibility_regions"]
        expected_rain_count = map_meta["regional_polygons"]["rainfall_contribution_regions"]

        self.assertEqual(res["riskPolygonsCount"], expected_risk_count)
        self.assertEqual(res["terrainPolygonsCount"], expected_terrain_count)
        self.assertEqual(res["rainPolygonsCount"], expected_rain_count)
        self.assertEqual(res["cellPolygonsCount"], EXPECTED_LOCATIONS_COUNT)

        overlays = res["overlaySummary"]
        matched = {k.split(" (")[0]: v for k, v in overlays.items()}
        self.assertEqual(matched["Dynamic Hazard Screening Regions"]["count"], expected_risk_count)
        self.assertTrue(matched["Dynamic Hazard Screening Regions"]["addedToMap"])
        self.assertEqual(matched["Terrain Susceptibility Regions"]["count"], expected_terrain_count)
        self.assertFalse(matched["Terrain Susceptibility Regions"]["addedToMap"])
        self.assertEqual(matched["Rainfall Contribution Regions"]["count"], expected_rain_count)
        self.assertFalse(matched["Rainfall Contribution Regions"]["addedToMap"])
        self.assertEqual(matched["5 km Spatial Grid Cells"]["count"], EXPECTED_LOCATIONS_COUNT)
        self.assertFalse(matched["5 km Spatial Grid Cells"]["addedToMap"])
        self.assertEqual(matched["Historical Landslide Events"]["count"], EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertTrue(matched["Historical Landslide Events"]["addedToMap"])
        self.assertEqual(matched["Validated Background Samples"]["count"], EXPECTED_BACKGROUND_SAMPLES_COUNT)
        self.assertFalse(matched["Validated Background Samples"]["addedToMap"])

        self.assertIsNotNone(res["cellSample"])
        self.assertEqual(res["cellSample"]["rings"], 4)
        for w in ("1h", "3h", "6h", "12h", "24h"):
            self.assertIn(f"Mean {w} Accumulation:", res["cellSample"]["popupHtml"])
        self.assertIn("14 validated Western Ghats DEM tiles", res["cellSample"]["popupHtml"])

        res_stale_cell = self._execute_node_test(
            """
            var obsMs = Date.parse('2026-11-11T00:00:00Z');
            Date.now = function() { return obsMs + 13 * 3600000; };
            window.setObservationEndTimeForTesting('2026-11-11T00:00:00Z');
            return {
                cellColor: window.cellPolygons[0].layer.style.fillColor,
                cellTip: window.cellPolygons[0].layer.tooltip
            };
            """
        )
        self.assertEqual(res_stale_cell["cellColor"], "#95a5a6")
        self.assertIn("UNASSESSED", res_stale_cell["cellTip"])

    def test_10_crs_conversion_and_spatial_cells_geojson_validity(self) -> None:
        with open(DEFAULT_OUTPUT_CELLS_GEOJSON, "r", encoding="utf-8") as f:
            cells_doc = json.load(f)
        self.assertEqual(cells_doc["metadata"]["crs"], "urn:ogc:def:crs:OGC:1.3:CRS84")
        self.assertEqual(len(cells_doc["features"]), EXPECTED_LOCATIONS_COUNT)
        for feat in cells_doc["features"][::50]:
            geom = shape(feat["geometry"])
            self.assertTrue(geom.is_valid and not geom.is_empty)
            minx, miny, maxx, maxy = geom.bounds
            self.assertTrue(LON_BOUNDS[0] - 0.5 <= minx <= maxx <= LON_BOUNDS[1] + 0.5)
            self.assertTrue(LAT_BOUNDS[0] - 0.5 <= miny <= maxy <= LAT_BOUNDS[1] + 0.5)


if __name__ == "__main__":
    unittest.main()
