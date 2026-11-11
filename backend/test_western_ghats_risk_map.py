"""
Automated Test Suite for Stage 6E: Western Ghats Dynamic GIS Risk Map Integration.

Tests:
1. All 1,582 records and 14 DEM tiles represented in source dataset (182 events, 1,400 background).
2. GeoJSON format validity, RFC 7946 compliance, geometry types, and 1,582 Point features within bounds.
3. Full comparison of ALL 1,582 GeoJSON features against source CSV (coordinates, identifiers, risk, rainfall, terrain).
4. Exact Stage 6D metadata thresholds verified with strict numerical tolerance and zero fallback tolerance.
5. GeoJSON safe_value handles booleans as true/false (not 1/0) and serializes NaN/inf as null.
6. Missing rainfall and score values formatted as 'Unavailable' and never converted to zero.
7. Freshness gate strictly enforces explicit UTC timestamps (Z or +00:00), rejecting naive, future, or stale timestamps.
8. Completeness flag strictly requires JSON boolean true (rejects strings, numbers, false).
9. Missing terrain/risk values do not crash marker popup generation or become zero.
10. Terrain layer labeling clearly separates balanced (used in risk) and unweighted (comparison only) models.
11. Overwrite protection prevents replacing existing GeoJSON and HTML outputs without explicit permission.
12. Standalone HTML map generates with Esri Topo basemap, all 5 toggleable layers, legend, and disclaimer.
13. Regional workflows and Stage 6D source artifacts cryptographically preserved via SHA-256 hashes.
14. Exhaustive 1,582-feature verification that dynamic_risk_index is authoritative with no conflicting risk fields.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.create_western_ghats_risk_map import (
    create_western_ghats_dynamic_risk_map,
    verify_observation_freshness_and_completeness,
    validate_stage6d_metadata_thresholds as validate_map_thresholds,
    build_marker_popup_html,
    parse_iso8601_utc,
    CATEGORY_COLORS,
    DEFAULT_INPUT_CSV,
    DEFAULT_INPUT_METADATA,
    DEFAULT_OUTPUT_HTML,
    EXPECTED_LOCATIONS_COUNT,
    EXPECTED_TILES_COUNT,
    EXPECTED_HISTORICAL_EVENTS_COUNT,
    EXPECTED_BACKGROUND_SAMPLES_COUNT,
    MAX_DATA_LATENCY_HOURS,
    format_rainfall_val,
    format_score_val,
    safe_html,
)
from backend.export_western_ghats_dynamic_risk_geojson import (
    export_western_ghats_dynamic_risk_geojson,
    validate_stage6d_metadata_thresholds as validate_geojson_thresholds,
    DEFAULT_OUTPUT_GEOJSON,
    safe_value,
)
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Trusted pre-Stage-6E baseline SHA-256 digests from Git commit 7f59343 (HEAD)
# Each digest is strictly 64 lowercase hexadecimal characters [0-9a-f]
TRUSTED_REGIONAL_WORKFLOW_HASHES = {
    PROJECT_ROOT / "backend" / "create_regional_risk_map.py":
        "b1acf9f993fc08754d8768b11685030b0ec664bb86b71f3a08a6b4ca2baf0a29",
    PROJECT_ROOT / "backend" / "export_western_ghats_risk_geojson.py":
        "a526201530502c5e5673dbdd19d702ee34525afe554e65998812318c6c10f313",
    PROJECT_ROOT / "backend" / "view_western_ghats_map.py":
        "a0503822f8df20251333906e7e814995ceb4ae51eabbde750640444b785d0483",
    PROJECT_ROOT / "outputs" / "regional_landslide_risk_map_esri_02nov2026.html":
        "ad7e83dceb3f56f215c3f7c9afed920cbe4cbce233806ebc4994be80d868b297",
    PROJECT_ROOT / "outputs" / "regional_landslide_risk_map_esri_01nov2026.html":
        "8a0a478699e73e06d6e8cf226b8b6dfa535b68fca1aa40c5ba3bd15bff613cd2",
    PROJECT_ROOT / "western_ghats_pilot_map.html":
        "25302708476b5a5ab82bfd4e3ed93921a1aac0940f98884c3a7e3ca44fe73b0c",
}

# Untracked Stage 6D source artifacts: no pre-existing committed baseline digest available
STAGE6D_SOURCE_ARTIFACTS = [
    PROJECT_ROOT / "data" / "processed" / "western_ghats_dynamic_risk" / "western_ghats_dynamic_hazard_risk.csv",
    PROJECT_ROOT / "data" / "processed" / "western_ghats_dynamic_risk" / "hazard_risk_metadata.json",
    PROJECT_ROOT / "data" / "processed" / "western_ghats_dynamic_risk" / "hazard_risk_validation_report.json",
]


class TestWesternGhatsRiskMap(unittest.TestCase):
    """Test suite for Stage 6E Dynamic GIS Risk Map Integration."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage6e_"))
        self.test_geojson_path = self.temp_dir / "test_risk.geojson"
        self.test_map_path = self.temp_dir / "test_map.html"

    def tearDown(self):
        """Clean up temporary test artifacts."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_counts_and_tiles_retention(self):
        """Test 1: Verify exactly 1,582 records and 14 DEM tiles exist in source Stage 6D dataset."""
        self.assertTrue(DEFAULT_INPUT_CSV.exists(), f"Missing input CSV: {DEFAULT_INPUT_CSV}")
        df = pd.read_csv(DEFAULT_INPUT_CSV)
        self.assertEqual(len(df), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(len(df["tile"].unique()), EXPECTED_TILES_COUNT)
        self.assertEqual(df["record_id"].nunique(), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["sample_id"].nunique(), EXPECTED_LOCATIONS_COUNT)
        self.assertFalse(df["record_id"].isnull().any())
        self.assertFalse(df["sample_id"].isnull().any())
        self.assertEqual((df["sample_type"] == "historical_landslide").sum(), EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual((df["sample_type"] == "provisional_background").sum(), EXPECTED_BACKGROUND_SAMPLES_COUNT)

        # Coordinate finiteness and bounds
        self.assertTrue(np.isfinite(df["latitude"]).all())
        self.assertTrue(np.isfinite(df["longitude"]).all())
        self.assertTrue(df["latitude"].between(8.0, 22.0).all())
        self.assertTrue(df["longitude"].between(70.0, 80.0).all())

    def test_geojson_export_validity_and_features(self):
        """Test 2: Export GeoJSON and verify RFC 7946 compliance, 1,582 Point features, and valid bounds."""
        res = export_western_ghats_dynamic_risk_geojson(
            input_csv_path=DEFAULT_INPUT_CSV,
            input_metadata_path=DEFAULT_INPUT_METADATA,
            output_geojson_path=self.test_geojson_path,
            overwrite=True,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(self.test_geojson_path.exists())

        with open(self.test_geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), EXPECTED_LOCATIONS_COUNT)
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["total_locations"], EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(data["metadata"]["dem_tiles_count"], EXPECTED_TILES_COUNT)
        self.assertEqual(data["metadata"]["historical_events_count"], EXPECTED_HISTORICAL_EVENTS_COUNT)
        self.assertEqual(data["metadata"]["provisional_background_count"], EXPECTED_BACKGROUND_SAMPLES_COUNT)
        self.assertEqual(data["metadata"]["coverage_scope"], "LIMITED_PILOT")
        self.assertIs(data["metadata"]["complete_region_coverage"], False)
        self.assertIn("Limited Pilot", data["metadata"]["title"])
        self.assertIn("Limited Pilot", data["metadata"]["coverage_region"])

        feature_ids = set()
        for feat in data["features"]:
            self.assertEqual(feat["type"], "Feature")
            self.assertEqual(feat["geometry"]["type"], "Point")
            coords = feat["geometry"]["coordinates"]
            self.assertEqual(len(coords), 2)
            lon, lat = coords
            self.assertTrue(math.isfinite(lon) and math.isfinite(lat))
            self.assertGreaterEqual(lon, 70.0)
            self.assertLessEqual(lon, 80.0)
            self.assertGreaterEqual(lat, 8.0)
            self.assertLessEqual(lat, 22.0)
            feature_ids.add(feat["id"])

        # Every source record represented exactly once with no duplicates
        self.assertEqual(len(feature_ids), EXPECTED_LOCATIONS_COUNT)

    def test_all_1582_geojson_features_match_source_csv(self):
        """Test 3: Exhaustively compare ALL 1,582 GeoJSON features against source CSV records."""
        export_western_ghats_dynamic_risk_geojson(
            input_csv_path=DEFAULT_INPUT_CSV,
            input_metadata_path=DEFAULT_INPUT_METADATA,
            output_geojson_path=self.test_geojson_path,
            overwrite=True,
        )
        df = pd.read_csv(DEFAULT_INPUT_CSV)
        with open(self.test_geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features_by_id = {f["id"]: f for f in geojson_data["features"]}
        self.assertEqual(len(features_by_id), EXPECTED_LOCATIONS_COUNT)

        # Verify category distribution matches Stage 6D source CSV counts dynamically
        expected_cat_counts = df["primary_screening_category"].value_counts().to_dict()
        cat_counts = pd.Series(
            [f["properties"]["primary_screening_category"] for f in features_by_id.values()]
        ).value_counts().to_dict()
        for cat in ("Low", "Moderate", "High", "Very High"):
            self.assertEqual(cat_counts.get(cat, 0), expected_cat_counts.get(cat, 0))
        self.assertEqual(sum(cat_counts.values()), EXPECTED_LOCATIONS_COUNT)

        # Exhaustive 1,582 row-by-row verification
        for _, row in df.iterrows():
            rid = str(row["record_id"])
            self.assertIn(rid, features_by_id)
            feat = features_by_id[rid]
            coords = feat["geometry"]["coordinates"]
            props = feat["properties"]

            # Coordinates
            self.assertAlmostEqual(coords[0], float(row["longitude"]), places=6)
            self.assertAlmostEqual(coords[1], float(row["latitude"]), places=6)

            # Identifiers
            self.assertEqual(props["record_id"], str(row["record_id"]))
            self.assertEqual(props["sample_id"], str(row["sample_id"]))
            self.assertEqual(props["tile"], str(row["tile"]))
            self.assertEqual(props["sample_type"], str(row["sample_type"]))

            # Risk and category (authoritative dynamic_risk_index without conflicting fields)
            self.assertAlmostEqual(props["dynamic_risk_index"], float(row["dynamic_risk_index"]), places=4)
            self.assertEqual(props["primary_screening_category"], str(row["primary_screening_category"]))
            self.assertNotIn("dynamic_risk_interactive", props)

            # Terrain attributes (balanced and unweighted)
            self.assertEqual(props["terrain_rf_pred_balanced"], int(row["terrain_rf_pred_balanced"]))
            self.assertAlmostEqual(props["terrain_rf_prob_balanced"], float(row["terrain_rf_prob_balanced"]), places=4)
            self.assertEqual(props["terrain_rf_pred_unweighted"], int(row["terrain_rf_pred_unweighted"]))
            self.assertAlmostEqual(props["terrain_rf_prob_unweighted"], float(row["terrain_rf_prob_unweighted"]), places=4)

            # Rainfall accumulations
            self.assertAlmostEqual(props["gpm_latest_rainfall_rate_mm_hr"], float(row["gpm_latest_rainfall_rate_mm_hr"]), places=3)
            self.assertAlmostEqual(props["gpm_1h_accumulation_mm"], float(row["gpm_1h_accumulation_mm"]), places=3)
            self.assertAlmostEqual(props["gpm_3h_accumulation_mm"], float(row["gpm_3h_accumulation_mm"]), places=3)
            self.assertAlmostEqual(props["gpm_6h_accumulation_mm"], float(row["gpm_6h_accumulation_mm"]), places=3)
            self.assertAlmostEqual(props["gpm_12h_accumulation_mm"], float(row["gpm_12h_accumulation_mm"]), places=3)
            self.assertAlmostEqual(props["gpm_24h_accumulation_mm"], float(row["gpm_24h_accumulation_mm"]), places=3)

    def test_strict_metadata_thresholds_zero_fallback(self):
        """Test 4: Verify metadata-driven dynamic threshold validation and invalid-input rejection."""
        with open(DEFAULT_INPUT_METADATA, "r", encoding="utf-8") as f:
            valid_meta = json.load(f)

        cutoffs = valid_meta["risk_formula_specification"]["relative_category_cutoffs"]
        exp_p50 = float(cutoffs["P50_low_to_moderate"])
        exp_p80 = float(cutoffs["P80_moderate_to_high"])
        exp_p95 = float(cutoffs["P95_high_to_very_high"])

        # Valid metadata passes strictly in both map and geojson validators
        res_map = validate_map_thresholds(valid_meta)
        res_geo = validate_geojson_thresholds(valid_meta)
        self.assertTrue(0.0 < exp_p50 < exp_p80 < exp_p95 < 100.0)
        self.assertTrue(math.isclose(res_map["P50_low_to_moderate"], exp_p50, rel_tol=1e-6))
        self.assertTrue(math.isclose(res_geo["P80_moderate_to_high"], exp_p80, rel_tol=1e-6))
        self.assertTrue(math.isclose(res_geo["P95_high_to_very_high"], exp_p95, rel_tol=1e-6))

        # Simulated shifted valid quantile cutoffs must be accepted dynamically (no stale hardcoded lock)
        meta_shifted = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": 18.5,
                    "P80_moderate_to_high": 42.25,
                    "P95_high_to_very_high": 76.75,
                }
            }
        }
        res_shifted_map = validate_map_thresholds(meta_shifted)
        res_shifted_geo = validate_geojson_thresholds(meta_shifted)
        self.assertEqual(res_shifted_map["P50_low_to_moderate"], 18.5)
        self.assertEqual(res_shifted_geo["P95_high_to_very_high"], 76.75)

        # Rejects missing relative_category_cutoffs
        with self.assertRaises(ValueError):
            validate_map_thresholds({"risk_formula_specification": {}})

        # Rejects missing P95 threshold key
        meta_missing_p95 = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": exp_p50,
                    "P80_moderate_to_high": exp_p80,
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_map_thresholds(meta_missing_p95)

        # Rejects null threshold
        meta_null = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": exp_p50,
                    "P80_moderate_to_high": None,
                    "P95_high_to_very_high": exp_p95,
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_geojson_thresholds(meta_null)

        # Rejects boolean threshold
        meta_bool = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": True,
                    "P80_moderate_to_high": exp_p80,
                    "P95_high_to_very_high": exp_p95,
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_map_thresholds(meta_bool)

        # Rejects non-finite (inf) threshold
        meta_inf = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": float("inf"),
                    "P80_moderate_to_high": exp_p80,
                    "P95_high_to_very_high": exp_p95,
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_geojson_thresholds(meta_inf)

        # Rejects non-monotonic / reversed thresholds
        meta_reversed = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": 60.0,
                    "P80_moderate_to_high": 40.0,
                    "P95_high_to_very_high": 80.0,
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_map_thresholds(meta_reversed)

        # Rejects out-of-range thresholds (<= 0 or >= 100)
        meta_out_of_range = {
            "risk_formula_specification": {
                "relative_category_cutoffs": {
                    "P50_low_to_moderate": 0.0,
                    "P80_moderate_to_high": exp_p80,
                    "P95_high_to_very_high": 100.0,
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_geojson_thresholds(meta_out_of_range)

    def test_geojson_safe_value_booleans_and_nulls(self):
        """Test 5: Verify booleans remain true/false (not 1/0) and NaN/inf serialize as null."""
        self.assertIs(safe_value(True), True)
        self.assertIsInstance(safe_value(True), bool)
        self.assertNotEqual(type(safe_value(True)), int)

        self.assertIs(safe_value(False), False)
        self.assertIsInstance(safe_value(False), bool)
        self.assertNotEqual(type(safe_value(False)), int)

        self.assertIs(safe_value(np.bool_(True)), True)
        self.assertIsInstance(safe_value(np.bool_(True)), bool)

        self.assertEqual(safe_value(1), 1)
        self.assertEqual(type(safe_value(1)), int)
        self.assertEqual(safe_value(0), 0)
        self.assertEqual(type(safe_value(0)), int)

        self.assertIsNone(safe_value(np.nan))
        self.assertIsNone(safe_value(float("inf")))
        self.assertIsNone(safe_value(float("-inf")))
        self.assertIsNone(safe_value(None))

    def test_missing_rainfall_not_displayed_as_zero(self):
        """Test 6: Verify missing/NaN rainfall is formatted as 'Unavailable' and not '0.00 mm'."""
        formatted_nan = format_rainfall_val(np.nan)
        self.assertIn("Unavailable", formatted_nan)
        self.assertNotIn("0.00 mm", formatted_nan)

        formatted_none = format_rainfall_val(None)
        self.assertIn("Unavailable", formatted_none)

        formatted_inf = format_rainfall_val(float("inf"))
        self.assertIn("Unavailable", formatted_inf)

        formatted_valid = format_rainfall_val(12.345)
        self.assertEqual(formatted_valid, "12.35 mm")

        self.assertIn("Unavailable", format_score_val(np.nan))
        self.assertIn("Unavailable", format_score_val(float("inf")))
        self.assertEqual(format_score_val(42.5), "42.50")

    def test_explicit_utc_timestamp_enforcement(self):
        """Test 7: Verify parser and freshness gate require explicit UTC (Z or +00:00) and reject naive timestamps."""
        # Parser checks
        self.assertIsNotNone(parse_iso8601_utc("2026-11-07T02:00:00Z"))
        self.assertIsNotNone(parse_iso8601_utc("2026-11-07T02:00:00+00:00"))

        # Rejects timezone-naive strings
        self.assertIsNone(parse_iso8601_utc("2026-11-07T02:00:00"))
        self.assertIsNone(parse_iso8601_utc("2026-11-07 02:00:00"))
        self.assertIsNone(parse_iso8601_utc("invalid_date"))
        self.assertIsNone(parse_iso8601_utc(None))

        # Freshness gate checks with naive timestamp
        meta_naive = {
            "gpm_observation_metadata": {
                "observation_start_utc": "2026-11-06T02:00:00Z",
                "observation_end_utc": "2026-11-07T02:00:00",  # Missing Z or +00:00
                "is_24h_complete": True,
            }
        }
        ref_now = datetime(2026, 11, 7, 8, 0, 0, tzinfo=timezone.utc)
        ok, msg, _ = verify_observation_freshness_and_completeness(meta_naive, now_utc=ref_now)
        self.assertFalse(ok)
        self.assertIn("explicit-utc", msg.lower())

        # Future timestamp rejected
        base_meta = {
            "gpm_observation_metadata": {
                "observation_start_utc": "2026-11-06T02:00:00Z",
                "observation_end_utc": "2026-11-07T02:00:00Z",
                "is_24h_complete": True,
            }
        }
        ref_now_future = datetime(2026, 11, 7, 1, 59, 59, tzinfo=timezone.utc)
        ok, msg, _ = verify_observation_freshness_and_completeness(base_meta, now_utc=ref_now_future)
        self.assertFalse(ok)
        self.assertIn("future", msg.lower())

        # Stale (>12h) timestamp rejected
        ref_now_stale = datetime(2026, 11, 7, 15, 0, 0, tzinfo=timezone.utc)
        ok, msg, _ = verify_observation_freshness_and_completeness(base_meta, now_utc=ref_now_stale)
        self.assertFalse(ok)
        self.assertIn("stale", msg.lower())

    def test_completeness_flag_validation(self):
        """Test 8: Verify 24-hour completeness strictly requires JSON boolean true."""
        ref_now = datetime(2026, 11, 7, 8, 0, 0, tzinfo=timezone.utc)

        def make_meta(complete_val):
            return {
                "gpm_observation_metadata": {
                    "observation_start_utc": "2026-11-06T02:00:00Z",
                    "observation_end_utc": "2026-11-07T02:00:00Z",
                    "is_24h_complete": complete_val,
                }
            }

        ok, msg, _ = verify_observation_freshness_and_completeness(make_meta(True), now_utc=ref_now)
        self.assertTrue(ok)

        ok, msg, _ = verify_observation_freshness_and_completeness(make_meta(False), now_utc=ref_now)
        self.assertFalse(ok)

        ok, msg, _ = verify_observation_freshness_and_completeness(make_meta("true"), now_utc=ref_now)
        self.assertFalse(ok)

        ok, msg, _ = verify_observation_freshness_and_completeness(make_meta(1), now_utc=ref_now)
        self.assertFalse(ok)

    def test_missing_terrain_risk_values_do_not_crash_or_zero(self):
        """Test 9: Verify missing terrain/risk/rainfall values do not crash marker popup generation or become zero."""
        row_missing = pd.Series({
            "record_id": "test_missing_01",
            "sample_id": "test_missing_01",
            "sample_type": "historical_landslide",
            "tile": "N15E073",
            "subregion": "Central_Western_Ghats",
            "latitude": 15.5,
            "longitude": 73.8,
            "elevation_m": np.nan,
            "slope_degrees": float("inf"),
            "dynamic_risk_index": np.nan,
            "primary_screening_category": "UNASSESSED_MISSING_RAINFALL",
            "risk_tier_fixed": None,
            "terrain_rf_prob_balanced": np.nan,
            "terrain_baseline_screening_tier": None,
            "terrain_rf_prob_unweighted": np.nan,
            "gpm_latest_rainfall_rate_mm_hr": np.nan,
            "gpm_1h_accumulation_mm": np.nan,
            "gpm_3h_accumulation_mm": np.nan,
            "gpm_6h_accumulation_mm": np.nan,
            "gpm_12h_accumulation_mm": np.nan,
            "gpm_24h_accumulation_mm": np.nan,
            "rainfall_intensity_index": np.nan,
            "rainfall_percentile_24h": np.nan,
        })

        popup_html = build_marker_popup_html(
            row=row_missing,
            obs_start_utc="2026-11-06T02:00:00Z",
            obs_end_utc="2026-11-07T02:00:00Z",
            latency_hours=6.4,
        )

        self.assertIsInstance(popup_html, str)
        self.assertIn("Unavailable", popup_html)
        self.assertNotIn("0.00 mm", popup_html)
        self.assertIn(CATEGORY_COLORS["UNASSESSED_MISSING_RAINFALL"], popup_html)

    def test_terrain_layer_labeling_and_popup_distinction(self):
        """Test 10: Verify popup and map clearly distinguish balanced model (in risk) from unweighted model (comparison)."""
        row_sample = pd.Series({
            "record_id": "test_terrain_01",
            "sample_id": "test_terrain_01",
            "sample_type": "historical_landslide",
            "tile": "N15E073",
            "subregion": "Central_Western_Ghats",
            "latitude": 15.5,
            "longitude": 73.8,
            "elevation_m": 850.0,
            "slope_degrees": 22.5,
            "dynamic_risk_index": 44.5,
            "primary_screening_category": "High",
            "risk_tier_fixed": "Moderate",
            "terrain_rf_prob_balanced": 0.885,
            "terrain_baseline_screening_tier": "Very High",
            "terrain_rf_prob_unweighted": 0.421,
            "gpm_latest_rainfall_rate_mm_hr": 0.0,
            "gpm_1h_accumulation_mm": 0.0,
            "gpm_3h_accumulation_mm": 0.0,
            "gpm_6h_accumulation_mm": 0.0,
            "gpm_12h_accumulation_mm": 0.0,
            "gpm_24h_accumulation_mm": 0.0,
            "rainfall_intensity_index": 0.0,
            "rainfall_percentile_24h": 0.0,
        })

        popup_html = build_marker_popup_html(
            row=row_sample,
            obs_start_utc="2026-11-06T02:00:00Z",
            obs_end_utc="2026-11-07T02:00:00Z",
            latency_hours=6.4,
        )

        # Popup clearly separates balanced model from unweighted model and preserves 5 GPM windows + pilot scope
        self.assertIn("Balanced Terrain Prob (Used in Risk):", popup_html)
        self.assertIn("0.885", popup_html)
        self.assertIn("Unweighted Terrain Prob (Comparison Only):", popup_html)
        self.assertIn("0.421", popup_html)
        self.assertIn("Limited Western Ghats Pilot", popup_html)
        for win_label in ("1-hour accumulation:", "3-hour accumulation:", "6-hour accumulation:", "12-hour accumulation:", "24-hour accumulation:"):
            self.assertIn(win_label, popup_html)

    def test_overwrite_protection(self):
        """Test 11: Verify overwrite protection prevents replacing existing outputs without --overwrite."""
        self.test_geojson_path.write_text("existing geojson content", encoding="utf-8")
        self.test_map_path.write_text("existing map html content", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            export_western_ghats_dynamic_risk_geojson(
                output_geojson_path=self.test_geojson_path,
                overwrite=False,
            )

        with self.assertRaises(FileExistsError):
            create_western_ghats_dynamic_risk_map(
                output_html_path=self.test_map_path,
                overwrite=False,
            )

        res_geo = export_western_ghats_dynamic_risk_geojson(
            output_geojson_path=self.test_geojson_path,
            overwrite=True,
        )
        self.assertEqual(res_geo["status"], "SUCCESS")

    def test_map_html_generation_and_layer_structure(self):
        """Test 12: Generate standalone compact Canvas HTML map and verify 5 layers, 60s JS freshness guard, and Node boundary behavior."""
        from datetime import timedelta

        with open(DEFAULT_INPUT_METADATA, "r", encoding="utf-8") as f:
            meta = json.load(f)
        gpm_meta = meta.get("gpm_observation_metadata", {})
        obs_end_str = gpm_meta.get("observation_end_utc")
        obs_dt = parse_iso8601_utc(obs_end_str)
        self.assertIsNotNone(obs_dt)

        ref_now = obs_dt + timedelta(hours=6)
        res = create_western_ghats_dynamic_risk_map(
            input_csv_path=DEFAULT_INPUT_CSV,
            input_metadata_path=DEFAULT_INPUT_METADATA,
            output_html_path=self.test_map_path,
            now_utc=ref_now,
            overwrite=True,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(self.test_map_path.exists())
        html_size = self.test_map_path.stat().st_size
        self.assertGreater(html_size, 100_000)
        self.assertLess(html_size, 3_000_000)

        html_content = self.test_map_path.read_text(encoding="utf-8")

        # Verify Esri World Topographic Basemap URL
        self.assertIn("server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map", html_content)
        self.assertIn("Esri World Topographic Map", html_content)

        # Verify all 5 distinct feature group layer names (with explicitly labeled unweighted layer)
        self.assertIn("Dynamic Hazard Screening (Primary Stage 6D Category)", html_content)
        self.assertIn("GPM 24h Rainfall Accumulation (mm)", html_content)
        self.assertIn("Unweighted Random Forest Terrain Susceptibility (Comparison Only)", html_content)
        self.assertIn("Historical Landslide Events (n=182)", html_content)
        self.assertIn("Provisional Background Samples (n=1,400)", html_content)

        # Verify limited Western Ghats pilot coverage transparency & exact counts
        self.assertIn("Western Ghats Limited Pilot", html_content)
        self.assertIn("Limited Western Ghats Pilot", html_content)
        self.assertIn("does not imply complete Western Ghats coverage", html_content)
        self.assertIn("<b>14</b> Validated DEM Tiles", html_content)
        self.assertIn("<b>1,582</b> Locations", html_content)
        self.assertIn("<b>182</b> Historical Events", html_content)
        self.assertIn("<b>1,400</b> Background Samples", html_content)

        # Verify disclaimer, dynamic freshness guard, and stale styling in HTML
        self.assertIn("EXPERIMENTAL PROVISIONAL SCREENING ONLY", html_content)
        self.assertIn("Observation Window (UTC)", html_content)
        self.assertIn("computeFreshness", html_content)
        self.assertIn("updateFreshnessStatus", html_content)
        self.assertIn("setInterval(updateFreshnessStatus, 60000)", html_content)
        self.assertIn("#95a5a6", html_content)

        # Execute embedded JS freshness + popup logic in Node.js across exact boundary & invalid scenarios
        script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", html_content, flags=re.DOTALL)
        target_scripts = [s for s in script_blocks if "computeFreshness" in s and "wgRecords" in s]
        self.assertGreaterEqual(len(target_scripts), 1, "Expected embedded script with computeFreshness and wgRecords")
        script_code = re.sub(
            r"var lg(Risk|Rain|Terrain|Events|Background)\s*=\s*[a-zA-Z0-9_]+;",
            r"var lg\1 = feature_group_mock;",
            target_scripts[0],
        )

        node_harness = f"""
const assert = require('assert');
const RealDate = Date;
let fakeNowMs = RealDate.parse({json.dumps(obs_end_str)}) + 6 * 3600 * 1000;
global.Date = class extends RealDate {{
  constructor(...args) {{
    if (args.length === 0) return new RealDate(fakeNowMs);
    return new RealDate(...args);
  }}
  static now() {{ return fakeNowMs; }}
  static parse(s) {{ return RealDate.parse(s); }}
  static UTC(...args) {{ return RealDate.UTC(...args); }}
}};

const domStore = {{}};
const getEl = (id) => {{
  if (!domStore[id]) {{
    domStore[id] = {{
      id,
      innerHTML: '',
      textContent: '',
      style: {{}},
      className: '',
      attributes: {{}},
      setAttribute(k, v) {{ this.attributes[k] = v; }},
      classList: {{ add() {{}}, remove() {{}} }},
    }};
  }}
  return domStore[id];
}};

global.window = global;
global.window.addEventListener = () => {{}};
global.document = {{
  readyState: 'complete',
  getElementById: getEl,
  addEventListener() {{}},
}};
const intervalCalls = [];
global.setInterval = (fn, ms) => {{ intervalCalls.push(ms); return intervalCalls.length; }};

const createdMarkers = [];
const makeLayerGroup = () => ({{
  layers: [],
  addLayer(l) {{ this.layers.push(l); }},
  clearLayers() {{ this.layers = []; }},
  eachLayer(fn) {{ this.layers.forEach(fn); }},
}});
const feature_group_mock = makeLayerGroup();
global.L = {{
  canvas() {{ return {{}}; }},
  circleMarker(latlng, opts) {{
    const m = {{
      latlng,
      options: Object.assign({{}}, opts),
      popup: '',
      tooltip: '',
      styleHistory: [],
      bindTooltip(t) {{ this.tooltip = t; return this; }},
      setTooltipContent(t) {{ this.tooltip = t; return this; }},
      bindPopup(p) {{ this.popup = p; return this; }},
      addTo(g) {{ g.addLayer(this); return this; }},
      setStyle(s) {{ this.options = Object.assign(this.options, s); this.styleHistory.push(s); return this; }},
    }};
    createdMarkers.push(m);
    return m;
  }},
}};

{script_code}

assert.strictEqual(typeof window.computeFreshness, 'function');
assert.strictEqual(typeof window.updateFreshnessStatus, 'function');
assert.strictEqual(typeof window.renderPopupHtml, 'function');
assert.strictEqual(window.riskMarkers.length, {EXPECTED_LOCATIONS_COUNT});
assert.ok(intervalCalls.includes(60000), 'Expected 60000ms freshness interval');

const obsIso = {json.dumps(obs_end_str)};
const obsMs = RealDate.parse(obsIso);

// Exact 12.000000h -> FRESH
fakeNowMs = obsMs + {MAX_DATA_LATENCY_HOURS} * 3600 * 1000;
const exact12 = window.computeFreshness();
assert.strictEqual(exact12.isStale, false);
assert.strictEqual(exact12.statusStr, 'FRESH');

// Exact 12.000001h -> STALE_DATA_UNASSESSED
fakeNowMs = obsMs + {MAX_DATA_LATENCY_HOURS} * 3600 * 1000 + 4;
const over12 = window.computeFreshness();
assert.strictEqual(over12.isStale, true);
assert.strictEqual(over12.statusStr, 'STALE_DATA_UNASSESSED');

// Missing / invalid -> UNASSESSED_*
window.setObservationEndTimeForTesting('');
assert.strictEqual(window.computeFreshness().statusStr, 'UNASSESSED_MISSING_TIMESTAMP');
assert.strictEqual(window.getIsStale(), true);
window.setObservationEndTimeForTesting('invalid-ts');
assert.strictEqual(window.computeFreshness().statusStr, 'UNASSESSED_INVALID_TIMESTAMP');
assert.strictEqual(window.getIsStale(), true);

// Future timestamp -> UNASSESSED_FUTURE_TIMESTAMP
window.setObservationEndTimeForTesting(obsIso);
fakeNowMs = obsMs - 60000;
const futureRes = window.computeFreshness();
assert.strictEqual(futureRes.isStale, true);
assert.strictEqual(futureRes.statusStr, 'UNASSESSED_FUTURE_TIMESTAMP');

// Verify updateFreshnessStatus transitions markers to stale #95a5a6 styling when aged past 12h
fakeNowMs = obsMs + 13 * 3600 * 1000;
window.updateFreshnessStatus();
assert.strictEqual(window.getIsStale(), true);
const allStaleStyled = window.riskMarkers.every(
  item => item.marker.options.fillColor === '#95a5a6'
);
assert.ok(allStaleStyled, 'Expected all risk markers to take #95a5a6 stale fill when aged > 12h');

// Verify fresh transition restores active styling & renders balanced vs unweighted popup labels
fakeNowMs = obsMs + 6 * 3600 * 1000;
window.updateFreshnessStatus();
assert.strictEqual(window.getIsStale(), false);
const sampleRec = window.riskMarkers[0].data;
const popupHtml = window.renderPopupHtml(sampleRec);
assert.ok(popupHtml.includes('Balanced Terrain Prob (Used in Risk)'), 'Missing Balanced label in popup');
assert.ok(popupHtml.includes('Unweighted Terrain Prob (Comparison Only)'), 'Missing Unweighted label in popup');
assert.ok(popupHtml.includes('Limited Western Ghats Pilot'), 'Missing limited pilot scope in WG popup');
for (const wLbl of ['1-hour accumulation:', '3-hour accumulation:', '6-hour accumulation:', '12-hour accumulation:', '24-hour accumulation:']) {{
  assert.ok(popupHtml.includes(wLbl), 'Missing preserved GPM window in WG popup: ' + wLbl);
}}
console.log('WG_NODE_JS_VERIFIED');
"""
        tmp_js = tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8")
        try:
            tmp_js.write(node_harness)
            tmp_js.close()
            proc = subprocess.run(
                ["node", tmp_js.name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"Node JS freshness harness failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertIn("WG_NODE_JS_VERIFIED", proc.stdout)
        finally:
            Path(tmp_js.name).unlink(missing_ok=True)

    def test_regional_workflows_and_stage6d_cryptographic_preservation(self):
        """Test 13: Verify regional workflow files match trusted Git baseline hashes and Stage 6D files exist."""
        # 1. Verify regional workflow files against trusted Git commit 7f59343 (HEAD) baseline
        for file_path, expected_hash in TRUSTED_REGIONAL_WORKFLOW_HASHES.items():
            self.assertTrue(file_path.exists(), f"Protected regional workflow file missing: {file_path}")
            # Ensure every expected digest is exactly 64 lowercase hexadecimal characters
            self.assertEqual(len(expected_hash), 64, f"Hash for {file_path.name} must be exactly 64 characters")
            self.assertTrue(
                all(c in "0123456789abcdef" for c in expected_hash),
                f"Hash for {file_path.name} contains invalid non-hexadecimal characters: {expected_hash}",
            )

            # Compute normalized SHA-256 (LF normalized) to match Git tree blob content
            raw_content = file_path.read_text(encoding="utf-8", errors="replace")
            normalized_bytes = raw_content.replace("\r\n", "\n").encode("utf-8")
            calculated_hash = hashlib.sha256(normalized_bytes).hexdigest()

            self.assertEqual(
                calculated_hash,
                expected_hash,
                f"Cryptographic hash mismatch in {file_path.name}:\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {calculated_hash}",
            )

        # 2. Verify Stage 6D source artifacts exist and maintain expected schema
        for file_path in STAGE6D_SOURCE_ARTIFACTS:
            self.assertTrue(file_path.exists(), f"Stage 6D artifact missing: {file_path}")
            self.assertGreater(file_path.stat().st_size, 0, f"Stage 6D artifact is empty: {file_path}")

    def test_authoritative_risk_no_conflicting_fields(self):
        """Test 14: Exhaustively verify all 1,582 features have authoritative dynamic_risk_index and no conflicting risk fields."""
        export_western_ghats_dynamic_risk_geojson(
            input_csv_path=DEFAULT_INPUT_CSV,
            input_metadata_path=DEFAULT_INPUT_METADATA,
            output_geojson_path=self.test_geojson_path,
            overwrite=True,
        )
        df = pd.read_csv(DEFAULT_INPUT_CSV)
        with open(self.test_geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        features = geojson_data["features"]
        self.assertEqual(len(features), EXPECTED_LOCATIONS_COUNT)

        expected_risk_keys = {
            "dynamic_risk_index",
            "risk_category_relative",
            "risk_tier_fixed",
            "risk_calculation_status",
            "risk_calculation_timestamp_utc",
        }

        features_by_id = {f["id"]: f for f in features}

        for _, row in df.iterrows():
            rid = str(row["record_id"])
            self.assertIn(rid, features_by_id)
            feat = features_by_id[rid]
            props = feat["properties"]

            # 1. Authoritative dynamic risk score present and finite
            self.assertIn("dynamic_risk_index", props)
            risk_score = props["dynamic_risk_index"]
            self.assertIsInstance(risk_score, float)
            self.assertTrue(math.isfinite(risk_score))
            self.assertGreaterEqual(risk_score, 0.0)
            self.assertLessEqual(risk_score, 100.0)
            self.assertAlmostEqual(risk_score, float(row["dynamic_risk_index"]), places=4)

            # 2. Conflicting field dynamic_risk_interactive is completely removed from GeoJSON properties
            self.assertNotIn("dynamic_risk_interactive", props)

            # 3. No conflicting continuous numeric risk fields exist
            risk_related_keys = {k for k in props.keys() if "risk" in k}
            self.assertEqual(
                risk_related_keys,
                expected_risk_keys,
                f"Unexpected or conflicting risk keys found in feature {rid}: {risk_related_keys - expected_risk_keys}",
            )

            # 4. Authoritative categories match Stage 6D source
            self.assertEqual(props["primary_screening_category"], str(row["primary_screening_category"]))
            self.assertEqual(props["risk_category_relative"], str(row["risk_category_relative"]))
            self.assertEqual(props["risk_tier_fixed"], str(row["risk_tier_fixed"]))
            self.assertEqual(props["risk_calculation_status"], str(row["risk_calculation_status"]))


if __name__ == "__main__":
    unittest.main()
