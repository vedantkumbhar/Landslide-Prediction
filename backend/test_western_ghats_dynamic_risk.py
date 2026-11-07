"""
Automated Test Suite for Western Ghats Dynamic Landslide Hazard Risk Screening Pipeline.

Stage 6D Targeted Audit Validation Suite:
Tests independent timestamp freshness parsing, stale status rejection, non-finite
rainfall rejection (+inf, -inf), missing data handling, zero-rainfall baseline invariance,
terrain feature contract preservation, relative category differentiation, and end-to-end execution.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.predict_western_ghats_dynamic_risk import (
    WesternGhatsDynamicRiskEngine,
    verify_rainfall_freshness_gate,
    parse_iso8601_utc,
    run_western_ghats_dynamic_risk_pipeline,
    DEFAULT_INTEGRATED_CSV,
    DEFAULT_REFRESH_STATUS_JSON,
    TERRAIN_MODEL_PREDICTORS,
    EXPECTED_LOCATIONS_COUNT,
    EXPECTED_TILES_COUNT,
    DISCLAIMER_TEXT,
)


class TestWesternGhatsDynamicRisk(unittest.TestCase):
    """Test suite for Stage 6D dynamic landslide hazard screening system."""

    def setUp(self):
        """Prepare sample test DataFrames with controlled conditions."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage6d_"))
        
        # Create a synthetic 10-row dataset modeling real conditions
        self.sample_df = pd.DataFrame({
            "record_id": [f"rec_{i:02d}" for i in range(10)],
            "sample_id": [f"samp_{i:02d}" for i in range(10)],
            "tile": ["N10E076"] * 5 + ["N12E075"] * 5,
            "dem_region": ["Western Ghats"] * 10,
            "subregion": ["Kerala"] * 10,
            "latitude": [10.5 + i * 0.1 for i in range(10)],
            "longitude": [76.5 + i * 0.1 for i in range(10)],
            "elevation_m": [100.0 + i * 50.0 for i in range(10)],
            "slope_degrees": [5.0 + i * 4.0 for i in range(10)],
            "aspect_sin": [0.5] * 10,
            "aspect_cos": [-0.5] * 10,
            "aspect_flat": [0] * 10,
            "terrain_rf_prob_balanced": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "gpm_latest_rainfall_rate_mm_hr": [0.0] * 10,
            "gpm_1h_accumulation_mm": [0.0, 0.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 0.0, 0.0],
            "gpm_3h_accumulation_mm": [0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0],
            "gpm_6h_accumulation_mm": [0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 0.0, 0.0],
            "gpm_12h_accumulation_mm": [0.0, 0.0, 0.0, 20.0, 40.0, 60.0, 80.0, 90.0, 0.0, 0.0],
            "gpm_24h_accumulation_mm": [0.0, 0.0, 0.0, 25.0, 50.0, 75.0, 100.0, 120.0, 0.0, 0.0],
            "gpm_24h_complete": [True] * 10,
        })

    def tearDown(self):
        """Clean up temporary test directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -----------------------------------------------------------------
    # FRESHNESS GATE TESTS (ISSUE 1)
    # -----------------------------------------------------------------

    def test_old_success_status_rejected_via_timestamp(self):
        """Test 1: Verify an old SUCCESS status file is rejected using its actual observation timestamp."""
        # Status file claiming SUCCESS and complete, but observation ended 138 hours ago
        old_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            "observation_end_utc": "2026-11-01T12:00:00Z",
            "data_latency_hours": 4.0,  # Falsely claims 4.0h
        }
        old_path = self.temp_dir / "old_success.json"
        with open(old_path, "w") as f:
            json.dump(old_status, f)

        # Evaluate against reference time 2026-11-07T08:00:00Z (latency = 140 hours)
        ref_time = datetime(2026, 11, 7, 8, 0, 0, tzinfo=timezone.utc)
        is_fresh, msg, data = verify_rainfall_freshness_gate(
            old_path, max_latency_hours=12.0, now_utc=ref_time
        )
        self.assertFalse(is_fresh)
        self.assertIn("stale", msg.lower())
        self.assertGreater(data.get("calculated_latency_hours", 0), 12.0)

    def test_fresh_success_status_accepted(self):
        """Test 2: Verify fresh SUCCESS status file with valid observation timestamp is accepted."""
        fresh_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            "observation_end_utc": "2026-11-07T02:00:00Z",
        }
        fresh_path = self.temp_dir / "fresh_success.json"
        with open(fresh_path, "w") as f:
            json.dump(fresh_status, f)

        ref_time = datetime(2026, 11, 7, 7, 30, 0, tzinfo=timezone.utc)  # 5.5 hours latency
        is_fresh, msg, data = verify_rainfall_freshness_gate(
            fresh_path, max_latency_hours=12.0, now_utc=ref_time
        )
        self.assertTrue(is_fresh)
        self.assertIn("fresh and complete", msg)
        self.assertAlmostEqual(data.get("calculated_latency_hours", 0), 5.5, places=2)

    def test_missing_observation_timestamp_rejected(self):
        """Test 3: Verify refresh status file missing observation timestamp is rejected."""
        missing_ts_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            # No observation_end_utc
        }
        status_path = self.temp_dir / "missing_ts.json"
        with open(status_path, "w") as f:
            json.dump(missing_ts_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path)
        self.assertFalse(is_fresh)
        self.assertIn("Missing 'observation_end_utc'", msg)

    def test_invalid_observation_timestamp_rejected(self):
        """Test 4: Verify refresh status file with invalid/malformed timestamp format is rejected."""
        invalid_ts_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            "observation_end_utc": "invalid-non-iso-timestamp-string",
        }
        status_path = self.temp_dir / "invalid_ts.json"
        with open(status_path, "w") as f:
            json.dump(invalid_ts_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path)
        self.assertFalse(is_fresh)
        self.assertIn("Invalid 'observation_end_utc'", msg)

    def test_future_timestamp_by_one_second_rejected(self):
        """Verify observation timestamp one second into the future is rejected without tolerance."""
        ref_time = datetime(2026, 11, 7, 8, 0, 0, tzinfo=timezone.utc)
        future_ts = (ref_time + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

        future_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            "observation_end_utc": future_ts,
        }
        status_path = self.temp_dir / "future_1s.json"
        with open(status_path, "w") as f:
            json.dump(future_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path, now_utc=ref_time)
        self.assertFalse(is_fresh)
        self.assertIn("in the future", msg)

    def test_future_timestamp_by_six_minutes_rejected(self):
        """Verify observation timestamp six minutes into the future is strictly rejected."""
        ref_time = datetime(2026, 11, 7, 8, 0, 0, tzinfo=timezone.utc)
        future_ts = (ref_time + timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

        future_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            "observation_end_utc": future_ts,
        }
        status_path = self.temp_dir / "future_6m.json"
        with open(status_path, "w") as f:
            json.dump(future_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path, now_utc=ref_time)
        self.assertFalse(is_fresh)
        self.assertIn("in the future", msg)

    def test_actual_boolean_true_completeness_accepted(self):
        """Verify actual JSON boolean true for is_24h_complete is accepted when other conditions pass."""
        ref_time = datetime(2026, 11, 7, 8, 0, 0, tzinfo=timezone.utc)
        valid_status = {
            "status": "SUCCESS",
            "is_24h_complete": True,
            "observation_end_utc": "2026-11-07T02:00:00Z",
        }
        status_path = self.temp_dir / "bool_true.json"
        with open(status_path, "w") as f:
            json.dump(valid_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path, now_utc=ref_time)
        self.assertTrue(is_fresh)
        self.assertIn("fresh and complete", msg)

    def test_string_true_completeness_rejected(self):
        """Verify string 'true' for is_24h_complete is strictly rejected."""
        str_true_status = {
            "status": "SUCCESS",
            "is_24h_complete": "true",
            "observation_end_utc": "2026-11-07T02:00:00Z",
        }
        status_path = self.temp_dir / "str_true.json"
        with open(status_path, "w") as f:
            json.dump(str_true_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path)
        self.assertFalse(is_fresh)
        self.assertIn("Must be actual boolean true", msg)

    def test_string_false_completeness_rejected(self):
        """Verify string 'false' for is_24h_complete is strictly rejected."""
        str_false_status = {
            "status": "SUCCESS",
            "is_24h_complete": "false",
            "observation_end_utc": "2026-11-07T02:00:00Z",
        }
        status_path = self.temp_dir / "str_false.json"
        with open(status_path, "w") as f:
            json.dump(str_false_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path)
        self.assertFalse(is_fresh)
        self.assertIn("Must be actual boolean true", msg)

    def test_missing_completeness_flag_rejected(self):
        """Verify missing is_24h_complete flag is rejected."""
        missing_flag_status = {
            "status": "SUCCESS",
            "observation_end_utc": "2026-11-07T02:00:00Z",
        }
        status_path = self.temp_dir / "missing_flag.json"
        with open(status_path, "w") as f:
            json.dump(missing_flag_status, f)

        is_fresh, msg, _ = verify_rainfall_freshness_gate(status_path)
        self.assertFalse(is_fresh)
        self.assertIn("Missing 'is_24h_complete'", msg)

    # -----------------------------------------------------------------
    # NON-FINITE RAINFALL VALIDATION TESTS (ISSUE 2)
    # -----------------------------------------------------------------

    def test_positive_infinity_rainfall_rejected(self):
        """Test 5: Verify positive infinity (+inf) in rainfall accumulation is rejected and produces unassessed risk."""
        engine = WesternGhatsDynamicRiskEngine()
        inf_df = self.sample_df.copy()

        # Inject +inf in different windows across rows
        inf_df.loc[3, "gpm_1h_accumulation_mm"] = float("inf")
        inf_df.loc[4, "gpm_24h_accumulation_mm"] = np.inf

        out_df, report = engine.compute_combined_hazard_risk(inf_df)

        # Row 3 checks (+inf in 1h accumulation)
        row3 = out_df.iloc[3]
        self.assertTrue(np.isnan(row3["norm_rain_1h"]))
        self.assertTrue(np.isnan(row3["rainfall_intensity_index"]))
        self.assertTrue(np.isnan(row3["dynamic_risk_index"]))
        self.assertTrue(np.isnan(row3["dynamic_risk_interactive"]))
        self.assertEqual(row3["primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")
        self.assertEqual(row3["risk_calculation_status"], "INCOMPLETE_RAINFALL_WINDOW")

        # Row 4 checks (+inf in 24h accumulation)
        row4 = out_df.iloc[4]
        self.assertTrue(np.isnan(row4["norm_rain_24h"]))
        self.assertTrue(np.isnan(row4["rainfall_intensity_index"]))
        self.assertTrue(np.isnan(row4["dynamic_risk_index"]))
        self.assertEqual(row4["primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")
        self.assertEqual(row4["risk_calculation_status"], "INCOMPLETE_RAINFALL_WINDOW")

    def test_negative_infinity_rainfall_rejected(self):
        """Test 6: Verify negative infinity (-inf) in rainfall accumulation is rejected and produces unassessed risk."""
        engine = WesternGhatsDynamicRiskEngine()
        neginf_df = self.sample_df.copy()

        # Inject -inf in 6h accumulation
        neginf_df.loc[5, "gpm_6h_accumulation_mm"] = float("-inf")
        neginf_df.loc[6, "gpm_12h_accumulation_mm"] = -np.inf

        out_df, report = engine.compute_combined_hazard_risk(neginf_df)

        # Row 5 checks (-inf in 6h)
        row5 = out_df.iloc[5]
        self.assertTrue(np.isnan(row5["norm_rain_6h"]))
        self.assertTrue(np.isnan(row5["rainfall_intensity_index"]))
        self.assertTrue(np.isnan(row5["dynamic_risk_index"]))
        self.assertEqual(row5["primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")

        # Row 6 checks (-inf in 12h)
        row6 = out_df.iloc[6]
        self.assertTrue(np.isnan(row6["norm_rain_12h"]))
        self.assertTrue(np.isnan(row6["rainfall_intensity_index"]))
        self.assertTrue(np.isnan(row6["dynamic_risk_index"]))
        self.assertEqual(row6["primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")

    def test_missing_rainfall_produces_unassessed_risk(self):
        """Test 7: Verify missing/NaN rainfall still produces unassessed risk and incomplete status."""
        engine = WesternGhatsDynamicRiskEngine()
        nan_df = self.sample_df.copy()
        nan_df.loc[7, "gpm_3h_accumulation_mm"] = np.nan

        out_df, report = engine.compute_combined_hazard_risk(nan_df)
        row7 = out_df.iloc[7]
        self.assertTrue(np.isnan(row7["norm_rain_3h"]))
        self.assertTrue(np.isnan(row7["rainfall_intensity_index"]))
        self.assertTrue(np.isnan(row7["dynamic_risk_index"]))
        self.assertEqual(row7["primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")
        self.assertEqual(row7["risk_calculation_status"], "INCOMPLETE_RAINFALL_WINDOW")

    def test_zero_rainfall_behavior_unchanged(self):
        """Test 8: Verify zero-rainfall behavior remains strictly unchanged and matches dry terrain baseline."""
        engine = WesternGhatsDynamicRiskEngine(weight_terrain=0.50, weight_rainfall=0.50)
        out_df, report = engine.compute_combined_hazard_risk(self.sample_df)

        # Rows 0, 1, 2 have strictly 0.0 mm rainfall across all windows
        zero_rows = out_df.iloc[0:3]
        for idx, row in zero_rows.iterrows():
            self.assertEqual(row["norm_rain_1h"], 0.0)
            self.assertEqual(row["norm_rain_3h"], 0.0)
            self.assertEqual(row["norm_rain_6h"], 0.0)
            self.assertEqual(row["norm_rain_12h"], 0.0)
            self.assertEqual(row["norm_rain_24h"], 0.0)
            self.assertEqual(row["rainfall_intensity_index"], 0.0)
            self.assertEqual(row["rainfall_percentile_24h"], 0.0)

            # Dry baseline formula: 100 * w_terrain * S_terrain
            expected_dry_risk = 100.0 * 0.50 * row["terrain_rf_prob_balanced"]
            self.assertAlmostEqual(row["dynamic_risk_index"], expected_dry_risk, places=5)

    # -----------------------------------------------------------------
    # NORMALIZATION & MODEL INVARIANCE TESTS (ISSUE 3 & EXISTING CHECKS)
    # -----------------------------------------------------------------

    def test_rainfall_normalization_and_clipping(self):
        """Verify rainfall normalization clips properly at benchmark scales."""
        engine = WesternGhatsDynamicRiskEngine()
        out_df, _ = engine.compute_combined_hazard_risk(self.sample_df)

        # Row 7 has 24h accumulation = 120.0 mm (scale is 100.0 mm)
        row7 = out_df.iloc[7]
        self.assertEqual(row7["norm_rain_24h"], 1.0)
        self.assertLessEqual(row7["rainfall_intensity_index"], 1.0)
        self.assertGreaterEqual(row7["rainfall_intensity_index"], 0.0)

    def test_relative_screening_category_differentiation(self):
        """Verify relative category assignment populates all 4 screening levels with monotonic properties."""
        engine = WesternGhatsDynamicRiskEngine()
        out_df, report = engine.compute_combined_hazard_risk(self.sample_df)

        rel_cats = out_df["risk_category_relative"].unique()
        self.assertIn("Low", rel_cats)
        self.assertIn("Moderate", rel_cats)
        self.assertIn("High", rel_cats)
        self.assertIn("Very High", rel_cats)

        # Check mean risk score monotonicity
        means = out_df.groupby("risk_category_relative")["dynamic_risk_index"].mean()
        self.assertLess(means["Low"], means["Moderate"])
        self.assertLess(means["Moderate"], means["High"])
        self.assertLess(means["High"], means["Very High"])

    def test_terrain_feature_contract_preservation(self):
        """Verify that all five terrain model predictors remain intact."""
        engine = WesternGhatsDynamicRiskEngine()
        out_df, _ = engine.compute_combined_hazard_risk(self.sample_df)

        for col in TERRAIN_MODEL_PREDICTORS:
            self.assertIn(col, out_df.columns)
            self.assertTrue(np.allclose(out_df[col], self.sample_df[col]))

    def test_disclaimer_stamped_on_every_row(self):
        """Verify the operational disclaimer string is stamped on all rows."""
        engine = WesternGhatsDynamicRiskEngine()
        out_df, _ = engine.compute_combined_hazard_risk(self.sample_df)

        self.assertIn("screening_disclaimer", out_df.columns)
        self.assertTrue((out_df["screening_disclaimer"] == DISCLAIMER_TEXT).all())

    def test_end_to_end_production_integration(self):
        """Execute end-to-end pipeline using production Stage 6C input files into a temporary directory."""
        if not DEFAULT_INTEGRATED_CSV.exists() or not DEFAULT_REFRESH_STATUS_JSON.exists():
            self.skipTest("Production input datasets not available for integration test.")

        prod_out_dir = self.temp_dir / "prod_run"
        result = run_western_ghats_dynamic_risk_pipeline(
            integrated_csv_path=DEFAULT_INTEGRATED_CSV,
            status_json_path=DEFAULT_REFRESH_STATUS_JSON,
            output_dir=prod_out_dir,
            max_latency_hours=12.0,
        )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["csv_path"].exists())
        self.assertTrue(result["metadata_path"].exists())
        self.assertTrue(result["report_path"].exists())

        # Verify output CSV records and columns
        prod_df = pd.read_csv(result["csv_path"])
        self.assertEqual(len(prod_df), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(len(prod_df["tile"].unique()), EXPECTED_TILES_COUNT)
        self.assertIn("dynamic_risk_index", prod_df.columns)
        self.assertIn("risk_category_relative", prod_df.columns)
        self.assertIn("screening_disclaimer", prod_df.columns)


if __name__ == "__main__":
    unittest.main()
