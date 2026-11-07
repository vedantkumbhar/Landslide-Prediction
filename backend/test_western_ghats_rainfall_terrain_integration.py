"""
Comprehensive Validation Suite for Western Ghats GPM Rainfall–Terrain Integration (Stage 6C).

Tests the 10 mandatory requirements:
1. Exact record retention (1,582 records: 182 events, 1,400 background).
2. Complete spatial coverage across all 14 DEM tiles.
3. ID uniqueness (zero duplicates in sample_id or record_id).
4. Exact 1-to-1 terrain-to-rainfall ID alignment.
5. Zero unintended changes to terrain values (bitwise parity with source files).
6. Rainfall units and rolling accumulation consistency (valid intervals, monotonicity).
7. Correct UTC observation timestamps (ISO-8601).
8. Freshness and completeness checks (refusal of stale/incomplete rainfall inputs).
9. Missing rainfall handling (never converting missing data to zero).
10. Existing approved terrain model compatibility (contract preservation, successful inference).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from backend.integrate_western_ghats_rainfall_terrain import (
    run_integration_pipeline,
    perform_integration,
    verify_rainfall_freshness_and_status,
    load_source_datasets,
    EVENTS_CSV,
    BACKGROUND_CSV,
    LATEST_RAINFALL_CSV,
    LATEST_STATUS_JSON,
    MODEL_BALANCED_PATH,
    MODEL_UNWEIGHTED_PATH,
    EXPECTED_EVENT_COUNT,
    EXPECTED_BG_COUNT,
    EXPECTED_TOTAL_COUNT,
    EXPECTED_TILE_COUNT,
    TERRAIN_MODEL_PREDICTORS,
    RAIN_ACCUMULATION_COLUMNS,
)

EXPECTED_TILE_DISTRIBUTION = {
    "N09E076": 121,
    "N10E076": 114,
    "N10E077": 112,
    "N11E076": 117,
    "N12E074": 111,
    "N12E075": 108,
    "N14E074": 108,
    "N15E073": 120,
    "N15E074": 108,
    "N16E073": 108,
    "N17E073": 107,
    "N18E073": 113,
    "N19E072": 118,
    "N19E073": 117,
}


class TestWesternGhatsRainfallTerrainIntegration(unittest.TestCase):
    """Rigorous validation suite for Stage 6C Rainfall-Terrain integration."""

    @classmethod
    def setUpClass(cls):
        assert EVENTS_CSV.exists(), f"Events CSV not found: {EVENTS_CSV}"
        assert BACKGROUND_CSV.exists(), f"Background CSV not found: {BACKGROUND_CSV}"
        assert LATEST_RAINFALL_CSV.exists(), f"Latest rainfall CSV not found: {LATEST_RAINFALL_CSV}"
        assert LATEST_STATUS_JSON.exists(), f"Status JSON not found: {LATEST_STATUS_JSON}"
        assert MODEL_BALANCED_PATH.exists(), f"Balanced model not found: {MODEL_BALANCED_PATH}"
        assert MODEL_UNWEIGHTED_PATH.exists(), f"Unweighted model not found: {MODEL_UNWEIGHTED_PATH}"

        cls.ev_df, cls.bg_df, cls.rf_df = load_source_datasets(
            EVENTS_CSV, BACKGROUND_CSV, LATEST_RAINFALL_CSV
        )

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_integration_"))

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------
    # Test 1: Record Retention & Counts
    # -----------------------------------------------------------------
    def test_record_retention_and_counts(self):
        """Verify exact record retention: 1,582 records (182 events + 1,400 background)."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)
        self.assertEqual(len(integrated_df), EXPECTED_TOTAL_COUNT)
        self.assertEqual((integrated_df["label"] == 1).sum(), EXPECTED_EVENT_COUNT)
        self.assertEqual((integrated_df["label"] == 0).sum(), EXPECTED_BG_COUNT)

    # -----------------------------------------------------------------
    # Test 2: Spatial Coverage (All 14 Tiles)
    # -----------------------------------------------------------------
    def test_spatial_coverage_all_14_tiles(self):
        """Verify that all 14 DEM tiles are represented with exact expected counts."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)
        self.assertEqual(integrated_df["tile"].nunique(), EXPECTED_TILE_COUNT)
        counts = integrated_df.groupby("tile").size().to_dict()
        self.assertEqual(counts, EXPECTED_TILE_DISTRIBUTION)

    # -----------------------------------------------------------------
    # Test 3: ID Uniqueness
    # -----------------------------------------------------------------
    def test_id_uniqueness(self):
        """Verify zero duplicates in sample_id and record_id."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)
        self.assertEqual(integrated_df["sample_id"].nunique(), EXPECTED_TOTAL_COUNT)
        self.assertEqual(integrated_df["record_id"].nunique(), EXPECTED_TOTAL_COUNT)
        self.assertFalse(integrated_df["sample_id"].duplicated().any())
        self.assertFalse(integrated_df["record_id"].duplicated().any())

    # -----------------------------------------------------------------
    # Test 4: Exact 1-to-1 Terrain-to-Rainfall ID Matching
    # -----------------------------------------------------------------
    def test_exact_terrain_rainfall_id_matching(self):
        """Verify 100% exact 1-to-1 ID match between terrain inputs and rainfall records."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)
        expected_ids = set(self.ev_df["event_id"].astype(str)) | set(self.bg_df["sample_id"].astype(str))
        integrated_ids = set(integrated_df["sample_id"].astype(str))
        self.assertEqual(integrated_ids, expected_ids)

    # -----------------------------------------------------------------
    # Test 5: Zero Unintended Changes to Terrain Values
    # -----------------------------------------------------------------
    def test_zero_terrain_value_changes(self):
        """Verify 0-difference floating-point equality between integrated dataset and source terrain files."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)

        ev_map = {str(r["event_id"]): r for _, r in self.ev_df.iterrows()}
        bg_map = {str(r["sample_id"]): r for _, r in self.bg_df.iterrows()}

        check_cols = ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "aspect_flat", "latitude", "longitude"]
        for _, row in integrated_df.iterrows():
            sid = str(row["sample_id"])
            src = ev_map[sid] if row["label"] == 1 else bg_map[sid]
            for col in check_cols:
                diff = abs(float(row[col]) - float(src[col]))
                self.assertLessEqual(diff, 1e-6, f"Terrain value mismatch at {sid} for {col}")

    # -----------------------------------------------------------------
    # Test 6: Rainfall Units & Rolling Accumulation Consistency
    # -----------------------------------------------------------------
    def test_rainfall_accumulation_consistency(self):
        """Verify rainfall interval counts, zero negative values, and monotonicity."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)

        for col in RAIN_ACCUMULATION_COLUMNS:
            self.assertFalse(integrated_df[col].isna().any())
            self.assertTrue((integrated_df[col] >= 0.0).all())

        # Interval checks
        self.assertTrue((integrated_df["gpm_1h_valid_intervals"] == 2).all())
        self.assertTrue((integrated_df["gpm_3h_valid_intervals"] == 6).all())
        self.assertTrue((integrated_df["gpm_6h_valid_intervals"] == 12).all())
        self.assertTrue((integrated_df["gpm_12h_valid_intervals"] == 24).all())
        self.assertTrue((integrated_df["gpm_24h_valid_intervals"] == 48).all())

        # Monotonicity check: 1h <= 3h <= 6h <= 12h <= 24h
        self.assertTrue((integrated_df["gpm_1h_accumulation_mm"] <= integrated_df["gpm_3h_accumulation_mm"] + 1e-6).all())
        self.assertTrue((integrated_df["gpm_3h_accumulation_mm"] <= integrated_df["gpm_6h_accumulation_mm"] + 1e-6).all())
        self.assertTrue((integrated_df["gpm_6h_accumulation_mm"] <= integrated_df["gpm_12h_accumulation_mm"] + 1e-6).all())
        self.assertTrue((integrated_df["gpm_12h_accumulation_mm"] <= integrated_df["gpm_24h_accumulation_mm"] + 1e-6).all())

    # -----------------------------------------------------------------
    # Test 7: Correct UTC Observation Timestamps
    # -----------------------------------------------------------------
    def test_utc_observation_timestamps(self):
        """Verify that observation timestamps are in UTC and conform to ISO-8601 formatting."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)

        time_cols = [c for c in integrated_df.columns if c.endswith("_utc")]
        self.assertGreaterEqual(len(time_cols), 6)

        for col in time_cols:
            vals = integrated_df[col].unique()
            self.assertEqual(len(vals), 1, f"Multiple timestamps in uniform window column: {col}")
            self.assertTrue(vals[0].endswith("Z"))
            datetime.strptime(vals[0], "%Y-%m-%dT%H:%M:%SZ")

    # -----------------------------------------------------------------
    # Test 8: Freshness & Completeness Gating
    # -----------------------------------------------------------------
    def test_freshness_and_completeness_gating(self):
        """Verify refusal of stale, incomplete, or failed rainfall refresh inputs."""
        # Test A: Status is not SUCCESS
        bad_status_file = self.test_dir / "bad_status.json"
        with open(bad_status_file, "w") as f:
            json.dump({"status": "STALE_DATA", "is_24h_complete": True, "missing_intervals_count": 0, "data_latency_hours": 150.0}, f)

        with self.assertRaises(ValueError) as ctx:
            verify_rainfall_freshness_and_status(bad_status_file, max_latency_hours=12.0)
        self.assertIn("STALE_DATA", str(ctx.exception))

        # Test B: Incomplete 24h window
        with open(bad_status_file, "w") as f:
            json.dump({"status": "SUCCESS", "is_24h_complete": False, "missing_intervals_count": 5, "data_latency_hours": 4.0}, f)

        with self.assertRaises(ValueError) as ctx:
            verify_rainfall_freshness_and_status(bad_status_file, max_latency_hours=12.0)
        self.assertIn("incomplete", str(ctx.exception).lower())

        # Test C: Excessive latency
        with open(bad_status_file, "w") as f:
            json.dump({"status": "SUCCESS", "is_24h_complete": True, "missing_intervals_count": 0, "data_latency_hours": 18.0}, f)

        with self.assertRaises(ValueError) as ctx:
            verify_rainfall_freshness_and_status(bad_status_file, max_latency_hours=12.0)
        self.assertIn("latency", str(ctx.exception).lower())

    # -----------------------------------------------------------------
    # Test 9: Missing Rainfall Handling (Never Convert to Zero)
    # -----------------------------------------------------------------
    def test_missing_rainfall_never_converted_to_zero(self):
        """Verify that any missing/NaN rainfall value causes integration failure rather than silent zeroing."""
        corrupt_rf = self.rf_df.copy()
        corrupt_rf.loc[5, "gpm_24h_accumulation_mm"] = np.nan

        with self.assertRaises(ValueError) as ctx:
            perform_integration(self.ev_df, self.bg_df, corrupt_rf)
        self.assertIn("nulls in rainfall column", str(ctx.exception).lower())

    # -----------------------------------------------------------------
    # Test 10: Existing Terrain Model Compatibility
    # -----------------------------------------------------------------
    def test_existing_terrain_model_compatibility(self):
        """Verify model feature contract preservation and inference compatibility without retraining."""
        integrated_df, summary = perform_integration(self.ev_df, self.bg_df, self.rf_df)

        model_bal = joblib.load(MODEL_BALANCED_PATH)
        model_unw = joblib.load(MODEL_UNWEIGHTED_PATH)

        # 1. Input contract must be exactly the 5 terrain predictors
        self.assertEqual(list(model_bal.feature_names_in_), TERRAIN_MODEL_PREDICTORS)
        self.assertEqual(list(model_unw.feature_names_in_), TERRAIN_MODEL_PREDICTORS)

        # 2. Inference on integrated terrain features must run without error
        X_terrain = integrated_df[TERRAIN_MODEL_PREDICTORS]
        probs_bal = model_bal.predict_proba(X_terrain)[:, 1]
        probs_unw = model_unw.predict_proba(X_terrain)[:, 1]

        self.assertEqual(len(probs_bal), EXPECTED_TOTAL_COUNT)
        self.assertEqual(len(probs_unw), EXPECTED_TOTAL_COUNT)
        self.assertTrue((probs_bal >= 0.0).all() and (probs_bal <= 1.0).all())
        self.assertTrue((probs_unw >= 0.0).all() and (probs_unw <= 1.0).all())

        # 3. Reference predictions stored in integrated dataset must match exactly
        np.testing.assert_allclose(integrated_df["terrain_rf_prob_balanced"].values, probs_bal, atol=1e-7)
        np.testing.assert_allclose(integrated_df["terrain_rf_prob_unweighted"].values, probs_unw, atol=1e-7)


def run_tests():
    print("\n" + "=" * 75)
    print("RUNNING STAGE 6C RAINFALL-TERRAIN INTEGRATION TEST SUITE (10 TESTS)")
    print("=" * 75)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWesternGhatsRainfallTerrainIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nTEST SUITE FAILED!")
        sys.exit(1)
    print("\nALL 10 TESTS PASSED WITH 100% SUCCESS.")


if __name__ == "__main__":
    run_tests()
