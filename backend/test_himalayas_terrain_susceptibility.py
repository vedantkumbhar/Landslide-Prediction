"""
Automated Test Suite for Stage 8.2: Himalayas Grid Terrain Susceptibility Prediction.

Tests:
1. Model loading and contract verification: Validates RandomForestClassifier and 5 mandatory features without retraining.
2. Input grid validation: Rejects missing columns, null grid_ids, and non-finite features.
3. Prediction execution and exact row counts: Generates 12,052 predictions matching input grid.
4. Probability bounds and finiteness: All probabilities strictly within [0.0, 1.0].
5. Binary classification threshold consistency: Validates pred == (prob >= 0.5).
6. Screening tier assignment: Verifies Low, Moderate, High, Very High cutoffs and methodology.
7. Provenance retention: Coordinates, grid_id, and tile match input grid row-by-row.
8. Validation report completeness: Schema, statistics, class breakdown, tile distribution, and disclaimer.
9. Overwrite protection: Enforces FileExistsError when outputs exist without --overwrite.
10. Protection of previous approved project stages: Stage 8.1 grid, Northeast India, and Western Ghats files remain intact.
"""

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.predict_himalayas_terrain_susceptibility import (
    load_regional_terrain_model,
    validate_input_grid,
    assign_screening_tiers,
    predict_himalayas_terrain_susceptibility,
    _validate_predictions,
    DEFAULT_INPUT_GRID,
    DEFAULT_MODEL_PATH,
    MANDATORY_TERRAIN_FEATURES,
    TIER_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reference paths for regression / safeguard tests
STAGE_8_1_GRID_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "himalayas_spatial_grid"
    / "himalayas_5km_spatial_grid.csv"
)
STAGE_8_1_REPORT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "himalayas_spatial_grid"
    / "himalayas_spatial_grid_validation_report.json"
)
NORTHEAST_SUSCEPTIBILITY_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_terrain_predictions"
    / "northeast_india_5km_terrain_susceptibility.csv"
)
WESTERN_GHATS_GRID_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_grid"
    / "western_ghats_5km_spatial_grid.csv"
)


class TestHimalayasTerrainSusceptibility(unittest.TestCase):
    """Test suite for Himalayas terrain landslide susceptibility prediction engine."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level temporary directory and run predictions once."""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage8_2_"))
        cls.test_csv_path = cls.temp_dir / "test_himalayas_susceptibility.csv"
        cls.test_report_path = cls.temp_dir / "test_susceptibility_report.json"

        cls.report = predict_himalayas_terrain_susceptibility(
            input_grid_path=DEFAULT_INPUT_GRID,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=cls.test_csv_path,
            validation_report_path=cls.test_report_path,
            overwrite=True,
        )
        cls.df_out = pd.read_csv(cls.test_csv_path)
        cls.df_in = pd.read_csv(DEFAULT_INPUT_GRID)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_model_loading_and_contract(self):
        """Test 1: Verify trained model loads cleanly and matches exact 5-feature contract without retraining."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model path missing: {DEFAULT_MODEL_PATH}")
        model = load_regional_terrain_model(DEFAULT_MODEL_PATH)
        self.assertIsNotNone(model)

        features = list(getattr(model, "feature_names_in_", []))
        self.assertEqual(features, MANDATORY_TERRAIN_FEATURES)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(len(model.classes_), 2)
        self.assertEqual(len(model.estimators_), 300)

        # Rejects non-existent path
        with self.assertRaises(FileNotFoundError):
            load_regional_terrain_model(self.temp_dir / "non_existent_model.joblib")

    def test_02_input_grid_validation(self):
        """Test 2: Verify input grid validation rejects corrupt, missing, or non-finite inputs."""
        valid_df = pd.DataFrame({
            "grid_id": ["HIM_001", "HIM_002"],
            "latitude": [30.5, 31.0],
            "longitude": [78.0, 79.0],
            "tile": ["N30E078", "N31E079"],
            "elevation_m": [1200.0, 2450.0],
            "slope_degrees": [15.5, 28.0],
            "aspect_sin": [0.0, 0.7071],
            "aspect_cos": [1.0, 0.7071],
            "aspect_flat": [0, 0],
        })
        # Valid grid passes cleanly
        validate_input_grid(valid_df)

        # Missing required column
        missing_df = valid_df.drop(columns=["slope_degrees"])
        with self.assertRaises(ValueError):
            validate_input_grid(missing_df)

        # Duplicate grid_id
        dup_df = valid_df.copy()
        dup_df["grid_id"] = ["HIM_001", "HIM_001"]
        with self.assertRaises(ValueError):
            validate_input_grid(dup_df)

        # Non-finite slope
        nan_df = valid_df.copy()
        nan_df.loc[0, "slope_degrees"] = np.nan
        with self.assertRaises(ValueError):
            validate_input_grid(nan_df)

        # Out-of-bounds slope
        bad_slope_df = valid_df.copy()
        bad_slope_df.loc[0, "slope_degrees"] = 95.0
        with self.assertRaises(ValueError):
            validate_input_grid(bad_slope_df)

    def test_03_screening_tier_assignment(self):
        """Test 3: Verify continuous probabilities map accurately to discrete screening tiers."""
        test_probs = np.array([0.0, 0.15, 0.249, 0.25, 0.40, 0.499, 0.50, 0.65, 0.749, 0.75, 0.95, 1.0])
        tiers = assign_screening_tiers(test_probs)

        expected = [
            "Low", "Low", "Low",
            "Moderate", "Moderate", "Moderate",
            "High", "High", "High",
            "Very High", "Very High", "Very High",
        ]
        self.assertEqual(list(tiers), expected)

    def test_04_predictions_generation_and_row_count(self):
        """Test 4: Execute predictions on full grid and verify exact row count of 12,052."""
        self.assertEqual(self.report["status"], "SUCCESS")
        self.assertTrue(self.test_csv_path.exists())
        self.assertTrue(self.test_report_path.exists())

        self.assertEqual(len(self.df_out), 12052)
        self.assertEqual(self.report["total_grid_locations"], 12052)
        self.assertEqual(self.df_out["tile"].nunique(), 29)

    def test_05_probability_bounds_and_finiteness(self):
        """Test 5: Verify all probabilities are finite and strictly within [0.0, 1.0]."""
        probs = self.df_out["terrain_rf_prob_balanced"]
        self.assertTrue(np.isfinite(probs).all())
        self.assertFalse(probs.isnull().any())
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())
        self.assertGreater(probs.min(), -1e-6)
        self.assertLessEqual(probs.max(), 1.0)

    def test_06_binary_prediction_threshold_consistency(self):
        """Test 6: Verify binary predictions are in {0, 1} and strictly consistent with 0.5 threshold."""
        preds = self.df_out["terrain_rf_pred_balanced"]
        probs = self.df_out["terrain_rf_prob_balanced"]

        self.assertTrue(set(preds.unique()).issubset({0, 1}))
        expected_preds = (probs >= 0.5).astype(int)
        self.assertTrue((preds == expected_preds).all())

    def test_07_provenance_retention(self):
        """Test 7: Verify grid_id, coordinates, and tile provenance match input grid row-by-row."""
        self.assertEqual(len(self.df_in), len(self.df_out))
        self.assertTrue((self.df_in["grid_id"] == self.df_out["grid_id"]).all())
        self.assertTrue((self.df_in["tile"] == self.df_out["tile"]).all())
        self.assertTrue(np.allclose(self.df_in["latitude"], self.df_out["latitude"], atol=1e-6))
        self.assertTrue(np.allclose(self.df_in["longitude"], self.df_out["longitude"], atol=1e-6))

        for feat in MANDATORY_TERRAIN_FEATURES:
            self.assertTrue(np.allclose(self.df_in[feat], self.df_out[feat], atol=1e-4))

    def test_08_validation_report_schema_and_metrics(self):
        """Test 8: Verify validation report structure, stats, tier distributions, and disclaimer."""
        with open(self.test_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["region"], "Himalayas")
        self.assertEqual(report["total_grid_locations"], 12052)
        self.assertEqual(report["geographic_extent"]["dem_tile_count"], 29)

        # Check probability stats keys
        stats = report["susceptibility_probability_statistics"]
        for k in ["min", "max", "mean", "std", "P25", "P50", "P75"]:
            self.assertIn(k, stats)
            self.assertTrue(math.isfinite(stats[k]))

        # Check binary class breakdown
        cls_summary = report["binary_classification_summary"]
        self.assertEqual(cls_summary["class_0_count"] + cls_summary["class_1_count"], 12052)
        self.assertAlmostEqual(cls_summary["class_0_pct"] + cls_summary["class_1_pct"], 100.0, places=1)

        # Check tier distribution
        tier_summary = report["screening_tier_distribution"]
        for label in TIER_LABELS:
            self.assertIn(label, tier_summary)

        total_tier_points = sum(tier_summary[l]["count"] for l in TIER_LABELS)
        self.assertEqual(total_tier_points, 12052)

        # Check tile-level stats
        tile_stats = report["tile_level_susceptibility_summary"]
        self.assertEqual(len(tile_stats), 29)

        self.assertIn("EXPERIMENTAL", report["disclaimer"])

    def test_09_overwrite_protection(self):
        """Test 9: Verify FileExistsError when output exists and overwrite=False."""
        temp_dir_ow = Path(tempfile.mkdtemp(prefix="test_ow_stage8_2_"))
        csv_ow = temp_dir_ow / "existing.csv"
        rep_ow = temp_dir_ow / "existing.json"
        try:
            csv_ow.write_text("existing content", encoding="utf-8")
            rep_ow.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                predict_himalayas_terrain_susceptibility(
                    input_grid_path=DEFAULT_INPUT_GRID,
                    model_path=DEFAULT_MODEL_PATH,
                    output_csv_path=csv_ow,
                    validation_report_path=rep_ow,
                    overwrite=False,
                )
        finally:
            shutil.rmtree(temp_dir_ow, ignore_errors=True)

    def test_10_protection_of_previous_approved_project_stages(self):
        """Test 10: Verify Stage 8.1 grid, Northeast India, and Western Ghats files remain intact."""
        self.assertTrue(STAGE_8_1_GRID_CSV.exists(), "Stage 8.1 grid CSV must exist")
        self.assertTrue(STAGE_8_1_REPORT_JSON.exists(), "Stage 8.1 validation report must exist")
        df_8_1 = pd.read_csv(STAGE_8_1_GRID_CSV)
        self.assertEqual(len(df_8_1), 12052)

        self.assertTrue(NORTHEAST_SUSCEPTIBILITY_CSV.exists(), "Northeast susceptibility CSV must exist")
        df_ne = pd.read_csv(NORTHEAST_SUSCEPTIBILITY_CSV)
        self.assertEqual(len(df_ne), 14550)

        if WESTERN_GHATS_GRID_CSV.exists():
            df_wg = pd.read_csv(WESTERN_GHATS_GRID_CSV)
            self.assertGreater(len(df_wg), 1000)


if __name__ == "__main__":
    unittest.main()
