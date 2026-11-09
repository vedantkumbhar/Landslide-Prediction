"""
Automated Test Suite for Stage 7.2: Northeast India Grid Terrain Susceptibility Prediction.

Tests:
1. Model loading and contract verification: Validates RandomForestClassifier and 5 mandatory features.
2. Input grid validation: Rejects missing columns, null grid_ids, and non-finite features.
3. Prediction execution and exact row counts: Generates 14,550 predictions matching input grid.
4. Probability bounds and finiteness: All probabilities strictly within [0.0, 1.0].
5. Binary classification threshold consistency: Validates pred == (prob >= 0.5).
6. Screening tier assignment: Verifies Low, Moderate, High, Very High cutoffs.
7. Provenance retention: Coordinates, grid_id, and tile match input grid row-by-row.
8. Validation report completeness: Schema, statistics, class breakdown, and disclaimer.
9. Overwrite protection: Enforces FileExistsError when outputs exist without --overwrite.
"""

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.predict_northeast_terrain_susceptibility import (
    load_regional_terrain_model,
    validate_input_grid,
    assign_screening_tiers,
    predict_northeast_terrain_susceptibility,
    _validate_predictions,
    DEFAULT_INPUT_GRID,
    DEFAULT_MODEL_PATH,
    MANDATORY_TERRAIN_FEATURES,
    TIER_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestNortheastTerrainSusceptibility(unittest.TestCase):
    """Test suite for Northeast India terrain landslide susceptibility prediction engine."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage7_2_"))
        self.test_csv_path = self.temp_dir / "test_northeast_susceptibility.csv"
        self.test_report_path = self.temp_dir / "test_susceptibility_report.json"

    def tearDown(self):
        """Clean up temporary test directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_loading_and_contract(self):
        """Test 1: Verify trained model loads cleanly and matches exact 5-feature contract."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model path missing: {DEFAULT_MODEL_PATH}")
        model = load_regional_terrain_model(DEFAULT_MODEL_PATH)
        self.assertIsNotNone(model)

        features = list(getattr(model, "feature_names_in_", []))
        self.assertEqual(features, MANDATORY_TERRAIN_FEATURES)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(len(model.classes_), 2)

        # Rejects non-existent path
        with self.assertRaises(FileNotFoundError):
            load_regional_terrain_model(self.temp_dir / "non_existent_model.joblib")

    def test_input_grid_validation(self):
        """Test 2: Verify input grid validation rejects corrupt, missing, or non-finite inputs."""
        valid_df = pd.DataFrame({
            "grid_id": ["NE_001", "NE_002"],
            "latitude": [25.5, 26.0],
            "longitude": [91.0, 92.0],
            "tile": ["N25E091", "N26E092"],
            "elevation_m": [120.0, 450.0],
            "slope_degrees": [10.5, 25.0],
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
        dup_df["grid_id"] = ["NE_001", "NE_001"]
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

    def test_screening_tier_assignment(self):
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

    def test_predictions_generation_and_row_count(self):
        """Test 4: Execute predictions on full grid and verify exact row count of 14,550."""
        report = predict_northeast_terrain_susceptibility(
            input_grid_path=DEFAULT_INPUT_GRID,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=self.test_csv_path,
            validation_report_path=self.test_report_path,
            overwrite=True,
        )

        self.assertEqual(report["status"], "SUCCESS")
        self.assertTrue(self.test_csv_path.exists())
        self.assertTrue(self.test_report_path.exists())

        df_out = pd.read_csv(self.test_csv_path)
        self.assertEqual(len(df_out), 14550)
        self.assertEqual(report["total_grid_locations"], 14550)
        self.assertEqual(df_out["tile"].nunique(), 33)

    def test_probability_bounds_and_finiteness(self):
        """Test 5: Verify all probabilities are finite and strictly within [0.0, 1.0]."""
        predict_northeast_terrain_susceptibility(
            input_grid_path=DEFAULT_INPUT_GRID,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=self.test_csv_path,
            validation_report_path=self.test_report_path,
            overwrite=True,
        )
        df = pd.read_csv(self.test_csv_path)

        probs = df["terrain_rf_prob_balanced"]
        self.assertTrue(np.isfinite(probs).all())
        self.assertFalse(probs.isnull().any())
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())
        self.assertGreater(probs.min(), -1e-6)
        self.assertLessEqual(probs.max(), 1.0)

    def test_binary_prediction_threshold_consistency(self):
        """Test 6: Verify binary predictions are in {0, 1} and strictly consistent with 0.5 threshold."""
        predict_northeast_terrain_susceptibility(
            input_grid_path=DEFAULT_INPUT_GRID,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=self.test_csv_path,
            validation_report_path=self.test_report_path,
            overwrite=True,
        )
        df = pd.read_csv(self.test_csv_path)

        preds = df["terrain_rf_pred_balanced"]
        probs = df["terrain_rf_prob_balanced"]

        self.assertTrue(set(preds.unique()).issubset({0, 1}))
        expected_preds = (probs >= 0.5).astype(int)
        self.assertTrue((preds == expected_preds).all())

    def test_provenance_retention(self):
        """Test 7: Verify grid_id, coordinates, and tile provenance match input grid row-by-row."""
        predict_northeast_terrain_susceptibility(
            input_grid_path=DEFAULT_INPUT_GRID,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=self.test_csv_path,
            validation_report_path=self.test_report_path,
            overwrite=True,
        )
        df_in = pd.read_csv(DEFAULT_INPUT_GRID)
        df_out = pd.read_csv(self.test_csv_path)

        self.assertEqual(len(df_in), len(df_out))
        self.assertTrue((df_in["grid_id"] == df_out["grid_id"]).all())
        self.assertTrue((df_in["tile"] == df_out["tile"]).all())
        self.assertTrue(np.allclose(df_in["latitude"], df_out["latitude"], atol=1e-6))
        self.assertTrue(np.allclose(df_in["longitude"], df_out["longitude"], atol=1e-6))

    def test_validation_report_schema_and_metrics(self):
        """Test 8: Verify validation report structure, stats, tier distributions, and disclaimer."""
        predict_northeast_terrain_susceptibility(
            input_grid_path=DEFAULT_INPUT_GRID,
            model_path=DEFAULT_MODEL_PATH,
            output_csv_path=self.test_csv_path,
            validation_report_path=self.test_report_path,
            overwrite=True,
        )
        with open(self.test_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["region"], "Northeast India")
        self.assertEqual(report["total_grid_locations"], 14550)
        self.assertEqual(report["geographic_extent"]["dem_tile_count"], 33)

        # Check probability stats keys
        stats = report["susceptibility_probability_statistics"]
        for k in ["min", "max", "mean", "std", "P25", "P50", "P75"]:
            self.assertIn(k, stats)
            self.assertTrue(math.isfinite(stats[k]))

        # Check binary class breakdown
        cls_summary = report["binary_classification_summary"]
        self.assertEqual(cls_summary["class_0_count"] + cls_summary["class_1_count"], 14550)
        self.assertAlmostEqual(cls_summary["class_0_pct"] + cls_summary["class_1_pct"], 100.0, places=1)

        # Check tier distribution
        tier_summary = report["screening_tier_distribution"]
        for label in TIER_LABELS:
            self.assertIn(label, tier_summary)

        total_tier_points = sum(tier_summary[l]["count"] for l in TIER_LABELS)
        self.assertEqual(total_tier_points, 14550)

        # Check tile-level stats
        tile_stats = report["tile_level_susceptibility_summary"]
        self.assertEqual(len(tile_stats), 33)

        self.assertIn("EXPERIMENTAL", report["disclaimer"])

    def test_overwrite_protection(self):
        """Test 9: Verify FileExistsError when output exists and overwrite=False."""
        self.test_csv_path.write_text("existing content", encoding="utf-8")
        self.test_report_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            predict_northeast_terrain_susceptibility(
                input_grid_path=DEFAULT_INPUT_GRID,
                model_path=DEFAULT_MODEL_PATH,
                output_csv_path=self.test_csv_path,
                validation_report_path=self.test_report_path,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
