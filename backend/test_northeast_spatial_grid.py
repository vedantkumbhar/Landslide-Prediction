"""
Automated Test Suite for Stage 7.1: Northeast India Spatial Prediction Grid.

Tests:
1. Active DEM tile discovery: Exactly 33 active tiles discovered, excluding backups.
2. Multi-zone UTM support: Detects and processes all 3 UTM zones (EPSG:32645, 32646, 32647).
3. Aspect derivative calculations: Verifies flat vs directional aspect transformations.
4. Output grid generation and dimensions: Verifies >14,000 valid points across all 33 tiles.
5. Model contract feature compliance: Strict verification of 5 mandatory features, units, and ranges.
6. Geographic bounds compliance: Verifies all coordinates fall within Northeast India extent.
7. Zero duplicate coordinates: Enforces unique spatial locations across tile boundaries.
8. Trained regional model inference compatibility: Proves scikit-learn RF model consumes grid features cleanly.
9. Validation report completeness: Verifies JSON report structure, stats, and metadata schema.
10. Overwrite protection: Verifies FileExistsError when outputs exist without --overwrite.
"""

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import rasterio

from backend.generate_northeast_spatial_grid import (
    discover_northeast_dem_tiles,
    get_tile_raster_paths,
    compute_aspect_derivatives,
    generate_northeast_spatial_grid,
    _validate_grid_dataset,
    DEFAULT_DEM_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_VALIDATION_REPORT,
    MANDATORY_TERRAIN_FEATURES,
    NORTHEAST_LAT_BOUNDS,
    NORTHEAST_LON_BOUNDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORTHEAST_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regional_models_30oct"
    / "northeast_india"
    / "random_forest_model.joblib"
)


class TestNortheastSpatialGrid(unittest.TestCase):
    """Test suite for Northeast India 5 km spatial prediction grid generation."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage7_1_"))
        self.test_csv_path = self.temp_dir / "test_northeast_grid.csv"
        self.test_report_path = self.temp_dir / "test_validation_report.json"

    def tearDown(self):
        """Clean up temporary test directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tile_discovery_and_active_count(self):
        """Test 1: Verify exactly 33 active DEM tiles discovered and backups excluded."""
        tiles = discover_northeast_dem_tiles(DEFAULT_DEM_DIR)
        self.assertEqual(len(tiles), 33)
        self.assertNotIn("N27E088_backup_before_fix", tiles)
        self.assertIn("N22E088", tiles)
        self.assertIn("N27E088", tiles)
        self.assertIn("N28E096", tiles)

    def test_multizone_utm_detection_and_handling(self):
        """Test 2: Verify all three UTM zones (45N, 46N, 47N) are detected across active tiles."""
        tiles = discover_northeast_dem_tiles(DEFAULT_DEM_DIR)
        zones = set()
        for t in tiles:
            paths = get_tile_raster_paths(DEFAULT_DEM_DIR, t)
            with rasterio.open(paths["elevation"]) as src:
                zones.add(str(src.crs))

        expected_zones = {"EPSG:32645", "EPSG:32646", "EPSG:32647"}
        self.assertEqual(zones, expected_zones)

    def test_aspect_derivative_transformations(self):
        """Test 3: Verify aspect angle to sin/cos/flat transformations."""
        # Flat terrain (negative values in GDAL aspect)
        s, c, f = compute_aspect_derivatives(-1.0)
        self.assertEqual(f, 1)
        self.assertEqual(s, 0.0)
        self.assertEqual(c, 0.0)

        s, c, f = compute_aspect_derivatives(-9999.0)
        self.assertEqual(f, 1)
        self.assertEqual(s, 0.0)
        self.assertEqual(c, 0.0)

        # North (0 deg)
        s, c, f = compute_aspect_derivatives(0.0)
        self.assertEqual(f, 0)
        self.assertAlmostEqual(s, 0.0, places=6)
        self.assertAlmostEqual(c, 1.0, places=6)

        # East (90 deg)
        s, c, f = compute_aspect_derivatives(90.0)
        self.assertEqual(f, 0)
        self.assertAlmostEqual(s, 1.0, places=6)
        self.assertAlmostEqual(c, 0.0, places=6)

        # South (180 deg)
        s, c, f = compute_aspect_derivatives(180.0)
        self.assertEqual(f, 0)
        self.assertAlmostEqual(s, 0.0, places=6)
        self.assertAlmostEqual(c, -1.0, places=6)

        # West (270 deg)
        s, c, f = compute_aspect_derivatives(270.0)
        self.assertEqual(f, 0)
        self.assertAlmostEqual(s, -1.0, places=6)
        self.assertAlmostEqual(c, 0.0, places=6)

    def test_grid_generation_and_output_dimensions(self):
        """Test 4: Generate grid in temporary directory and verify output dimensions."""
        report = generate_northeast_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=self.test_csv_path,
            validation_report_path=self.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )

        self.assertEqual(report["status"], "SUCCESS")
        self.assertTrue(self.test_csv_path.exists())
        self.assertTrue(self.test_report_path.exists())

        df = pd.read_csv(self.test_csv_path)
        self.assertGreater(len(df), 14000)
        self.assertEqual(report["total_valid_grid_locations"], len(df))

        # Check all 33 tiles represented
        self.assertEqual(df["tile"].nunique(), 33)

        # Check unique grid IDs
        self.assertEqual(df["grid_id"].nunique(), len(df))
        self.assertTrue(df["grid_id"].str.startswith("NE_GRID_").all())

    def test_model_contract_feature_compliance(self):
        """Test 5: Verify all 5 terrain features conform strictly to model contract."""
        generate_northeast_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=self.test_csv_path,
            validation_report_path=self.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )
        df = pd.read_csv(self.test_csv_path)

        for col in MANDATORY_TERRAIN_FEATURES:
            self.assertIn(col, df.columns)
            self.assertTrue(np.isfinite(df[col]).all())
            self.assertFalse((df[col] == -9999.0).any())

        # Slope range [0, 90]
        self.assertTrue((df["slope_degrees"] >= 0.0).all())
        self.assertTrue((df["slope_degrees"] <= 90.0).all())

        # Aspect sin/cos range [-1, 1]
        self.assertTrue((df["aspect_sin"] >= -1.0001).all())
        self.assertTrue((df["aspect_sin"] <= 1.0001).all())
        self.assertTrue((df["aspect_cos"] >= -1.0001).all())
        self.assertTrue((df["aspect_cos"] <= 1.0001).all())

        # Aspect flat binary {0, 1}
        self.assertTrue(set(df["aspect_flat"].unique()).issubset({0, 1}))

        # Pythagorean identity for non-flat aspect
        non_flat = df[df["aspect_flat"] == 0]
        sum_sq = non_flat["aspect_sin"] ** 2 + non_flat["aspect_cos"] ** 2
        self.assertTrue(np.allclose(sum_sq, 1.0, atol=1e-4))

        # Flat terrain check
        flat = df[df["aspect_flat"] == 1]
        if len(flat) > 0:
            self.assertTrue((flat["aspect_sin"] == 0.0).all())
            self.assertTrue((flat["aspect_cos"] == 0.0).all())

    def test_geographic_bounds_compliance(self):
        """Test 6: Verify all grid coordinates fall within Northeast India extent."""
        generate_northeast_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=self.test_csv_path,
            validation_report_path=self.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )
        df = pd.read_csv(self.test_csv_path)

        self.assertTrue(df["latitude"].between(NORTHEAST_LAT_BOUNDS[0], NORTHEAST_LAT_BOUNDS[1]).all())
        self.assertTrue(df["longitude"].between(NORTHEAST_LON_BOUNDS[0], NORTHEAST_LON_BOUNDS[1]).all())

    def test_no_duplicate_spatial_coordinates(self):
        """Test 7: Enforce zero duplicate (latitude, longitude) coordinates across entire grid."""
        generate_northeast_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=self.test_csv_path,
            validation_report_path=self.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )
        df = pd.read_csv(self.test_csv_path)

        dup_count = df.duplicated(subset=["latitude", "longitude"]).sum()
        self.assertEqual(dup_count, 0, f"Found {dup_count} duplicate coordinate pairs")

    def test_trained_model_inference_compatibility(self):
        """Test 8: Verify trained Northeast Random Forest model consumes generated grid seamlessly."""
        self.assertTrue(NORTHEAST_MODEL_PATH.exists(), f"Model not found: {NORTHEAST_MODEL_PATH}")
        model = joblib.load(NORTHEAST_MODEL_PATH)

        generate_northeast_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=self.test_csv_path,
            validation_report_path=self.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )
        df = pd.read_csv(self.test_csv_path)

        # Take a representative sample across multiple tiles
        sample_df = df.groupby("tile").head(5)[MANDATORY_TERRAIN_FEATURES]
        self.assertEqual(len(sample_df), 33 * 5)

        # Run inference
        probs = model.predict_proba(sample_df)[:, 1]
        self.assertEqual(len(probs), len(sample_df))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_validation_report_completeness(self):
        """Test 9: Verify validation report structure, stats, and metadata schema."""
        generate_northeast_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=self.test_csv_path,
            validation_report_path=self.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )
        with open(self.test_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["region"], "Northeast India")
        self.assertEqual(report["grid_spacing_km"], 5.0)
        self.assertEqual(report["dem_tile_count"], 33)
        self.assertIn("excluded_samples_breakdown", report)
        self.assertIn("terrain_feature_statistics", report)
        self.assertIn("tile_point_distribution", report)
        self.assertTrue(report["model_contract_verification"]["all_features_present"])
        self.assertTrue(report["model_contract_verification"]["all_values_finite"])

    def test_overwrite_protection(self):
        """Test 10: Verify FileExistsError when outputs exist without --overwrite."""
        self.test_csv_path.write_text("existing content", encoding="utf-8")
        self.test_report_path.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            generate_northeast_spatial_grid(
                dem_dir=DEFAULT_DEM_DIR,
                output_csv=self.test_csv_path,
                validation_report_path=self.test_report_path,
                spacing_m=5000.0,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
