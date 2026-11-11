"""
Automated Test Suite for Stage 8.1: Himalayas Spatial Prediction Grid.

Tests:
1. Active DEM tile discovery: Exactly 29 active tiles discovered.
2. Multi-zone UTM support: Detects and processes both UTM zones (EPSG:32643, EPSG:32644).
3. Aspect derivative calculations: Verifies flat vs directional aspect transformations.
4. Output grid generation and dimensions: Verifies exactly 12,052 valid points across all 29 tiles.
5. Model contract feature compliance: Strict verification of 5 mandatory features, units, and ranges.
6. Geographic bounds compliance: Verifies all coordinates fall within Himalayas extent.
7. Zero duplicate coordinates: Enforces unique spatial locations across tile boundaries.
8. Trained regional model inference compatibility: Proves scikit-learn RF model consumes grid features cleanly.
9. Validation report completeness: Verifies JSON report structure, stats, and metadata schema.
10. Overwrite protection: Verifies FileExistsError when outputs exist without --overwrite.
11. System protection boundaries: Verifies Western Ghats and Northeast India baseline files remain intact.
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

from backend.generate_himalayas_spatial_grid import (
    discover_himalayas_dem_tiles,
    get_tile_raster_paths,
    compute_aspect_derivatives,
    generate_himalayas_spatial_grid,
    _validate_grid_dataset,
    DEFAULT_DEM_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_VALIDATION_REPORT,
    MANDATORY_TERRAIN_FEATURES,
    HIMALAYAS_LAT_BOUNDS,
    HIMALAYAS_LON_BOUNDS,
    GRID_PREFIX,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIMALAYAS_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regional_models_30oct"
    / "himalayas"
    / "random_forest_model.joblib"
)

# Reference files for system protection boundaries
NORTHEAST_GRID_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_spatial_grid"
    / "northeast_india_5km_spatial_grid.csv"
)
NORTHEAST_RISK_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_dynamic_risk"
    / "northeast_india_5km_dynamic_hazard_risk.csv"
)
WESTERN_GHATS_GRID_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spatial_grid"
    / "western_ghats_5km_spatial_grid.csv"
)


class TestHimalayasSpatialGrid(unittest.TestCase):
    """Test suite for Himalayas 5 km spatial prediction grid generation."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level temporary directory and generate test grid once."""
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_stage8_1_"))
        cls.test_csv_path = cls.temp_dir / "test_himalayas_grid.csv"
        cls.test_report_path = cls.temp_dir / "test_validation_report.json"

        # Generate test grid once for all tests to evaluate
        cls.report = generate_himalayas_spatial_grid(
            dem_dir=DEFAULT_DEM_DIR,
            output_csv=cls.test_csv_path,
            validation_report_path=cls.test_report_path,
            spacing_m=5000.0,
            overwrite=True,
        )
        cls.df = pd.read_csv(cls.test_csv_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_tile_discovery_and_active_count(self):
        """Test 1: Verify exactly 29 active DEM tiles discovered."""
        tiles = discover_himalayas_dem_tiles(DEFAULT_DEM_DIR)
        self.assertEqual(len(tiles), 29)
        self.assertIn("N27E078", tiles)
        self.assertIn("N30E078", tiles)
        self.assertIn("N34E077", tiles)

    def test_02_multizone_utm_detection_and_handling(self):
        """Test 2: Verify both UTM zones (43N, 44N) are detected across active tiles."""
        tiles = discover_himalayas_dem_tiles(DEFAULT_DEM_DIR)
        zones = set()
        for t in tiles:
            paths = get_tile_raster_paths(DEFAULT_DEM_DIR, t)
            with rasterio.open(paths["elevation"]) as src:
                zones.add(str(src.crs))

        expected_zones = {"EPSG:32643", "EPSG:32644"}
        self.assertEqual(zones, expected_zones)

    def test_03_aspect_derivative_transformations(self):
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

    def test_04_grid_generation_and_output_dimensions(self):
        """Test 4: Verify generated grid output dimensions and unique IDs."""
        self.assertEqual(self.report["status"], "SUCCESS")
        self.assertTrue(self.test_csv_path.exists())
        self.assertTrue(self.test_report_path.exists())

        self.assertEqual(len(self.df), 12052)
        self.assertEqual(self.report["total_valid_grid_locations"], 12052)

        # Check all 29 tiles represented
        self.assertEqual(self.df["tile"].nunique(), 29)

        # Check unique grid IDs
        self.assertEqual(self.df["grid_id"].nunique(), len(self.df))
        self.assertTrue(self.df["grid_id"].str.startswith(GRID_PREFIX).all())

    def test_05_model_contract_feature_compliance(self):
        """Test 5: Verify all 5 terrain features conform strictly to model contract."""
        for col in MANDATORY_TERRAIN_FEATURES:
            self.assertIn(col, self.df.columns)
            self.assertTrue(np.isfinite(self.df[col]).all())
            self.assertFalse((self.df[col] == -9999.0).any())

        # Slope range [0, 90]
        self.assertTrue((self.df["slope_degrees"] >= 0.0).all())
        self.assertTrue((self.df["slope_degrees"] <= 90.0).all())

        # Aspect sin/cos range [-1, 1]
        self.assertTrue((self.df["aspect_sin"] >= -1.0001).all())
        self.assertTrue((self.df["aspect_sin"] <= 1.0001).all())
        self.assertTrue((self.df["aspect_cos"] >= -1.0001).all())
        self.assertTrue((self.df["aspect_cos"] <= 1.0001).all())

        # Aspect flat binary {0, 1}
        self.assertTrue(set(self.df["aspect_flat"].unique()).issubset({0, 1}))

        # Pythagorean identity for non-flat aspect
        non_flat = self.df[self.df["aspect_flat"] == 0]
        sum_sq = non_flat["aspect_sin"] ** 2 + non_flat["aspect_cos"] ** 2
        self.assertTrue(np.allclose(sum_sq, 1.0, atol=1e-4))

        # Flat terrain check
        flat = self.df[self.df["aspect_flat"] == 1]
        if len(flat) > 0:
            self.assertTrue((flat["aspect_sin"] == 0.0).all())
            self.assertTrue((flat["aspect_cos"] == 0.0).all())

    def test_06_geographic_bounds_compliance(self):
        """Test 6: Verify all grid coordinates fall within Himalayas extent."""
        self.assertTrue(
            self.df["latitude"].between(HIMALAYAS_LAT_BOUNDS[0], HIMALAYAS_LAT_BOUNDS[1]).all()
        )
        self.assertTrue(
            self.df["longitude"].between(HIMALAYAS_LON_BOUNDS[0], HIMALAYAS_LON_BOUNDS[1]).all()
        )

    def test_07_no_duplicate_spatial_coordinates(self):
        """Test 7: Enforce zero duplicate (latitude, longitude) coordinates across entire grid."""
        dup_count = self.df.duplicated(subset=["latitude", "longitude"]).sum()
        self.assertEqual(dup_count, 0, f"Found {dup_count} duplicate coordinate pairs")

    def test_08_trained_model_inference_compatibility(self):
        """Test 8: Verify trained Himalayas Random Forest model consumes generated grid seamlessly."""
        self.assertTrue(HIMALAYAS_MODEL_PATH.exists(), f"Model not found: {HIMALAYAS_MODEL_PATH}")
        model = joblib.load(HIMALAYAS_MODEL_PATH)

        # Take a representative sample across multiple tiles
        sample_df = self.df.groupby("tile").head(5)[MANDATORY_TERRAIN_FEATURES]
        self.assertEqual(len(sample_df), 29 * 5)

        # Run inference
        probs = model.predict_proba(sample_df)[:, 1]
        self.assertEqual(len(probs), len(sample_df))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_09_validation_report_completeness(self):
        """Test 9: Verify validation report structure, stats, and metadata schema."""
        with open(self.test_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["region"], "Himalayas")
        self.assertEqual(report["grid_spacing_km"], 5.0)
        self.assertEqual(report["dem_tile_count"], 29)
        self.assertEqual(report["total_valid_grid_locations"], 12052)
        self.assertIn("excluded_samples_breakdown", report)
        self.assertIn("terrain_feature_statistics", report)
        self.assertIn("tile_point_distribution", report)
        self.assertTrue(report["model_contract_verification"]["all_features_present"])
        self.assertTrue(report["model_contract_verification"]["all_values_finite"])

    def test_10_overwrite_protection(self):
        """Test 10: Verify FileExistsError when outputs exist without --overwrite."""
        temp_dir_ow = Path(tempfile.mkdtemp(prefix="test_ow_"))
        csv_ow = temp_dir_ow / "existing.csv"
        rep_ow = temp_dir_ow / "existing.json"
        try:
            csv_ow.write_text("dummy,csv", encoding="utf-8")
            rep_ow.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                generate_himalayas_spatial_grid(
                    dem_dir=DEFAULT_DEM_DIR,
                    output_csv=csv_ow,
                    validation_report_path=rep_ow,
                    spacing_m=5000.0,
                    overwrite=False,
                )
        finally:
            shutil.rmtree(temp_dir_ow, ignore_errors=True)

    def test_11_system_protection_boundaries(self):
        """Test 11: Verify that Western Ghats and Northeast India baseline files remain intact."""
        self.assertTrue(NORTHEAST_GRID_CSV.exists(), "Northeast India grid file must exist")
        self.assertTrue(NORTHEAST_RISK_CSV.exists(), "Northeast India risk file must exist")
        if WESTERN_GHATS_GRID_CSV.exists():
            df_wg = pd.read_csv(WESTERN_GHATS_GRID_CSV)
            self.assertGreater(len(df_wg), 1000)


if __name__ == "__main__":
    unittest.main()
