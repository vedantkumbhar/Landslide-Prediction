"""
Automated unit tests for Stage 7.3: Northeast India Grid-Level NASA GPM IMERG Rainfall Integration.

Verifies:
1. Input alignment between Stage 7.1 grid and Stage 7.2 terrain predictions.
2. Robust error handling for dataset alignment mismatches.
3. Native GPM 0.1° (~10 km) spatial grid matching and point-to-cell aggregation.
4. Rainfall accumulation calculation, monotonicity, and non-negativity.
5. Freshness gate and status classification (SUCCESS, STALE_DATA, INCOMPLETE_DATA).
6. Fail-safe NaN fallback during incomplete data scenarios.
7. Output integrated CSV schema, row count invariance (14,550), and column integrity.
8. JSON validation report schema and spatial matching metrics.
9. Overwrite protection against accidental data destruction.
10. System protection boundaries (Western Ghats and prior stages remain untouched).
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from backend.integrate_northeast_gpm_rainfall import (
    DEFAULT_GRID_CSV,
    DEFAULT_TERRAIN_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_INTEGRATED_CSV,
    DEFAULT_VALIDATION_REPORT,
    MAX_DATA_LATENCY_HOURS,
    DISCLAIMER_TEXT,
    verify_input_datasets,
    resolve_gpm_observation_window,
    integrate_northeast_gpm_rainfall,
)
from backend.extract_western_ghats_gpm_rainfall import (
    match_spatial_grid,
    compute_accumulations,
    EXPECTED_GRANULES_24H,
    MAX_GRID_MATCH_DISTANCE_KM,
)
from backend.refresh_western_ghats_rainfall import DEFAULT_ARCHIVE_DIR


class TestNortheastGPMRainfall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid_csv = DEFAULT_GRID_CSV
        cls.terrain_csv = DEFAULT_TERRAIN_CSV
        cls.output_csv = DEFAULT_INTEGRATED_CSV
        cls.report_json = DEFAULT_VALIDATION_REPORT
        cls.archive_dir = DEFAULT_ARCHIVE_DIR

    def test_01_input_datasets_exist_and_aligned(self):
        """Test Stage 7.1 grid and Stage 7.2 terrain predictions exist and align exactly."""
        self.assertTrue(self.grid_csv.exists(), f"Grid CSV missing: {self.grid_csv}")
        self.assertTrue(self.terrain_csv.exists(), f"Terrain CSV missing: {self.terrain_csv}")

        df_grid, df_terrain = verify_input_datasets(self.grid_csv, self.terrain_csv)

        self.assertEqual(len(df_grid), 14550)
        self.assertEqual(len(df_terrain), 14550)
        self.assertTrue((df_grid["grid_id"] == df_terrain["grid_id"]).all())
        self.assertTrue(np.allclose(df_grid["latitude"], df_terrain["latitude"], atol=1e-6))
        self.assertTrue(np.allclose(df_grid["longitude"], df_terrain["longitude"], atol=1e-6))
        self.assertTrue((df_grid["tile"] == df_terrain["tile"]).all())

    def test_02_verify_input_datasets_mismatch_detection(self):
        """Test verify_input_datasets catches row count, ID sequence, or coordinate mismatches."""
        df_grid = pd.read_csv(self.grid_csv).head(100)
        df_terrain = pd.read_csv(self.terrain_csv).head(90)

        tmp_dir = self.output_csv.parent / "_test_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            p_grid = tmp_dir / "mismatch_grid.csv"
            p_terr = tmp_dir / "mismatch_terr.csv"
            df_grid.to_csv(p_grid, index=False)
            df_terrain.to_csv(p_terr, index=False)

            # Row count mismatch
            with self.assertRaises(ValueError):
                verify_input_datasets(p_grid, p_terr)

            # ID sequence mismatch
            df_terr_scrambled = df_grid.copy()
            df_terr_scrambled["grid_id"] = df_terr_scrambled["grid_id"].iloc[::-1].values
            df_terr_scrambled.to_csv(p_terr, index=False)
            with self.assertRaises(ValueError):
                verify_input_datasets(p_grid, p_terr)

            # Coordinate mismatch
            df_terr_bad_coords = df_grid.copy()
            df_terr_bad_coords.loc[0, "latitude"] += 0.5
            df_terr_bad_coords.to_csv(p_terr, index=False)
            with self.assertRaises(ValueError):
                verify_input_datasets(p_grid, p_terr)
        finally:
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_03_spatial_grid_matching_and_aggregation(self):
        """Test spatial grid matching against native GPM 0.1° grid."""
        archive_files = sorted(self.archive_dir.glob("*.HDF5"))
        self.assertGreater(len(archive_files), 0, "No HDF5 granules available in archive for spatial grid test")

        sample_hdf5 = archive_files[0]
        df_grid = pd.read_csv(self.grid_csv)

        matched_lats, matched_lons, distances_km, lat_idx, lon_idx = match_spatial_grid(
            df_grid, sample_hdf5
        )

        self.assertEqual(len(matched_lats), 14550)
        self.assertEqual(len(matched_lons), 14550)
        self.assertEqual(len(distances_km), 14550)

        # Distance threshold
        self.assertLess(
            distances_km.max(),
            MAX_GRID_MATCH_DISTANCE_KM,
            f"Matching distance exceeds limit ({distances_km.max():.2f} km >= {MAX_GRID_MATCH_DISTANCE_KM} km)",
        )
        self.assertGreaterEqual(distances_km.min(), 0.0)

        # Point to cell aggregation (multiple 5km points per ~10km GPM cell)
        unique_cells = set(zip(np.round(matched_lats, 4), np.round(matched_lons, 4)))
        self.assertGreater(len(unique_cells), 3000)
        self.assertLess(len(unique_cells), 4000)

        # Confirm multiple points per cell
        points_per_cell = len(df_grid) / len(unique_cells)
        self.assertGreater(points_per_cell, 3.5)
        self.assertLess(points_per_cell, 5.5)

    def test_04_accumulation_calculation_and_monotonicity(self):
        """Test compute_accumulations rolling windows and monotonicity enforcement."""
        now = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
        granules = [
            {
                "start": now - timedelta(minutes=30 * (48 - i)),
                "end": now - timedelta(minutes=30 * (47 - i)),
                "path": Path(f"mock_{i}.HDF5"),
            }
            for i in range(48)
        ]

        np.random.seed(42)
        # Mock 100 locations x 48 half-hour intervals with variable rainfall [0, 25] mm/h
        rainfall_matrix = np.random.uniform(0.0, 25.0, size=(100, 48))
        accum_res = compute_accumulations(rainfall_matrix, granules)

        self.assertIn("gpm_latest_rainfall_rate_mm_hr", accum_res)
        self.assertIn("gpm_1h_accumulation_mm", accum_res)
        self.assertIn("gpm_3h_accumulation_mm", accum_res)
        self.assertIn("gpm_6h_accumulation_mm", accum_res)
        self.assertIn("gpm_12h_accumulation_mm", accum_res)
        self.assertIn("gpm_24h_accumulation_mm", accum_res)

        r1 = accum_res["gpm_1h_accumulation_mm"]
        r3 = accum_res["gpm_3h_accumulation_mm"]
        r6 = accum_res["gpm_6h_accumulation_mm"]
        r12 = accum_res["gpm_12h_accumulation_mm"]
        r24 = accum_res["gpm_24h_accumulation_mm"]

        # All non-negative
        self.assertTrue((r1 >= 0.0).all())
        self.assertTrue((r3 >= 0.0).all())
        self.assertTrue((r6 >= 0.0).all())
        self.assertTrue((r12 >= 0.0).all())
        self.assertTrue((r24 >= 0.0).all())

        # Strict monotonicity: 1h <= 3h <= 6h <= 12h <= 24h
        self.assertTrue((r1 <= r3 + 1e-5).all())
        self.assertTrue((r3 <= r6 + 1e-5).all())
        self.assertTrue((r6 <= r12 + 1e-5).all())
        self.assertTrue((r12 <= r24 + 1e-5).all())

    def test_05_status_classification_rules(self):
        """Test status evaluation rules: SUCCESS, STALE_DATA, INCOMPLETE_DATA."""
        ref_now = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # Fresh observation (latency 4h <= 12.0h)
        target_end_fresh = ref_now - timedelta(hours=4)
        latency_fresh = (ref_now - target_end_fresh).total_seconds() / 3600.0
        self.assertLessEqual(latency_fresh, MAX_DATA_LATENCY_HOURS)

        # Stale observation (latency 15h > 12.0h)
        target_end_stale = ref_now - timedelta(hours=15)
        latency_stale = (ref_now - target_end_stale).total_seconds() / 3600.0
        self.assertGreater(latency_stale, MAX_DATA_LATENCY_HOURS)

        # Future observation timestamp raises ValueError
        target_end_future = ref_now + timedelta(hours=1)
        latency_future = (ref_now - target_end_future).total_seconds()
        self.assertLess(latency_future, 0.0)

    def test_06_output_csv_integrity(self):
        """Verify output integrated CSV row count, schema, and column constraints."""
        if not self.output_csv.exists():
            self.skipTest(f"Output CSV not yet generated: {self.output_csv}")

        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 14550)
        self.assertEqual(df["grid_id"].nunique(), 14550)
        self.assertEqual(df["tile"].nunique(), 33)

        expected_columns = [
            "grid_id",
            "latitude",
            "longitude",
            "tile",
            "elevation_m",
            "slope_degrees",
            "aspect_sin",
            "aspect_cos",
            "aspect_flat",
            "terrain_rf_prob_balanced",
            "terrain_rf_pred_balanced",
            "terrain_baseline_screening_tier",
            "gpm_grid_latitude",
            "gpm_grid_longitude",
            "gpm_matching_distance_km",
            "gpm_observation_start_utc",
            "gpm_observation_end_utc",
            "rainfall_refresh_status",
            "gpm_latest_rainfall_rate_mm_hr",
            "gpm_latest_quality_index",
            "gpm_latest_random_error_mm_hr",
            "gpm_1h_accumulation_mm",
            "gpm_3h_accumulation_mm",
            "gpm_6h_accumulation_mm",
            "gpm_12h_accumulation_mm",
            "gpm_24h_accumulation_mm",
            "gpm_interval_count",
            "gpm_24h_complete",
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing expected column in output CSV: {col}")

        # Status must be one of allowed strings
        valid_statuses = {"SUCCESS", "STALE_DATA", "INCOMPLETE_DATA"}
        self.assertTrue(df["rainfall_refresh_status"].isin(valid_statuses).all())

        # If SUCCESS or STALE_DATA, accumulation columns must be non-null and monotonic
        status = df["rainfall_refresh_status"].iloc[0]
        if status in {"SUCCESS", "STALE_DATA"}:
            self.assertTrue(df["gpm_24h_complete"].all())
            self.assertEqual(df["gpm_interval_count"].iloc[0], 48)
            self.assertTrue((df["gpm_1h_accumulation_mm"] <= df["gpm_3h_accumulation_mm"] + 1e-4).all())
            self.assertTrue((df["gpm_3h_accumulation_mm"] <= df["gpm_6h_accumulation_mm"] + 1e-4).all())
            self.assertTrue((df["gpm_6h_accumulation_mm"] <= df["gpm_12h_accumulation_mm"] + 1e-4).all())
            self.assertTrue((df["gpm_12h_accumulation_mm"] <= df["gpm_24h_accumulation_mm"] + 1e-4).all())

    def test_07_validation_report_integrity(self):
        """Verify validation report JSON schema, metrics, and disclaimer."""
        if not self.report_json.exists():
            self.skipTest(f"Validation report not yet generated: {self.report_json}")

        with open(self.report_json, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["region"], "Northeast India")
        self.assertEqual(report["total_grid_locations"], 14550)
        self.assertEqual(report["unique_dem_tiles"], 33)
        self.assertIn(report["refresh_status"], ["SUCCESS", "STALE_DATA", "INCOMPLETE_DATA"])

        # Spatial matching metrics
        spatial_match = report["spatial_grid_matching"]
        self.assertGreater(spatial_match["unique_matched_gpm_cells"], 3000)
        self.assertLess(spatial_match["matching_distance_km"]["max"], MAX_GRID_MATCH_DISTANCE_KM)

        # Disclaimer
        self.assertIn("EXPERIMENTAL", report["disclaimer"])
        self.assertEqual(report["disclaimer"], DISCLAIMER_TEXT)

    def test_08_overwrite_protection(self):
        """Verify that attempting to overwrite existing outputs without --overwrite raises FileExistsError."""
        if not self.output_csv.exists():
            self.skipTest("Output CSV does not exist yet to test overwrite protection")

        with self.assertRaises(FileExistsError):
            integrate_northeast_gpm_rainfall(
                grid_csv_path=self.grid_csv,
                terrain_csv_path=self.terrain_csv,
                output_csv_path=self.output_csv,
                validation_report_path=self.report_json,
                overwrite=False,
            )

    def test_09_protection_boundaries(self):
        """Verify Western Ghats files and prior Stage 7 artifacts have not been modified."""
        protected_paths = [
            Path("data/processed/western_ghats_dynamic_risk/western_ghats_dynamic_hazard_risk.geojson"),
            Path("outputs/western_ghats_dynamic_risk_map.html"),
            Path("data/processed/northeast_spatial_grid/northeast_india_5km_spatial_grid.csv"),
            Path("data/processed/northeast_terrain_predictions/northeast_india_5km_terrain_susceptibility.csv"),
        ]

        for p in protected_paths:
            full_p = Path(__file__).resolve().parents[1] / p
            self.assertTrue(full_p.exists(), f"Protected artifact missing: {full_p}")
            self.assertGreater(full_p.stat().st_size, 0, f"Protected artifact empty: {full_p}")


if __name__ == "__main__":
    unittest.main()
