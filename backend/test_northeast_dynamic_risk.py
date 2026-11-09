"""
Automated test suite for Stage 7.4: Northeast India Dynamic Hazard Risk Formulation and Calibration.

Verifies:
1. Input dataset alignment and integrity (14,550 locations, 33 DEM tiles).
2. Formula correctness: H_risk = Terrain_Contribution + Rainfall_Contribution.
3. Multi-window hydrometeorological scaling, clipping, and monotonicity.
4. Zero-rainfall rule (matches dry terrain baseline identically).
5. Missing, incomplete, and non-finite data handling (strict NaN fallback and unassessed status).
6. Category assignments (relative empirical quantiles and fixed benchmark tiers).
7. Freshness gate evaluation and latency limits.
8. Output CSV row count invariance, uniqueness of grid_id, and provenance retention.
9. Metadata and validation report schema, metrics, and experimental disclaimer.
10. Overwrite protection against accidental data destruction.
11. Protection boundaries (Western Ghats and prior stages remain untouched).
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from backend.predict_northeast_dynamic_risk import (
    DEFAULT_INTEGRATED_CSV,
    DEFAULT_GPM_REPORT_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_METADATA_JSON,
    DEFAULT_VALIDATION_REPORT_JSON,
    EXPECTED_LOCATIONS_COUNT,
    EXPECTED_TILES_COUNT,
    DEFAULT_WINDOW_SCALES_MM,
    DEFAULT_WINDOW_WEIGHTS,
    DEFAULT_WEIGHT_TERRAIN,
    DEFAULT_WEIGHT_RAINFALL,
    MAX_DATA_LATENCY_HOURS,
    DISCLAIMER_TEXT,
    TERRAIN_MODEL_PREDICTORS,
    verify_northeast_rainfall_freshness_gate,
    NortheastDynamicRiskEngine,
    run_northeast_dynamic_risk_pipeline,
)


class TestNortheastDynamicRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.integrated_csv = DEFAULT_INTEGRATED_CSV
        cls.gpm_report_json = DEFAULT_GPM_REPORT_JSON
        cls.output_csv = DEFAULT_OUTPUT_CSV
        cls.metadata_json = DEFAULT_METADATA_JSON
        cls.report_json = DEFAULT_VALIDATION_REPORT_JSON

    def test_01_input_dataset_integrity(self):
        """Verify input rainfall-terrain integrated dataset has 14,550 rows and expected schema."""
        self.assertTrue(self.integrated_csv.exists(), f"Input CSV missing: {self.integrated_csv}")
        df = pd.read_csv(self.integrated_csv)

        self.assertEqual(len(df), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["grid_id"].nunique(), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["tile"].nunique(), EXPECTED_TILES_COUNT)

        required_cols = [
            "grid_id",
            "latitude",
            "longitude",
            "tile",
            *TERRAIN_MODEL_PREDICTORS,
            "terrain_rf_prob_balanced",
            "gpm_1h_accumulation_mm",
            "gpm_3h_accumulation_mm",
            "gpm_6h_accumulation_mm",
            "gpm_12h_accumulation_mm",
            "gpm_24h_accumulation_mm",
            "gpm_24h_complete",
            "rainfall_refresh_status",
            "gpm_observation_end_utc",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required input column: {col}")

    def test_02_formula_correctness_and_contributions(self):
        """Verify H_risk = Terrain_Contribution + Rainfall_Contribution = 100 * (w_t * S_t + w_r * R_r)."""
        engine = NortheastDynamicRiskEngine(
            weight_terrain=0.50,
            weight_rainfall=0.50,
        )

        # Create synthetic test DataFrame
        df_synth = pd.DataFrame(
            {
                "grid_id": ["TEST_001", "TEST_002", "TEST_003"],
                "terrain_rf_prob_balanced": [0.40, 0.80, 0.20],
                "gpm_1h_accumulation_mm": [2.5, 12.5, 0.0],
                "gpm_3h_accumulation_mm": [5.0, 25.0, 0.0],
                "gpm_6h_accumulation_mm": [7.5, 37.5, 0.0],
                "gpm_12h_accumulation_mm": [10.0, 50.0, 0.0],
                "gpm_24h_accumulation_mm": [15.0, 75.0, 0.0],
                "gpm_24h_complete": [True, True, True],
                "rainfall_refresh_status": ["SUCCESS", "SUCCESS", "SUCCESS"],
            }
        )

        res_df, report = engine.compute_combined_hazard_risk(df_synth)

        # In TEST_001: each window is exactly 10% of scale (2.5/25, 5/50, 7.5/75, 10/100, 15/150 = 0.10)
        # So R_rain = 0.10
        # Terrain contrib = 100 * 0.5 * 0.40 = 20.0
        # Rain contrib = 100 * 0.5 * 0.10 = 5.0
        # Dynamic risk = 25.0
        self.assertAlmostEqual(res_df.loc[0, "terrain_contribution"], 20.0, places=3)
        self.assertAlmostEqual(res_df.loc[0, "rainfall_contribution"], 5.0, places=3)
        self.assertAlmostEqual(res_df.loc[0, "dynamic_risk_index"], 25.0, places=3)

        # In TEST_002: each window is exactly 50% of scale (0.50)
        # Terrain contrib = 100 * 0.5 * 0.80 = 40.0
        # Rain contrib = 100 * 0.5 * 0.50 = 25.0
        # Dynamic risk = 65.0
        self.assertAlmostEqual(res_df.loc[1, "terrain_contribution"], 40.0, places=3)
        self.assertAlmostEqual(res_df.loc[1, "rainfall_contribution"], 25.0, places=3)
        self.assertAlmostEqual(res_df.loc[1, "dynamic_risk_index"], 65.0, places=3)

        # Verify additivity: dynamic_risk == terrain_contribution + rainfall_contribution
        for i in range(len(res_df)):
            self.assertAlmostEqual(
                res_df.loc[i, "dynamic_risk_index"],
                res_df.loc[i, "terrain_contribution"] + res_df.loc[i, "rainfall_contribution"],
                places=3,
            )

    def test_03_rainfall_normalization_and_monotonicity(self):
        """Verify that increasing rainfall in any window monotonically increases rainfall intensity."""
        engine = NortheastDynamicRiskEngine()

        base_row = {
            "gpm_1h_accumulation_mm": 5.0,
            "gpm_3h_accumulation_mm": 10.0,
            "gpm_6h_accumulation_mm": 15.0,
            "gpm_12h_accumulation_mm": 20.0,
            "gpm_24h_accumulation_mm": 30.0,
            "gpm_24h_complete": True,
            "rainfall_refresh_status": "SUCCESS",
            "terrain_rf_prob_balanced": 0.5,
        }

        df1 = pd.DataFrame([base_row])
        res1, _ = engine.compute_normalized_rainfall(df1)
        intensity_base = res1.loc[0, "rainfall_intensity_index"]

        # Increase 24h accumulation
        row_inc_24 = base_row.copy()
        row_inc_24["gpm_24h_accumulation_mm"] = 60.0
        df2 = pd.DataFrame([row_inc_24])
        res2, _ = engine.compute_normalized_rainfall(df2)
        intensity_inc_24 = res2.loc[0, "rainfall_intensity_index"]
        self.assertGreater(intensity_inc_24, intensity_base)

        # Increase 1h accumulation
        row_inc_1h = base_row.copy()
        row_inc_1h["gpm_1h_accumulation_mm"] = 15.0
        df3 = pd.DataFrame([row_inc_1h])
        res3, _ = engine.compute_normalized_rainfall(df3)
        intensity_inc_1h = res3.loc[0, "rainfall_intensity_index"]
        self.assertGreater(intensity_inc_1h, intensity_base)

        # Test clipping at scale (values exceeding scale saturate at 1.0)
        row_sat = {
            "gpm_1h_accumulation_mm": 100.0,  # scale 25
            "gpm_3h_accumulation_mm": 200.0,  # scale 50
            "gpm_6h_accumulation_mm": 300.0,  # scale 75
            "gpm_12h_accumulation_mm": 400.0, # scale 100
            "gpm_24h_accumulation_mm": 500.0, # scale 150
            "gpm_24h_complete": True,
            "rainfall_refresh_status": "SUCCESS",
            "terrain_rf_prob_balanced": 0.5,
        }
        df_sat = pd.DataFrame([row_sat])
        res_sat, _ = engine.compute_normalized_rainfall(df_sat)
        self.assertAlmostEqual(res_sat.loc[0, "rainfall_intensity_index"], 1.0, places=4)
        for w in ["1h", "3h", "6h", "12h", "24h"]:
            self.assertAlmostEqual(res_sat.loc[0, f"norm_rain_{w}"], 1.0, places=4)

    def test_04_zero_rainfall_handling(self):
        """Verify that zero rainfall yields R_rainfall=0.0 and H_risk exactly matches dry baseline."""
        engine = NortheastDynamicRiskEngine()

        zero_row = {
            "terrain_rf_prob_balanced": 0.65,
            "gpm_1h_accumulation_mm": 0.0,
            "gpm_3h_accumulation_mm": 0.0,
            "gpm_6h_accumulation_mm": 0.0,
            "gpm_12h_accumulation_mm": 0.0,
            "gpm_24h_accumulation_mm": 0.0,
            "gpm_24h_complete": True,
            "rainfall_refresh_status": "SUCCESS",
        }
        df_zero = pd.DataFrame([zero_row])
        res_zero, _ = engine.compute_combined_hazard_risk(df_zero)

        self.assertAlmostEqual(res_zero.loc[0, "rainfall_intensity_index"], 0.0, places=5)
        self.assertAlmostEqual(res_zero.loc[0, "rainfall_contribution"], 0.0, places=5)
        expected_dry_score = 100.0 * 0.50 * 0.65  # 32.5
        self.assertAlmostEqual(res_zero.loc[0, "terrain_contribution"], expected_dry_score, places=3)
        self.assertAlmostEqual(res_zero.loc[0, "dynamic_risk_index"], expected_dry_score, places=3)
        self.assertEqual(res_zero.loc[0, "risk_calculation_status"], "VALIDATED_SCREENING_CALCULATED")

    def test_05_missing_and_invalid_data_handling(self):
        """Verify incomplete, NaN, or non-finite rainfall evaluates to NaN and unassessed status."""
        engine = NortheastDynamicRiskEngine()

        rows = [
            # Incomplete 24h interval
            {
                "terrain_rf_prob_balanced": 0.50,
                "gpm_1h_accumulation_mm": 5.0,
                "gpm_3h_accumulation_mm": 10.0,
                "gpm_6h_accumulation_mm": 15.0,
                "gpm_12h_accumulation_mm": 20.0,
                "gpm_24h_accumulation_mm": 25.0,
                "gpm_24h_complete": False,  # INCOMPLETE
                "rainfall_refresh_status": "SUCCESS",
            },
            # NaN in accumulation
            {
                "terrain_rf_prob_balanced": 0.50,
                "gpm_1h_accumulation_mm": np.nan,
                "gpm_3h_accumulation_mm": 10.0,
                "gpm_6h_accumulation_mm": 15.0,
                "gpm_12h_accumulation_mm": 20.0,
                "gpm_24h_accumulation_mm": 25.0,
                "gpm_24h_complete": True,
                "rainfall_refresh_status": "SUCCESS",
            },
            # Negative accumulation
            {
                "terrain_rf_prob_balanced": 0.50,
                "gpm_1h_accumulation_mm": -1.0,
                "gpm_3h_accumulation_mm": 10.0,
                "gpm_6h_accumulation_mm": 15.0,
                "gpm_12h_accumulation_mm": 20.0,
                "gpm_24h_accumulation_mm": 25.0,
                "gpm_24h_complete": True,
                "rainfall_refresh_status": "SUCCESS",
            },
            # Infinite accumulation (+inf)
            {
                "terrain_rf_prob_balanced": 0.50,
                "gpm_1h_accumulation_mm": np.inf,
                "gpm_3h_accumulation_mm": 10.0,
                "gpm_6h_accumulation_mm": 15.0,
                "gpm_12h_accumulation_mm": 20.0,
                "gpm_24h_accumulation_mm": 25.0,
                "gpm_24h_complete": True,
                "rainfall_refresh_status": "SUCCESS",
            },
        ]

        df_inv = pd.DataFrame(rows)
        res_inv, _ = engine.compute_combined_hazard_risk(df_inv)

        for i in range(len(rows)):
            self.assertTrue(np.isnan(res_inv.loc[i, "rainfall_intensity_index"]))
            self.assertTrue(np.isnan(res_inv.loc[i, "rainfall_contribution"]))
            self.assertTrue(np.isnan(res_inv.loc[i, "dynamic_risk_index"]))
            self.assertTrue(np.isnan(res_inv.loc[i, "dynamic_risk_interactive"]))
            self.assertEqual(res_inv.loc[i, "primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")
            self.assertEqual(res_inv.loc[i, "risk_calculation_status"], "INCOMPLETE_RAINFALL_WINDOW")

    def test_06_relative_and_fixed_category_assignment(self):
        """Verify empirical percentile thresholds partition distribution and fixed tiers behave correctly."""
        engine = NortheastDynamicRiskEngine(
            relative_percentiles={"low_moderate": 50.0, "moderate_high": 80.0, "high_very_high": 95.0}
        )

        n = 1000
        np.random.seed(42)
        df_test = pd.DataFrame(
            {
                "terrain_rf_prob_balanced": np.linspace(0.05, 0.95, n),
                "gpm_1h_accumulation_mm": np.random.uniform(0, 10, n),
                "gpm_3h_accumulation_mm": np.random.uniform(0, 20, n),
                "gpm_6h_accumulation_mm": np.random.uniform(0, 30, n),
                "gpm_12h_accumulation_mm": np.random.uniform(0, 40, n),
                "gpm_24h_accumulation_mm": np.random.uniform(0, 60, n),
                "gpm_24h_complete": [True] * n,
                "rainfall_refresh_status": ["SUCCESS"] * n,
            }
        )

        res, report = engine.compute_combined_hazard_risk(df_test)
        cuts = report["relative_category_thresholds"]

        self.assertIn("P50_low_to_moderate", cuts)
        self.assertIn("P80_moderate_to_high", cuts)
        self.assertIn("P95_high_to_very_high", cuts)
        self.assertLess(cuts["P50_low_to_moderate"], cuts["P80_moderate_to_high"])
        self.assertLess(cuts["P80_moderate_to_high"], cuts["P95_high_to_very_high"])

        cat_counts = res["risk_category_relative"].value_counts()
        self.assertAlmostEqual(cat_counts["Low"] / n, 0.50, delta=0.03)
        self.assertAlmostEqual(cat_counts["Moderate"] / n, 0.30, delta=0.03)
        self.assertAlmostEqual(cat_counts["High"] / n, 0.15, delta=0.03)
        self.assertAlmostEqual(cat_counts["Very High"] / n, 0.05, delta=0.03)

    def test_07_freshness_gate(self):
        """Verify freshness gate accepts fresh data and rejects stale or future timestamps."""
        ref_now = datetime(2026, 11, 9, 15, 0, 0, tzinfo=timezone.utc)

        # Fresh data (latency 5h <= 12h)
        df_fresh = pd.DataFrame(
            {
                "rainfall_refresh_status": ["SUCCESS"],
                "gpm_24h_complete": [True],
                "gpm_observation_end_utc": ["2026-11-09T10:00:00Z"],
            }
        )
        is_fresh, msg, _ = verify_northeast_rainfall_freshness_gate(df_fresh, now_utc=ref_now)
        self.assertTrue(is_fresh, f"Fresh data rejected: {msg}")

        # Stale data (latency 16h > 12h)
        df_stale = pd.DataFrame(
            {
                "rainfall_refresh_status": ["SUCCESS"],
                "gpm_24h_complete": [True],
                "gpm_observation_end_utc": ["2026-11-08T23:00:00Z"],
            }
        )
        is_fresh, msg, _ = verify_northeast_rainfall_freshness_gate(df_stale, now_utc=ref_now)
        self.assertFalse(is_fresh, "Stale data passed gate unexpectedly")

        # Future timestamp
        df_future = pd.DataFrame(
            {
                "rainfall_refresh_status": ["SUCCESS"],
                "gpm_24h_complete": [True],
                "gpm_observation_end_utc": ["2026-11-09T16:00:00Z"],
            }
        )
        is_fresh, msg, _ = verify_northeast_rainfall_freshness_gate(df_future, now_utc=ref_now)
        self.assertFalse(is_fresh, "Future timestamp passed gate unexpectedly")

    def test_08_output_csv_schema_and_provenance(self):
        """Verify generated dynamic risk CSV schema, row count invariance, and provenance retention."""
        if not self.output_csv.exists():
            self.skipTest(f"Output CSV not yet generated: {self.output_csv}")

        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["grid_id"].nunique(), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["tile"].nunique(), EXPECTED_TILES_COUNT)

        expected_columns = [
            "grid_id",
            "latitude",
            "longitude",
            "tile",
            *TERRAIN_MODEL_PREDICTORS,
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
            "norm_rain_1h",
            "norm_rain_3h",
            "norm_rain_6h",
            "norm_rain_12h",
            "norm_rain_24h",
            "rainfall_intensity_index",
            "rainfall_percentile_24h",
            "terrain_contribution",
            "rainfall_contribution",
            "dynamic_risk_index",
            "dynamic_risk_interactive",
            "risk_tier_fixed",
            "risk_category_relative",
            "primary_screening_category",
            "risk_calculation_status",
            "risk_calculation_timestamp_utc",
            "screening_disclaimer",
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing expected column in dynamic risk CSV: {col}")

        # Check disclaimer
        self.assertTrue((df["screening_disclaimer"] == DISCLAIMER_TEXT).all())

        # Check finite ranges
        valid_risk = df["dynamic_risk_index"].dropna()
        self.assertTrue((valid_risk >= 0.0).all() and (valid_risk <= 100.0).all())

        valid_contrib_t = df["terrain_contribution"].dropna()
        self.assertTrue((valid_contrib_t >= 0.0).all() and (valid_contrib_t <= 50.0).all())

        valid_contrib_r = df["rainfall_contribution"].dropna()
        self.assertTrue((valid_contrib_r >= 0.0).all() and (valid_contrib_r <= 50.0).all())

    def test_09_metadata_and_validation_report_schema(self):
        """Verify metadata JSON and validation report JSON have required keys and metrics."""
        if not self.metadata_json.exists():
            self.skipTest(f"Metadata JSON not yet generated: {self.metadata_json}")
        if not self.report_json.exists():
            self.skipTest(f"Validation report JSON not yet generated: {self.report_json}")

        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["coverage_region"], "Northeast India")
        self.assertEqual(meta["total_locations"], EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(meta["dem_tiles_count"], EXPECTED_TILES_COUNT)
        self.assertIn("gpm_observation_metadata", meta)
        self.assertIn("risk_formula_specification", meta)
        self.assertIn("summary_statistics", meta)
        self.assertEqual(meta["disclaimer"], DISCLAIMER_TEXT)

        with open(self.report_json, "r", encoding="utf-8") as f:
            rep = json.load(f)

        self.assertEqual(rep["status"], "PASSED")
        self.assertTrue(rep["checks"]["record_count_retained_14550"])
        self.assertTrue(rep["checks"]["all_33_tiles_present"])
        self.assertTrue(rep["checks"]["formula_components_sum_to_risk_index"])
        self.assertTrue(rep["checks"]["disclaimer_stamped_on_all_rows"])

    def test_10_overwrite_protection(self):
        """Verify attempting to overwrite without --overwrite raises FileExistsError."""
        if not self.output_csv.exists():
            self.skipTest("Output CSV does not exist yet to test overwrite protection")

        with self.assertRaises(FileExistsError):
            run_northeast_dynamic_risk_pipeline(
                integrated_csv_path=self.integrated_csv,
                gpm_report_path=self.gpm_report_json,
                output_csv_path=self.output_csv,
                metadata_json_path=self.metadata_json,
                validation_report_path=self.report_json,
                overwrite=False,
            )

    def test_11_protection_boundaries(self):
        """Verify Western Ghats files and prior Stage 7 files remain intact."""
        protected = [
            Path("data/processed/western_ghats_dynamic_risk/western_ghats_dynamic_hazard_risk.csv"),
            Path("data/processed/western_ghats_dynamic_risk/western_ghats_dynamic_hazard_risk.geojson"),
            Path("outputs/western_ghats_dynamic_risk_map.html"),
            Path("data/processed/northeast_spatial_grid/northeast_india_5km_spatial_grid.csv"),
            Path("data/processed/northeast_terrain_predictions/northeast_india_5km_terrain_susceptibility.csv"),
            Path("data/processed/northeast_gpm_rainfall/northeast_india_5km_rainfall_terrain_integrated.csv"),
        ]

        for p in protected:
            full_path = Path(__file__).resolve().parents[1] / p
            self.assertTrue(full_path.exists(), f"Protected file missing: {full_path}")
            self.assertGreater(full_path.stat().st_size, 0, f"Protected file empty: {full_path}")


if __name__ == "__main__":
    unittest.main()
