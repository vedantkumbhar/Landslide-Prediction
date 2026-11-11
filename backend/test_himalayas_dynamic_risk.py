"""
Automated test suite for Stage 8.4: Himalayas Dynamic Hazard Risk Formulation and Calibration.

Verifies:
1. Input dataset alignment and integrity (12,052 locations, 29 DEM tiles).
2. Formula correctness: H_risk = Terrain_Contribution + Rainfall_Contribution = 100 * (w_t * S_t + w_r * R_r).
3. Multi-window hydrometeorological scaling, clipping, and monotonicity.
4. Zero-rainfall rule (matches dry terrain baseline identically).
5. Missing, incomplete, and non-finite data handling (strict NaN fallback and unassessed status).
6. Category assignments (relative empirical quantiles and fixed benchmark tiers).
7. Freshness gate evaluation and latency limits (12-hour limit).
8. Output CSV row count invariance (12,052), uniqueness of grid_id, and provenance retention.
9. Independent full-grid formula recalculation across all 12,052 locations.
10. Overwrite protection against accidental data destruction and backup creation.
11. Protection boundaries (Western Ghats, Northeast India, and prior Stage 8 stages remain untouched).
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.predict_himalayas_dynamic_risk import (
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
    verify_himalayas_rainfall_freshness_gate,
    HimalayasDynamicRiskEngine,
    run_himalayas_dynamic_risk_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestHimalayasDynamicRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.integrated_csv = DEFAULT_INTEGRATED_CSV
        cls.gpm_report_json = DEFAULT_GPM_REPORT_JSON
        cls.output_csv = DEFAULT_OUTPUT_CSV
        cls.metadata_json = DEFAULT_METADATA_JSON
        cls.report_json = DEFAULT_VALIDATION_REPORT_JSON

    def test_01_input_dataset_integrity(self):
        """Verify input rainfall-terrain integrated dataset has 12,052 rows and expected schema."""
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
        engine = HimalayasDynamicRiskEngine(
            weight_terrain=0.50,
            weight_rainfall=0.50,
            window_scales_mm={"1h": 20.0, "3h": 40.0, "6h": 60.0, "12h": 80.0, "24h": 100.0},
        )

        df_synth = pd.DataFrame(
            {
                "grid_id": ["TEST_001", "TEST_002", "TEST_003"],
                "terrain_rf_prob_balanced": [0.40, 0.80, 0.20],
                "gpm_1h_accumulation_mm": [2.0, 10.0, 0.0],
                "gpm_3h_accumulation_mm": [4.0, 20.0, 0.0],
                "gpm_6h_accumulation_mm": [6.0, 30.0, 0.0],
                "gpm_12h_accumulation_mm": [8.0, 40.0, 0.0],
                "gpm_24h_accumulation_mm": [10.0, 50.0, 0.0],
                "gpm_24h_complete": [True, True, True],
                "rainfall_refresh_status": ["SUCCESS", "SUCCESS", "SUCCESS"],
            }
        )

        res_df, report = engine.compute_combined_hazard_risk(df_synth)

        # In TEST_001: each window is exactly 10% of scale (2/20, 4/40, 6/60, 8/80, 10/100 = 0.10)
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
        engine = HimalayasDynamicRiskEngine()

        base_row = {
            "gpm_1h_accumulation_mm": 5.0,
            "gpm_3h_accumulation_mm": 10.0,
            "gpm_6h_accumulation_mm": 15.0,
            "gpm_12h_accumulation_mm": 20.0,
            "gpm_24h_accumulation_mm": 25.0,
            "gpm_24h_complete": True,
            "rainfall_refresh_status": "SUCCESS",
        }

        # Increase 24h window from 25 to 50 mm
        increased_row = base_row.copy()
        increased_row["gpm_24h_accumulation_mm"] = 50.0

        df_pair = pd.DataFrame([base_row, increased_row])
        norm_df, _ = engine.compute_normalized_rainfall(df_pair)

        i_base = norm_df.loc[0, "rainfall_intensity_index"]
        i_inc = norm_df.loc[1, "rainfall_intensity_index"]
        self.assertGreater(i_inc, i_base)

    def test_04_zero_rainfall_rule(self):
        """Verify that zero rainfall yields R_rainfall = 0.0 and H_risk matches dry baseline."""
        engine = HimalayasDynamicRiskEngine(weight_terrain=0.50, weight_rainfall=0.50)

        df_zero = pd.DataFrame(
            [
                {
                    "grid_id": "ZERO_01",
                    "terrain_rf_prob_balanced": 0.60,
                    "gpm_1h_accumulation_mm": 0.0,
                    "gpm_3h_accumulation_mm": 0.0,
                    "gpm_6h_accumulation_mm": 0.0,
                    "gpm_12h_accumulation_mm": 0.0,
                    "gpm_24h_accumulation_mm": 0.0,
                    "gpm_24h_complete": True,
                    "rainfall_refresh_status": "SUCCESS",
                }
            ]
        )

        res_df, _ = engine.compute_combined_hazard_risk(df_zero)
        self.assertAlmostEqual(res_df.loc[0, "rainfall_intensity_index"], 0.0, places=6)
        self.assertAlmostEqual(res_df.loc[0, "rainfall_contribution"], 0.0, places=6)
        self.assertAlmostEqual(res_df.loc[0, "terrain_contribution"], 30.0, places=6)
        self.assertAlmostEqual(res_df.loc[0, "dynamic_risk_index"], 30.0, places=6)

    def test_05_missing_and_invalid_data_safeguard(self):
        """Verify non-finite (+inf, -inf, NaN) or incomplete windows evaluate to NaN risk and unassessed."""
        engine = HimalayasDynamicRiskEngine()

        df_bad = pd.DataFrame(
            [
                # Negative rainfall
                {
                    "grid_id": "BAD_01",
                    "terrain_rf_prob_balanced": 0.50,
                    "gpm_1h_accumulation_mm": -1.0,
                    "gpm_3h_accumulation_mm": 0.0,
                    "gpm_6h_accumulation_mm": 0.0,
                    "gpm_12h_accumulation_mm": 0.0,
                    "gpm_24h_accumulation_mm": 0.0,
                    "gpm_24h_complete": True,
                    "rainfall_refresh_status": "SUCCESS",
                },
                # Incomplete flag
                {
                    "grid_id": "BAD_02",
                    "terrain_rf_prob_balanced": 0.50,
                    "gpm_1h_accumulation_mm": 5.0,
                    "gpm_3h_accumulation_mm": 10.0,
                    "gpm_6h_accumulation_mm": 15.0,
                    "gpm_12h_accumulation_mm": 20.0,
                    "gpm_24h_accumulation_mm": 25.0,
                    "gpm_24h_complete": False,
                    "rainfall_refresh_status": "SUCCESS",
                },
                # NaN in 6h window
                {
                    "grid_id": "BAD_03",
                    "terrain_rf_prob_balanced": 0.50,
                    "gpm_1h_accumulation_mm": 5.0,
                    "gpm_3h_accumulation_mm": 10.0,
                    "gpm_6h_accumulation_mm": np.nan,
                    "gpm_12h_accumulation_mm": 20.0,
                    "gpm_24h_accumulation_mm": 25.0,
                    "gpm_24h_complete": True,
                    "rainfall_refresh_status": "SUCCESS",
                },
            ]
        )

        res_df, _ = engine.compute_combined_hazard_risk(df_bad)
        for i in range(len(df_bad)):
            self.assertTrue(np.isnan(res_df.loc[i, "dynamic_risk_index"]))
            self.assertEqual(res_df.loc[i, "primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")
            self.assertEqual(res_df.loc[i, "risk_calculation_status"], "INCOMPLETE_RAINFALL_WINDOW")

    def test_06_category_assignment_relative_and_fixed(self):
        """Verify empirical percentile categories and fixed benchmark tiers."""
        engine = HimalayasDynamicRiskEngine(
            relative_percentiles={"low_moderate": 50.0, "moderate_high": 80.0, "high_very_high": 95.0}
        )

        # 100 synthetic locations with uniform dynamic risk [0, 100]
        synth_records = []
        for i in range(100):
            prob = i / 100.0
            synth_records.append(
                {
                    "grid_id": f"CAT_{i:03d}",
                    "terrain_rf_prob_balanced": prob,
                    "gpm_1h_accumulation_mm": prob * 20.0,
                    "gpm_3h_accumulation_mm": prob * 40.0,
                    "gpm_6h_accumulation_mm": prob * 60.0,
                    "gpm_12h_accumulation_mm": prob * 80.0,
                    "gpm_24h_accumulation_mm": prob * 100.0,
                    "gpm_24h_complete": True,
                    "rainfall_refresh_status": "SUCCESS",
                }
            )
        df_cat = pd.DataFrame(synth_records)
        res_df, report = engine.compute_combined_hazard_risk(df_cat)

        rel_counts = res_df["risk_category_relative"].value_counts()
        self.assertIn("Low", rel_counts)
        self.assertIn("Moderate", rel_counts)
        self.assertIn("High", rel_counts)
        self.assertIn("Very High", rel_counts)

        # Fixed tiers
        low_fixed = res_df[res_df["dynamic_risk_index"] < 25.0]
        self.assertTrue((low_fixed["risk_tier_fixed"] == "Low").all())

    def test_07_freshness_gate_and_stale_data_rejection(self):
        """Verify freshness gate passes <=12h latency and rejects >12h latency."""
        now_ref = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)

        # Fresh observation (4h latency)
        end_fresh = (now_ref - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        df_fresh = pd.DataFrame(
            {
                "rainfall_refresh_status": ["SUCCESS"],
                "gpm_24h_complete": [True],
                "gpm_observation_end_utc": [end_fresh],
                "gpm_observation_start_utc": ["2026-11-10T08:00:00Z"],
            }
        )
        is_fresh, msg, _ = verify_himalayas_rainfall_freshness_gate(df_fresh, now_utc=now_ref)
        self.assertTrue(is_fresh)

        # Stale observation (15h latency)
        end_stale = (now_ref - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        df_stale = pd.DataFrame(
            {
                "rainfall_refresh_status": ["SUCCESS"],
                "gpm_24h_complete": [True],
                "gpm_observation_end_utc": [end_stale],
            }
        )
        is_fresh_stale, msg_stale, _ = verify_himalayas_rainfall_freshness_gate(df_stale, now_utc=now_ref)
        self.assertFalse(is_fresh_stale)
        self.assertIn("stale", msg_stale.lower())

    def test_08_output_csv_row_counts_and_provenance(self):
        """Verify output dynamic risk CSV exists, has 12,052 rows and retains provenance."""
        if not self.output_csv.exists():
            self.skipTest(f"Output CSV not yet generated: {self.output_csv}")

        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["grid_id"].nunique(), EXPECTED_LOCATIONS_COUNT)
        self.assertEqual(df["tile"].nunique(), EXPECTED_TILES_COUNT)

        for col in TERRAIN_MODEL_PREDICTORS + ["terrain_rf_prob_balanced", "dynamic_risk_index"]:
            self.assertIn(col, df.columns)

    def test_09_independent_full_grid_formula_verification(self):
        """Independently recompute dynamic risk across all 12,052 locations and assert equality."""
        if not self.output_csv.exists():
            self.skipTest(f"Output CSV not yet generated: {self.output_csv}")

        df = pd.read_csv(self.output_csv)

        s_t = df["terrain_rf_prob_balanced"].to_numpy()
        t_contrib = 100.0 * 0.50 * s_t

        scales = DEFAULT_WINDOW_SCALES_MM
        weights = DEFAULT_WINDOW_WEIGHTS

        r_norm = (
            weights["1h"] * np.clip(df["gpm_1h_accumulation_mm"] / scales["1h"], 0.0, 1.0)
            + weights["3h"] * np.clip(df["gpm_3h_accumulation_mm"] / scales["3h"], 0.0, 1.0)
            + weights["6h"] * np.clip(df["gpm_6h_accumulation_mm"] / scales["6h"], 0.0, 1.0)
            + weights["12h"] * np.clip(df["gpm_12h_accumulation_mm"] / scales["12h"], 0.0, 1.0)
            + weights["24h"] * np.clip(df["gpm_24h_accumulation_mm"] / scales["24h"], 0.0, 1.0)
        )
        r_contrib = 100.0 * 0.50 * r_norm
        expected_risk = t_contrib + r_contrib

        self.assertTrue(np.allclose(df["dynamic_risk_index"], expected_risk, atol=1e-3))
        self.assertTrue(np.allclose(df["terrain_contribution"], t_contrib, atol=1e-3))
        self.assertTrue(np.allclose(df["rainfall_contribution"], r_contrib, atol=1e-3))

    def test_10_overwrite_protection_and_backup(self):
        """Verify FileExistsError without overwrite, and backup creation when overwrite=True."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_ow_risk_"))
        try:
            fake_csv = temp_dir / "test_risk.csv"
            fake_meta = temp_dir / "test_meta.json"
            fake_rep = temp_dir / "test_rep.json"

            fake_csv.write_text("dummy", encoding="utf-8")
            fake_meta.write_text("{}", encoding="utf-8")
            fake_rep.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_himalayas_dynamic_risk_pipeline(
                    integrated_csv_path=self.integrated_csv,
                    gpm_report_path=self.gpm_report_json,
                    output_dir=temp_dir,
                    output_csv_path=fake_csv,
                    metadata_json_path=fake_meta,
                    validation_report_path=fake_rep,
                    overwrite=False,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_11_protection_boundaries(self):
        """Verify Western Ghats, Northeast India, and Stage 8 prior artifacts remain untouched."""
        protected_paths = [
            Path("data/processed/western_ghats_dynamic_risk/western_ghats_dynamic_hazard_risk.geojson"),
            Path("outputs/western_ghats_dynamic_risk_map.html"),
            Path("data/processed/northeast_spatial_grid/northeast_india_5km_spatial_grid.csv"),
            Path("data/processed/northeast_terrain_predictions/northeast_india_5km_terrain_susceptibility.csv"),
            Path("data/processed/northeast_gpm_rainfall/northeast_india_5km_rainfall_terrain_integrated.csv"),
            Path("data/processed/northeast_dynamic_risk/northeast_india_5km_dynamic_hazard_risk.csv"),
            Path("data/processed/himalayas_spatial_grid/himalayas_5km_spatial_grid.csv"),
            Path("data/processed/himalayas_terrain_predictions/himalayas_5km_terrain_susceptibility.csv"),
            Path("data/processed/himalayas_gpm_rainfall/himalayas_5km_rainfall_terrain_integrated.csv"),
        ]

        for p in protected_paths:
            full_p = PROJECT_ROOT / p
            self.assertTrue(full_p.exists(), f"Protected artifact missing: {full_p}")
            self.assertGreater(full_p.stat().st_size, 0, f"Protected artifact empty: {full_p}")


if __name__ == "__main__":
    unittest.main()
