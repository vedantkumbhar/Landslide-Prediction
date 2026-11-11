"""
Tests for Western Ghats Limited Pilot — 5 km Grid Dynamic Risk Calculation (Stage 6.4 Grid).
"""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.gis_refresh_utils import parse_iso8601_utc, validate_dynamic_metadata_thresholds
from backend.integrate_western_ghats_gpm_rainfall import (
    DEFAULT_INTEGRATED_CSV,
    DEFAULT_VALIDATION_REPORT as GPM_REPORT,
)
from backend.predict_western_ghats_grid_dynamic_risk import (
    DEFAULT_METADATA_JSON,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_VALIDATION_REPORT,
    predict_western_ghats_grid_dynamic_risk,
)


class TestWesternGhatsGridDynamicRisk(unittest.TestCase):
    def test_01_production_outputs_and_metadata_thresholds(self) -> None:
        self.assertTrue(DEFAULT_OUTPUT_CSV.exists())
        self.assertTrue(DEFAULT_METADATA_JSON.exists())
        self.assertTrue(DEFAULT_VALIDATION_REPORT.exists())

        in_df = pd.read_csv(DEFAULT_INTEGRATED_CSV)
        out_df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        with open(DEFAULT_METADATA_JSON, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(len(out_df), len(in_df))
        self.assertEqual(metadata["dem_tiles_count"], 14)
        self.assertIn("Limited Pilot", metadata["coverage_disclosure"])
        self.assertIn("does not imply complete Western Ghats coverage", metadata["coverage_disclosure"])
        self.assertTrue(all(report["validation_checks"].values()))

        cutoffs = validate_dynamic_metadata_thresholds(metadata)
        self.assertLess(0.0, cutoffs["P50_low_to_moderate"])
        self.assertLess(cutoffs["P50_low_to_moderate"], cutoffs["P80_moderate_to_high"])
        self.assertLess(cutoffs["P80_moderate_to_high"], cutoffs["P95_high_to_very_high"])
        self.assertLess(cutoffs["P95_high_to_very_high"], 100.0)

    def test_02_additive_risk_formula_and_zero_rainfall_rule(self) -> None:
        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        valid = df["dynamic_risk_index"].notna()
        self.assertTrue(valid.all())

        diff = (df["terrain_contribution"] + df["rainfall_contribution"] - df["dynamic_risk_index"]).abs()
        self.assertTrue((diff <= 1e-3).all())

        zero_rain = df[
            (df["gpm_1h_accumulation_mm"] == 0.0)
            & (df["gpm_3h_accumulation_mm"] == 0.0)
            & (df["gpm_6h_accumulation_mm"] == 0.0)
            & (df["gpm_12h_accumulation_mm"] == 0.0)
            & (df["gpm_24h_accumulation_mm"] == 0.0)
        ]
        if len(zero_rain) > 0:
            self.assertTrue((zero_rain["rainfall_intensity_index"] == 0.0).all())
            self.assertTrue((zero_rain["rainfall_contribution"] == 0.0).all())

    def test_03_stale_gate_and_missing_rainfall_safeguards(self) -> None:
        df_in = pd.read_csv(DEFAULT_INTEGRATED_CSV)
        obs_end = parse_iso8601_utc(str(df_in["gpm_observation_end_utc"].iloc[0]))
        self.assertIsNotNone(obs_end)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_csv = tmp / "risk.csv"
            meta_json = tmp / "meta.json"
            rep_json = tmp / "rep.json"

            # Stale (>12h latency)
            stale_df, stale_meta, stale_rep = predict_western_ghats_grid_dynamic_risk(
                integrated_csv_path=DEFAULT_INTEGRATED_CSV,
                gpm_report_path=GPM_REPORT,
                output_dir=tmp,
                output_csv_path=out_csv,
                metadata_json_path=meta_json,
                validation_report_path=rep_json,
                now_utc=obs_end + timedelta(hours=13),
                overwrite=True,
            )
            self.assertEqual(stale_rep["status"], "FRESHNESS_GATE_FAILED")
            self.assertFalse(stale_rep["is_operational_success"])
            self.assertEqual(stale_meta["gpm_observation_metadata"]["freshness_status"], "STALE_DATA_UNASSESSED")
            self.assertTrue((stale_df["primary_screening_category"] == "STALE_DATA_UNASSESSED").all())

            # Partial missing NaN rainfall row -> only affected rows UNASSESSED_MISSING_RAINFALL
            bad_csv = tmp / "partial_nan.csv"
            mod_df = df_in.copy()
            mod_df.loc[0, "gpm_24h_accumulation_mm"] = np.nan
            mod_df.to_csv(bad_csv, index=False)

            nan_df, _, _ = predict_western_ghats_grid_dynamic_risk(
                integrated_csv_path=bad_csv,
                gpm_report_path=GPM_REPORT,
                output_dir=tmp,
                output_csv_path=out_csv,
                metadata_json_path=meta_json,
                validation_report_path=rep_json,
                now_utc=obs_end + timedelta(hours=2),
                overwrite=True,
            )
            self.assertTrue(pd.isna(nan_df.loc[0, "dynamic_risk_index"]))
            self.assertEqual(nan_df.loc[0, "primary_screening_category"], "UNASSESSED_MISSING_RAINFALL")


if __name__ == "__main__":
    unittest.main()
