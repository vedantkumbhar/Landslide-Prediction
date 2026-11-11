"""
Tests for Western Ghats Limited Pilot — Frozen RF Terrain Susceptibility (Stage 6.2 Grid).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from backend.generate_western_ghats_spatial_grid import DEFAULT_OUTPUT_CSV as GRID_CSV
from backend.predict_western_ghats_terrain_susceptibility import (
    DEFAULT_BALANCED_MODEL,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_VALIDATION_REPORT,
    RF_FEATURE_COLUMNS,
    classify_terrain_tier,
    predict_western_ghats_terrain_susceptibility,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestWesternGhatsTerrainSusceptibility(unittest.TestCase):
    def test_01_frozen_model_contract_and_no_retraining(self) -> None:
        self.assertTrue(DEFAULT_BALANCED_MODEL.exists())
        pre_sha = _sha256(DEFAULT_BALANCED_MODEL)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["frozen_models"]["primary_balanced_sha256"], pre_sha)
        self.assertFalse(report["frozen_models"]["retrained"])
        self.assertEqual(report["feature_contract"], RF_FEATURE_COLUMNS)

    def test_02_production_predictions_match_grid_1_to_1(self) -> None:
        grid_df = pd.read_csv(GRID_CSV)
        pred_df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertTrue(report["validation_passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(pred_df), len(grid_df))
        self.assertListEqual(pred_df["grid_id"].tolist(), grid_df["grid_id"].tolist())
        self.assertIn("Limited Pilot", report["coverage_disclosure"])
        self.assertIn("does not imply complete Western Ghats coverage", report["coverage_disclosure"])

    def test_03_probabilities_tiers_and_threshold_consistency(self) -> None:
        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        probs = df["terrain_rf_prob_balanced"]
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        self.assertEqual(int(probs.isna().sum()), 0)
        self.assertTrue(((probs >= 0.50).astype(int) == df["terrain_rf_pred_balanced"]).all())

        for p, tier in zip(probs, df["terrain_baseline_screening_tier"]):
            self.assertEqual(tier, classify_terrain_tier(float(p)))

    def test_04_missing_or_nan_features_rejected_and_overwrite_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_csv = tmp / "bad_grid.csv"
            out_csv = tmp / "out.csv"
            rep_json = tmp / "rep.json"

            pd.DataFrame({"grid_id": ["WG_GRID_00001"], "tile": ["N09E076"]}).to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError):
                predict_western_ghats_terrain_susceptibility(
                    grid_csv=bad_csv,
                    output_csv=out_csv,
                    validation_report_path=rep_json,
                )

            nan_df = pd.read_csv(GRID_CSV).head(5).copy()
            nan_df.loc[0, "slope_degrees"] = float("nan")
            nan_df.to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError):
                predict_western_ghats_terrain_susceptibility(
                    grid_csv=bad_csv,
                    output_csv=out_csv,
                    validation_report_path=rep_json,
                )

            out_csv.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                predict_western_ghats_terrain_susceptibility(
                    grid_csv=GRID_CSV,
                    output_csv=out_csv,
                    validation_report_path=rep_json,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
