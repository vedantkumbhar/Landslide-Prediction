"""
Tests for Western Ghats Limited Pilot — 5 km Grid GPM Rainfall Integration (Stage 6.3 Grid).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from backend.extract_western_ghats_gpm_rainfall import MAX_GRID_MATCH_DISTANCE_KM
from backend.gis_refresh_utils import parse_iso8601_utc
from backend.integrate_western_ghats_gpm_rainfall import (
    DEFAULT_INTEGRATED_CSV,
    DEFAULT_VALIDATION_REPORT,
    integrate_western_ghats_gpm_rainfall,
)
from backend.predict_western_ghats_terrain_susceptibility import (
    DEFAULT_OUTPUT_CSV as TERRAIN_CSV,
)


class TestWesternGhatsGpmGridRainfall(unittest.TestCase):
    def test_01_production_integrated_csv_matches_terrain_grid(self) -> None:
        self.assertTrue(DEFAULT_INTEGRATED_CSV.exists())
        self.assertTrue(DEFAULT_VALIDATION_REPORT.exists())

        t_df = pd.read_csv(TERRAIN_CSV)
        r_df = pd.read_csv(DEFAULT_INTEGRATED_CSV)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(len(r_df), len(t_df))
        self.assertListEqual(r_df["grid_id"].tolist(), t_df["grid_id"].tolist())
        self.assertEqual(report["unique_dem_tiles"], 14)
        self.assertIn("Limited Pilot", report["coverage_disclosure"])
        self.assertIn("does not imply complete Western Ghats coverage", report["coverage_disclosure"])
        self.assertTrue(all(report["validation_checks"].values()))

    def test_02_spatial_matching_within_8km(self) -> None:
        df = pd.read_csv(DEFAULT_INTEGRATED_CSV)
        self.assertTrue((df["gpm_matching_distance_km"] <= MAX_GRID_MATCH_DISTANCE_KM).all())
        self.assertTrue((df["gpm_matching_distance_km"] >= 0.0).all())
        self.assertEqual(int(df["gpm_grid_latitude"].isna().sum()), 0)
        self.assertEqual(int(df["gpm_grid_longitude"].isna().sum()), 0)

    def test_03_five_rainfall_windows_valid_and_monotonic(self) -> None:
        df = pd.read_csv(DEFAULT_INTEGRATED_CSV)
        cols = [
            "gpm_1h_accumulation_mm",
            "gpm_3h_accumulation_mm",
            "gpm_6h_accumulation_mm",
            "gpm_12h_accumulation_mm",
            "gpm_24h_accumulation_mm",
        ]
        for c in cols:
            self.assertIn(c, df.columns)
            self.assertEqual(int(df[c].isna().sum()), 0)
            self.assertTrue((df[c] >= 0.0).all())

        self.assertTrue((df["gpm_1h_accumulation_mm"] <= df["gpm_3h_accumulation_mm"] + 1e-4).all())
        self.assertTrue((df["gpm_3h_accumulation_mm"] <= df["gpm_6h_accumulation_mm"] + 1e-4).all())
        self.assertTrue((df["gpm_6h_accumulation_mm"] <= df["gpm_12h_accumulation_mm"] + 1e-4).all())
        self.assertTrue((df["gpm_12h_accumulation_mm"] <= df["gpm_24h_accumulation_mm"] + 1e-4).all())
        self.assertTrue((df["gpm_interval_count"] == 48).all())
        self.assertTrue(df["gpm_24h_complete"].astype(bool).all())

    def test_04_stale_and_incomplete_failsafes(self) -> None:
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            prod_rep = json.load(f)
        obs_end_str = prod_rep["gpm_observation_metadata"]["observation_end_utc"]
        obs_end_dt = parse_iso8601_utc(obs_end_str)
        self.assertIsNotNone(obs_end_dt)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_csv = tmp / "integrated.csv"
            rep_json = tmp / "report.json"

            # Stale reference time (15 hours past observation end)
            rep_stale = integrate_western_ghats_gpm_rainfall(
                output_dir=tmp,
                output_csv_path=out_csv,
                validation_report_path=rep_json,
                target_end_time=obs_end_dt,
                allow_download=False,
                now_utc=obs_end_dt + timedelta(hours=15),
                overwrite=True,
            )
            self.assertEqual(rep_stale["refresh_status"], "STALE_DATA")
            self.assertFalse(rep_stale["is_operational_success"])

            # Future timestamp fail-safe: ref_now earlier than latest observation raises ValueError
            with self.assertRaises(ValueError):
                integrate_western_ghats_gpm_rainfall(
                    output_dir=tmp,
                    output_csv_path=out_csv,
                    validation_report_path=rep_json,
                    target_end_time=obs_end_dt,
                    allow_download=False,
                    now_utc=obs_end_dt - timedelta(hours=1),
                    overwrite=True,
                )

            # Incomplete fail-safe: only 1 granule in isolated archive -> INCOMPLETE_DATA & NaN accumulations
            inc_archive = tmp / "inc_archive"
            inc_archive.mkdir(parents=True, exist_ok=True)
            sample_h5 = next(
                (Path(__file__).resolve().parent.parent / "data" / "raw" / "satellite" / "gpm_archive").glob("*.HDF5")
            )
            shutil.copy2(sample_h5, inc_archive / sample_h5.name)
            with patch("backend.refresh_western_ghats_rainfall.EXISTING_RAW_DIRS", [inc_archive]):
                rep_inc = integrate_western_ghats_gpm_rainfall(
                    output_dir=tmp,
                    output_csv_path=out_csv,
                    validation_report_path=rep_json,
                    archive_dir=inc_archive,
                    allow_download=False,
                    now_utc=obs_end_dt + timedelta(hours=1),
                    overwrite=True,
                )
            self.assertEqual(rep_inc["refresh_status"], "INCOMPLETE_DATA")
            self.assertFalse(rep_inc["is_operational_success"])
            self.assertFalse(rep_inc["gpm_observation_metadata"]["is_24h_complete"])
            inc_df = pd.read_csv(out_csv)
            self.assertTrue(inc_df["gpm_24h_accumulation_mm"].isna().all())
            self.assertTrue((~inc_df["gpm_24h_complete"].astype(bool)).all())

    def test_05_overwrite_guard_raises_when_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_csv = tmp / "existing.csv"
            rep_json = tmp / "report.json"
            out_csv.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                integrate_western_ghats_gpm_rainfall(
                    output_dir=tmp,
                    output_csv_path=out_csv,
                    validation_report_path=rep_json,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
