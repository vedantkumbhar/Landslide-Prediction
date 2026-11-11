"""
Tests for Western Ghats Limited Pilot — 5 km UTM Spatial Grid (Stage 6.1 Grid).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

from backend.generate_western_ghats_spatial_grid import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_VALIDATION_REPORT,
    EXPECTED_TILE_IDS,
    WG_UTM_EPSG,
    _compute_aspect_derivatives,
    discover_validated_tile_paths,
    generate_western_ghats_spatial_grid,
)
from backend.gis_refresh_utils import build_5km_spatial_cells_utm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN_RF_MODEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_regional_model"
    / "western_ghats_regional_terrain_random_forest.joblib"
)


class TestWesternGhatsSpatialGrid(unittest.TestCase):
    def test_01_discover_exact_14_validated_tiles_epsg_32643(self) -> None:
        registry = discover_validated_tile_paths()
        self.assertEqual(len(registry), 14)
        self.assertEqual(tuple(sorted(registry.keys())), EXPECTED_TILE_IDS)
        for tid, info in registry.items():
            self.assertEqual(info["epsg"], WG_UTM_EPSG, f"Unexpected EPSG for {tid}")
            self.assertTrue(info["elevation_tif"].exists())
            self.assertTrue(info["slope_tif"].exists())
            self.assertTrue(info["aspect_tif"].exists())

    def test_02_production_grid_integrity_and_validation_report(self) -> None:
        self.assertTrue(DEFAULT_OUTPUT_CSV.exists(), f"Missing {DEFAULT_OUTPUT_CSV}")
        self.assertTrue(DEFAULT_VALIDATION_REPORT.exists(), f"Missing {DEFAULT_VALIDATION_REPORT}")

        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertTrue(report["validation_passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(len(df), report["total_grid_locations"])
        self.assertEqual(report["validated_tiles_count"], 14)
        self.assertEqual(tuple(report["validated_tiles"]), EXPECTED_TILE_IDS)
        self.assertEqual(report["crs"], "EPSG:32643")
        self.assertIn("Limited Pilot", report["coverage_disclosure"])
        self.assertIn("does not imply complete Western Ghats coverage", report["coverage_disclosure"])

        self.assertTrue((df["utm_epsg"] == 32643).all())
        self.assertEqual(df["grid_id"].nunique(), len(df))
        self.assertEqual(int(df.duplicated(subset=["utm_x", "utm_y"]).sum()), 0)
        self.assertTrue(((df["utm_x"] % 5000.0).abs() < 1e-6).all())
        self.assertTrue(((df["utm_y"] % 5000.0).abs() < 1e-6).all())

    def test_03_ocean_exclusion_and_below_sea_level_land_retention(self) -> None:
        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        with open(DEFAULT_VALIDATION_REPORT, "r", encoding="utf-8") as f:
            report = json.load(f)

        filtering = report["land_mask_specification"]["filtering_summary"]
        self.assertGreater(filtering["rejected_beyond_coastal_tolerance_ocean"], 1000)
        self.assertGreater(filtering["rejected_buffer_nonpositive_ocean"], 0)
        self.assertGreater(filtering["retained_below_sea_level_exact_land"], 0)

        # Coastal tiles like N12E074 and N19E072 must have ocean excluded (well under full 1x1 deg ~484 cells)
        self.assertLess(int((df["tile"] == "N12E074").sum()), 200)
        self.assertLess(int((df["tile"] == "N19E072").sum()), 150)

        # Below-sea-level coastal lowland inside exact Natural Earth land (e.g. Kuttanad) must be preserved
        below_sea = df[df["elevation_m"] < 0.0]
        self.assertGreater(len(below_sea), 0)
        self.assertTrue(below_sea["in_ne_exact_land"].all())

        # Any point retained in the 1 km coastal buffer outside exact land must have positive elevation
        buffer_only = df[~df["in_ne_exact_land"].astype(bool)]
        self.assertGreater(len(buffer_only), 0)
        self.assertTrue((buffer_only["elevation_m"] > 0.0).all())
        self.assertTrue((buffer_only["distance_to_ne_coast_m"] <= 1000.0).all())

    def test_04_aspect_derivatives_and_seamless_5km_cells(self) -> None:
        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        s_flat, c_flat, f_flat = _compute_aspect_derivatives(-1.0)
        self.assertEqual((s_flat, c_flat, f_flat), (0.0, 0.0, 1))

        gdf_cells = build_5km_spatial_cells_utm(df, expected_count=len(df), spacing_m=5000.0)
        self.assertEqual(len(gdf_cells), len(df))
        self.assertTrue(gdf_cells.geometry.is_valid.all())
        # Each 5 km square cell in UTM zone 43N has exact area 25,000,000 m^2
        utm_gdf = gdf_cells.to_crs("EPSG:32643")
        areas = utm_gdf.geometry.area
        self.assertTrue(((areas - 25_000_000.0).abs() < 1e-3).all())

    def test_05_frozen_rf_feature_contract_compatibility(self) -> None:
        self.assertTrue(FROZEN_RF_MODEL.exists(), f"Missing frozen model: {FROZEN_RF_MODEL}")
        model_bundle = joblib.load(FROZEN_RF_MODEL)
        df = pd.read_csv(DEFAULT_OUTPUT_CSV)
        feature_cols = ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "aspect_flat"]
        for col in feature_cols:
            self.assertIn(col, df.columns)
            self.assertEqual(int(df[col].isna().sum()), 0)

        if hasattr(model_bundle, "feature_names_in_"):
            self.assertEqual(list(model_bundle.feature_names_in_), feature_cols)
        elif isinstance(model_bundle, dict) and "feature_names" in model_bundle:
            self.assertEqual(list(model_bundle["feature_names"]), feature_cols)

    def test_06_overwrite_guard_raises_when_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "grid.csv"
            json_path = Path(tmpdir) / "report.json"
            csv_path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_western_ghats_spatial_grid(
                    output_csv=csv_path,
                    validation_report_path=json_path,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
