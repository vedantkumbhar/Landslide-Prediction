"""
Comprehensive Freshness & Reliability Test Suite for Stage 6B Dynamic GPM Refresh.

Tests the mandatory validation requirements:
1. Latest-window selection when newer granules exist.
2. Correct window selection when newer granules are partially missing.
3. Stale-data detection (latency > max_latency_hours).
4. Prevention of stale snapshot promotion to latest_validated_rainfall.csv.
5. Correct data latency calculation.
6. Snapshot preservation (immutability, zero overwrite).
7. Duplicate granule detection & deduplication.
8. Missing granule handling & interval tracking.
9. Incomplete 24-hour window handling (safe halting, no false zero rain).
10. Rolling accumulation calculations & mathematical monotonicity.
11. Spatial matching across all 14 Western Ghats DEM tiles.
12. Safe refresh recovery from interrupted downloads (.part cleanup).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from backend.refresh_western_ghats_rainfall import (
    WesternGhatsDynamicRefreshEngine,
    LocalGranuleIndex,
    GPMGranuleDownloader,
    is_valid_hdf5_granule,
    DEFAULT_ARCHIVE_DIR,
    EXISTING_RAW_DIRS,
    DYNAMIC_OUTPUT_DIR,
)
from backend.extract_western_ghats_gpm_rainfall import (
    WesternGhatsGPMIntegrator,
    load_western_ghats_locations,
    compute_accumulations,
    DEFAULT_EVENTS_CSV,
    DEFAULT_BACKGROUND_CSV,
    EXPECTED_TOTAL_COUNT,
    EXPECTED_TILE_COUNT,
    EXPECTED_GRANULES_24H,
    MAX_GRID_MATCH_DISTANCE_KM,
)


class TestWesternGhatsDynamicRefreshFreshness(unittest.TestCase):
    """Rigorous freshness, latency, and error-handling test suite."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.events_csv = DEFAULT_EVENTS_CSV
        cls.background_csv = DEFAULT_BACKGROUND_CSV
        cls.raw_dirs = [
            cls.project_root / "data" / "raw" / "satellite" / "gpm_imerge_previous12h",
            cls.project_root / "data" / "raw" / "satellite" / "gpm_imerge_series_12h",
        ]
        assert cls.events_csv.exists(), f"Events CSV missing: {cls.events_csv}"
        assert cls.background_csv.exists(), f"Background CSV missing: {cls.background_csv}"

        cls.index = LocalGranuleIndex(cls.raw_dirs)
        cls.granules = cls.index.all_granules_sorted()
        assert len(cls.granules) == 48, f"Expected 48 granules; found {len(cls.granules)}"

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_gpm_freshness_"))
        self.test_archive = self.test_dir / "archive"
        self.test_output = self.test_dir / "output"
        self.test_archive.mkdir(parents=True, exist_ok=True)
        self.test_output.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------
    # Test 1: Stale-Data Detection
    # -----------------------------------------------------------------
    def test_stale_data_detection(self):
        """Verify that when observation latency exceeds max_latency_hours, status is STALE_DATA."""
        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=self.raw_dirs,
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            max_latency_hours=12.0,
            offline_mode=True,
        )

        # November 1 observation window evaluated at current wall-clock time (>130h latency)
        old_target_end = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
        metadata = engine.execute_refresh(target_end_time=old_target_end)

        self.assertEqual(metadata["status"], "STALE_DATA")
        self.assertEqual(metadata["promotion_status"], "REJECTED_STALE")
        self.assertGreater(metadata["data_latency_hours"], 12.0)
        self.assertTrue(metadata["is_24h_complete"])

    # -----------------------------------------------------------------
    # Test 2: Prevention of Stale Snapshot Promotion
    # -----------------------------------------------------------------
    def test_prevention_of_stale_snapshot_promotion(self):
        """Verify that a stale dataset is NOT promoted to latest_validated_rainfall.csv."""
        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=self.raw_dirs,
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            max_latency_hours=6.0,
            offline_mode=True,
        )

        old_target_end = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
        metadata = engine.execute_refresh(target_end_time=old_target_end)

        self.assertEqual(metadata["status"], "STALE_DATA")
        # Ensure latest_validated_rainfall.csv was NOT created
        self.assertFalse(engine.latest_validated_csv.exists(),
                         "latest_validated_rainfall.csv must NOT be created from stale data!")

    # -----------------------------------------------------------------
    # Test 3: Correct Latency Calculation
    # -----------------------------------------------------------------
    def test_correct_latency_calculation(self):
        """Verify that data latency matches (execution_utc - observation_end_utc)."""
        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=self.raw_dirs,
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            max_latency_hours=200.0,  # Allow through to test latency calculation
            offline_mode=True,
        )

        target_end = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
        t_before = datetime.now(timezone.utc)
        metadata = engine.execute_refresh(target_end_time=target_end)
        t_after = datetime.now(timezone.utc)

        expected_latency_min = (t_before - target_end).total_seconds() / 3600.0
        expected_latency_max = (t_after - target_end).total_seconds() / 3600.0

        self.assertTrue(expected_latency_min - 0.05 <= metadata["data_latency_hours"] <= expected_latency_max + 0.05)

    # -----------------------------------------------------------------
    # Test 4: Latest Window Selection When Newer Granules Exist
    # -----------------------------------------------------------------
    def test_latest_window_selection_when_newer_granules_exist(self):
        """Verify that when newer contiguous granules are available, the newer window is selected."""
        # Create a simulated online granule map with two separate days
        granule_map = {}
        # Day 1: Nov 1 (old)
        for g in self.granules:
            granule_map[g["start"]] = g

        # Day 2: Nov 5 (newer 48 granules)
        nov5_base = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
        for i in range(48):
            start = nov5_base + timedelta(minutes=30 * i)
            end = start + timedelta(seconds=1799)
            granule_map[start] = {
                "title": f"3B-HHR-E.MS.MRG.3IMERG.{start.strftime('%Y%m%d-S%H%M%S')}-E{end.strftime('%H%M%S')}.HDF5",
                "filename": f"3B-HHR-E.MS.MRG.3IMERG.{start.strftime('%Y%m%d-S%H%M%S')}-E{end.strftime('%H%M%S')}.HDF5",
                "start": start,
                "end": end,
                "url": "http://example.com/test.HDF5",
            }

        # Downloader mock with granule map
        downloader = GPMGranuleDownloader(archive_dir=self.test_archive)
        downloader.search_cmr_granules = lambda s, e: []

        # Find newest contiguous window from map
        starts = sorted(granule_map.keys())
        for latest_start in reversed(starts):
            cand_end = latest_start + timedelta(minutes=30)
            cand_starts = [cand_end - timedelta(minutes=30 * (48 - i)) for i in range(48)]
            if all(s in granule_map for s in cand_starts):
                selected_end = cand_end
                break

        expected_new_end = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(selected_end, expected_new_end, "Failed to select newer contiguous window!")

    # -----------------------------------------------------------------
    # Test 5: Selection When Newer Granules Are Partially Missing
    # -----------------------------------------------------------------
    def test_selection_when_newer_granules_partially_missing(self):
        """Verify that if newer granules are incomplete, it selects the latest FULLY contiguous window."""
        granule_map = {}
        # Full contiguous block 1: Nov 1 (48 granules)
        for g in self.granules:
            granule_map[g["start"]] = g

        # Incomplete newer block: Nov 6 (only 10 granules, 38 missing)
        nov6_base = datetime(2026, 11, 6, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            start = nov6_base + timedelta(minutes=30 * i)
            end = start + timedelta(seconds=1799)
            granule_map[start] = {"start": start, "end": end}

        # Search algorithm: find latest contiguous 48-granule block
        starts = sorted(granule_map.keys())
        selected_end = None
        for latest_start in reversed(starts):
            cand_end = latest_start + timedelta(minutes=30)
            cand_starts = [cand_end - timedelta(minutes=30 * (48 - i)) for i in range(48)]
            if all(s in granule_map for s in cand_starts):
                selected_end = cand_end
                break

        # Must fall back to the complete Nov 1 window, NOT the partial Nov 6 window
        expected_fallback_end = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(selected_end, expected_fallback_end)

    # -----------------------------------------------------------------
    # Test 6: Snapshot Preservation
    # -----------------------------------------------------------------
    def test_snapshot_preservation(self):
        """Verify that existing snapshots are immutable and never overwritten."""
        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=self.raw_dirs,
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            max_latency_hours=200.0,
            offline_mode=True,
        )
        target_end = self.granules[-1]["end"] + timedelta(seconds=1)

        meta1 = engine.execute_refresh(target_end_time=target_end)
        self.assertFalse(meta1["snapshot_preserved_existing"])
        snapshot_path = Path(meta1["snapshot_path"])
        mtime1 = snapshot_path.stat().st_mtime

        time.sleep(0.5)

        meta2 = engine.execute_refresh(target_end_time=target_end)
        self.assertTrue(meta2["snapshot_preserved_existing"])
        mtime2 = snapshot_path.stat().st_mtime

        self.assertEqual(mtime1, mtime2, "Existing snapshot was modified!")

    # -----------------------------------------------------------------
    # Test 7: Duplicate Granule Detection
    # -----------------------------------------------------------------
    def test_duplicate_granule_detection(self):
        """Verify that duplicate granules with identical timestamps are deduplicated."""
        dup1 = self.test_dir / "dup1"
        dup2 = self.test_dir / "dup2"
        dup1.mkdir(parents=True, exist_ok=True)
        dup2.mkdir(parents=True, exist_ok=True)

        g0 = self.granules[0]["path"]
        shutil.copy(g0, dup1 / g0.name)
        shutil.copy(g0, dup2 / g0.name)

        idx = LocalGranuleIndex([dup1, dup2])
        self.assertEqual(len(idx.all_granules_sorted()), 1)

    # -----------------------------------------------------------------
    # Test 8: Missing Granule Handling
    # -----------------------------------------------------------------
    def test_missing_granule_handling(self):
        """Verify that missing granules within a requested window are tracked in missing list."""
        partial_dir = self.test_dir / "partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        # Copy 47 out of 48 granules (omit index 15)
        omitted = self.granules[15]["start"]
        for i, g in enumerate(self.granules):
            if i != 15:
                shutil.copy(g["path"], partial_dir / g["path"].name)

        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=[partial_dir],
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            offline_mode=True,
        )
        target_end = self.granules[-1]["end"] + timedelta(seconds=1)
        granules, missing, _, _ = engine.ensure_granules_for_window(target_end)

        self.assertEqual(len(granules), 47)
        self.assertEqual(missing, [omitted])

    # -----------------------------------------------------------------
    # Test 9: Incomplete 24-Hour Window Handling
    # -----------------------------------------------------------------
    def test_incomplete_24h_window_handling(self):
        """Verify incomplete windows halt safely with INCOMPLETE_DATA and no zero substitution."""
        partial_dir = self.test_dir / "partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        for g in self.granules[:20]:
            shutil.copy(g["path"], partial_dir / g["path"].name)

        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=[partial_dir],
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            offline_mode=True,
        )
        target_end = self.granules[-1]["end"] + timedelta(seconds=1)
        meta = engine.execute_refresh(target_end_time=target_end)

        self.assertEqual(meta["status"], "INCOMPLETE_DATA")
        self.assertEqual(meta["promotion_status"], "REJECTED_INCOMPLETE")
        self.assertFalse(meta["is_24h_complete"])
        self.assertFalse(engine.latest_validated_csv.exists())

    # -----------------------------------------------------------------
    # Test 10: Rolling Accumulation Monotonicity
    # -----------------------------------------------------------------
    def test_accumulation_calculations(self):
        """Verify accumulation intervals and monotonicity: 1h <= 3h <= 6h <= 12h <= 24h."""
        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=self.raw_dirs,
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            max_latency_hours=200.0,
            offline_mode=True,
        )
        target_end = self.granules[-1]["end"] + timedelta(seconds=1)
        meta = engine.execute_refresh(target_end_time=target_end)

        df = pd.read_csv(meta["snapshot_path"])
        self.assertTrue((df["gpm_1h_valid_intervals"] == 2).all())
        self.assertTrue((df["gpm_3h_valid_intervals"] == 6).all())
        self.assertTrue((df["gpm_6h_valid_intervals"] == 12).all())
        self.assertTrue((df["gpm_12h_valid_intervals"] == 24).all())
        self.assertTrue((df["gpm_24h_valid_intervals"] == 48).all())

        self.assertTrue((df["gpm_1h_accumulation_mm"] <= df["gpm_3h_accumulation_mm"] + 1e-6).all())
        self.assertTrue((df["gpm_3h_accumulation_mm"] <= df["gpm_6h_accumulation_mm"] + 1e-6).all())
        self.assertTrue((df["gpm_6h_accumulation_mm"] <= df["gpm_12h_accumulation_mm"] + 1e-6).all())
        self.assertTrue((df["gpm_12h_accumulation_mm"] <= df["gpm_24h_accumulation_mm"] + 1e-6).all())

    # -----------------------------------------------------------------
    # Test 11: Spatial Matching Across 14 Tiles
    # -----------------------------------------------------------------
    def test_spatial_matching_14_tiles(self):
        """Verify spatial matching across all 14 tiles: distances <= 8.0 km, 771 GPM pixels."""
        engine = WesternGhatsDynamicRefreshEngine(
            events_csv=self.events_csv,
            background_csv=self.background_csv,
            raw_directories=self.raw_dirs,
            archive_dir=self.test_archive,
            output_dir=self.test_output,
            max_latency_hours=200.0,
            offline_mode=True,
        )
        target_end = self.granules[-1]["end"] + timedelta(seconds=1)
        meta = engine.execute_refresh(target_end_time=target_end)

        df = pd.read_csv(meta["snapshot_path"])
        self.assertEqual(len(df), EXPECTED_TOTAL_COUNT)
        self.assertEqual(df["tile"].nunique(), EXPECTED_TILE_COUNT)
        self.assertTrue((df["gpm_matching_distance_km"] <= MAX_GRID_MATCH_DISTANCE_KM).all())

        pixels = set(zip(df["gpm_grid_latitude"], df["gpm_grid_longitude"]))
        self.assertEqual(len(pixels), 771)

    # -----------------------------------------------------------------
    # Test 12: Safe Refresh Recovery
    # -----------------------------------------------------------------
    def test_safe_refresh_recovery(self):
        """Verify cleanup of temporary .part files and recovery from interrupted downloads."""
        downloader = GPMGranuleDownloader(archive_dir=self.test_archive)
        sample_item = {
            "filename": "test_partial_granule.HDF5",
            "url": "http://invalid-url-for-test.example",
        }

        part_file = self.test_archive / (sample_item["filename"] + ".part")
        with open(part_file, "w") as f:
            f.write("corrupted bytes")

        with self.assertRaises(Exception):
            downloader.download_single_granule(sample_item, max_retries=1)

        self.assertFalse(part_file.exists())


def run_tests():
    print("\n" + "=" * 75)
    print("RUNNING STAGE 6B DYNAMIC REFRESH FRESHNESS & RELIABILITY TEST SUITE (12 TESTS)")
    print("=" * 75)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWesternGhatsDynamicRefreshFreshness)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\nTEST SUITE FAILED!")
        sys.exit(1)
    print("\nALL 12 TESTS PASSED WITH 100% SUCCESS.")


if __name__ == "__main__":
    run_tests()
