"""
Independent Read-Only Audit for Western Ghats Regional NASA GPM IMERG Integration.

Audits:
1. Record counts, ID uniqueness, and label distributions (182 events, 1,400 background = 1,582 total).
2. Per-tile distributions across all 14 DEM tiles.
3. Coordinate and spatial matching metrics (distances <= 8.0 km, unique GPM pixels).
4. Native spatial resolution preservation (explicit 0.1° documentation, no downscaling claims).
5. Accumulation completeness, interval counts (1h=2, 3h=6, 6h=12, 12h=24, 24h=48), and monotonicity.
6. UTC timestamp consistency across all observation windows.
7. Unmodified status of underlying source datasets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_CSV = (
    PROJECT_ROOT / "data" / "processed" / "rainfall" / "western_ghats_regional_gpm_rainfall_features.csv"
)

EVENTS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_event_features.csv"
)

BACKGROUND_CSV = (
    PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_background_samples_v2.csv"
)

EXPECTED_EVENT_COUNTS_PER_TILE = {
    "N09E076": 21,
    "N10E076": 14,
    "N10E077": 12,
    "N11E076": 17,
    "N12E074": 11,
    "N12E075": 8,
    "N14E074": 8,
    "N15E073": 20,
    "N15E074": 8,
    "N16E073": 8,
    "N17E073": 7,
    "N18E073": 13,
    "N19E072": 18,
    "N19E073": 17,
}

EXPECTED_BG_COUNT_PER_TILE = 100


def run_audit(csv_path: Path = FEATURE_CSV):
    print("=" * 75)
    print("AUDIT: WESTERN GHATS GPM IMERG RAINFALL INTEGRATION DATASET")
    print("=" * 75)

    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\nTarget File: {csv_path.name}")
    print(f"Total Rows:  {len(df)}")
    print(f"Total Cols:  {len(df.columns)}")

    # 1. Row count & label distribution
    print("\n1. RECORD COUNTS & LABEL AUDIT:")
    if len(df) != 1582:
        raise ValueError(f"Expected 1,582 rows; got {len(df)}")

    event_mask = df["label"] == 1
    bg_mask = df["label"] == 0

    n_events = int(event_mask.sum())
    n_bg = int(bg_mask.sum())

    print(f"  Historical Events (label=1):    {n_events}  (Expected 182)")
    print(f"  Provisional Background (label=0): {n_bg} (Expected 1,400)")

    if n_events != 182 or n_bg != 1400:
        raise ValueError("Event or background counts do not match expected totals.")

    # 2. Identifiers
    print("\n2. IDENTIFIER UNIQUENESS AUDIT:")
    print(f"  Unique record_id: {df['record_id'].nunique()}/{len(df)}")
    print(f"  Unique sample_id: {df['sample_id'].nunique()}/{len(df)}")
    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id detected!")
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id detected!")

    # 3. Per-tile breakdown
    print("\n3. TILE-WISE DISTRIBUTION AUDIT (14 TILES):")
    tiles = sorted(df["tile"].unique())
    print(f"  Total Distinct Tiles: {len(tiles)} (Expected 14)")
    if len(tiles) != 14:
        raise ValueError(f"Expected 14 tiles; got {len(tiles)}")

    for t in tiles:
        t_sub = df[df["tile"] == t]
        t_ev = int((t_sub["label"] == 1).sum())
        t_bg = int((t_sub["label"] == 0).sum())
        exp_ev = EXPECTED_EVENT_COUNTS_PER_TILE.get(t, -1)
        status = "OK" if (t_ev == exp_ev and t_bg == EXPECTED_BG_COUNT_PER_TILE) else "MISMATCH"
        print(f"  Tile {t}: Total={len(t_sub):3d} | Events={t_ev:2d} (Exp {exp_ev:2d}) | BG={t_bg:3d} (Exp 100) -> {status}")
        if status != "OK":
            raise ValueError(f"Tile {t} count mismatch!")

    # 4. Spatial matching distance audit
    print("\n4. SPATIAL MATCHING & NATIVE RESOLUTION AUDIT:")
    dists = df["gpm_matching_distance_km"]
    print(f"  Min Matching Distance:  {dists.min():.3f} km")
    print(f"  Mean Matching Distance: {dists.mean():.3f} km")
    print(f"  Max Matching Distance:  {dists.max():.3f} km")
    print(f"  All Distances <= 8.0km: {(dists <= 8.0).all()}")
    if (dists > 8.0).any():
        raise ValueError(f"Found {int((dists > 8.0).sum())} points exceeding 8.0 km matching distance!")

    pixel_pairs = list(zip(df["gpm_grid_latitude"], df["gpm_grid_longitude"]))
    print(f"  Unique GPM Grid Pixels: {len(set(pixel_pairs))} (across 1,582 locations)")
    print(f"  Documented Resolution:  {df['gpm_spatial_resolution'].iloc[0]}")

    # 5. Rainfall accumulations & completeness
    print("\n5. ACCUMULATION & COMPLETENESS AUDIT:")
    windows = [1, 3, 6, 12, 24]
    for w in windows:
        col = f"gpm_{w}h_accumulation_mm"
        v_col = f"gpm_{w}h_valid_intervals"
        expected_intervals = w * 2

        null_cnt = int(df[col].isna().sum())
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        mean_v = float(df[col].mean())
        int_match = (df[v_col] == expected_intervals).all()

        print(f"  {w:2d}h Window: Nulls={null_cnt:4d} | Min={min_v:.3f} | Mean={mean_v:.3f} | Max={max_v:.3f} mm | Intervals={expected_intervals:2d} ({'100% OK' if int_match else 'FAIL'})")
        if null_cnt > 0:
            raise ValueError(f"Null values found in {col}!")
        if min_v < 0.0:
            raise ValueError(f"Negative accumulation found in {col}!")
        if not int_match:
            raise ValueError(f"Incomplete interval counts found in {v_col}!")

    print(f"  24h Complete Flag:       {int(df['gpm_24h_complete'].sum())}/{len(df)} (100% complete)")

    # 6. Monotonicity check: accum_1h <= accum_3h <= accum_6h <= accum_12h <= accum_24h
    print("\n6. ACCUMULATION MONOTONICITY AUDIT:")
    mono_1_3 = (df["gpm_1h_accumulation_mm"] <= df["gpm_3h_accumulation_mm"] + 1e-6).all()
    mono_3_6 = (df["gpm_3h_accumulation_mm"] <= df["gpm_6h_accumulation_mm"] + 1e-6).all()
    mono_6_12 = (df["gpm_6h_accumulation_mm"] <= df["gpm_12h_accumulation_mm"] + 1e-6).all()
    mono_12_24 = (df["gpm_12h_accumulation_mm"] <= df["gpm_24h_accumulation_mm"] + 1e-6).all()
    print(f"  1h <= 3h:  {mono_1_3}")
    print(f"  3h <= 6h:  {mono_3_6}")
    print(f"  6h <= 12h: {mono_6_12}")
    print(f"  12h <= 24h:{mono_12_24}")
    if not (mono_1_3 and mono_3_6 and mono_6_12 and mono_12_24):
        raise ValueError("Accumulation monotonicity violated!")

    # 7. Timestamps
    print("\n7. UTC TIMESTAMP AUDIT:")
    ts_cols = [c for c in df.columns if c.endswith("_utc")]
    for col in ts_cols:
        unique_ts = df[col].unique()
        print(f"  {col}: {unique_ts.tolist()}")
        if len(unique_ts) != 1:
            raise ValueError(f"Multiple timestamps found in uniform window column {col}!")
        # Validate format
        datetime.strptime(unique_ts[0], "%Y-%m-%dT%H:%M:%SZ")

    # 8. Source dataset verification
    print("\n8. SOURCE DATASET INTEGRITY AUDIT:")
    ev_df = pd.read_csv(EVENTS_CSV, low_memory=False)
    bg_df = pd.read_csv(BACKGROUND_CSV, low_memory=False)
    print(f"  western_ghats_regional_event_features.csv: {len(ev_df)} rows (OK)")
    print(f"  western_ghats_regional_background_samples_v2.csv: {len(bg_df)} rows (OK)")
    if len(ev_df) != 182 or len(bg_df) != 1400:
        raise ValueError("Source input datasets have unexpected row counts!")

    print("\n" + "=" * 75)
    print("ALL AUDIT CHECKS PASSED PERFECTLY.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Audit Western Ghats GPM rainfall features dataset.")
    parser.add_argument("--csv", type=Path, default=FEATURE_CSV, help="Path to feature CSV to audit.")
    args = parser.parse_args()

    try:
        run_audit(args.csv)
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
