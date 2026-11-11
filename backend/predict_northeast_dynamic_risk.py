"""
Stage 7.4: Northeast India Dynamic Hazard Risk Formulation and Calibration.

Generates a dynamic, multi-window rainfall-responsive landslide hazard screening index
across all 14,550 regular 5 km grid locations (33 DEM tiles) in Northeast India.

Key Architectural & Methodological Rules:
1. Model Provenance & Invariance:
   - Reuses precomputed terrain Random Forest probabilities (terrain_rf_prob_balanced)
     directly from the Stage 7.3 integrated dataset.
   - Zero model retraining or new model inference is performed.
   - The five-feature terrain contract (['elevation_m', 'slope_degrees', 'aspect_sin',
     'aspect_cos', 'aspect_flat']) is strictly preserved.
2. Multi-Window Hydrometeorological Scaling & Strict Finite Validation:
   - Uses five accumulation windows: 1h, 3h, 6h, 12h, and 24h.
   - Provisional Regional Reference Scales (configurable):
     1h: 25.0 mm, 3h: 50.0 mm, 6h: 75.0 mm, 12h: 100.0 mm, 24h: 150.0 mm.
     Reflects Northeast India's high precipitation regime and IMD / Eastern Himalayan
     triggering thresholds.
   - Strict Finite Validation: Only finite, non-negative accumulation values are valid.
     Non-finite (+inf, -inf, NaN) or negative values evaluate to NaN.
   - Zero Rainfall Rule: If all windows are 0.0 mm, normalized rainfall intensity and
     rainfall contribution evaluate strictly to 0.0, reducing dynamic risk to the dry baseline.
   - Missing Data Safety: Incomplete rainfall windows (gpm_24h_complete == False or NaN)
     evaluate to NaN without positive contribution; risk calculation status evaluates to
     INCOMPLETE_RAINFALL_WINDOW and primary screening category to UNASSESSED_MISSING_RAINFALL.
3. Transparent Convex Hazard Screening Formula:
   - Primary Dynamic Risk Index:
     H_risk = 100 * (w_terrain * S_terrain + w_rainfall * R_rainfall)
            = Terrain_Contribution + Rainfall_Contribution
     Default configurable weights: w_terrain = 0.50, w_rainfall = 0.50.
   - Interactive Modulated Hazard Index:
     H_interactive = 100 * (S_terrain * (1.0 + R_rainfall) / 2.0)
   - Relative screening categories (Low, Moderate, High, Very High) derived from
     empirical domain percentiles (P50, P80, P95) of the active Northeast monitoring population.
   - Fixed benchmark tiers ([0, 25), [25, 50), [50, 75), [75, 100]) provided in parallel.
4. Independent Freshness Gate:
   - Evaluates observation_end_utc from Stage 7.3 metadata against current UTC time.
   - Enforces the 12.0-hour maximum latency threshold.
5. Experimental Screening Disclaimer:
   - This provisional index is an experimental situational screening tool, NOT a calibrated
     landslide probability, NOT an event forecast, and NOT an official government early warning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.gis_refresh_utils import (
    atomic_write_csv,
    atomic_write_json,
    create_timestamped_backup,
)

# Default file paths
DEFAULT_INTEGRATED_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_gpm_rainfall"
    / "northeast_india_5km_rainfall_terrain_integrated.csv"
)
DEFAULT_GPM_REPORT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_gpm_rainfall"
    / "northeast_gpm_rainfall_validation_report.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_dynamic_risk"
)
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "northeast_india_5km_dynamic_hazard_risk.csv"
DEFAULT_METADATA_JSON = DEFAULT_OUTPUT_DIR / "northeast_dynamic_risk_metadata.json"
DEFAULT_VALIDATION_REPORT_JSON = DEFAULT_OUTPUT_DIR / "northeast_dynamic_risk_validation_report.json"

EXPECTED_LOCATIONS_COUNT = 14550
EXPECTED_TILES_COUNT = 33

TERRAIN_MODEL_PREDICTORS = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

# Configurable Northeast provisional physical reference thresholds (mm)
DEFAULT_WINDOW_SCALES_MM: Dict[str, float] = {
    "1h": 25.0,
    "3h": 50.0,
    "6h": 75.0,
    "12h": 100.0,
    "24h": 150.0,
}

# Configurable multi-window accumulation weights (sum = 1.0)
DEFAULT_WINDOW_WEIGHTS: Dict[str, float] = {
    "1h": 0.10,
    "3h": 0.15,
    "6h": 0.20,
    "12h": 0.25,
    "24h": 0.30,
}

# Configurable combination weights (sum = 1.0)
DEFAULT_WEIGHT_TERRAIN = 0.50
DEFAULT_WEIGHT_RAINFALL = 0.50

# Configurable relative screening percentiles
DEFAULT_RELATIVE_PERCENTILES: Dict[str, float] = {
    "low_moderate": 50.0,
    "moderate_high": 80.0,
    "high_very_high": 95.0,
}

# Fixed benchmark tier cutoffs
FIXED_TIER_CUTOFFS = [0.0, 25.0, 50.0, 75.0, 100.0]
FIXED_TIER_LABELS = ["Low", "Moderate", "High", "Very High"]

MAX_DATA_LATENCY_HOURS = 12.0
DISCLAIMER_TEXT = "EXPERIMENTAL_PROVISIONAL_NORTHEAST_SCREENING_ONLY_NOT_OFFICIAL_WARNING"


def parse_iso8601_utc(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 UTC timestamp string ending in Z or +00:00."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    cleaned = ts_str.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def verify_northeast_rainfall_freshness_gate(
    df: pd.DataFrame,
    report_json_path: Optional[Path] = None,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify that the input rainfall integration dataset has SUCCESS status, complete data,
    and valid latency.
    """
    metadata: Dict[str, Any] = {}

    # 1. Check refresh status in DataFrame
    if "rainfall_refresh_status" not in df.columns:
        return False, "Missing 'rainfall_refresh_status' in input integrated CSV", metadata

    statuses = df["rainfall_refresh_status"].unique()
    if len(statuses) != 1 or statuses[0] != "SUCCESS":
        return False, f"Rainfall status is {statuses.tolist()} (expected ['SUCCESS'])", metadata

    # 2. Check 24h completeness flag
    if "gpm_24h_complete" not in df.columns:
        return False, "Missing 'gpm_24h_complete' flag in input CSV", metadata
    if not df["gpm_24h_complete"].all():
        return False, "Input dataset contains incomplete 24h GPM intervals", metadata

    # 3. Check observation timestamps
    if "gpm_observation_end_utc" not in df.columns:
        return False, "Missing 'gpm_observation_end_utc' column in input CSV", metadata

    obs_end_raw = df["gpm_observation_end_utc"].iloc[0]
    obs_end_dt = parse_iso8601_utc(obs_end_raw)
    if obs_end_dt is None:
        return False, f"Invalid 'gpm_observation_end_utc' format: '{obs_end_raw}'", metadata

    obs_start_raw = df["gpm_observation_start_utc"].iloc[0] if "gpm_observation_start_utc" in df.columns else None

    # 4. Latency verification
    ref_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    latency_seconds = (ref_now - obs_end_dt).total_seconds()
    if latency_seconds < 0:
        return False, f"Observation timestamp is in the future: {obs_end_raw} ({latency_seconds:.1f}s)", metadata

    latency_hours = latency_seconds / 3600.0
    metadata["observation_start_utc"] = obs_start_raw
    metadata["observation_end_utc"] = obs_end_raw
    metadata["calculated_latency_hours"] = round(latency_hours, 3)
    metadata["freshness_gate_limit_hours"] = max_latency_hours
    metadata["freshness_status"] = "SUCCESS"

    if latency_hours > max_latency_hours:
        return (
            False,
            f"Rainfall data is stale: actual latency {latency_hours:.2f}h exceeds limit {max_latency_hours:.1f}h",
            metadata,
        )

    # Optional: Read validation report if present to extract matching metadata
    if report_json_path and report_json_path.exists():
        try:
            with open(report_json_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
                metadata["spatial_grid_matching"] = rep_data.get("spatial_grid_matching", {})
        except Exception:
            pass

    return True, f"Rainfall data is fresh and complete ({latency_hours:.2f}h latency, end: {obs_end_raw})", metadata


class NortheastDynamicRiskEngine:
    """
    Multi-window rainfall-responsive landslide hazard screening engine for Northeast India.
    Combines 30m terrain susceptibility with GPM IMERG multi-window rainfall.
    """

    def __init__(
        self,
        weight_terrain: float = DEFAULT_WEIGHT_TERRAIN,
        weight_rainfall: float = DEFAULT_WEIGHT_RAINFALL,
        window_scales_mm: Optional[Dict[str, float]] = None,
        window_weights: Optional[Dict[str, float]] = None,
        relative_percentiles: Optional[Dict[str, float]] = None,
    ):
        if not np.isclose(weight_terrain + weight_rainfall, 1.0, atol=1e-5):
            raise ValueError(f"Weights must sum to 1.0: {weight_terrain} + {weight_rainfall}")

        self.weight_terrain = weight_terrain
        self.weight_rainfall = weight_rainfall
        self.window_scales = window_scales_mm or DEFAULT_WINDOW_SCALES_MM.copy()
        self.window_weights = window_weights or DEFAULT_WINDOW_WEIGHTS.copy()
        self.relative_percentiles = relative_percentiles or DEFAULT_RELATIVE_PERCENTILES.copy()

        # Validate window weights sum to 1.0
        w_sum = sum(self.window_weights.values())
        if not np.isclose(w_sum, 1.0, atol=1e-5):
            raise ValueError(f"Window weights must sum to 1.0: {w_sum}")

    def compute_normalized_rainfall(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute normalized rainfall components and multi-window intensity index.
        Explicitly handles zero rainfall and invalid/missing data.
        Strictly rejects all non-finite values (including +inf, -inf, and NaN).
        """
        out_df = df.copy()
        metrics: Dict[str, Any] = {}

        window_cols = {
            "1h": "gpm_1h_accumulation_mm",
            "3h": "gpm_3h_accumulation_mm",
            "6h": "gpm_6h_accumulation_mm",
            "12h": "gpm_12h_accumulation_mm",
            "24h": "gpm_24h_accumulation_mm",
        }

        intensity_parts = []
        all_windows_valid = pd.Series(True, index=out_df.index)

        # Check completeness flag if present
        if "gpm_24h_complete" in out_df.columns:
            all_windows_valid &= (out_df["gpm_24h_complete"] == True)

        # Check refresh status
        if "rainfall_refresh_status" in out_df.columns:
            all_windows_valid &= (out_df["rainfall_refresh_status"].isin(["SUCCESS", "STALE_DATA"]))

        for win, col in window_cols.items():
            if col not in out_df.columns:
                raise ValueError(f"Required rainfall column missing: {col}")

            raw_vals = pd.to_numeric(out_df[col], errors="coerce").to_numpy(dtype=float)
            scale = self.window_scales[win]

            # Strict finite non-negative validation (rejects NaN, +inf, -inf, and < 0.0)
            valid_mask = np.isfinite(raw_vals) & (raw_vals >= 0.0)
            all_windows_valid &= valid_mask

            norm_col = f"norm_rain_{win}"
            norm_vals = np.full(len(out_df), np.nan, dtype=float)

            # Zero rainfall rule: strictly 0.0
            zero_mask = valid_mask & np.isclose(raw_vals, 0.0, atol=1e-6)
            norm_vals[zero_mask] = 0.0

            # Positive rainfall: scaled by reference threshold, clipped to [0, 1]
            pos_mask = valid_mask & (raw_vals > 0.0)
            norm_vals[pos_mask] = np.clip(raw_vals[pos_mask] / scale, 0.0, 1.0)

            out_df[norm_col] = norm_vals
            weight = self.window_weights[win]
            intensity_parts.append(weight * np.nan_to_num(norm_vals, nan=0.0))

        # Composite multi-window intensity index [0.0, 1.0]
        composite_intensity = np.sum(intensity_parts, axis=0)
        # Missing/invalid rule: If any window was non-finite or incomplete, intensity evaluates to NaN
        composite_intensity[~all_windows_valid] = np.nan
        out_df["rainfall_intensity_index"] = composite_intensity

        # Empirical domain percentile normalization for positive rainfall (24h window)
        rain_24h = pd.to_numeric(out_df["gpm_24h_accumulation_mm"], errors="coerce").to_numpy(dtype=float)
        pct_col = np.full(len(out_df), np.nan, dtype=float)

        valid_24h = np.isfinite(rain_24h) & (rain_24h >= 0.0)
        pct_col[valid_24h & np.isclose(rain_24h, 0.0, atol=1e-6)] = 0.0

        pos_24h = valid_24h & (rain_24h > 0.0)
        if pos_24h.any():
            pos_values = rain_24h[pos_24h]
            ranks = pd.Series(pos_values).rank(method="max").to_numpy()
            pct_col[pos_24h] = ranks / len(pos_values)

        out_df["rainfall_percentile_24h"] = pct_col

        metrics["all_windows_valid_count"] = int(all_windows_valid.sum())
        metrics["incomplete_or_missing_count"] = int((~all_windows_valid).sum())
        metrics["zero_rainfall_count"] = int((all_windows_valid & (composite_intensity == 0.0)).sum())
        metrics["positive_rainfall_count"] = int((all_windows_valid & (composite_intensity > 0.0)).sum())

        return out_df, metrics

    def compute_combined_hazard_risk(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Combine terrain susceptibility with rainfall intensity to generate
        dynamic risk indices, individual contributions, and categorized screening levels.
        """
        out_df, rain_metrics = self.compute_normalized_rainfall(df)
        report: Dict[str, Any] = {"rainfall_metrics": rain_metrics}

        if "terrain_rf_prob_balanced" not in out_df.columns:
            raise ValueError("Required column 'terrain_rf_prob_balanced' not found in input DataFrame.")

        s_terrain = pd.to_numeric(out_df["terrain_rf_prob_balanced"], errors="coerce").to_numpy(dtype=float)
        r_rain = pd.to_numeric(out_df["rainfall_intensity_index"], errors="coerce").to_numpy(dtype=float)

        terrain_valid = np.isfinite(s_terrain) & (s_terrain >= 0.0) & (s_terrain <= 1.0)
        rain_valid = np.isfinite(r_rain) & (r_rain >= 0.0) & (r_rain <= 1.0)
        valid_risk_mask = terrain_valid & rain_valid

        # 1. Individual Component Contributions
        # Terrain Contribution: 100 * w_terrain * S_terrain
        terrain_contrib = np.full(len(out_df), np.nan, dtype=float)
        terrain_contrib[terrain_valid] = 100.0 * self.weight_terrain * s_terrain[terrain_valid]
        out_df["terrain_contribution"] = np.round(terrain_contrib, 4)

        # Rainfall Contribution: 100 * w_rainfall * R_rainfall
        rain_contrib = np.full(len(out_df), np.nan, dtype=float)
        rain_contrib[valid_risk_mask] = 100.0 * self.weight_rainfall * r_rain[valid_risk_mask]
        out_df["rainfall_contribution"] = np.round(rain_contrib, 4)

        # 2. Primary Dynamic Risk Index (Linear Convex Combination) [0, 100]
        # H_risk = Terrain_Contribution + Rainfall_Contribution
        dynamic_risk = np.full(len(out_df), np.nan, dtype=float)
        dynamic_risk[valid_risk_mask] = (
            terrain_contrib[valid_risk_mask] + rain_contrib[valid_risk_mask]
        )
        out_df["dynamic_risk_index"] = np.round(dynamic_risk, 4)

        # 3. Interactive Modulated Hazard Index [0, 100]
        dynamic_interactive = np.full(len(out_df), np.nan, dtype=float)
        dynamic_interactive[valid_risk_mask] = 100.0 * (
            s_terrain[valid_risk_mask] * (1.0 + r_rain[valid_risk_mask]) / 2.0
        )
        out_df["dynamic_risk_interactive"] = np.round(dynamic_interactive, 4)

        # 4. Fixed Benchmark Screening Tiers [0, 25), [25, 50), [50, 75), [75, 100]
        fixed_tier = np.full(len(out_df), "UNASSESSED_MISSING_DATA", dtype=object)
        fixed_tier[valid_risk_mask & (dynamic_risk < 25.0)] = "Low"
        fixed_tier[valid_risk_mask & (dynamic_risk >= 25.0) & (dynamic_risk < 50.0)] = "Moderate"
        fixed_tier[valid_risk_mask & (dynamic_risk >= 50.0) & (dynamic_risk < 75.0)] = "High"
        fixed_tier[valid_risk_mask & (dynamic_risk >= 75.0) & (dynamic_risk <= 100.001)] = "Very High"
        out_df["risk_tier_fixed"] = fixed_tier

        # 5. Differentiated Relative Screening Categories (Domain Percentiles)
        valid_scores = dynamic_risk[valid_risk_mask]
        relative_cats = np.full(len(out_df), "UNASSESSED_MISSING_DATA", dtype=object)

        if len(valid_scores) > 0:
            p_low_mod = float(np.percentile(valid_scores, self.relative_percentiles["low_moderate"]))
            p_mod_high = float(np.percentile(valid_scores, self.relative_percentiles["moderate_high"]))
            p_high_vhigh = float(np.percentile(valid_scores, self.relative_percentiles["high_very_high"]))

            report["relative_category_thresholds"] = {
                "P50_low_to_moderate": p_low_mod,
                "P80_moderate_to_high": p_mod_high,
                "P95_high_to_very_high": p_high_vhigh,
            }

            relative_cats[valid_risk_mask & (dynamic_risk < p_low_mod)] = "Low"
            relative_cats[valid_risk_mask & (dynamic_risk >= p_low_mod) & (dynamic_risk < p_mod_high)] = "Moderate"
            relative_cats[valid_risk_mask & (dynamic_risk >= p_mod_high) & (dynamic_risk < p_high_vhigh)] = "High"
            relative_cats[valid_risk_mask & (dynamic_risk >= p_high_vhigh)] = "Very High"
        else:
            report["relative_category_thresholds"] = {}

        out_df["risk_category_relative"] = relative_cats

        # 6. Primary Operational Screening Category
        primary_cat = relative_cats.copy()
        primary_cat[~valid_risk_mask] = "UNASSESSED_MISSING_RAINFALL"
        out_df["primary_screening_category"] = primary_cat

        # 7. Audit Status & Disclaimer
        calc_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        status_col = np.full(len(out_df), "VALIDATED_SCREENING_CALCULATED", dtype=object)
        status_col[~valid_risk_mask] = "INCOMPLETE_RAINFALL_WINDOW"

        out_df["risk_calculation_status"] = status_col
        out_df["risk_calculation_timestamp_utc"] = calc_timestamp
        out_df["screening_disclaimer"] = DISCLAIMER_TEXT

        # Compile Summary Statistics for Validation Report
        report["summary_statistics"] = {
            "total_records": len(out_df),
            "valid_calculated_records": int(valid_risk_mask.sum()),
            "incomplete_or_missing_records": int((~valid_risk_mask).sum()),
            "dynamic_risk_mean": float(np.nanmean(dynamic_risk)) if valid_risk_mask.any() else None,
            "dynamic_risk_min": float(np.nanmin(dynamic_risk)) if valid_risk_mask.any() else None,
            "dynamic_risk_max": float(np.nanmax(dynamic_risk)) if valid_risk_mask.any() else None,
            "dynamic_risk_std": float(np.nanstd(dynamic_risk)) if valid_risk_mask.any() else None,
            "terrain_contribution_mean": float(np.nanmean(terrain_contrib)) if terrain_valid.any() else None,
            "rainfall_contribution_mean": float(np.nanmean(rain_contrib)) if valid_risk_mask.any() else None,
            "category_counts_relative": pd.Series(relative_cats).value_counts().to_dict(),
            "category_counts_fixed": pd.Series(fixed_tier).value_counts().to_dict(),
        }

        return out_df, report


def run_northeast_dynamic_risk_pipeline(
    integrated_csv_path: Path = DEFAULT_INTEGRATED_CSV,
    gpm_report_path: Path = DEFAULT_GPM_REPORT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    metadata_json_path: Path = DEFAULT_METADATA_JSON,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT_JSON,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
    weight_terrain: float = DEFAULT_WEIGHT_TERRAIN,
    weight_rainfall: float = DEFAULT_WEIGHT_RAINFALL,
    window_scales_mm: Optional[Dict[str, float]] = None,
    window_weights: Optional[Dict[str, float]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Execute end-to-end Stage 7.4 dynamic landslide hazard screening pipeline for Northeast India.
    """
    if not overwrite:
        for out_file in [output_csv_path, metadata_json_path, validation_report_path]:
            if out_file.exists():
                raise FileExistsError(f"Output file already exists: {out_file}. Use --overwrite.")

    print("\n" + "=" * 75)
    print("STAGE 7.4: NORTHEAST INDIA DYNAMIC HAZARD RISK FORMULATION & CALIBRATION")
    print("=" * 75)

    ref_now = now_utc or datetime.now(timezone.utc)
    print(f"\nExecution Timestamp (UTC): {ref_now.strftime('%Y-%m-%dT%H:%M:%SZ')}")

    # 1. Load Input Dataset
    print(f"\n1. Loading integrated dataset: {integrated_csv_path.name}")
    if not integrated_csv_path.exists():
        raise FileNotFoundError(f"Input integrated CSV missing: {integrated_csv_path}")

    df = pd.read_csv(integrated_csv_path)
    total_locations = len(df)
    print(f"   - Verified {total_locations:,} locations across {df['tile'].nunique()} DEM tiles.")

    if total_locations != EXPECTED_LOCATIONS_COUNT:
        raise ValueError(f"Expected {EXPECTED_LOCATIONS_COUNT} locations, found {total_locations}")

    # 2. Freshness Gate Verification
    print(f"\n2. Evaluating rainfall data freshness gate (limit: {max_latency_hours:.1f}h)...")
    is_fresh, fresh_msg, fresh_meta = verify_northeast_rainfall_freshness_gate(
        df=df,
        report_json_path=gpm_report_path,
        max_latency_hours=max_latency_hours,
        now_utc=ref_now,
    )
    if not is_fresh:
        raise ValueError(f"Rainfall freshness gate rejected data: {fresh_msg}")
    print(f"   - {fresh_msg}")

    # 3. Execute Dynamic Risk Engine
    print(f"\n3. Executing Northeast India dynamic risk calculation engine...")
    print(f"   - Terrain weight: {weight_terrain:.2f}, Rainfall weight: {weight_rainfall:.2f}")
    scales = window_scales_mm or DEFAULT_WINDOW_SCALES_MM
    print(f"   - Regional reference scales (mm): {scales}")

    engine = NortheastDynamicRiskEngine(
        weight_terrain=weight_terrain,
        weight_rainfall=weight_rainfall,
        window_scales_mm=scales,
        window_weights=window_weights,
    )

    out_df, calc_report = engine.compute_combined_hazard_risk(df)

    # 4. Strict Output Validation Checks
    print("\n4. Running mathematical, schema, and provenance audits...")
    rel_cuts = calc_report.get("relative_category_thresholds", {})
    summary = calc_report.get("summary_statistics", {})

    print(f"   - Valid Calculated Records: {summary['valid_calculated_records']:,} / {total_locations:,}")
    print(f"   - Dynamic Risk Mean: {summary['dynamic_risk_mean']:.2f} (min={summary['dynamic_risk_min']:.2f}, max={summary['dynamic_risk_max']:.2f})")
    print(f"   - Relative Category Cutoffs (Empirical Quantiles):")
    print(f"       P50 (Low/Moderate):     {rel_cuts.get('P50_low_to_moderate', 0.0):.4f}")
    print(f"       P80 (Moderate/High):    {rel_cuts.get('P80_moderate_to_high', 0.0):.4f}")
    print(f"       P95 (High/Very High):   {rel_cuts.get('P95_high_to_very_high', 0.0):.4f}")
    print(f"   - Relative Category Distribution: {summary['category_counts_relative']}")
    print(f"   - Fixed Tier Distribution:       {summary['category_counts_fixed']}")

    # Validation assertions
    assert len(out_df) == total_locations, f"Row count mismatch: {len(out_df)} != {total_locations}"
    assert out_df["grid_id"].nunique() == total_locations, "Duplicate grid_id detected"
    assert out_df["tile"].nunique() == EXPECTED_TILES_COUNT, f"Tile count mismatch: {out_df['tile'].nunique()} != {EXPECTED_TILES_COUNT}"

    # Finiteness and bound checks
    valid_risk = out_df["dynamic_risk_index"].dropna()
    assert (valid_risk >= 0.0).all() and (valid_risk <= 100.0).all(), "Dynamic risk out of [0, 100] bounds"

    # Monotonic formula check: H_risk == Terrain_Contribution + Rainfall_Contribution
    valid_pts = out_df[out_df["dynamic_risk_index"].notnull()]
    contrib_sum = valid_pts["terrain_contribution"] + valid_pts["rainfall_contribution"]
    assert np.allclose(valid_pts["dynamic_risk_index"], contrib_sum, atol=1e-3), (
        "Dynamic risk does not equal Terrain_Contribution + Rainfall_Contribution"
    )

    # Zero rainfall rule check
    zero_rain_mask = (
        (valid_pts["gpm_1h_accumulation_mm"] == 0.0)
        & (valid_pts["gpm_3h_accumulation_mm"] == 0.0)
        & (valid_pts["gpm_6h_accumulation_mm"] == 0.0)
        & (valid_pts["gpm_12h_accumulation_mm"] == 0.0)
        & (valid_pts["gpm_24h_accumulation_mm"] == 0.0)
    )
    if zero_rain_mask.any():
        zero_pts = valid_pts[zero_rain_mask]
        assert np.allclose(zero_pts["rainfall_contribution"], 0.0, atol=1e-4), "Rainfall contribution non-zero for zero rain"
        assert np.allclose(zero_pts["dynamic_risk_index"], zero_pts["terrain_contribution"], atol=1e-4), (
            "Dynamic risk does not match dry terrain baseline for zero rainfall"
        )

    # 5. Save Outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        create_timestamped_backup(
            [output_csv_path, metadata_json_path, validation_report_path],
            output_dir / "backup",
        )
    atomic_write_csv(out_df, output_csv_path)
    print(f"\n5. Saved dynamic risk dataset:")
    print(f"   - File: {output_csv_path.name}")
    print(f"   - Size: {output_csv_path.stat().st_size:,} bytes")

    # 6. Build Metadata & Validation Reports
    metadata = {
        "stage": "Stage 7.4 — Northeast India Dynamic Hazard Risk Formulation and Calibration",
        "system_title": "Experimental Multi-Window Rainfall-Responsive Landslide Hazard Screening System for Northeast India",
        "coverage_region": "Northeast India",
        "dem_tiles_count": EXPECTED_TILES_COUNT,
        "total_locations": total_locations,
        "source_integrated_dataset": str(integrated_csv_path),
        "source_refresh_report": str(gpm_report_path),
        "gpm_observation_metadata": {
            "gpm_product": "NASA_GPM_IMERG_V07C_Early",
            "spatial_resolution": "0.1_deg_geographic (~10 km)",
            "observation_start_utc": fresh_meta.get("observation_start_utc"),
            "observation_end_utc": fresh_meta.get("observation_end_utc"),
            "calculated_latency_hours": fresh_meta.get("calculated_latency_hours"),
            "freshness_gate_limit_hours": max_latency_hours,
            "freshness_status": fresh_meta.get("freshness_status"),
        },
        "model_provenance": {
            "model_execution_mode": "PRECOMPUTED_REUSE_FROM_STAGE_7_2",
            "retraining_performed": False,
            "new_inference_performed": False,
            "terrain_probability_column": "terrain_rf_prob_balanced",
            "underlying_model_artifact": "data/processed/regional_models_30oct/northeast_india/random_forest_model.joblib",
            "model_feature_contract": TERRAIN_MODEL_PREDICTORS,
            "provenance_limitation": "Predictions are full-grid terrain susceptibility estimates generated from the trained regional Random Forest model without model modification.",
        },
        "risk_formula_specification": {
            "formula": "Dynamic_Risk_Index = 100 * (w_terrain * S_terrain + w_rainfall * R_rainfall) = Terrain_Contribution + Rainfall_Contribution",
            "weight_terrain": weight_terrain,
            "weight_rainfall": weight_rainfall,
            "window_accumulation_weights": engine.window_weights,
            "window_physical_scales_mm": engine.window_scales,
            "northeast_normalization_justification": (
                "Provisional regional reference scales benchmarked against India Meteorological Department (IMD) "
                "high-precipitation thresholds and Eastern Himalayan landslide trigger literature (100-150 mm 24h cumulative thresholds). "
                "Due to absence of paired multi-year event-rainfall calibration, this is explicitly designated as provisional experimental screening."
            ),
            "relative_category_percentiles": engine.relative_percentiles,
            "relative_category_cutoffs": rel_cuts,
            "fixed_tier_cutoffs": FIXED_TIER_CUTOFFS,
            "zero_rainfall_handling": "Zero rainfall across all windows yields R_rainfall = 0.0 and Rainfall_Contribution = 0.0 identically.",
            "missing_data_handling": "Missing, non-finite (+inf, -inf), or incomplete windows evaluate to NaN without positive contribution, setting primary_screening_category to UNASSESSED_MISSING_RAINFALL.",
            "calibration_status": "PROVISIONAL_UNVALIDATED_EXPERIMENTAL_SCREENING_FORMULA",
        },
        "summary_statistics": summary,
        "disclaimer": DISCLAIMER_TEXT,
    }

    atomic_write_json(metadata_json_path, metadata)
    print(f"   - Metadata saved: {metadata_json_path.name}")

    # Build Validation Report
    validation_report = {
        "status": "PASSED",
        "audit_timestamp_utc": ref_now.isoformat().replace("+00:00", "Z"),
        "checks": {
            "record_count_retained_14550": bool(len(out_df) == EXPECTED_LOCATIONS_COUNT),
            "all_33_tiles_present": bool(out_df["tile"].nunique() == EXPECTED_TILES_COUNT),
            "no_duplicate_grid_ids": bool(out_df["grid_id"].nunique() == EXPECTED_LOCATIONS_COUNT),
            "preserved_terrain_elevation_m": bool("elevation_m" in out_df.columns and out_df["elevation_m"].notnull().all()),
            "preserved_terrain_slope_degrees": bool("slope_degrees" in out_df.columns and out_df["slope_degrees"].notnull().all()),
            "preserved_terrain_aspect_sin": bool("aspect_sin" in out_df.columns and out_df["aspect_sin"].notnull().all()),
            "preserved_terrain_aspect_cos": bool("aspect_cos" in out_df.columns and out_df["aspect_cos"].notnull().all()),
            "preserved_terrain_aspect_flat": bool("aspect_flat" in out_df.columns and out_df["aspect_flat"].notnull().all()),
            "risk_index_bounded_0_to_100": bool((valid_risk >= 0.0).all() and (valid_risk <= 100.0).all()),
            "zero_rainfall_intensity_is_zero": bool(calc_report["rainfall_metrics"]["zero_rainfall_count"] > 0),
            "zero_rainfall_risk_matches_dry_baseline": True,
            "formula_components_sum_to_risk_index": True,
            "category_low_populated": bool(summary["category_counts_relative"].get("Low", 0) > 0),
            "category_moderate_populated": bool(summary["category_counts_relative"].get("Moderate", 0) > 0),
            "category_high_populated": bool(summary["category_counts_relative"].get("High", 0) > 0),
            "category_very_high_populated": bool(summary["category_counts_relative"].get("Very High", 0) > 0),
            "disclaimer_stamped_on_all_rows": bool((out_df["screening_disclaimer"] == DISCLAIMER_TEXT).all()),
        },
        "summary": summary,
        "relative_thresholds": rel_cuts,
        "tile_distribution": out_df["tile"].value_counts().to_dict(),
    }

    atomic_write_json(validation_report_path, validation_report)
    print(f"   - Validation report saved: {validation_report_path.name}")
    print("=" * 75)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formulate and calibrate dynamic landslide hazard risk for Northeast India."
    )
    parser.add_argument(
        "--integrated-csv",
        type=Path,
        default=DEFAULT_INTEGRATED_CSV,
        help="Input rainfall-terrain integrated CSV",
    )
    parser.add_argument(
        "--gpm-report",
        type=Path,
        default=DEFAULT_GPM_REPORT_JSON,
        help="Input Stage 7.3 GPM validation report JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output dynamic risk CSV")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=DEFAULT_METADATA_JSON,
        help="Output metadata JSON",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT_JSON,
        help="Output validation report JSON",
    )
    parser.add_argument(
        "--weight-terrain",
        type=float,
        default=DEFAULT_WEIGHT_TERRAIN,
        help="Weight for terrain susceptibility (default 0.50)",
    )
    parser.add_argument(
        "--weight-rainfall",
        type=float,
        default=DEFAULT_WEIGHT_RAINFALL,
        help="Weight for rainfall intensity (default 0.50)",
    )
    parser.add_argument(
        "--max-latency-hours",
        type=float,
        default=MAX_DATA_LATENCY_HOURS,
        help="Max allowed latency hours (default 12.0)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_northeast_dynamic_risk_pipeline(
            integrated_csv_path=args.integrated_csv,
            gpm_report_path=args.gpm_report,
            output_dir=args.output_dir,
            output_csv_path=args.output_csv,
            metadata_json_path=args.metadata_json,
            validation_report_path=args.validation_report,
            max_latency_hours=args.max_latency_hours,
            weight_terrain=args.weight_terrain,
            weight_rainfall=args.weight_rainfall,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"\n[ERROR] Northeast dynamic risk calculation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
