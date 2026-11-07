"""
Western Ghats Regional Dynamic Landslide Hazard Risk Screening Engine.

Stage 6D Deliverable:
Integrates validated 30m terrain derivatives (via the approved Stage 5B Random
Forest baseline model) with dynamic NASA GPM IMERG Early Run V07C rolling rainfall
accumulations (1h, 3h, 6h, 12h, 24h) across all 14 Western Ghats DEM tiles and
1,582 locations (182 historical landslide events + 1,400 provisional background samples).

Methodological & Physical Considerations:
1. Model Provenance & Invariance:
   - Stage 6D reuses the precomputed terrain Random Forest predictions (terrain_rf_prob_balanced)
     directly from the approved Stage 6C integrated dataset (western_ghats_terrain_rainfall_integrated.csv).
   - No model retraining and no new model inference is performed in Stage 6D.
   - Predictions retain their previously documented in-sample baseline limitation (from Stage 5B/6C)
     and are not represented as out-of-fold spatial cross-validation results.
   - The five-feature terrain contract (['elevation_m', 'slope_degrees', 'aspect_sin', 'aspect_cos', 'aspect_flat'])
     is strictly preserved without modification.
2. Multi-Window Hydrometeorological Scaling & Strict Finite Validation:
   - Physical reference scales benchmark rolling accumulations:
     1h: 20 mm, 3h: 40 mm, 6h: 60 mm, 12h: 80 mm, 24h: 100 mm.
   - Strict Finite Validation: Only finite, non-negative accumulation values are valid.
     Positive infinity (+inf), negative infinity (-inf), NaN, and negative values (< 0.0) are
     strictly rejected and evaluate to NaN, producing unassessed risk results.
   - Zero rainfall rule: If precipitation is 0.0 mm across all windows, normalized rainfall intensity
     is 0.0 identically, simplifying risk to the dry terrain baseline.
   - Missing data safety: Missing, non-finite, negative, or incomplete rainfall intervals evaluate to NaN.
     No positive rainfall contribution is ever assigned to missing or invalid data.
3. Transparent Relative Hazard Screening:
   - Primary Dynamic Risk Index:
     H_risk = 100 * (w_terrain * S_terrain + w_rainfall * R_rainfall)
     Default configurable weights: w_terrain = 0.50, w_rainfall = 0.50.
   - Differentiated relative categories (Low, Moderate, High, Very High) derived from
     empirical regional quantiles (P50, P80, P95) of the active monitoring distribution.
   - Fixed benchmark tiers ([0, 25), [25, 50), [50, 75), [75, 100]) provided in parallel.
4. Independent Data Freshness Gate:
   - Gated execution: Independently parses the actual observation_end_utc timestamp from refresh metadata
     and calculates data latency against current UTC time.
   - Data older than the configured 12-hour limit, missing timestamps, invalid formats, or incomplete
     windows are strictly rejected. An old SUCCESS status file cannot pass indefinitely.
5. Historical Integrity & Disclaimer:
   - Current November 2026 satellite rainfall is an operational situational snapshot and is
     NOT treated as historical event-date rainfall for landslides recorded between 2007 and 2021.
   - This provisional relative screening index is NOT a calibrated probability, NOT a validated
     forecasting model, and NOT an official government early-warning system.
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


# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INTEGRATED_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_integrated"
    / "western_ghats_terrain_rainfall_integrated.csv"
)

DEFAULT_REFRESH_STATUS_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall"
    / "western_ghats_refresh"
    / "metadata"
    / "latest_refresh_status.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_dynamic_risk"
)

DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_hazard_risk.csv"
DEFAULT_METADATA_JSON = DEFAULT_OUTPUT_DIR / "hazard_risk_metadata.json"
DEFAULT_VALIDATION_REPORT_JSON = DEFAULT_OUTPUT_DIR / "hazard_risk_validation_report.json"

EXPECTED_LOCATIONS_COUNT = 1582
EXPECTED_HISTORICAL_EVENTS = 182
EXPECTED_BACKGROUND_SAMPLES = 1400
EXPECTED_TILES_COUNT = 14

TERRAIN_MODEL_PREDICTORS = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

# Configurable physical reference thresholds (mm)
DEFAULT_WINDOW_SCALES_MM: Dict[str, float] = {
    "1h": 20.0,
    "3h": 40.0,
    "6h": 60.0,
    "12h": 80.0,
    "24h": 100.0,
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
DISCLAIMER_TEXT = "EXPERIMENTAL_PROVISIONAL_SCREENING_ONLY_NOT_OFFICIAL_WARNING"


# =====================================================================
# FRESHNESS & DATA AUDIT GATES
# =====================================================================

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


def verify_rainfall_freshness_gate(
    status_path: Path,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify that the latest rainfall refresh status is SUCCESS, complete, and fresh.
    Independently parses observation_end_utc from status metadata, calculates latency
    relative to current UTC time (or provided now_utc), and validates within max_latency_hours.
    Rejects stale, missing, invalid timestamps or incomplete windows.
    An old SUCCESS status file cannot pass indefinitely.
    """
    if not status_path.exists():
        return False, f"Refresh status file not found: {status_path}", {}

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse refresh status JSON: {e}", {}

    status_str = status_data.get("status", "")
    obs_end_raw = status_data.get("observation_end_utc")

    # 1. Verify status string
    if status_str != "SUCCESS":
        return False, f"Rainfall status is '{status_str}' (expected 'SUCCESS')", status_data

    # 2. Strictly validate completeness flag: must be actual JSON boolean True
    if "is_24h_complete" not in status_data:
        return (
            False,
            "Missing 'is_24h_complete' flag in refresh status metadata (must be JSON boolean true)",
            status_data,
        )

    is_24h_complete_val = status_data["is_24h_complete"]
    if not isinstance(is_24h_complete_val, bool) or is_24h_complete_val is not True:
        val_type = type(is_24h_complete_val).__name__
        return (
            False,
            f"Invalid 'is_24h_complete' flag: {is_24h_complete_val!r} (type: {val_type}). Must be actual boolean true",
            status_data,
        )

    # 3. Verify observation_end_utc presence and validity
    if not obs_end_raw:
        return False, "Missing 'observation_end_utc' in refresh status metadata", status_data

    obs_end_dt = parse_iso8601_utc(obs_end_raw)
    if obs_end_dt is None:
        return False, f"Invalid 'observation_end_utc' timestamp format: '{obs_end_raw}'", status_data

    # 4. Independently calculate data latency against current UTC time (or now_utc)
    reference_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)

    actual_latency_seconds = (reference_now - obs_end_dt).total_seconds()
    actual_latency_hours = actual_latency_seconds / 3600.0

    # Strictly reject all future observation timestamps without clock-skew tolerance
    if actual_latency_seconds < 0:
        return (
            False,
            f"Observation timestamp is in the future: {obs_end_raw} (latency: {actual_latency_seconds:.1f}s)",
            status_data,
        )

    # Store calculated latency in returned status dictionary for audit trail
    status_data["calculated_latency_hours"] = round(actual_latency_hours, 3)

    if actual_latency_hours > max_latency_hours:
        return (
            False,
            f"Rainfall data is stale: actual latency {actual_latency_hours:.2f}h exceeds limit {max_latency_hours:.1f}h (observation end: {obs_end_raw})",
            status_data,
        )

    return True, f"Rainfall data is fresh and complete ({actual_latency_hours:.2f}h latency, end: {obs_end_raw})", status_data


# =====================================================================
# DYNAMIC RISK SCREENING ENGINE
# =====================================================================

class WesternGhatsDynamicRiskEngine:
    """
    Experimental rainfall-responsive landslide hazard screening engine.
    Combines 30m terrain susceptibility with GPM IMERG multi-window rainfall.
    Designed for Western Ghats regional coverage with full modularity.
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

        # 1. Normalize individual accumulation windows
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

            # Positive rainfall: scaled by benchmark threshold, clipped to [0, 1]
            pos_mask = valid_mask & (raw_vals > 0.0)
            norm_vals[pos_mask] = np.clip(raw_vals[pos_mask] / scale, 0.0, 1.0)

            out_df[norm_col] = norm_vals
            weight = self.window_weights[win]
            intensity_parts.append(weight * np.nan_to_num(norm_vals, nan=0.0))

        # 2. Multi-window composite intensity index
        composite_intensity = np.sum(intensity_parts, axis=0)
        # Missing/invalid rule: If any window was non-finite or incomplete, intensity evaluates to NaN
        composite_intensity[~all_windows_valid] = np.nan
        out_df["rainfall_intensity_index"] = composite_intensity

        # 3. Empirical domain percentile normalization for positive rainfall (24h window)
        # Reference Population: Active Western Ghats monitoring locations in current snapshot
        rain_24h = pd.to_numeric(out_df["gpm_24h_accumulation_mm"], errors="coerce").to_numpy(dtype=float)
        pct_col = np.full(len(out_df), np.nan, dtype=float)

        valid_24h = np.isfinite(rain_24h) & (rain_24h >= 0.0)
        pct_col[valid_24h & np.isclose(rain_24h, 0.0, atol=1e-6)] = 0.0

        pos_24h = valid_24h & (rain_24h > 0.0)
        if pos_24h.any():
            pos_values = rain_24h[pos_24h]
            # Calculate rank-based percentile among positive observations
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
        dynamic risk indices and categorized screening levels.
        """
        out_df, rain_metrics = self.compute_normalized_rainfall(df)
        report: Dict[str, Any] = {"rainfall_metrics": rain_metrics}

        # Validate terrain probability column
        if "terrain_rf_prob_balanced" not in out_df.columns:
            raise ValueError("Required column 'terrain_rf_prob_balanced' not found in input DataFrame.")

        s_terrain = pd.to_numeric(out_df["terrain_rf_prob_balanced"], errors="coerce").to_numpy(dtype=float)
        r_rain = pd.to_numeric(out_df["rainfall_intensity_index"], errors="coerce").to_numpy(dtype=float)

        # Valid mask for risk calculation: both terrain and rainfall must be finite and within [0, 1]
        valid_risk_mask = (
            np.isfinite(s_terrain)
            & (s_terrain >= 0.0)
            & (s_terrain <= 1.0)
            & np.isfinite(r_rain)
            & (r_rain >= 0.0)
            & (r_rain <= 1.0)
        )

        # 1. Primary Dynamic Risk Index (Linear Convex Combination) [0, 100]
        dynamic_risk = np.full(len(out_df), np.nan, dtype=float)
        dynamic_risk[valid_risk_mask] = 100.0 * (
            self.weight_terrain * s_terrain[valid_risk_mask]
            + self.weight_rainfall * r_rain[valid_risk_mask]
        )
        out_df["dynamic_risk_index"] = dynamic_risk

        # 2. Interactive Modulated Hazard Index [0, 100]
        # Models terrain susceptibility amplified by current rainfall trigger
        dynamic_interactive = np.full(len(out_df), np.nan, dtype=float)
        dynamic_interactive[valid_risk_mask] = 100.0 * (
            s_terrain[valid_risk_mask] * (1.0 + r_rain[valid_risk_mask]) / 2.0
        )
        out_df["dynamic_risk_interactive"] = dynamic_interactive

        # 3. Terrain Baseline Screening Tier (Fallback for unassessed/dry conditions)
        terrain_tier = np.full(len(out_df), "Unassessed", dtype=object)
        terrain_valid = np.isfinite(s_terrain) & (s_terrain >= 0.0) & (s_terrain <= 1.0)
        terrain_tier[terrain_valid & (s_terrain < 0.25)] = "Low"
        terrain_tier[terrain_valid & (s_terrain >= 0.25) & (s_terrain < 0.50)] = "Moderate"
        terrain_tier[terrain_valid & (s_terrain >= 0.50) & (s_terrain < 0.75)] = "High"
        terrain_tier[terrain_valid & (s_terrain >= 0.75)] = "Very High"
        out_df["terrain_baseline_screening_tier"] = terrain_tier

        # 4. Fixed Benchmark Screening Tiers [0, 25), [25, 50), [50, 75), [75, 100]
        fixed_tier = np.full(len(out_df), "UNASSESSED_MISSING_DATA", dtype=object)
        fixed_tier[valid_risk_mask & (dynamic_risk < 25.0)] = "Low"
        fixed_tier[valid_risk_mask & (dynamic_risk >= 25.0) & (dynamic_risk < 50.0)] = "Moderate"
        fixed_tier[valid_risk_mask & (dynamic_risk >= 50.0) & (dynamic_risk < 75.0)] = "High"
        fixed_tier[valid_risk_mask & (dynamic_risk >= 75.0) & (dynamic_risk < 100.001)] = "Very High"
        out_df["risk_tier_fixed"] = fixed_tier

        # 5. Differentiated Relative Screening Categories (Domain Percentiles)
        # Reference Population: Valid locations in current Western Ghats observation domain
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
            "category_counts_relative": pd.Series(relative_cats).value_counts().to_dict(),
            "category_counts_fixed": pd.Series(fixed_tier).value_counts().to_dict(),
        }

        return out_df, report


# =====================================================================
# INTEGRATION & EXECUTION PIPELINE
# =====================================================================

def run_western_ghats_dynamic_risk_pipeline(
    integrated_csv_path: Path = DEFAULT_INTEGRATED_CSV,
    status_json_path: Path = DEFAULT_REFRESH_STATUS_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
    weight_terrain: float = DEFAULT_WEIGHT_TERRAIN,
    weight_rainfall: float = DEFAULT_WEIGHT_RAINFALL,
) -> Dict[str, Any]:
    """
    Execute end-to-end Stage 6D dynamic landslide hazard screening pipeline.
    Independently validates data freshness from observation_end_utc, executes risk engine,
    performs audits, and saves outputs.
    """
    print("\n" + "=" * 75)
    print("STAGE 6D: WESTERN GHATS DYNAMIC LANDSLIDE HAZARD RISK PREDICTION")
    print("=" * 75)

    # 1. Freshness Gate Verification
    print(f"\n1. Verifying GPM rainfall freshness gate: {status_json_path.name}")
    is_fresh, gate_message, status_data = verify_rainfall_freshness_gate(
        status_json_path, max_latency_hours=max_latency_hours, now_utc=now_utc
    )
    print(f"   Status: {status_data.get('status', 'UNKNOWN')}")
    print(f"   Observation End: {status_data.get('observation_end_utc', 'N/A')}")
    print(f"   Calculated Latency: {status_data.get('calculated_latency_hours', 'N/A')} hours (limit: {max_latency_hours:.1f}h)")
    print(f"   Gate Result: {gate_message}")

    if not is_fresh:
        raise RuntimeError(f"GPM rainfall freshness gate failed: {gate_message}")

    # 2. Load Integrated Dataset
    print(f"\n2. Loading Stage 6C integrated dataset: {integrated_csv_path.name}")
    if not integrated_csv_path.exists():
        raise FileNotFoundError(f"Integrated dataset not found: {integrated_csv_path}")

    df = pd.read_csv(integrated_csv_path, low_memory=False)
    print(f"   Loaded {len(df)} records across {len(df['tile'].unique())} DEM tiles.")

    if len(df) != EXPECTED_LOCATIONS_COUNT:
        raise ValueError(f"Expected {EXPECTED_LOCATIONS_COUNT} records; found {len(df)}")

    # 3. Execute Dynamic Risk Calculation Engine
    print("\n3. Computing dynamic multi-window hazard screening index...")
    engine = WesternGhatsDynamicRiskEngine(
        weight_terrain=weight_terrain,
        weight_rainfall=weight_rainfall,
    )
    risk_df, report = engine.compute_combined_hazard_risk(df)

    # 4. Perform Rigorous Quality Audit
    print("\n4. Running validation checks on output dataset...")
    validation_checks: Dict[str, bool] = {}

    validation_checks["record_count_retained_1582"] = len(risk_df) == EXPECTED_LOCATIONS_COUNT
    validation_checks["all_14_tiles_present"] = len(risk_df["tile"].unique()) == EXPECTED_TILES_COUNT
    validation_checks["no_duplicate_sample_ids"] = not risk_df["sample_id"].duplicated().any()

    # Terrain feature contract preservation
    for feat in TERRAIN_MODEL_PREDICTORS:
        validation_checks[f"preserved_terrain_{feat}"] = (
            feat in risk_df.columns and not risk_df[feat].isna().any()
        )

    # Risk bounds check
    valid_scores = risk_df.loc[risk_df["dynamic_risk_index"].notna(), "dynamic_risk_index"]
    validation_checks["risk_index_bounded_0_to_100"] = bool(
        (valid_scores >= 0.0).all() and (valid_scores <= 100.0).all()
    )

    # Zero rainfall handling check
    zero_rain_mask = (
        (risk_df["gpm_1h_accumulation_mm"] == 0.0)
        & (risk_df["gpm_3h_accumulation_mm"] == 0.0)
        & (risk_df["gpm_6h_accumulation_mm"] == 0.0)
        & (risk_df["gpm_12h_accumulation_mm"] == 0.0)
        & (risk_df["gpm_24h_accumulation_mm"] == 0.0)
    )
    zero_rain_df = risk_df[zero_rain_mask]
    if len(zero_rain_df) > 0:
        validation_checks["zero_rainfall_intensity_is_zero"] = bool(
            (zero_rain_df["rainfall_intensity_index"] == 0.0).all()
        )
        expected_dry_risk = 100.0 * (weight_terrain * zero_rain_df["terrain_rf_prob_balanced"])
        validation_checks["zero_rainfall_risk_matches_dry_baseline"] = bool(
            np.allclose(zero_rain_df["dynamic_risk_index"], expected_dry_risk, atol=1e-5)
        )
    else:
        validation_checks["zero_rainfall_intensity_is_zero"] = True
        validation_checks["zero_rainfall_risk_matches_dry_baseline"] = True

    # Relative category differentiation check
    cat_counts = risk_df["risk_category_relative"].value_counts()
    for cat in ["Low", "Moderate", "High", "Very High"]:
        validation_checks[f"category_{cat.lower()}_populated"] = bool(cat_counts.get(cat, 0) > 0)

    # Freshness metadata preservation check
    validation_checks["disclaimer_stamped_on_all_rows"] = bool(
        (risk_df["screening_disclaimer"] == DISCLAIMER_TEXT).all()
    )

    all_passed = all(validation_checks.values())
    print(f"   Validation Status: {'PASSED' if all_passed else 'FAILED'}")
    for chk, res in validation_checks.items():
        print(f"   - {chk:<45}: {'OK' if res else 'FAIL'}")

    if not all_passed:
        raise RuntimeError("One or more validation checks failed in Stage 6D!")

    # 5. Save Outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "western_ghats_dynamic_hazard_risk.csv"
    output_meta_path = output_dir / "hazard_risk_metadata.json"
    output_report_path = output_dir / "hazard_risk_validation_report.json"

    try:
        disp_dir = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        disp_dir = output_dir
    print(f"\n5. Saving production outputs to: {disp_dir}")

    risk_df.to_csv(output_csv_path, index=False)
    print(f"   - Location Risk CSV: {output_csv_path.name} ({len(risk_df)} rows, {len(risk_df.columns)} columns)")

    # Metadata JSON
    metadata = {
        "stage": "Stage 6D — Dynamic Landslide Hazard Risk Prediction",
        "system_title": "Experimental Multi-Window Rainfall-Responsive Landslide Hazard Screening System",
        "coverage_region": "Western Ghats, India",
        "dem_tiles_count": EXPECTED_TILES_COUNT,
        "total_locations": EXPECTED_LOCATIONS_COUNT,
        "historical_events_count": EXPECTED_HISTORICAL_EVENTS,
        "background_samples_count": EXPECTED_BACKGROUND_SAMPLES,
        "source_integrated_dataset": str(integrated_csv_path),
        "source_refresh_status": str(status_json_path),
        "gpm_observation_metadata": {
            "gpm_product": status_data.get("gpm_product", "NASA_GPM_IMERG_V07C_Early"),
            "spatial_resolution": status_data.get("spatial_resolution", "0.1_deg_geographic"),
            "observation_start_utc": status_data.get("observation_start_utc"),
            "observation_end_utc": status_data.get("observation_end_utc"),
            "reported_latency_hours": status_data.get("data_latency_hours"),
            "independently_calculated_latency_hours": status_data.get("calculated_latency_hours"),
            "freshness_gate_limit_hours": max_latency_hours,
            "freshness_status": status_data.get("status"),
        },
        "model_provenance": {
            "model_execution_mode": "PRECOMPUTED_REUSE_FROM_STAGE_6C",
            "retraining_performed": False,
            "new_inference_performed": False,
            "terrain_probability_column": "terrain_rf_prob_balanced",
            "source_dataset": str(integrated_csv_path),
            "underlying_model_artifact": "data/processed/western_ghats_regional_model/western_ghats_regional_terrain_random_forest.joblib",
            "model_architecture": "RandomForestClassifier(n_estimators=300, min_samples_split=4, min_samples_leaf=2, class_weight='balanced')",
            "model_feature_contract": TERRAIN_MODEL_PREDICTORS,
            "provenance_limitation": "Predictions are full-model in-sample outputs evaluated on the training dataset (Stage 5B/6C); they represent baseline terrain propensity, not independent spatial out-of-fold cross-validation.",
        },
        "risk_formula_specification": {
            "formula": "Dynamic_Risk_Index = 100 * (w_terrain * S_terrain + w_rainfall * R_rainfall)",
            "weight_terrain": weight_terrain,
            "weight_rainfall": weight_rainfall,
            "window_accumulation_weights": engine.window_weights,
            "window_physical_scales_mm": engine.window_scales,
            "relative_category_percentiles": engine.relative_percentiles,
            "relative_category_cutoffs": report.get("relative_category_thresholds", {}),
            "fixed_tier_cutoffs": FIXED_TIER_CUTOFFS,
            "zero_rainfall_handling": "Zero rainfall across all windows yields R_rainfall = 0.0 identically.",
            "missing_data_handling": "Missing, non-finite (+inf, -inf), or incomplete windows evaluate to NaN without positive contribution.",
            "calibration_status": "UNVALIDATED_EXPERIMENTAL_SCREENING_FORMULA",
        },
        "summary_statistics": report["summary_statistics"],
        "disclaimer": DISCLAIMER_TEXT,
    }
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"   - Metadata JSON:     {output_meta_path.name}")

    # Validation Report JSON
    val_report = {
        "status": "PASSED" if all_passed else "FAILED",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": validation_checks,
        "summary": report["summary_statistics"],
        "relative_thresholds": report.get("relative_category_thresholds", {}),
        "tile_distribution": risk_df["tile"].value_counts().to_dict(),
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    print(f"   - Validation Report: {output_report_path.name}")

    print("\n" + "=" * 75)
    print("STAGE 6D DYNAMIC HAZARD RISK PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 75)

    return {
        "status": "SUCCESS",
        "csv_path": output_csv_path,
        "metadata_path": output_meta_path,
        "report_path": output_report_path,
        "checks": validation_checks,
        "summary": report["summary_statistics"],
    }


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Western Ghats Dynamic Landslide Hazard Risk Prediction")
    parser.add_argument("--integrated-csv", type=Path, default=DEFAULT_INTEGRATED_CSV, help="Path to Stage 6C CSV")
    parser.add_argument("--status-json", type=Path, default=DEFAULT_REFRESH_STATUS_JSON, help="Path to refresh status")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--max-latency", type=float, default=MAX_DATA_LATENCY_HOURS, help="Max latency hours")
    parser.add_argument("--weight-terrain", type=float, default=DEFAULT_WEIGHT_TERRAIN, help="Terrain weight")
    parser.add_argument("--weight-rainfall", type=float, default=DEFAULT_WEIGHT_RAINFALL, help="Rainfall weight")

    args = parser.parse_args()

    try:
        run_western_ghats_dynamic_risk_pipeline(
            integrated_csv_path=args.integrated_csv,
            status_json_path=args.status_json,
            output_dir=args.output_dir,
            max_latency_hours=args.max_latency,
            weight_terrain=args.weight_terrain,
            weight_rainfall=args.weight_rainfall,
        )
    except Exception as e:
        print(f"\n[ERROR] Stage 6D execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
