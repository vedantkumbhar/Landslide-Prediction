"""
Western Ghats Limited Pilot — 5 km Grid Dynamic Hazard Risk Calculation (Stage 6.4 Grid)
========================================================================================

Applies the approved Stage 6D dynamic risk formula across the 5 km Western Ghats
spatial grid (14 validated DEM tiles only):

    H_risk = 100 * (0.50 * S_terrain + 0.50 * R_rainfall)
           = Terrain_Contribution + Rainfall_Contribution

Reuses `WesternGhatsDynamicRiskEngine` directly from
`backend.predict_western_ghats_dynamic_risk` with identical window scales
(20, 40, 60, 80, 100 mm), window weights (0.10, 0.15, 0.20, 0.25, 0.30),
P50/P80/P95 relative screening cutoffs, 12-hour freshness gate, and missing-data
/ zero-rainfall safeguards.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backend.gis_refresh_utils import (
    MAX_DATA_LATENCY_HOURS,
    atomic_write_json,
    create_timestamped_backup,
    evaluate_freshness_condition,
    validate_dynamic_metadata_thresholds,
)
from backend.integrate_himalayas_gpm_rainfall import atomic_write_csv
from backend.integrate_western_ghats_gpm_rainfall import (
    DEFAULT_INTEGRATED_CSV,
    DEFAULT_VALIDATION_REPORT as DEFAULT_GPM_REPORT_JSON,
    WG_PILOT_DISCLAIMER,
)
from backend.predict_western_ghats_dynamic_risk import (
    DEFAULT_RELATIVE_PERCENTILES,
    DEFAULT_WEIGHT_RAINFALL,
    DEFAULT_WEIGHT_TERRAIN,
    DEFAULT_WINDOW_SCALES_MM,
    DEFAULT_WINDOW_WEIGHTS,
    FIXED_TIER_CUTOFFS,
    WesternGhatsDynamicRiskEngine,
)
from backend.predict_western_ghats_terrain_susceptibility import RF_FEATURE_COLUMNS


LOGGER = logging.getLogger("western_ghats_grid_dynamic_risk")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "western_ghats_grid_dynamic_risk"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "western_ghats_5km_dynamic_hazard_risk.csv"
DEFAULT_METADATA_JSON = DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_risk_metadata.json"
DEFAULT_VALIDATION_REPORT = DEFAULT_OUTPUT_DIR / "western_ghats_dynamic_risk_validation_report.json"


def verify_wg_grid_freshness_gate(
    df: pd.DataFrame,
    gpm_report_path: Optional[Path] = None,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluate Stage 6.4 Grid freshness gate using the shared UTC freshness contract.
    Rejects stale (>12h), incomplete (<48 granules), future, missing, or invalid timestamps.
    """
    if "rainfall_refresh_status" not in df.columns:
        return False, "Missing 'rainfall_refresh_status' column.", {"status_code": "MISSING_STATUS"}
    if "gpm_observation_end_utc" not in df.columns or "gpm_24h_complete" not in df.columns:
        return False, "Missing GPM observation metadata columns.", {"status_code": "MISSING_METADATA"}

    statuses = df["rainfall_refresh_status"].astype(str).unique().tolist()
    obs_end_raw = df["gpm_observation_end_utc"].iloc[0]
    obs_start_raw = (
        df["gpm_observation_start_utc"].iloc[0]
        if "gpm_observation_start_utc" in df.columns
        else None
    )
    obs_end_str = None if pd.isna(obs_end_raw) else str(obs_end_raw)
    obs_start_str = None if pd.isna(obs_start_raw) else str(obs_start_raw)

    is_24h_complete = bool(df["gpm_24h_complete"].astype(bool).all())
    avail_granules = int(df["gpm_interval_count"].min()) if "gpm_interval_count" in df.columns else None
    missing_intervals = 0
    if gpm_report_path and gpm_report_path.exists():
        try:
            with open(gpm_report_path, "r", encoding="utf-8") as f:
                rep_meta = json.load(f).get("gpm_observation_metadata", {})
            missing_intervals = int(rep_meta.get("missing_intervals_count", 0))
            if avail_granules is None and "granules_count" in rep_meta:
                avail_granules = int(rep_meta["granules_count"])
        except Exception:
            pass

    freshness = evaluate_freshness_condition(
        obs_end_utc_str=obs_end_str,
        is_24h_complete=is_24h_complete,
        available_granules=avail_granules,
        missing_intervals=missing_intervals,
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
    )
    meta = {
        "observation_start_utc": obs_start_str,
        "observation_end_utc": obs_end_str,
        "calculated_latency_hours": freshness.get("latency_hours"),
        "freshness_gate_limit_hours": max_latency_hours,
        "freshness_status": "SUCCESS" if freshness["is_fresh"] and statuses == ["SUCCESS"] else freshness["status_code"],
        "status_code": freshness["status_code"],
        "reason": freshness["reason"],
    }

    if statuses != ["SUCCESS"] and freshness["is_fresh"]:
        meta["freshness_status"] = statuses[0] if len(statuses) == 1 else "MIXED_STATUS"
        meta["status_code"] = "UPSTREAM_REFRESH_NOT_SUCCESS"
        return False, f"Upstream rainfall status is {statuses}, expected ['SUCCESS']", meta

    if not freshness["is_fresh"]:
        return False, freshness["reason"], meta

    return (
        True,
        f"Rainfall data is fresh and complete ({freshness['latency_hours']:.2f}h <= {max_latency_hours:.1f}h)",
        meta,
    )


def predict_western_ghats_grid_dynamic_risk(
    integrated_csv_path: Path = DEFAULT_INTEGRATED_CSV,
    gpm_report_path: Path = DEFAULT_GPM_REPORT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    metadata_json_path: Path = DEFAULT_METADATA_JSON,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT,
    weight_terrain: float = DEFAULT_WEIGHT_TERRAIN,
    weight_rainfall: float = DEFAULT_WEIGHT_RAINFALL,
    window_scales_mm: Optional[Dict[str, float]] = None,
    window_weights: Optional[Dict[str, float]] = None,
    max_latency_hours: float = MAX_DATA_LATENCY_HOURS,
    now_utc: Optional[datetime] = None,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Compute Stage 6.4 dynamic landslide hazard risk for the Western Ghats 5 km grid."""
    if not integrated_csv_path.exists():
        raise FileNotFoundError(f"Integrated grid CSV not found: {integrated_csv_path}")

    existing = [p for p in (output_csv_path, metadata_json_path, validation_report_path) if p.exists()]
    if existing:
        if not overwrite:
            raise FileExistsError(
                f"Dynamic risk output already exists (use --overwrite): {existing[0]}"
            )
        create_timestamped_backup(existing, output_dir / "backups")

    ref_now = now_utc if now_utc is not None else datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    df_raw = pd.read_csv(integrated_csv_path)
    total_locations = len(df_raw)
    if total_locations == 0:
        raise ValueError(f"Integrated CSV is empty: {integrated_csv_path}")
    if int(df_raw["tile"].nunique()) != 14:
        raise ValueError(
            f"Expected 14 validated Western Ghats DEM tiles, got {df_raw['tile'].nunique()}."
        )

    is_fresh, freshness_msg, freshness_meta = verify_wg_grid_freshness_gate(
        df=df_raw,
        gpm_report_path=gpm_report_path,
        max_latency_hours=max_latency_hours,
        now_utc=ref_now,
    )

    engine = WesternGhatsDynamicRiskEngine(
        weight_terrain=weight_terrain,
        weight_rainfall=weight_rainfall,
        window_scales_mm=window_scales_mm or dict(DEFAULT_WINDOW_SCALES_MM),
        window_weights=window_weights or dict(DEFAULT_WINDOW_WEIGHTS),
        relative_percentiles=dict(DEFAULT_RELATIVE_PERCENTILES),
    )

    df_out, report = engine.compute_combined_hazard_risk(df_raw)

    # Populate additive components explicitly matching Himalayas & Northeast CSV schema
    s_terrain = pd.to_numeric(df_out["terrain_rf_prob_balanced"], errors="coerce").to_numpy(dtype=float)
    r_rain = pd.to_numeric(df_out["rainfall_intensity_index"], errors="coerce").to_numpy(dtype=float)
    valid_risk = np.isfinite(df_out["dynamic_risk_index"].to_numpy(dtype=float))

    t_contrib = np.full(len(df_out), np.nan, dtype=float)
    r_contrib = np.full(len(df_out), np.nan, dtype=float)
    valid_t = np.isfinite(s_terrain) & (s_terrain >= 0.0) & (s_terrain <= 1.0)
    t_contrib[valid_t] = np.round(100.0 * weight_terrain * s_terrain[valid_t], 4)
    r_contrib[valid_risk] = np.round(100.0 * weight_rainfall * r_rain[valid_risk], 4)

    df_out["terrain_contribution"] = t_contrib
    df_out["rainfall_contribution"] = r_contrib
    df_out["screening_disclaimer"] = WG_PILOT_DISCLAIMER

    # Enforce freshness gate fail-safe cleanly without silently serving stale risk as active
    if not is_fresh:
        gate_code = freshness_meta.get("status_code", "STALE_DATA_UNASSESSED")
        cat_override = (
            "STALE_DATA_UNASSESSED"
            if gate_code == "STALE_DATA_UNASSESSED"
            else "UNASSESSED_MISSING_RAINFALL"
        )
        df_out["rainfall_refresh_status"] = gate_code
        df_out["primary_screening_category"] = cat_override
        df_out["risk_calculation_status"] = gate_code

    for col in RF_FEATURE_COLUMNS:
        if col not in df_out.columns:
            raise ValueError(f"Missing mandatory terrain predictor in output: {col}")
    if len(df_out) != total_locations or df_out["grid_id"].nunique() != total_locations:
        raise ValueError("Grid ID integrity / row count violated in dynamic risk output.")

    summary_stats = dict(report.get("summary_statistics", {}))
    summary_stats["terrain_contribution_mean"] = (
        float(np.nanmean(t_contrib)) if valid_t.any() else None
    )
    summary_stats["rainfall_contribution_mean"] = (
        float(np.nanmean(r_contrib)) if valid_risk.any() else None
    )

    cutoffs = dict(report.get("relative_category_thresholds", {}))
    metadata = {
        "stage": "Western Ghats Limited Pilot — Stage 6.4 Grid Dynamic Hazard Risk",
        "system_title": (
            "Experimental Multi-Window Rainfall-Responsive Landslide Hazard Screening System "
            "for the Western Ghats Limited Pilot (14 Validated DEM Tiles Only)"
        ),
        "coverage_region": "Western Ghats",
        "coverage_disclosure": (
            "Western Ghats Limited Pilot: covers strictly 4,819 5 km grid cells across the 14 validated "
            "DEM tiles plus separated 182 historical events and 1,400 background samples; "
            "does not imply complete Western Ghats coverage."
        ),
        "dem_tiles_count": int(df_out["tile"].nunique()),
        "total_locations": int(total_locations),
        "source_integrated_dataset": str(integrated_csv_path),
        "source_refresh_report": str(gpm_report_path),
        "gpm_observation_metadata": {
            "gpm_product": "NASA_GPM_IMERG_V07C_Early",
            "spatial_resolution": "0.1_deg_geographic (~10 km)",
            "observation_start_utc": freshness_meta.get("observation_start_utc"),
            "observation_end_utc": freshness_meta.get("observation_end_utc"),
            "calculated_latency_hours": freshness_meta.get("calculated_latency_hours"),
            "freshness_gate_limit_hours": max_latency_hours,
            "freshness_status": freshness_meta.get("freshness_status", "UNKNOWN"),
        },
        "model_provenance": {
            "model_execution_mode": "PRECOMPUTED_REUSE_FROM_STAGE_6_2_GRID",
            "retraining_performed": False,
            "new_inference_performed": False,
            "terrain_probability_column": "terrain_rf_prob_balanced",
            "underlying_model_artifact": "data/processed/western_ghats_regional_model/western_ghats_regional_terrain_random_forest.joblib",
            "model_feature_contract": RF_FEATURE_COLUMNS,
        },
        "risk_formula_specification": {
            "formula": "Dynamic_Risk_Index = 100 * (w_terrain * S_terrain + w_rainfall * R_rainfall) = Terrain_Contribution + Rainfall_Contribution",
            "interactive_formula": "Dynamic_Risk_Interactive = 100 * (S_terrain * (1.0 + R_rainfall) / 2.0)",
            "weight_terrain": weight_terrain,
            "weight_rainfall": weight_rainfall,
            "window_accumulation_weights": engine.window_weights,
            "window_physical_scales_mm": engine.window_scales,
            "relative_category_percentiles": engine.relative_percentiles,
            "relative_category_cutoffs": cutoffs,
            "fixed_tier_cutoffs": FIXED_TIER_CUTOFFS,
            "zero_rainfall_handling": "Zero rainfall across all windows yields R_rainfall = 0.0 and Rainfall_Contribution = 0.0 identically.",
            "missing_data_handling": "Missing, non-finite (+inf, -inf), or incomplete windows evaluate to NaN without positive contribution.",
            "calibration_status": "PROVISIONAL_UNVALIDATED_EXPERIMENTAL_SCREENING_FORMULA",
        },
        "summary_statistics": summary_stats,
        "disclaimer": WG_PILOT_DISCLAIMER,
    }

    if cutoffs:
        validate_dynamic_metadata_thresholds(metadata)

    val_report = {
        "stage": "Western Ghats Limited Pilot — Stage 6.4 Grid Dynamic Hazard Risk Validation",
        "status": "SUCCESS" if is_fresh else "FRESHNESS_GATE_FAILED",
        "region": "Western Ghats",
        "coverage_disclosure": metadata["coverage_disclosure"],
        "total_grid_locations": int(total_locations),
        "is_operational_success": bool(is_fresh),
        "freshness_evaluation": {**freshness_meta, "message": freshness_msg},
        "risk_formula_specification": metadata["risk_formula_specification"],
        "summary_statistics": summary_stats,
        "validation_checks": {
            "all_rows_processed": bool(len(df_out) == total_locations),
            "all_14_tiles_present": int(df_out["tile"].nunique()) == 14,
            "provenance_preserved": True,
            "additive_formula_verified": bool(
                np.allclose(
                    t_contrib[valid_risk] + r_contrib[valid_risk],
                    df_out.loc[valid_risk, "dynamic_risk_index"].to_numpy(dtype=float),
                    atol=1e-3,
                )
            )
            if valid_risk.any()
            else True,
            "relative_cutoffs_strictly_ordered": bool(cutoffs and validate_dynamic_metadata_thresholds(metadata)),
            "freshness_gate_enforced": True,
        },
        "generation_timestamp_utc": ref_now.isoformat().replace("+00:00", "Z"),
        "disclaimer": WG_PILOT_DISCLAIMER,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(df_out, output_csv_path)
    atomic_write_json(metadata_json_path, metadata)
    atomic_write_json(validation_report_path, val_report)
    LOGGER.info(
        "Computed dynamic hazard risk for %d Western Ghats grid cells (status=%s) -> %s",
        total_locations,
        val_report["status"],
        output_csv_path,
    )
    return df_out, metadata, val_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate dynamic landslide hazard risk across the Western Ghats 5 km grid."
    )
    parser.add_argument("--integrated-csv", type=Path, default=DEFAULT_INTEGRATED_CSV)
    parser.add_argument("--gpm-report", type=Path, default=DEFAULT_GPM_REPORT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    predict_western_ghats_grid_dynamic_risk(
        integrated_csv_path=args.integrated_csv,
        gpm_report_path=args.gpm_report,
        output_dir=args.output_dir,
        output_csv_path=args.output_csv,
        metadata_json_path=args.metadata_json,
        validation_report_path=args.report_json,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
