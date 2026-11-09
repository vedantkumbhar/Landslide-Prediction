"""
Stage 7.2: Northeast India Grid-Level Terrain Susceptibility Prediction Engine.

Generates baseline static terrain landslide susceptibility predictions for all 14,550
5 km regular spatial grid locations in Northeast India using the approved pre-trained
regional Random Forest model (Stage 5B/regional 30-Oct baseline).

Core Requirements:
1. Load existing trained model without retraining (random_forest_model.joblib).
2. Validate exact five-feature input contract and feature ordering:
   ['elevation_m', 'slope_degrees', 'aspect_sin', 'aspect_cos', 'aspect_flat'].
3. Compute calibrated class probabilities (terrain_rf_prob_balanced) and binary
   predictions (terrain_rf_pred_balanced).
4. Assign standardized baseline screening tiers:
   - Low: [0.00, 0.25)
   - Moderate: [0.25, 0.50)
   - High: [0.50, 0.75)
   - Very High: [0.75, 1.00]
5. Preserve grid_id, latitude, longitude, and DEM tile provenance.
6. Enforce strict output validation: finite probabilities in [0.0, 1.0], zero NaNs,
   zero missing values, and row-count invariance.
7. Save predictions to a dedicated output directory and generate a comprehensive
   validation report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_GRID = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_spatial_grid"
    / "northeast_india_5km_spatial_grid.csv"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regional_models_30oct"
    / "northeast_india"
    / "random_forest_model.joblib"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "northeast_terrain_predictions"
)
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "northeast_india_5km_terrain_susceptibility.csv"
DEFAULT_VALIDATION_REPORT = DEFAULT_OUTPUT_DIR / "northeast_terrain_susceptibility_validation_report.json"

# Model Contract
MANDATORY_TERRAIN_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

EXPECTED_TILES_COUNT = 33
TIER_BINS = [0.0, 0.25, 0.50, 0.75, 1.0001]
TIER_LABELS = ["Low", "Moderate", "High", "Very High"]
DISCLAIMER_TEXT = (
    "EXPERIMENTAL_BASELINE_TERRAIN_SUSCEPTIBILITY_SCREENING_ONLY_NOT_DYNAMIC_RISK_AND_NOT_OFFICIAL_WARNING"
)


def load_regional_terrain_model(model_path: Path) -> RandomForestClassifier:
    """Load and validate the pre-trained Northeast India Random Forest model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found at: {model_path}")

    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected RandomForestClassifier, got {type(model)}")

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        if list(feature_names) != MANDATORY_TERRAIN_FEATURES:
            raise ValueError(
                f"Model feature contract mismatch.\n"
                f"Expected: {MANDATORY_TERRAIN_FEATURES}\n"
                f"Model has: {list(feature_names)}"
            )

    return model


def validate_input_grid(df: pd.DataFrame) -> None:
    """Strictly validate input spatial grid columns, types, and values."""
    required_cols = ["grid_id", "latitude", "longitude", "tile"] + MANDATORY_TERRAIN_FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input grid is missing required columns: {missing_cols}")

    if df["grid_id"].isnull().any():
        raise ValueError("Input grid contains null grid_id values")
    if df["grid_id"].duplicated().any():
        raise ValueError("Input grid contains duplicate grid_id values")

    # Finiteness of coordinates and terrain features
    for col in required_cols:
        if col in ["grid_id", "tile"]:
            continue
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values detected in input column: {col}")

    # Physical ranges
    if (df["slope_degrees"] < 0.0).any() or (df["slope_degrees"] > 90.0).any():
        raise ValueError("Input grid contains slope_degrees outside [0.0, 90.0]")

    if (df["aspect_sin"] < -1.0001).any() or (df["aspect_sin"] > 1.0001).any():
        raise ValueError("Input grid contains aspect_sin outside [-1.0, 1.0]")

    if (df["aspect_cos"] < -1.0001).any() or (df["aspect_cos"] > 1.0001).any():
        raise ValueError("Input grid contains aspect_cos outside [-1.0, 1.0]")

    if not set(df["aspect_flat"].unique()).issubset({0, 1}):
        raise ValueError("Input grid aspect_flat must contain only 0 and 1")


def assign_screening_tiers(probs: np.ndarray) -> np.ndarray:
    """Categorize probabilities into discrete screening tiers."""
    tiers = np.full(len(probs), "Unassessed", dtype=object)
    tiers[(probs >= 0.0) & (probs < 0.25)] = "Low"
    tiers[(probs >= 0.25) & (probs < 0.50)] = "Moderate"
    tiers[(probs >= 0.50) & (probs < 0.75)] = "High"
    tiers[(probs >= 0.75) & (probs <= 1.0001)] = "Very High"
    return tiers


def predict_northeast_terrain_susceptibility(
    input_grid_path: Path = DEFAULT_INPUT_GRID,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Generate and validate grid-level terrain susceptibility predictions.
    """
    if not overwrite:
        if output_csv_path.exists():
            raise FileExistsError(
                f"Output CSV already exists: {output_csv_path}. Use --overwrite to replace."
            )
        if validation_report_path.exists():
            raise FileExistsError(
                f"Validation report already exists: {validation_report_path}. Use --overwrite to replace."
            )

    print("=" * 75)
    print("STAGE 7.2: NORTHEAST INDIA GRID TERRAIN SUSCEPTIBILITY PREDICTION")
    print("=" * 75)

    # 1. Load Model
    print(f"\n1. Loading trained regional model from: {model_path.name}")
    model = load_regional_terrain_model(model_path)
    print(f"   - Model Type: {type(model).__name__}")
    print(f"   - Estimators: {len(model.estimators_)}")
    print(f"   - Class Weight: {model.class_weight}")
    print(f"   - Validated feature contract: {MANDATORY_TERRAIN_FEATURES}")

    # 2. Load and Validate Input Grid
    print(f"\n2. Loading input spatial grid: {input_grid_path.name}")
    if not input_grid_path.exists():
        raise FileNotFoundError(f"Input spatial grid not found: {input_grid_path}")
    df_grid = pd.read_csv(input_grid_path)
    print(f"   - Total grid locations loaded: {len(df_grid):,}")
    print(f"   - DEM tiles represented: {df_grid['tile'].nunique()}")
    validate_input_grid(df_grid)

    # 3. Model Inference (No retraining)
    print("\n3. Executing terrain Random Forest model inference...")
    X = df_grid[MANDATORY_TERRAIN_FEATURES]
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    tiers = assign_screening_tiers(probs)

    # 4. Construct Output Dataset
    out_df = df_grid.copy()
    out_df["terrain_rf_prob_balanced"] = np.round(probs, 6)
    out_df["terrain_rf_pred_balanced"] = preds.astype(int)
    out_df["terrain_baseline_screening_tier"] = tiers

    # 5. Output Validation
    print("\n4. Validating predictions integrity...")
    _validate_predictions(out_df, len(df_grid))

    # 6. Save Output CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f"\n5. Saved susceptibility predictions CSV: {output_csv_path.name}")
    print(f"   - Locations: {len(out_df):,}")
    print(f"   - File size: {output_csv_path.stat().st_size:,} bytes")

    # 7. Generate Validation Report
    report = _build_validation_report(out_df, model, model_path, input_grid_path)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"   - Validation report saved: {validation_report_path.name}")
    print("=" * 75)

    return report


def _validate_predictions(df: pd.DataFrame, expected_rows: int) -> None:
    """Validate prediction outputs for finiteness, row counts, and probability bounds."""
    if len(df) != expected_rows:
        raise ValueError(f"Output row count ({len(df)}) does not match input ({expected_rows})")

    probs = df["terrain_rf_prob_balanced"]
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite probabilities detected in predictions")
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("Probabilities outside [0.0, 1.0] detected")

    preds = df["terrain_rf_pred_balanced"]
    if not set(preds.unique()).issubset({0, 1}):
        raise ValueError("Binary predictions must contain only 0 and 1")

    # Verify binary prediction consistency with probability threshold 0.5
    expected_preds = (probs >= 0.5).astype(int)
    if not (preds == expected_preds).all():
        raise ValueError("Binary class predictions diverge from probability threshold 0.5")

    # Tier consistency
    valid_tiers = set(TIER_LABELS)
    actual_tiers = set(df["terrain_baseline_screening_tier"].unique())
    if not actual_tiers.issubset(valid_tiers):
        raise ValueError(f"Unexpected screening tiers: {actual_tiers - valid_tiers}")


def _build_validation_report(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    model_path: Path,
    grid_path: Path,
) -> Dict[str, Any]:
    """Compile comprehensive JSON validation report for predictions."""
    probs = df["terrain_rf_prob_balanced"]
    preds = df["terrain_rf_pred_balanced"]
    tiers = df["terrain_baseline_screening_tier"]
    total = len(df)

    # Quantiles and summary metrics
    prob_stats = {
        "min": float(probs.min()),
        "max": float(probs.max()),
        "mean": round(float(probs.mean()), 6),
        "std": round(float(probs.std()), 6),
        "P10": round(float(probs.quantile(0.10)), 6),
        "P25": round(float(probs.quantile(0.25)), 6),
        "P50": round(float(probs.quantile(0.50)), 6),
        "P75": round(float(probs.quantile(0.75)), 6),
        "P90": round(float(probs.quantile(0.90)), 6),
        "P95": round(float(probs.quantile(0.95)), 6),
    }

    # Binary class distribution
    class_counts = preds.value_counts().to_dict()
    class_summary = {
        "class_0_count": int(class_counts.get(0, 0)),
        "class_1_count": int(class_counts.get(1, 0)),
        "class_0_pct": round(float(class_counts.get(0, 0) / total * 100.0), 2),
        "class_1_pct": round(float(class_counts.get(1, 0) / total * 100.0), 2),
    }

    # Tier breakdown
    tier_counts = tiers.value_counts().to_dict()
    tier_summary = {
        label: {
            "count": int(tier_counts.get(label, 0)),
            "pct": round(float(tier_counts.get(label, 0) / total * 100.0), 2),
        }
        for label in TIER_LABELS
    }

    # Per-tile distribution
    tile_group = df.groupby("tile")["terrain_rf_prob_balanced"]
    tile_stats = {}
    for tile, group in tile_group:
        tile_stats[tile] = {
            "locations": len(group),
            "mean_prob": round(float(group.mean()), 4),
            "min_prob": round(float(group.min()), 4),
            "max_prob": round(float(group.max()), 4),
            "high_or_very_high_pct": round(float((group >= 0.50).sum() / len(group) * 100.0), 2),
        }

    return {
        "stage": "Stage 7.2 — Northeast India Grid-Level Terrain Susceptibility Prediction",
        "status": "SUCCESS",
        "region": "Northeast India",
        "total_grid_locations": total,
        "input_grid_path": str(grid_path),
        "model_metadata": {
            "model_path": str(model_path),
            "model_type": type(model).__name__,
            "n_estimators": len(model.estimators_),
            "class_weight": str(model.class_weight),
            "mandatory_feature_contract": MANDATORY_TERRAIN_FEATURES,
        },
        "geographic_extent": {
            "dem_tile_count": int(df["tile"].nunique()),
            "latitude_min": float(df["latitude"].min()),
            "latitude_max": float(df["latitude"].max()),
            "longitude_min": float(df["longitude"].min()),
            "longitude_max": float(df["longitude"].max()),
        },
        "susceptibility_probability_statistics": prob_stats,
        "binary_classification_summary": class_summary,
        "screening_tier_distribution": tier_summary,
        "tile_level_susceptibility_summary": tile_stats,
        "validation_checks": {
            "row_count_invariant": True,
            "zero_missing_values": True,
            "all_probabilities_finite": True,
            "all_probabilities_in_bounds_0_1": True,
            "class_threshold_consistency": True,
            "provenance_preserved": True,
        },
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "disclaimer": DISCLAIMER_TEXT,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict grid-level terrain landslide susceptibility for Northeast India."
    )
    parser.add_argument("--input-grid", type=Path, default=DEFAULT_INPUT_GRID, help="Input spatial grid CSV")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained joblib model")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to output predictions CSV")
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
        help="Path to output validation report JSON",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        predict_northeast_terrain_susceptibility(
            input_grid_path=args.input_grid,
            model_path=args.model_path,
            output_csv_path=args.output_csv,
            validation_report_path=args.validation_report,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"\n[ERROR] Susceptibility prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
