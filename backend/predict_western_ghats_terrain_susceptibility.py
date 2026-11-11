"""
Western Ghats Limited Pilot — Frozen Random Forest Terrain Susceptibility Inference
===================================================================================

Applies the frozen Western Ghats balanced Random Forest model (zero retraining)
across the 5 km EPSG:32643 spatial grid built strictly from the 14 validated
DEM tiles. Also computes the unweighted comparison probability column for
continuity with the Stage 6B regional pilot contract while anchoring primary
screening outputs on the balanced RF probability (`terrain_rf_prob_balanced`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.gis_refresh_utils import atomic_write_json, create_timestamped_backup


LOGGER = logging.getLogger("western_ghats_terrain_susceptibility")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRID_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "western_ghats_spatial_grid"
    / "western_ghats_5km_spatial_grid.csv"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "western_ghats_regional_model"
DEFAULT_BALANCED_MODEL = DEFAULT_MODEL_DIR / "western_ghats_regional_terrain_random_forest.joblib"
DEFAULT_UNWEIGHTED_MODEL = (
    DEFAULT_MODEL_DIR / "western_ghats_regional_terrain_random_forest_unweighted.joblib"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "western_ghats_terrain_predictions"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "western_ghats_5km_terrain_susceptibility.csv"
DEFAULT_VALIDATION_REPORT = DEFAULT_OUTPUT_DIR / "western_ghats_terrain_validation_report.json"

RF_FEATURE_COLUMNS: List[str] = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False, float_format="%.8f")
    os.replace(tmp_path, path)


def classify_terrain_tier(prob: float) -> str:
    """Classify balanced RF probability into the 4 standard terrain screening tiers."""
    if prob < 0.25:
        return "Low"
    if prob < 0.50:
        return "Moderate"
    if prob < 0.75:
        return "High"
    return "Very High"


def load_frozen_rf_model(model_path: Path, expected_class_weight: Optional[str]) -> Any:
    """Load frozen Random Forest artifact without retraining and verify 5-feature contract."""
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen Western Ghats RF model not found: {model_path}")
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Loaded object at {model_path} lacks predict_proba().")
    if hasattr(model, "feature_names_in_"):
        actual_features = list(model.feature_names_in_)
        if actual_features != RF_FEATURE_COLUMNS:
            raise ValueError(
                f"Model {model_path.name} feature mismatch: got {actual_features}, expected {RF_FEATURE_COLUMNS}"
            )
    if expected_class_weight is not None and getattr(model, "class_weight", None) != expected_class_weight:
        raise ValueError(
            f"Expected class_weight={expected_class_weight!r} on {model_path.name}, "
            f"got {getattr(model, 'class_weight', None)!r}"
        )
    return model


def predict_western_ghats_terrain_susceptibility(
    grid_csv: Path = DEFAULT_GRID_CSV,
    balanced_model_path: Path = DEFAULT_BALANCED_MODEL,
    unweighted_model_path: Optional[Path] = DEFAULT_UNWEIGHTED_MODEL,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run frozen RF inference across all Western Ghats 5 km grid points."""
    if not grid_csv.exists():
        raise FileNotFoundError(f"Western Ghats 5 km spatial grid CSV not found: {grid_csv}")

    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists (pass overwrite=True to replace): {output_csv}"
        )
    existing_to_backup = [p for p in (output_csv, validation_report_path) if p.exists()]
    if existing_to_backup and overwrite:
        create_timestamped_backup(existing_to_backup, output_csv.parent / "backups")

    df = pd.read_csv(grid_csv)
    if df.empty:
        raise ValueError(f"Input grid CSV is empty: {grid_csv}")

    missing_features = [c for c in RF_FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Input grid CSV missing RF features: {missing_features}")

    nan_counts = {c: int(df[c].isna().sum()) for c in RF_FEATURE_COLUMNS}
    if any(v > 0 for v in nan_counts.values()):
        raise ValueError(f"NaN values detected in RF feature columns: {nan_counts}")

    balanced_model = load_frozen_rf_model(balanced_model_path, expected_class_weight="balanced")
    features_df = df[RF_FEATURE_COLUMNS].astype(np.float64)
    prob_balanced = balanced_model.predict_proba(features_df)[:, 1].astype(np.float64)

    out_df = df.copy()
    out_df["terrain_rf_prob_balanced"] = np.clip(prob_balanced, 0.0, 1.0)
    out_df["terrain_rf_pred_balanced"] = (out_df["terrain_rf_prob_balanced"] >= 0.50).astype(int)
    out_df["terrain_baseline_screening_tier"] = [
        classify_terrain_tier(float(p)) for p in out_df["terrain_rf_prob_balanced"]
    ]

    unweighted_sha = None
    if unweighted_model_path is not None and unweighted_model_path.exists():
        unweighted_model = load_frozen_rf_model(unweighted_model_path, expected_class_weight=None)
        prob_unw = unweighted_model.predict_proba(features_df)[:, 1].astype(np.float64)
        out_df["terrain_rf_prob_unweighted"] = np.clip(prob_unw, 0.0, 1.0)
        unweighted_sha = _sha256_file(unweighted_model_path)

    report = validate_terrain_susceptibility_predictions(
        df=out_df,
        expected_count=len(df),
        balanced_model_path=balanced_model_path,
        balanced_model_sha256=_sha256_file(balanced_model_path),
        unweighted_model_sha256=unweighted_sha,
        output_csv=output_csv,
    )

    _atomic_write_csv(output_csv, out_df)
    atomic_write_json(validation_report_path, report)
    return out_df, report


def validate_terrain_susceptibility_predictions(
    df: pd.DataFrame,
    expected_count: int,
    balanced_model_path: Path,
    balanced_model_sha256: str,
    unweighted_model_sha256: Optional[str],
    output_csv: Path,
) -> Dict[str, Any]:
    """Validate frozen RF terrain susceptibility predictions across the Western Ghats grid."""
    if len(df) != expected_count:
        raise ValueError(f"Row count mismatch: expected {expected_count}, got {len(df)}")

    probs = df["terrain_rf_prob_balanced"]
    preds = df["terrain_rf_pred_balanced"]
    tiers = df["terrain_baseline_screening_tier"]

    valid_tiers = {"Low", "Moderate", "High", "Very High"}
    unexpected_tiers = set(tiers.unique().tolist()) - valid_tiers
    if unexpected_tiers:
        raise ValueError(f"Unexpected terrain screening tiers: {unexpected_tiers}")

    checks = {
        "row_count_preserved": len(df) == expected_count,
        "all_14_tiles_present": df["tile"].nunique() == 14 if "tile" in df.columns else False,
        "zero_nan_probabilities": int(probs.isna().sum()) == 0,
        "probabilities_in_unit_interval": bool((probs >= 0.0).all() and (probs <= 1.0).all()),
        "binary_pred_consistent_with_0_50": bool(((probs >= 0.50).astype(int) == preds).all()),
        "valid_screening_tiers_only": not unexpected_tiers,
        "zero_retraining_frozen_model_used": True,
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Terrain susceptibility validation failed checks: {failed}")

    tier_counts = {str(k): int(v) for k, v in tiers.value_counts().items()}
    return {
        "stage": "Western Ghats Limited Pilot — Stage 6.2 Terrain Susceptibility Inference",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_disclosure": (
            "Western Ghats Limited Pilot: terrain susceptibility evaluated strictly on the 5 km grid "
            "across 14 validated DEM tiles using frozen Random Forest models; does not imply complete Western Ghats coverage."
        ),
        "feature_contract": RF_FEATURE_COLUMNS,
        "total_grid_locations": int(len(df)),
        "frozen_models": {
            "primary_balanced_model": str(balanced_model_path),
            "primary_balanced_sha256": balanced_model_sha256,
            "unweighted_comparison_sha256": unweighted_model_sha256,
            "retrained": False,
        },
        "probability_summary_balanced": {
            "min": float(probs.min()),
            "mean": float(probs.mean()),
            "median": float(probs.median()),
            "max": float(probs.max()),
        },
        "positive_prediction_count_balanced": int(preds.sum()),
        "screening_tier_counts": tier_counts,
        "output_csv": str(output_csv),
        "checks": checks,
        "validation_passed": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen Western Ghats RF terrain susceptibility inference across the 5 km grid."
    )
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--balanced-model", type=Path, default=DEFAULT_BALANCED_MODEL)
    parser.add_argument("--unweighted-model", type=Path, default=DEFAULT_UNWEIGHTED_MODEL)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    df, report = predict_western_ghats_terrain_susceptibility(
        grid_csv=args.grid_csv,
        balanced_model_path=args.balanced_model,
        unweighted_model_path=args.unweighted_model,
        output_csv=args.output_csv,
        validation_report_path=args.report_json,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Predicted terrain susceptibility for %d Western Ghats grid points (mean=%.4f) -> %s",
        len(df),
        report["probability_summary_balanced"]["mean"],
        args.output_csv,
    )


if __name__ == "__main__":
    main()
