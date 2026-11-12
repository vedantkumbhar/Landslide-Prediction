"""
Landslide Prediction System — Unified Standard Refresh Run Procedure (Steps 1–11).

Executes the end-to-end operational refresh across the Indian Himalayas,
Northeast India, and Western Ghats using live NASA GPM IMERG Early Run V07C
observations, preserving validated terrain features and frozen regional Random
Forest models, recalculating metadata-driven relative hazard risk categories,
regenerating interactive GIS outputs (HTML + GeoJSONs), running cross-region
consistency & unit test verification, and emitting a 12-item JSON + Markdown
Run Summary Report under outputs/refresh_reports/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

from backend.gis_refresh_utils import (
    atomic_write_json,
    atomic_write_text,
    evaluate_freshness_condition,
    parse_iso8601_utc,
    resolve_gpm_window_with_telemetry,
    validate_dynamic_metadata_thresholds,
)
from backend.refresh_safety import (
    DEFAULT_LOCK_FILENAME,
    RefreshProcessLock,
    RegionTransaction,
    record_skipped_execution,
    recover_interrupted_transactions,
)
from backend.integrate_himalayas_gpm_rainfall import (
    DEFAULT_ARCHIVE_DIR as GPM_ARCHIVE_DIR,
    DEFAULT_GRID_CSV as HIM_GRID_CSV,
    DEFAULT_INTEGRATED_CSV as HIM_INTEGRATED_CSV,
    DEFAULT_TERRAIN_CSV as HIM_TERRAIN_CSV,
    DEFAULT_VALIDATION_REPORT as HIM_RAIN_REPORT_JSON,
    EXISTING_RAW_DIRS as GPM_RAW_DIRS,
    integrate_himalayas_gpm_rainfall,
)
from backend.predict_himalayas_dynamic_risk import (
    DEFAULT_METADATA_JSON as HIM_RISK_META_JSON,
    DEFAULT_OUTPUT_CSV as HIM_RISK_CSV,
    DEFAULT_VALIDATION_REPORT_JSON as HIM_RISK_REPORT_JSON,
    run_himalayas_dynamic_risk_pipeline,
)
from backend.create_himalayas_risk_map import (
    DEFAULT_OUTPUT_CELLS_GEOJSON as HIM_CELLS_GEOJSON,
    DEFAULT_OUTPUT_GEOJSON as HIM_POINTS_GEOJSON,
    DEFAULT_OUTPUT_HTML as HIM_HTML_MAP,
    DEFAULT_OUTPUT_METADATA as HIM_MAP_META_JSON,
    DEFAULT_OUTPUT_REGIONS_GEOJSON as HIM_REGIONS_GEOJSON,
    DEFAULT_TRAINING_DATASET as REGIONAL_TRAINING_DATASET,
    DEFAULT_VALIDATION_REPORT as HIM_MAP_REPORT_JSON,
    generate_himalayas_dynamic_risk_map,
)
from backend.integrate_northeast_gpm_rainfall import (
    DEFAULT_GRID_CSV as NE_GRID_CSV,
    DEFAULT_INTEGRATED_CSV as NE_INTEGRATED_CSV,
    DEFAULT_TERRAIN_CSV as NE_TERRAIN_CSV,
    DEFAULT_VALIDATION_REPORT as NE_RAIN_REPORT_JSON,
    integrate_northeast_gpm_rainfall,
)
from backend.predict_northeast_dynamic_risk import (
    DEFAULT_METADATA_JSON as NE_RISK_META_JSON,
    DEFAULT_OUTPUT_CSV as NE_RISK_CSV,
    DEFAULT_VALIDATION_REPORT_JSON as NE_RISK_REPORT_JSON,
    run_northeast_dynamic_risk_pipeline,
)
from backend.create_northeast_risk_map import (
    DEFAULT_OUTPUT_CELLS_GEOJSON as NE_STAGE_CELLS_GEOJSON,
    DEFAULT_OUTPUT_GEOJSON as NE_STAGE_POINTS_GEOJSON,
    DEFAULT_OUTPUT_HTML as NE_STAGE_HTML_MAP,
    DEFAULT_OUTPUT_META as NE_STAGE_MAP_META_JSON,
    DEFAULT_OUTPUT_REGIONS_GEOJSON as NE_STAGE_REGIONS_GEOJSON,
    DEFAULT_SYNC_OUTPUT_DIR as OUTPUTS_DIR,
    DEFAULT_VALIDATION_REPORT as NE_STAGE_MAP_REPORT_JSON,
    create_northeast_dynamic_risk_map,
)
from backend.generate_western_ghats_spatial_grid import (
    DEFAULT_OUTPUT_CSV as WG_GRID_CSV,
)
from backend.predict_western_ghats_terrain_susceptibility import (
    DEFAULT_BALANCED_MODEL as WG_MODEL_BALANCED,
    DEFAULT_OUTPUT_CSV as WG_TERRAIN_CSV,
    DEFAULT_UNWEIGHTED_MODEL as WG_MODEL_UNWEIGHTED,
)
from backend.integrate_western_ghats_gpm_rainfall import (
    DEFAULT_INTEGRATED_CSV as WG_RAIN_LATEST_CSV,
    DEFAULT_VALIDATION_REPORT as WG_RAIN_STATUS_JSON,
    integrate_western_ghats_gpm_rainfall,
)
from backend.predict_western_ghats_grid_dynamic_risk import (
    DEFAULT_METADATA_JSON as WG_RISK_META_JSON,
    DEFAULT_OUTPUT_CSV as WG_RISK_CSV,
    DEFAULT_VALIDATION_REPORT as WG_RISK_REPORT_JSON,
    predict_western_ghats_grid_dynamic_risk,
)
from backend.create_western_ghats_polygon_risk_map import (
    DEFAULT_BACKGROUND_CSV as WG_BACKGROUND_CSV,
    DEFAULT_EVENTS_CSV as WG_EVENTS_CSV,
    DEFAULT_OUTPUT_CELLS_GEOJSON as WG_CELLS_GEOJSON,
    DEFAULT_OUTPUT_GEOJSON as WG_POINTS_GEOJSON,
    DEFAULT_OUTPUT_HTML as WG_HTML_MAP,
    DEFAULT_OUTPUT_METADATA as WG_MAP_META_JSON,
    DEFAULT_OUTPUT_REGIONS_GEOJSON as WG_REGIONS_GEOJSON,
    DEFAULT_VALIDATION_REPORT as WG_MAP_REPORT_JSON,
    generate_western_ghats_polygon_risk_map,
)


DEFAULT_REPORT_DIR = OUTPUTS_DIR / "refresh_reports"
DEFAULT_MAX_LATENCY_HOURS = 12.0
EXPECTED_GRANULES_24H = 48
RAINFALL_ACCUM_WINDOWS = ["1h", "3h", "6h", "12h", "24h"]
RAINFALL_ACCUM_COLS = [f"gpm_{w}_accumulation_mm" for w in RAINFALL_ACCUM_WINDOWS]
ACTIVE_CATEGORIES = ("Low", "Moderate", "High", "Very High")
TERRAIN_MODEL_PREDICTORS = [
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "aspect_flat",
]

HIM_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "regional_models_30oct" / "himalayas" / "random_forest_model.joblib"
NE_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "regional_models_30oct" / "northeast_india" / "random_forest_model.joblib"

REGION_ORDER = ("himalayas", "northeast", "western_ghats")


def _build_region_specs() -> Dict[str, Dict[str, Any]]:
    """Build unified per-region configuration for pre-flight, execution, transaction backup, and validation."""
    return {
        "himalayas": {
            "display_name": "Himalayas",
            "stages": "Stages 8.3 -> 8.4 -> 8.5",
            "expected_locations": 12052,
            "expected_tiles": 29,
            "geometry_mode": "regional_polygons_and_5km_cells",
            "required_inputs": [
                (HIM_GRID_CSV, ["grid_id", "latitude", "longitude", "tile", *TERRAIN_MODEL_PREDICTORS]),
                (HIM_TERRAIN_CSV, ["grid_id", "terrain_rf_prob_balanced", "terrain_rf_pred_balanced"]),
                (REGIONAL_TRAINING_DATASET, ["dem_region", "tile", "latitude", "longitude", "label"]),
            ],
            "required_models": [HIM_MODEL_PATH],
            "tracked_outputs": [
                HIM_INTEGRATED_CSV,
                HIM_RAIN_REPORT_JSON,
                HIM_RISK_CSV,
                HIM_RISK_META_JSON,
                HIM_RISK_REPORT_JSON,
                HIM_HTML_MAP,
                HIM_REGIONS_GEOJSON,
                HIM_CELLS_GEOJSON,
                HIM_POINTS_GEOJSON,
                HIM_MAP_META_JSON,
                HIM_MAP_REPORT_JSON,
            ],
            "risk_csv": HIM_RISK_CSV,
            "risk_metadata_json": HIM_RISK_META_JSON,
            "risk_report_json": HIM_RISK_REPORT_JSON,
            "rain_report_json": HIM_RAIN_REPORT_JSON,
            "map_html": HIM_HTML_MAP,
            "map_metadata_json": HIM_MAP_META_JSON,
            "map_report_json": HIM_MAP_REPORT_JSON,
            "expected_geojson_counts": {
                HIM_POINTS_GEOJSON: 13037,
                HIM_CELLS_GEOJSON: 12052,
            },
            "regions_geojson": HIM_REGIONS_GEOJSON,
            "html_max_bytes": 15_000_000,
            "synced_outputs": [
                HIM_HTML_MAP,
                HIM_REGIONS_GEOJSON,
                HIM_CELLS_GEOJSON,
                HIM_POINTS_GEOJSON,
                HIM_MAP_META_JSON,
                HIM_MAP_REPORT_JSON,
            ],
            "test_modules": [
                "backend.test_himalayas_gpm_rainfall",
                "backend.test_himalayas_dynamic_risk",
                "backend.test_himalayas_dynamic_risk_map",
            ],
        },
        "northeast": {
            "display_name": "Northeast India",
            "stages": "Stages 7.3 -> 7.4 -> 7.5",
            "expected_locations": 14550,
            "expected_tiles": 33,
            "geometry_mode": "regional_polygons_and_5km_cells",
            "required_inputs": [
                (NE_GRID_CSV, ["grid_id", "latitude", "longitude", "tile", *TERRAIN_MODEL_PREDICTORS]),
                (NE_TERRAIN_CSV, ["grid_id", "terrain_rf_prob_balanced", "terrain_rf_pred_balanced"]),
                (REGIONAL_TRAINING_DATASET, ["dem_region", "tile", "latitude", "longitude", "label"]),
            ],
            "required_models": [NE_MODEL_PATH],
            "tracked_outputs": [
                NE_INTEGRATED_CSV,
                NE_RAIN_REPORT_JSON,
                NE_RISK_CSV,
                NE_RISK_META_JSON,
                NE_RISK_REPORT_JSON,
                NE_STAGE_HTML_MAP,
                NE_STAGE_REGIONS_GEOJSON,
                NE_STAGE_CELLS_GEOJSON,
                NE_STAGE_POINTS_GEOJSON,
                NE_STAGE_MAP_META_JSON,
                NE_STAGE_MAP_REPORT_JSON,
                OUTPUTS_DIR / NE_STAGE_HTML_MAP.name,
                OUTPUTS_DIR / NE_STAGE_REGIONS_GEOJSON.name,
                OUTPUTS_DIR / NE_STAGE_CELLS_GEOJSON.name,
                OUTPUTS_DIR / NE_STAGE_POINTS_GEOJSON.name,
                OUTPUTS_DIR / NE_STAGE_MAP_META_JSON.name,
                OUTPUTS_DIR / NE_STAGE_MAP_REPORT_JSON.name,
            ],
            "risk_csv": NE_RISK_CSV,
            "risk_metadata_json": NE_RISK_META_JSON,
            "risk_report_json": NE_RISK_REPORT_JSON,
            "rain_report_json": NE_RAIN_REPORT_JSON,
            "map_html": OUTPUTS_DIR / NE_STAGE_HTML_MAP.name,
            "map_metadata_json": OUTPUTS_DIR / NE_STAGE_MAP_META_JSON.name,
            "map_report_json": OUTPUTS_DIR / NE_STAGE_MAP_REPORT_JSON.name,
            "expected_geojson_counts": {
                NE_STAGE_POINTS_GEOJSON: 14550,
                NE_STAGE_CELLS_GEOJSON: 14550,
                OUTPUTS_DIR / NE_STAGE_POINTS_GEOJSON.name: 14550,
                OUTPUTS_DIR / NE_STAGE_CELLS_GEOJSON.name: 14550,
            },
            "regions_geojson": OUTPUTS_DIR / NE_STAGE_REGIONS_GEOJSON.name,
            "html_max_bytes": 25_000_000,
            "synced_outputs": [
                OUTPUTS_DIR / NE_STAGE_HTML_MAP.name,
                OUTPUTS_DIR / NE_STAGE_REGIONS_GEOJSON.name,
                OUTPUTS_DIR / NE_STAGE_CELLS_GEOJSON.name,
                OUTPUTS_DIR / NE_STAGE_POINTS_GEOJSON.name,
                OUTPUTS_DIR / NE_STAGE_MAP_META_JSON.name,
                OUTPUTS_DIR / NE_STAGE_MAP_REPORT_JSON.name,
            ],
            "test_modules": [
                "backend.test_northeast_gpm_rainfall",
                "backend.test_northeast_dynamic_risk",
                "backend.test_northeast_risk_map",
            ],
        },
        "western_ghats": {
            "display_name": "Western Ghats (Limited Pilot)",
            "coverage_note": "Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.",
            "stages": "Stages 6.3 -> 6.4 -> 6.5 (5 km Grid & Polygons)",
            "expected_locations": 4819,
            "expected_tiles": 14,
            "geometry_mode": "regional_polygons_and_5km_cells",
            "required_inputs": [
                (WG_GRID_CSV, ["grid_id", "tile", "latitude", "longitude", "utm_x", "utm_y", "utm_epsg", *TERRAIN_MODEL_PREDICTORS]),
                (WG_TERRAIN_CSV, ["grid_id", "tile", "latitude", "longitude", "terrain_rf_prob_balanced"]),
                (WG_EVENTS_CSV, ["event_id", "latitude", "longitude", "dem_tile", *TERRAIN_MODEL_PREDICTORS]),
                (WG_BACKGROUND_CSV, ["sample_id", "latitude", "longitude", "tile", *TERRAIN_MODEL_PREDICTORS]),
            ],
            "required_models": [WG_MODEL_BALANCED, WG_MODEL_UNWEIGHTED],
            "tracked_outputs": [
                WG_RAIN_LATEST_CSV,
                WG_RAIN_STATUS_JSON,
                WG_RISK_CSV,
                WG_RISK_META_JSON,
                WG_RISK_REPORT_JSON,
                WG_HTML_MAP,
                WG_POINTS_GEOJSON,
                WG_REGIONS_GEOJSON,
                WG_CELLS_GEOJSON,
                WG_MAP_META_JSON,
                WG_MAP_REPORT_JSON,
            ],
            "risk_csv": WG_RISK_CSV,
            "risk_metadata_json": WG_RISK_META_JSON,
            "risk_report_json": WG_RISK_REPORT_JSON,
            "rain_report_json": WG_RAIN_STATUS_JSON,
            "map_html": WG_HTML_MAP,
            "map_metadata_json": WG_MAP_META_JSON,
            "map_report_json": WG_MAP_REPORT_JSON,
            "expected_geojson_counts": {
                WG_POINTS_GEOJSON: 6401,
                WG_CELLS_GEOJSON: 4819,
            },
            "regions_geojson": WG_REGIONS_GEOJSON,
            "html_max_bytes": 15_000_000,
            "synced_outputs": [
                WG_HTML_MAP,
                WG_POINTS_GEOJSON,
                WG_REGIONS_GEOJSON,
                WG_CELLS_GEOJSON,
                WG_MAP_META_JSON,
                WG_MAP_REPORT_JSON,
            ],
            "test_modules": [
                "backend.test_western_ghats_spatial_grid",
                "backend.test_western_ghats_terrain_susceptibility",
                "backend.test_western_ghats_gpm_grid_rainfall",
                "backend.test_western_ghats_grid_dynamic_risk",
                "backend.test_western_ghats_polygon_risk_map",
                "backend.test_western_ghats_gpm_refresh",
                "backend.test_western_ghats_rainfall_terrain_integration",
                "backend.test_western_ghats_dynamic_risk",
                "backend.test_western_ghats_risk_map",
            ],
        },
    }


# RegionTransaction is imported from backend.refresh_safety to provide
# SHA-256 verified, journaled backup and crash recovery across runs.


def _utc_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_regions(raw_regions: Optional[Sequence[str]]) -> List[str]:
    """Validate and canonicalize requested regions in deterministic execution order."""
    if not raw_regions:
        return list(REGION_ORDER)
    tokens: List[str] = []
    for item in raw_regions:
        for part in str(item).split(","):
            norm = part.strip().lower()
            if norm:
                tokens.append(norm)
    if "all" in tokens:
        return list(REGION_ORDER)
    unknown = [t for t in tokens if t not in REGION_ORDER]
    if unknown:
        raise ValueError(f"Unsupported region(s): {unknown}. Valid choices: {list(REGION_ORDER)}")
    return [r for r in REGION_ORDER if r in tokens]


def check_preflight(
    regions: Sequence[str],
    region_specs: Dict[str, Dict[str, Any]],
    offline: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
    env_file: Optional[Path] = PROJECT_ROOT / ".env",
    verify_model_load: bool = True,
) -> Dict[str, Any]:
    """
    Step 1: Verify environment credentials, directories, input CSV contracts, and frozen RF models.
    """
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    token = (os.getenv("NASA_EARTHDATA_TOKEN") or "").strip()
    token_ok = True if offline else bool(token)
    checks.append({
        "check": "nasa_earthdata_token",
        "required": not offline,
        "configured": bool(token),
        "passed": token_ok,
    })
    if not token_ok:
        errors.append("Missing NASA_EARTHDATA_TOKEN in environment/.env while online mode is active.")

    required_dirs = [GPM_ARCHIVE_DIR, OUTPUTS_DIR, report_dir]
    for d in required_dirs:
        try:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
            checks.append({"check": f"dir:{d.name}", "path": str(d), "passed": d.is_dir()})
        except Exception as exc:
            checks.append({"check": f"dir:{d.name}", "path": str(d), "passed": False, "error": str(exc)})
            errors.append(f"Directory unavailable: {d} ({exc})")

    checked_paths: set = set()
    for rkey in regions:
        spec = region_specs[rkey]
        for csv_path, req_cols in spec["required_inputs"]:
            if csv_path in checked_paths:
                continue
            checked_paths.add(csv_path)
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                checks.append({"check": f"input:{csv_path.name}", "path": str(csv_path), "passed": False})
                errors.append(f"Missing or empty input dataset for {rkey}: {csv_path}")
                continue
            try:
                cols = list(pd.read_csv(csv_path, nrows=1).columns)
                missing_cols = [c for c in req_cols if c not in cols]
                ok = len(missing_cols) == 0
                checks.append({
                    "check": f"input:{csv_path.name}",
                    "path": str(csv_path),
                    "passed": ok,
                    "missing_columns": missing_cols,
                })
                if not ok:
                    errors.append(f"Missing required columns {missing_cols} in {csv_path.name}")
            except Exception as exc:
                checks.append({"check": f"input:{csv_path.name}", "path": str(csv_path), "passed": False, "error": str(exc)})
                errors.append(f"Unreadable CSV {csv_path}: {exc}")

        for model_path in spec["required_models"]:
            if not model_path.exists() or model_path.stat().st_size == 0:
                checks.append({"check": f"model:{model_path.name}", "path": str(model_path), "passed": False})
                errors.append(f"Missing frozen terrain RF model for {rkey}: {model_path}")
                continue
            model_ok = True
            model_err = None
            n_features = None
            if verify_model_load:
                try:
                    mdl = joblib.load(model_path)
                    n_features = getattr(mdl, "n_features_in_", None)
                    if n_features is not None and int(n_features) != len(TERRAIN_MODEL_PREDICTORS):
                        model_ok = False
                        model_err = f"Expected {len(TERRAIN_MODEL_PREDICTORS)} features, got {n_features}"
                except Exception as exc:
                    model_ok = False
                    model_err = str(exc)
            checks.append({
                "check": f"model:{model_path.name}",
                "path": str(model_path),
                "passed": model_ok,
                "n_features_in": n_features,
                "error": model_err,
            })
            if not model_ok:
                errors.append(f"Model contract failure for {model_path.name}: {model_err}")

    return {
        "status": "PASSED" if not errors else "FAILED",
        "token_configured": bool(token),
        "offline_mode": bool(offline),
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c.get("passed")),
        "errors": errors,
        "checks": checks,
    }


def discover_and_retrieve_gpm(
    target_end_time: Optional[datetime],
    offline: bool,
    workers: int,
    now_utc: datetime,
    max_latency_hours: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Steps 2 & 3: Discover the latest 48-granule (24h) GPM IMERG Early V07C window,
    download any missing granules (unless offline or dry-run), and enforce freshness.
    """
    from datetime import timedelta
    from backend.refresh_western_ghats_rainfall import GPMGranuleDownloader

    granules = None
    resolved_end_dt = target_end_time
    status_str = "UNKNOWN"
    missing_starts: List[datetime] = []
    telemetry: Dict[str, Any] = {}

    if dry_run and not offline and target_end_time is None:
        dl = GPMGranuleDownloader(archive_dir=GPM_ARCHIVE_DIR)
        if dl.token:
            c_end, online_granules, miss = dl.discover_latest_contiguous_window_online()
            resolved_end_dt = c_end
            granules = online_granules
            missing_starts = list(miss or [])
            avail = len(online_granules or [])
            telemetry = {
                "window_discovery_source": "NASA_CMR_DRY_RUN",
                "granules_discovered_cmr": getattr(dl, "last_discovered_cmr_count", avail),
                "granules_reused_cached": 0,
                "granules_downloaded": 0,
                "granules_validated": avail,
            }
            status_str = "INCOMPLETE_DATA" if (not online_granules or missing_starts) else "DRY_RUN_CMR_DISCOVERED"

    if not telemetry:
        allow_download = (not offline) and (not dry_run)
        granules, resolved_end_dt, status_str, _lat, missing_starts, telemetry = (
            resolve_gpm_window_with_telemetry(
                archive_dir=GPM_ARCHIVE_DIR,
                raw_dirs=GPM_RAW_DIRS,
                target_end_time=target_end_time,
                allow_download=allow_download,
                workers=workers,
                now_utc=now_utc,
                max_latency_hours=max_latency_hours,
            )
        )

    obs_end_str = _utc_iso(resolved_end_dt)
    obs_start_str = _utc_iso(resolved_end_dt - timedelta(hours=24)) if resolved_end_dt is not None else None

    avail_count = int(telemetry.get("granules_validated", len(granules) if granules else 0) or 0)
    missing_count = len(missing_starts) if missing_starts else (EXPECTED_GRANULES_24H - avail_count)
    is_complete = (granules is not None and avail_count == EXPECTED_GRANULES_24H and missing_count == 0)

    freshness = evaluate_freshness_condition(
        obs_end_utc_str=obs_end_str,
        is_24h_complete=is_complete,
        available_granules=avail_count,
        missing_intervals=max(0, int(missing_count)),
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
    )

    cmr_discovered = int(telemetry.get("granules_discovered_cmr", 0) or 0)
    downloaded = int(telemetry.get("granules_downloaded", 0) or 0)
    cached = int(
        telemetry.get(
            "granules_reused_cached",
            avail_count if not downloaded else max(0, avail_count - downloaded),
        )
        or 0
    )

    raw_lat = freshness.get("latency_hours")
    return {
        "resolved_end_dt": resolved_end_dt,
        "observation_start_utc": obs_start_str,
        "observation_end_utc": obs_end_str,
        "discovery_source": telemetry.get("window_discovery_source", "UNKNOWN"),
        "resolver_status": status_str,
        "latency_hours": round(float(raw_lat), 3) if raw_lat is not None and np.isfinite(raw_lat) else None,
        "max_latency_hours": float(max_latency_hours),
        "is_fresh": bool(freshness["is_fresh"]),
        "freshness_status": freshness["status_code"],
        "freshness_label": freshness["freshness_label"],
        "freshness_reason": freshness.get("reason"),
        "granules_discovered_cmr": cmr_discovered,
        "granules_downloaded": downloaded,
        "granules_cached": cached,
        "granules_validated": avail_count,
        "expected_granules": EXPECTED_GRANULES_24H,
        "missing_intervals_count": max(0, int(missing_count)),
        "is_24h_complete": bool(is_complete),
        "telemetry": telemetry,
    }


def _execute_himalayas_pipeline(
    resolved_end_dt: Optional[datetime],
    offline: bool,
    now_utc: datetime,
    max_latency_hours: float,
) -> Dict[str, Any]:
    rain_report = integrate_himalayas_gpm_rainfall(
        target_end_time=resolved_end_dt,
        allow_download=not offline,
        overwrite=True,
        now_utc=now_utc,
    )
    if rain_report.get("refresh_status") != "SUCCESS" or not rain_report.get("is_operational_success"):
        raise RuntimeError(f"Himalayas Stage 8.3 rainfall integration failed freshness/status: {rain_report.get('refresh_status')}")

    _, risk_meta, risk_val_report = run_himalayas_dynamic_risk_pipeline(
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
        overwrite=True,
    )
    if risk_val_report.get("status") != "SUCCESS" or not risk_val_report.get("is_operational_success"):
        raise RuntimeError(f"Himalayas Stage 8.4 risk prediction failed: {risk_val_report.get('status')}")

    map_res = generate_himalayas_dynamic_risk_map(
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
        overwrite=True,
    )
    if map_res.get("status") != "SUCCESS":
        raise RuntimeError(f"Himalayas Stage 8.5 map generation failed: {map_res.get('status')}")

    return {
        "rainfall_report": rain_report,
        "risk_metadata": risk_meta,
        "risk_validation_report": risk_val_report,
        "map_result": map_res,
    }


def _execute_northeast_pipeline(
    resolved_end_dt: Optional[datetime],
    offline: bool,
    now_utc: datetime,
    max_latency_hours: float,
) -> Dict[str, Any]:
    rain_report = integrate_northeast_gpm_rainfall(
        target_end_time=resolved_end_dt,
        allow_download=not offline,
        overwrite=True,
        now_utc=now_utc,
    )
    if rain_report.get("refresh_status") != "SUCCESS" or not rain_report.get("is_operational_success"):
        raise RuntimeError(f"Northeast India Stage 7.3 rainfall integration failed freshness/status: {rain_report.get('refresh_status')}")

    risk_meta = run_northeast_dynamic_risk_pipeline(
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
        overwrite=True,
    )
    fresh_status = risk_meta.get("gpm_observation_metadata", {}).get("freshness_status")
    if fresh_status != "SUCCESS":
        raise RuntimeError(f"Northeast India Stage 7.4 risk prediction freshness status is {fresh_status}")

    map_res = create_northeast_dynamic_risk_map(
        now_utc=now_utc,
        overwrite=True,
    )
    map_checks = map_res.get("validation_checks") or map_res.get("checks") or {}
    freshness_gate_passed = map_checks.get("freshness_gate_passed", False)
    if map_res.get("status") not in ("SUCCESS", "PASSED") or not freshness_gate_passed:
        raise RuntimeError(
            f"Northeast India Stage 7.5 map generation failed: status={map_res.get('status')}, "
            f"freshness_gate={freshness_gate_passed}"
        )

    return {
        "rainfall_report": rain_report,
        "risk_metadata": risk_meta,
        "map_result": map_res,
    }


def _execute_western_ghats_pipeline(
    resolved_end_dt: Optional[datetime],
    offline: bool,
    workers: int,
    now_utc: datetime,
    max_latency_hours: float,
) -> Dict[str, Any]:
    _ = workers
    rain_report = integrate_western_ghats_gpm_rainfall(
        target_end_time=resolved_end_dt,
        allow_download=not offline,
        overwrite=True,
        now_utc=now_utc,
    )
    if rain_report.get("refresh_status") != "SUCCESS" or not rain_report.get("is_operational_success"):
        raise RuntimeError(
            f"Western Ghats Stage 6.3 GPM rainfall failed gate: "
            f"status={rain_report.get('refresh_status')}, operational={rain_report.get('is_operational_success')}"
        )

    _, risk_meta, risk_report = predict_western_ghats_grid_dynamic_risk(
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
        overwrite=True,
    )
    if risk_report.get("status") != "SUCCESS":
        raise RuntimeError(f"Western Ghats Stage 6.4 dynamic risk prediction failed: {risk_report.get('status')}")

    map_report = generate_western_ghats_polygon_risk_map(
        max_latency_hours=max_latency_hours,
        now_utc=now_utc,
        overwrite=True,
    )
    if map_report.get("status") != "SUCCESS":
        raise RuntimeError(f"Western Ghats Stage 6.5 polygon map generation failed: {map_report.get('status')}")

    return {
        "rainfall_report": rain_report,
        "risk_metadata": risk_meta,
        "risk_report": risk_report,
        "map_report": map_report,
    }


def _inspect_geojson_features(geojson_path: Path, require_polygon: bool = False) -> Tuple[int, List[str]]:
    """Read a GeoJSON file and return (feature_count, unique_geometry_types)."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj.get("features", [])
    geom_types = sorted({
        (feat.get("geometry") or {}).get("type", "UNKNOWN")
        for feat in features
    })
    if require_polygon and features:
        allowed = {"Polygon", "MultiPolygon"}
        if not set(geom_types).issubset(allowed):
            raise ValueError(f"Expected Polygon/MultiPolygon geometries in {geojson_path.name}, got {geom_types}")
    return len(features), geom_types


def _collect_region_metrics_and_validate(
    region_key: str,
    spec: Dict[str, Any],
    expected_obs_end_utc: Optional[str],
    max_latency_hours: float,
) -> Dict[str, Any]:
    """
    Step 10 helper: Verify outputs, freshness alignment, 5-window monotonicity,
    category counts, metadata thresholds, GeoJSON counts, and HTML freshness script.
    """
    errors: List[str] = []
    warnings: List[str] = []

    risk_csv: Path = spec["risk_csv"]
    risk_meta_path: Path = spec["risk_metadata_json"]
    map_html: Path = spec["map_html"]
    map_meta_path: Path = spec["map_metadata_json"]
    map_report_path: Path = spec["map_report_json"]

    for req_path in [risk_csv, risk_meta_path, map_html, map_meta_path, map_report_path, *spec["synced_outputs"]]:
        if not req_path.exists() or req_path.stat().st_size == 0:
            errors.append(f"Missing or empty output: {req_path}")

    if errors:
        return {"validation_passed": False, "errors": errors, "warnings": warnings}

    # 1. Inspect Risk CSV & 5-window calculations
    df = pd.read_csv(risk_csv, low_memory=False)
    total_rows = int(len(df))
    unique_tiles = int(df["tile"].nunique()) if "tile" in df.columns else 0
    if total_rows != spec["expected_locations"]:
        errors.append(f"Row count mismatch: expected {spec['expected_locations']}, got {total_rows}")
    if unique_tiles != spec["expected_tiles"]:
        errors.append(f"Tile count mismatch: expected {spec['expected_tiles']}, got {unique_tiles}")

    window_stats: Dict[str, Dict[str, Any]] = {}
    for w_label, col in zip(RAINFALL_ACCUM_WINDOWS, RAINFALL_ACCUM_COLS):
        if col not in df.columns:
            errors.append(f"Missing rainfall window column {col}")
            window_stats[w_label] = {"present": False}
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        finite_vals = vals[np.isfinite(vals)]
        window_stats[w_label] = {
            "present": True,
            "finite_count": int(len(finite_vals)),
            "missing_count": int(total_rows - len(finite_vals)),
            "min_mm": round(float(finite_vals.min()), 4) if len(finite_vals) else None,
            "mean_mm": round(float(finite_vals.mean()), 4) if len(finite_vals) else None,
            "max_mm": round(float(finite_vals.max()), 4) if len(finite_vals) else None,
        }

    mono_ok = True
    if all(c in df.columns for c in RAINFALL_ACCUM_COLS):
        valid_mask = df[RAINFALL_ACCUM_COLS].apply(lambda s: pd.to_numeric(s, errors="coerce")).notna().all(axis=1)
        sub = df.loc[valid_mask, RAINFALL_ACCUM_COLS]
        if len(sub) == 0:
            mono_ok = False
            errors.append("No finite rows available for 5-window monotonicity check.")
        else:
            for a, b in zip(RAINFALL_ACCUM_COLS[:-1], RAINFALL_ACCUM_COLS[1:]):
                if not (sub[a] <= sub[b] + 1e-4).all():
                    mono_ok = False
                    errors.append(f"Monotonicity violated: {a} > {b}")
    else:
        mono_ok = False

    if "gpm_24h_complete" not in df.columns or not bool(df["gpm_24h_complete"].fillna(False).astype(bool).all()):
        errors.append("gpm_24h_complete is not True across all updated locations.")

    cat_series = df["primary_screening_category"].astype(str)
    raw_counts = cat_series.value_counts().to_dict()
    category_counts = {cat: int(raw_counts.get(cat, 0)) for cat in ACTIVE_CATEGORIES}
    assessed_count = sum(category_counts.values())
    unassessed_count = int(total_rows - assessed_count)
    category_counts["Unassessed"] = unassessed_count

    # 2. Validate Metadata Thresholds & Observation Synchronization
    with open(risk_meta_path, "r", encoding="utf-8") as f:
        risk_meta = json.load(f)
    cutoffs = validate_dynamic_metadata_thresholds(risk_meta)

    gpm_meta = risk_meta.get("gpm_observation_metadata", {})
    obs_start_utc = gpm_meta.get("observation_start_utc")
    obs_end_utc = gpm_meta.get("observation_end_utc")
    obs_end_dt = parse_iso8601_utc(obs_end_utc)
    if obs_end_dt is None:
        errors.append(f"Invalid or missing explicit UTC observation_end_utc in {risk_meta_path.name}")
    elif expected_obs_end_utc and obs_end_utc != expected_obs_end_utc:
        errors.append(f"Observation window mismatch: expected {expected_obs_end_utc}, got {obs_end_utc}")

    freshness_status = gpm_meta.get("freshness_status")
    if freshness_status != "SUCCESS":
        errors.append(f"Metadata freshness_status is {freshness_status}, expected SUCCESS.")

    latency_val = gpm_meta.get("calculated_latency_hours")
    if latency_val is None:
        latency_val = gpm_meta.get("independently_calculated_latency_hours")
    if latency_val is None:
        latency_val = gpm_meta.get("reported_latency_hours")
    if latency_val is None or float(latency_val) > max_latency_hours:
        errors.append(f"Metadata latency ({latency_val}) exceeds limit {max_latency_hours}h.")

    # Check map metadata alignment
    with open(map_meta_path, "r", encoding="utf-8") as f:
        map_meta = json.load(f)
    map_obs_end = map_meta.get("gpm_observation_metadata", {}).get("observation_end_utc")
    if obs_end_utc and map_obs_end and map_obs_end != obs_end_utc:
        errors.append(f"Map metadata obs_end_utc ({map_obs_end}) != risk metadata ({obs_end_utc})")

    # 3. Validate GeoJSON counts & HTML map properties
    geojson_inventory: Dict[str, Any] = {}
    for gj_path, exp_count in spec["expected_geojson_counts"].items():
        if not gj_path.exists():
            errors.append(f"Missing GeoJSON: {gj_path}")
            continue
        cnt, gtypes = _inspect_geojson_features(gj_path, require_polygon=False)
        geojson_inventory[str(gj_path)] = {
            "feature_count": cnt,
            "geometry_types": gtypes,
            "size_bytes": gj_path.stat().st_size,
        }
        if cnt != exp_count:
            errors.append(f"GeoJSON feature count mismatch in {gj_path.name}: expected {exp_count}, got {cnt}")

    if spec["regions_geojson"] is not None:
        reg_gj: Path = spec["regions_geojson"]
        if not reg_gj.exists():
            errors.append(f"Missing regional hazard polygon GeoJSON: {reg_gj}")
        else:
            cnt, gtypes = _inspect_geojson_features(reg_gj, require_polygon=True)
            geojson_inventory[str(reg_gj)] = {
                "feature_count": cnt,
                "geometry_types": gtypes,
                "size_bytes": reg_gj.stat().st_size,
            }
            if cnt <= 0:
                errors.append(f"Regional polygon GeoJSON has 0 features: {reg_gj.name}")

    html_size = map_html.stat().st_size
    if html_size > spec["html_max_bytes"]:
        errors.append(f"HTML map size {html_size} exceeds limit {spec['html_max_bytes']} bytes: {map_html.name}")

    html_text = map_html.read_text(encoding="utf-8")
    for needle in ("computeFreshness", "updateFreshnessStatus", "60000"):
        if needle not in html_text:
            errors.append(f"HTML map {map_html.name} missing runtime freshness guard token: {needle}")
    if obs_end_utc and obs_end_utc not in html_text:
        errors.append(f"HTML map {map_html.name} does not embed updated observation_end_utc {obs_end_utc}")

    return {
        "validation_passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_locations": total_rows,
        "unique_tiles": unique_tiles,
        **({"coverage_note": spec["coverage_note"]} if spec.get("coverage_note") else {}),
        "assessed_count": assessed_count,
        "unassessed_count": unassessed_count,
        "category_counts": category_counts,
        "relative_cutoffs": cutoffs,
        "observation_start_utc": obs_start_utc,
        "observation_end_utc": obs_end_utc,
        "latency_hours": round(float(latency_val), 3) if latency_val is not None else None,
        "windows_5_monotonic": mono_ok,
        "window_stats": window_stats,
        "gis_paths": {
            "html_map": str(map_html),
            "html_size_bytes": html_size,
            "map_metadata_json": str(map_meta_path),
            "map_validation_report_json": str(map_report_path),
            "risk_csv": str(risk_csv),
            "risk_metadata_json": str(risk_meta_path),
            "geojson_outputs": geojson_inventory,
            "synced_outputs": [str(p) for p in spec["synced_outputs"]],
        },
    }


def run_validation_tests(
    regions: Sequence[str],
    region_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Step 10 test runner: Execute unittest verification suites for refreshed regions.
    """
    modules: List[str] = []
    for rkey in regions:
        modules.extend(region_specs[rkey]["test_modules"])
    if Path(PROJECT_ROOT / "backend" / "test_standard_refresh_procedure.py").exists():
        # Do not invoke ourselves recursively if invoked inside test_standard_refresh_procedure
        if "unittest" not in sys.argv[0].lower():
            modules.append("backend.test_standard_refresh_procedure")

    suite_results: List[Dict[str, Any]] = []
    all_passed = True
    t_start = datetime.now(timezone.utc)

    for mod in modules:
        t0 = datetime.now(timezone.utc)
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-m", "unittest", mod],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        elapsed_s = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
        ok = proc.returncode == 0
        if not ok:
            all_passed = False
        combined = f"{proc.stdout}\n{proc.stderr}".strip()
        summary_line = ""
        for line in reversed(combined.splitlines()):
            if line.strip().startswith("Ran ") or line.strip().startswith("OK") or line.strip().startswith("FAILED"):
                summary_line = f"{line.strip()} | {summary_line}".strip(" |")
        suite_results.append({
            "module": mod,
            "passed": ok,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed_s,
            "summary": summary_line[:240],
        })

    return {
        "executed": True,
        "all_passed": all_passed,
        "suites_total": len(suite_results),
        "suites_passed": sum(1 for s in suite_results if s["passed"]),
        "total_elapsed_seconds": round((datetime.now(timezone.utc) - t_start).total_seconds(), 2),
        "suites": suite_results,
    }


def render_run_summary_markdown(report: Dict[str, Any]) -> str:
    """Render the mandatory 12-item Standard Refresh Run Summary in Markdown."""
    obs = report["observation_window"]
    lat = report["data_latency"]
    gran = report["granule_inventory"]
    rwin = report["rainfall_window_status"]
    locs = report["updated_locations_summary"]
    cov = report["assessment_coverage"]
    val_tests = report["validation_and_test_results"]

    lines = [
        f"# Standard Refresh Run Summary — `{report['run_id']}`",
        "",
        "## 1–6. Execution, Window, Latency & Granule Telemetry",
        "",
        "| Item | Metric | Value |",
        "|---|---|---|",
        f"| 1 | **Run ID** | `{report['run_id']}` |",
        f"| 2 | **Execution Timestamp (UTC)** | `{report['execution_timestamp_utc']}` |",
        f"| 3 | **Final Run Status** | **{report['status']}** |",
        f"| 4 | **Observation Window (UTC)** | `{obs.get('observation_start_utc')}` to `{obs.get('observation_end_utc')}` (`{obs.get('discovery_source')}`) |",
        f"| 5 | **Observed Data Latency** | `{lat.get('latency_hours')} h` (Limit: `{lat.get('max_latency_hours')} h`, Fresh: `{lat.get('is_fresh')}`) |",
        f"| 6 | **GPM Granules (24h Window)** | Discovered: `{gran.get('granules_discovered_cmr')}`, Downloaded: `{gran.get('granules_downloaded')}`, Cached: `{gran.get('granules_cached')}`, Validated: `{gran.get('granules_validated')}/{gran.get('expected_granules')}` |",
        f"| 7 | **5-Window Rainfall Status** | `{rwin.get('status')}` (`{', '.join(rwin.get('windows', []))}`) |",
        f"| 8 | **Updated Locations Total** | `{locs.get('total_updated_locations', 0):,}` |",
        f"| 9 | **Assessed vs Unassessed** | Assessed: `{cov.get('total_assessed_locations', 0):,}` / Unassessed: `{cov.get('total_unassessed_locations', 0):,}` |",
        "",
        "## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths",
        "",
    ]

    for rkey, rdata in report["regions"].items():
        lines.append(f"### {rdata.get('display_name', rkey)} — `{rdata.get('status')}`")
        lines.append("")
        if rdata.get("status") != "SUCCESS":
            lines.append(f"- **Error**: {rdata.get('error', 'Failed')}")
            lines.append(f"- **Rollback Performed**: `{rdata.get('rollback_performed', False)}`")
            lines.append("")
            continue
        cats = rdata.get("category_counts", {})
        cuts = rdata.get("relative_cutoffs", {})
        gis = rdata.get("gis_paths", {})
        lines.extend([
            *([f"- **Coverage Scope**: {rdata['coverage_note']}"] if rdata.get("coverage_note") else []),
            f"- **Updated Locations**: `{rdata.get('total_locations', 0):,}` across `{rdata.get('unique_tiles', 0)}` tiles (`Assessed={rdata.get('assessed_count', 0):,}`, `Unassessed={rdata.get('unassessed_count', 0):,}`)",
            f"- **Dynamic Cutoffs**: `P50={cuts.get('P50_low_to_moderate', 0):.2f}`, `P80={cuts.get('P80_moderate_to_high', 0):.2f}`, `P95={cuts.get('P95_high_to_very_high', 0):.2f}`",
            f"- **Category Counts**: Low=`{cats.get('Low', 0):,}`, Moderate=`{cats.get('Moderate', 0):,}`, High=`{cats.get('High', 0):,}`, Very High=`{cats.get('Very High', 0):,}`, Unassessed=`{cats.get('Unassessed', 0):,}`",
            f"- **HTML Map**: `{gis.get('html_map')}` (`{gis.get('html_size_bytes', 0):,}` bytes)",
            f"- **Risk CSV**: `{gis.get('risk_csv')}`",
            f"- **GeoJSON Outputs**:",
        ])
        for gpath, ginfo in gis.get("geojson_outputs", {}).items():
            lines.append(f"  - `{gpath}` (`{ginfo.get('feature_count', 0):,}` features, types=`{ginfo.get('geometry_types')}`, `{ginfo.get('size_bytes', 0):,}` bytes)")
        lines.append("")

    lines.extend([
        "## 12. Validation, Test Results, Errors & Skipped Regions",
        "",
        f"- **Pre-flight Check Status**: `{val_tests.get('preflight_status')}` (`{val_tests.get('preflight_passed')}/{val_tests.get('preflight_total')}` checks)",
        f"- **Cross-Region Consistency Passed**: `{val_tests.get('cross_region_validation_passed')}`",
        f"- **Unit Tests Executed**: `{val_tests.get('unit_tests', {}).get('executed', False)}` | **All Passed**: `{val_tests.get('unit_tests', {}).get('all_passed')}` (`{val_tests.get('unit_tests', {}).get('suites_passed', 0)}/{val_tests.get('unit_tests', {}).get('suites_total', 0)}` suites)",
        f"- **Skipped Regions**: `{val_tests.get('skipped_regions')}`",
        f"- **Errors**: `{val_tests.get('errors')}`",
        f"- **Warnings**: `{val_tests.get('warnings')}`",
    ])
    for s in val_tests.get("unit_tests", {}).get("suites", []):
        lines.append(f"  - `{s['module']}`: passed=`{s['passed']}` ({s['elapsed_seconds']}s) — {s['summary']}")

    lines.append("")
    return "\n".join(lines)


def build_run_summary_report(
    run_id: str,
    execution_dt: datetime,
    status: str,
    preflight: Dict[str, Any],
    discovery: Optional[Dict[str, Any]],
    region_results: Dict[str, Dict[str, Any]],
    skipped_regions: List[str],
    unit_tests: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    transaction_recovery: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical 12-item Standard Refresh Run Summary payload."""
    disc = discovery or {}
    total_updated = sum(
        int(r.get("total_locations", 0)) for r in region_results.values() if r.get("status") == "SUCCESS"
    )
    total_assessed = sum(
        int(r.get("assessed_count", 0)) for r in region_results.values() if r.get("status") == "SUCCESS"
    )
    total_unassessed = sum(
        int(r.get("unassessed_count", 0)) for r in region_results.values() if r.get("status") == "SUCCESS"
    )

    agg_categories = {cat: 0 for cat in (*ACTIVE_CATEGORIES, "Unassessed")}
    per_region_windows: Dict[str, Any] = {}
    all_mono = True
    has_any_success = False
    for rkey, rdata in region_results.items():
        if rdata.get("status") == "SUCCESS":
            has_any_success = True
            for cat in agg_categories:
                agg_categories[cat] += int(rdata.get("category_counts", {}).get(cat, 0))
            per_region_windows[rkey] = rdata.get("window_stats", {})
            if not rdata.get("windows_5_monotonic", False):
                all_mono = False

    cross_valid_ok = all(
        r.get("validation_passed", False)
        for r in region_results.values()
        if r.get("status") == "SUCCESS"
    ) if region_results else False

    if status == "DRY_RUN":
        win_status = "DRY_RUN_NOT_COMPUTED"
    elif has_any_success and all_mono:
        win_status = "VERIFIED_MONOTONIC"
    elif has_any_success:
        win_status = "MONOTONICITY_OR_WINDOW_WARNING"
    else:
        win_status = "NOT_COMPUTED_OR_FAILED"

    return {
        "schema_version": "standard_refresh_report_v1",
        "run_id": run_id,
        "execution_timestamp_utc": _utc_iso(execution_dt),
        "status": status,
        "observation_window": {
            "observation_start_utc": disc.get("observation_start_utc"),
            "observation_end_utc": disc.get("observation_end_utc"),
            "window_hours": 24,
            "discovery_source": disc.get("discovery_source"),
        },
        "data_latency": {
            "latency_hours": disc.get("latency_hours"),
            "max_latency_hours": disc.get("max_latency_hours"),
            "is_fresh": disc.get("is_fresh", False),
            "freshness_status": disc.get("freshness_status"),
        },
        "granule_inventory": {
            "granules_discovered_cmr": disc.get("granules_discovered_cmr", 0),
            "granules_downloaded": disc.get("granules_downloaded", 0),
            "granules_cached": disc.get("granules_cached", 0),
            "granules_validated": disc.get("granules_validated", 0),
            "expected_granules": EXPECTED_GRANULES_24H,
            "missing_intervals_count": disc.get("missing_intervals_count", 0),
            "is_24h_complete": disc.get("is_24h_complete", False),
        },
        "rainfall_window_status": {
            "windows": list(RAINFALL_ACCUM_WINDOWS),
            "status": win_status,
            "per_region_summary": per_region_windows,
        },
        "updated_locations_summary": {
            "total_updated_locations": total_updated,
            "by_region": {
                rkey: rdata.get("total_locations", 0)
                for rkey, rdata in region_results.items()
                if rdata.get("status") == "SUCCESS"
            },
        },
        "assessment_coverage": {
            "total_assessed_locations": total_assessed,
            "total_unassessed_locations": total_unassessed,
            "by_region": {
                rkey: {
                    "assessed_count": rdata.get("assessed_count", 0),
                    "unassessed_count": rdata.get("unassessed_count", 0),
                }
                for rkey, rdata in region_results.items()
                if rdata.get("status") == "SUCCESS"
            },
        },
        "category_distribution_and_thresholds": {
            "aggregate_category_counts": agg_categories,
            "by_region": {
                rkey: {
                    "category_counts": rdata.get("category_counts", {}),
                    "relative_cutoffs": rdata.get("relative_cutoffs", {}),
                }
                for rkey, rdata in region_results.items()
                if rdata.get("status") == "SUCCESS"
            },
        },
        "gis_outputs": {
            rkey: rdata.get("gis_paths", {})
            for rkey, rdata in region_results.items()
            if rdata.get("status") == "SUCCESS"
        },
        "validation_and_test_results": {
            "preflight_status": preflight.get("status"),
            "preflight_passed": preflight.get("checks_passed", 0),
            "preflight_total": preflight.get("checks_total", 0),
            "cross_region_validation_passed": cross_valid_ok,
            "unit_tests": unit_tests,
            "errors": errors,
            "warnings": warnings,
            "skipped_regions": skipped_regions,
            "transaction_recovery": transaction_recovery,
        },
        "regions": region_results,
    }


def _persist_run_reports(report: Dict[str, Any], report_dir: Path) -> Tuple[Path, Path]:
    """Write timestamped and latest JSON + Markdown run reports atomically."""
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = report["run_id"]
    md_text = render_run_summary_markdown(report)

    ts_json = report_dir / f"refresh_report_{run_id}.json"
    ts_md = report_dir / f"refresh_report_{run_id}.md"
    latest_json = report_dir / "latest_refresh_report.json"
    latest_md = report_dir / "latest_refresh_report.md"

    atomic_write_json(ts_json, report)
    atomic_write_text(ts_md, md_text)
    atomic_write_json(latest_json, report)
    atomic_write_text(latest_md, md_text)
    return ts_json, ts_md


def run_standard_refresh(
    regions: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    offline: bool = False,
    target_end_time: Optional[datetime] = None,
    max_latency_hours: float = DEFAULT_MAX_LATENCY_HOURS,
    workers: int = 4,
    skip_tests: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
    now_utc: Optional[datetime] = None,
    env_file: Optional[Path] = PROJECT_ROOT / ".env",
    verify_model_load: bool = True,
    region_executors: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
    region_validators: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
    tx_backup_root: Optional[Path] = None,
    test_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Acquire the process-wide refresh lock and execute Steps 1-11 safely.

    If another refresh process already holds the lock, records the skipped
    execution cleanly and returns status ``SKIPPED_LOCKED`` without modifying
    project datasets or overwriting the latest valid run report.
    """
    exec_now = now_utc or datetime.now(timezone.utc)
    run_id = f"std_refresh_{exec_now.strftime('%Y%m%dT%H%M%SZ')}"
    resolved_lock_path = lock_path or (report_dir / DEFAULT_LOCK_FILENAME)
    lock = RefreshProcessLock(resolved_lock_path, run_id=run_id)
    if not lock.acquire():
        skipped_log = record_skipped_execution(
            report_dir=report_dir,
            run_id=run_id,
            execution_dt=exec_now,
            lock_path=resolved_lock_path,
            lock_holder=lock.holder_info,
        )
        print(
            f"[LOCKED] Concurrent Standard Refresh already active "
            f"(lock={resolved_lock_path}, holder={lock.holder_info}). "
            f"Exiting cleanly without modifying project data; recorded in {skipped_log}."
        )
        return {
            "schema_version": "standard_refresh_report_v1",
            "run_id": run_id,
            "execution_timestamp_utc": _utc_iso(exec_now),
            "status": "SKIPPED_LOCKED",
            "lock_path": str(resolved_lock_path),
            "lock_holder": lock.holder_info,
            "skipped_runs_log": str(skipped_log),
            "reason": "Another Standard Refresh process currently holds the refresh lock.",
        }

    try:
        return _run_standard_refresh_locked(
            run_id=run_id,
            exec_now=exec_now,
            regions=regions,
            dry_run=dry_run,
            offline=offline,
            target_end_time=target_end_time,
            max_latency_hours=max_latency_hours,
            workers=workers,
            skip_tests=skip_tests,
            report_dir=report_dir,
            env_file=env_file,
            verify_model_load=verify_model_load,
            region_executors=region_executors,
            region_validators=region_validators,
            tx_backup_root=tx_backup_root,
            test_runner=test_runner,
        )
    finally:
        lock.release()


def _run_standard_refresh_locked(
    run_id: str,
    exec_now: datetime,
    regions: Optional[Sequence[str]],
    dry_run: bool,
    offline: bool,
    target_end_time: Optional[datetime],
    max_latency_hours: float,
    workers: int,
    skip_tests: bool,
    report_dir: Path,
    env_file: Optional[Path],
    verify_model_load: bool,
    region_executors: Optional[Dict[str, Callable[..., Dict[str, Any]]]],
    region_validators: Optional[Dict[str, Callable[..., Dict[str, Any]]]],
    tx_backup_root: Optional[Path],
    test_runner: Optional[Callable[..., Dict[str, Any]]],
) -> Dict[str, Any]:
    """Execute Steps 1-11 while holding the process-wide Standard Refresh lock."""
    selected_regions = normalize_regions(regions)
    skipped_regions = [r for r in REGION_ORDER if r not in selected_regions]
    specs = _build_region_specs()

    errors: List[str] = []
    warnings: List[str] = []
    region_results: Dict[str, Dict[str, Any]] = {}
    discovery: Optional[Dict[str, Any]] = None
    unit_tests: Dict[str, Any] = {"executed": False, "all_passed": None, "suites": []}

    print("=" * 78)
    print(f"LANDSLIDE PREDICTION SYSTEM — STANDARD REFRESH RUN ({run_id})")
    print(f"Execution UTC: {exec_now.strftime('%Y-%m-%dT%H:%M:%SZ')} | Regions: {selected_regions}")
    print("=" * 78)

    tx_root_parent = tx_backup_root or (PROJECT_ROOT / "data" / "processed" / "_standard_refresh_tx_backups")
    recovery_summary = recover_interrupted_transactions(tx_root_parent)
    if recovery_summary["recovered_transactions"]:
        warnings.append(
            f"Recovered {len(recovery_summary['recovered_transactions'])} interrupted transaction(s) "
            f"from an earlier crash: {[t['region_key'] for t in recovery_summary['recovered_transactions']]}."
        )
    if recovery_summary["preserved_uncertain"]:
        warnings.append(
            f"Preserved {len(recovery_summary['preserved_uncertain'])} uncertain/failed transaction backup(s) "
            f"under {tx_root_parent} for operator inspection."
        )
    if recovery_summary["errors"]:
        errors.extend(recovery_summary["errors"])
        report = build_run_summary_report(
            run_id=run_id,
            execution_dt=exec_now,
            status="FAILED",
            preflight={
                "status": "FAILED",
                "checks_passed": 0,
                "checks_total": 0,
                "errors": list(recovery_summary["errors"]),
            },
            discovery=None,
            region_results={},
            skipped_regions=skipped_regions,
            unit_tests=unit_tests,
            errors=errors,
            warnings=warnings,
            transaction_recovery=recovery_summary,
        )
        _persist_run_reports(report, report_dir)
        print(f"CRASH RECOVERY HALTED SAFELY: {recovery_summary['errors']}")
        return report

    # Step 1: Pre-flight checks
    print("\n[STEP 1/11] Running pre-flight checks (environment, inputs, RF models, contracts)...")
    preflight = check_preflight(
        regions=selected_regions,
        region_specs=specs,
        offline=offline,
        report_dir=report_dir,
        env_file=env_file,
        verify_model_load=verify_model_load,
    )
    if preflight["status"] != "PASSED":
        errors.extend(preflight["errors"])
        report = build_run_summary_report(
            run_id=run_id,
            execution_dt=exec_now,
            status="FAILED",
            preflight=preflight,
            discovery=None,
            region_results={},
            skipped_regions=skipped_regions,
            unit_tests=unit_tests,
            errors=errors,
            warnings=warnings,
            transaction_recovery=recovery_summary,
        )
        _persist_run_reports(report, report_dir)
        print(f"PRE-FLIGHT FAILED: {errors}")
        return report

    # Steps 2 & 3: Data Discovery & Retrieval
    print("\n[STEPS 2–3/11] Resolving NASA GPM IMERG V07C 24h window and granules...")
    try:
        discovery = discover_and_retrieve_gpm(
            target_end_time=target_end_time,
            offline=offline,
            workers=workers,
            now_utc=exec_now,
            max_latency_hours=max_latency_hours,
            dry_run=dry_run,
        )
    except Exception as exc:
        errors.append(f"GPM discovery/retrieval error: {exc}")
        report = build_run_summary_report(
            run_id=run_id,
            execution_dt=exec_now,
            status="FAILED",
            preflight=preflight,
            discovery=discovery,
            region_results={},
            skipped_regions=skipped_regions,
            unit_tests=unit_tests,
            errors=errors,
            warnings=warnings,
            transaction_recovery=recovery_summary,
        )
        _persist_run_reports(report, report_dir)
        print(f"DISCOVERY/RETRIEVAL FAILED: {exc}")
        return report

    print(
        f"  Window: {discovery['observation_start_utc']} -> {discovery['observation_end_utc']} | "
        f"Latency: {discovery['latency_hours']}h | Granules: {discovery['granules_validated']}/48 | "
        f"Status: {discovery['freshness_status']}"
    )

    if dry_run:
        warnings.append("Dry-run requested: stopped after Step 2 discovery without modifying datasets or maps.")
        report = build_run_summary_report(
            run_id=run_id,
            execution_dt=exec_now,
            status="DRY_RUN",
            preflight=preflight,
            discovery=discovery,
            region_results={},
            skipped_regions=skipped_regions,
            unit_tests=unit_tests,
            errors=errors,
            warnings=warnings,
            transaction_recovery=recovery_summary,
        )
        _persist_run_reports(report, report_dir)
        return report

    # Freshness / Completeness Gate before modifying regional operational outputs
    if not discovery["is_24h_complete"] or not discovery["is_fresh"]:
        msg = (
            f"Freshness/completeness gate halted Steps 4-9 safely: "
            f"status={discovery['freshness_status']}, latency={discovery['latency_hours']}h, "
            f"granules={discovery['granules_validated']}/48 ({discovery.get('freshness_reason')}). "
            "Retained prior validated datasets untouched."
        )
        errors.append(msg)
        report = build_run_summary_report(
            run_id=run_id,
            execution_dt=exec_now,
            status="FAILED",
            preflight=preflight,
            discovery=discovery,
            region_results={},
            skipped_regions=skipped_regions,
            unit_tests=unit_tests,
            errors=errors,
            warnings=warnings,
            transaction_recovery=recovery_summary,
        )
        _persist_run_reports(report, report_dir)
        print(f"HALTED SAFELY: {msg}")
        return report

    # Steps 4–9 per region wrapped in transactional backup/rollback
    resolved_end_dt = discovery["resolved_end_dt"]
    default_executors: Dict[str, Callable[..., Dict[str, Any]]] = {
        "himalayas": lambda: _execute_himalayas_pipeline(
            resolved_end_dt=resolved_end_dt,
            offline=offline,
            now_utc=exec_now,
            max_latency_hours=max_latency_hours,
        ),
        "northeast": lambda: _execute_northeast_pipeline(
            resolved_end_dt=resolved_end_dt,
            offline=offline,
            now_utc=exec_now,
            max_latency_hours=max_latency_hours,
        ),
        "western_ghats": lambda: _execute_western_ghats_pipeline(
            resolved_end_dt=resolved_end_dt,
            offline=offline,
            workers=workers,
            now_utc=exec_now,
            max_latency_hours=max_latency_hours,
        ),
    }
    executors = dict(default_executors)
    if region_executors:
        executors.update(region_executors)

    tx_backup_base = tx_root_parent / run_id
    for rkey in selected_regions:
        spec = specs[rkey]
        print(f"\n[STEPS 4–9] Refreshing {spec['display_name']} ({spec['stages']})...")
        tx = RegionTransaction(rkey, spec["tracked_outputs"], tx_backup_base / rkey, run_id=run_id)
        try:
            with tx:
                executors[rkey]()
                if region_validators and rkey in region_validators:
                    metrics = region_validators[rkey](
                        region_key=rkey,
                        spec=spec,
                        expected_obs_end_utc=discovery["observation_end_utc"],
                        max_latency_hours=max_latency_hours,
                    )
                else:
                    metrics = _collect_region_metrics_and_validate(
                        region_key=rkey,
                        spec=spec,
                        expected_obs_end_utc=discovery["observation_end_utc"],
                        max_latency_hours=max_latency_hours,
                    )
                if not metrics.get("validation_passed"):
                    raise RuntimeError(
                        f"{spec['display_name']} Step 10 validation errors: {metrics.get('errors')}"
                    )
            region_results[rkey] = {
                "display_name": spec["display_name"],
                "status": "SUCCESS",
                "rollback_performed": False,
                **metrics,
            }
        except Exception as exc:
            err_msg = f"{spec['display_name']} refresh failed and was rolled back: {exc}"
            errors.append(err_msg)
            region_results[rkey] = {
                "display_name": spec["display_name"],
                "status": "FAILED",
                "rollback_performed": tx.rolled_back,
                "validation_passed": False,
                "error": str(exc),
                "traceback": traceback.format_exc()[-800:],
            }
            print(f"  [ERROR] {err_msg}")
    try:
        # Remove only empty directories so any preserved ROLLBACK_FAILED or
        # uncertain transaction backup directory stays intact for inspection.
        if tx_backup_base.exists() and not any(tx_backup_base.iterdir()):
            tx_backup_base.rmdir()
        if tx_backup_root is None and tx_root_parent.exists() and not any(tx_root_parent.iterdir()):
            tx_root_parent.rmdir()
    except OSError:
        pass

    # Step 10: Unit test suite verification
    succeeded_regions = [r for r in selected_regions if region_results.get(r, {}).get("status") == "SUCCESS"]
    if not skip_tests and succeeded_regions:
        print("\n[STEP 10/11] Running unittest verification suites across refreshed regions...")
        runner = test_runner or run_validation_tests
        unit_tests = runner(succeeded_regions, specs)
        if not unit_tests.get("all_passed", False):
            errors.append(
                f"Validation test suite failures: "
                f"{unit_tests.get('suites_passed', 0)}/{unit_tests.get('suites_total', 0)} suites passed."
            )
    elif skip_tests:
        warnings.append("Unit test verification skipped (--skip-tests).")

    # Final status determination: SUCCESS requires all selected regions + validation + executed tests passing
    n_success = len(succeeded_regions)
    n_selected = len(selected_regions)
    tests_ok = True if skip_tests else bool(unit_tests.get("all_passed", False))
    if n_success == n_selected and tests_ok and not errors:
        final_status = "SUCCESS"
    elif n_success > 0:
        final_status = "PARTIAL"
    else:
        final_status = "FAILED"

    # Step 11: Persist 12-item JSON + Markdown reports
    print(f"\n[STEP 11/11] Writing 12-item Standard Refresh Run Report ({final_status})...")
    report = build_run_summary_report(
        run_id=run_id,
        execution_dt=exec_now,
        status=final_status,
        preflight=preflight,
        discovery=discovery,
        region_results=region_results,
        skipped_regions=skipped_regions,
        unit_tests=unit_tests,
        errors=errors,
        warnings=warnings,
        transaction_recovery=recovery_summary,
    )
    ts_json, ts_md = _persist_run_reports(report, report_dir)
    print(f"  - JSON Report:     {ts_json}")
    print(f"  - Markdown Report: {ts_md}")
    print(f"  - Final Status:    {final_status}")
    print("=" * 78)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the 11-step Standard Refresh Run Procedure for Himalayas, Northeast India, and Western Ghats."
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(REGION_ORDER),
        help="Regions to refresh: himalayas, northeast, western_ghats, or all (default: all 3).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run Steps 1-2 discovery checks without modifying files.")
    parser.add_argument("--offline", action="store_true", help="Use local HDF5 archive only without CMR downloads.")
    parser.add_argument("--target-end-utc", type=str, default=None, help="Explicit UTC observation end (YYYY-MM-DDTHH:MM:SSZ).")
    parser.add_argument("--max-latency-hours", type=float, default=DEFAULT_MAX_LATENCY_HOURS, help="Maximum allowed observation age in hours (default: 12.0).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent GPM download workers (default: 4).")
    parser.add_argument("--skip-tests", action="store_true", help="Skip Step 10 unittest suites.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Output directory for run reports.")
    parser.add_argument("--lock-path", type=Path, default=None, help="Optional override for the process-wide refresh lock file.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    target_dt = None
    if args.target_end_utc:
        target_dt = parse_iso8601_utc(args.target_end_utc)
        if target_dt is None:
            print("ERROR: --target-end-utc must be explicit UTC ISO-8601 (e.g. 2026-11-11T12:00:00Z)", file=sys.stderr)
            return 2

    report = run_standard_refresh(
        regions=args.regions,
        dry_run=args.dry_run,
        offline=args.offline,
        target_end_time=target_dt,
        max_latency_hours=args.max_latency_hours,
        workers=args.workers,
        skip_tests=args.skip_tests,
        report_dir=args.report_dir,
        lock_path=args.lock_path,
    )
    return 0 if report["status"] in ("SUCCESS", "DRY_RUN", "SKIPPED_LOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
