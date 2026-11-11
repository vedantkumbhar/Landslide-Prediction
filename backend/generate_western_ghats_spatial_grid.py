"""
Western Ghats Limited Pilot — 5 km UTM Spatial Grid Generator
==============================================================

Generates a regular 5 km x 5 km UTM (EPSG:32643) spatial evaluation grid
strictly across the 14 validated Western Ghats DEM tiles.

Coverage Disclosure (Western Ghats Limited Pilot):
- Derived strictly from 14 validated 30 m ALOS/SRTM metric DEM tiles across
  `western_ghats`, `western_ghats_tier1`, and `western_maharashtra`.
- Does NOT represent complete Western Ghats mountain range coverage.

Ocean Exclusion & Land-Mask Rule:
- Uses Natural Earth 10m Land (`ne_10m_land.shp`) clipped in EPSG:4326 before
  reprojection to EPSG:32643, with a documented 1,000 m (1 km) coastal tolerance.
- Points strictly inside exact Natural Earth land boundaries with valid finite
  DEM pixels are retained (including valid below-sea-level coastal land such as
  Kerala's Kuttanad lowlands).
- Points strictly in the outer (0 m, 1,000 m] coastal buffer are retained ONLY if
  30 m DEM elevation > 0.0 m (real coastal headlands/cliffs resolved better by
  30 m DEM than 1:10M coastline), rejecting flat 0 m and negative sea pixels.
- Points beyond 1,000 m offshore are strictly excluded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep

from backend.gis_refresh_utils import atomic_write_json, create_timestamped_backup


LOGGER = logging.getLogger("western_ghats_spatial_grid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEM_ROOT = PROJECT_ROOT / "data" / "processed" / "dem"
DEFAULT_LAND_MASK_SHP = (
    PROJECT_ROOT / "data" / "raw" / "reference" / "naturalearth_land" / "ne_10m_land.shp"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "western_ghats_spatial_grid"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "western_ghats_5km_spatial_grid.csv"
DEFAULT_VALIDATION_REPORT = DEFAULT_OUTPUT_DIR / "western_ghats_spatial_grid_validation_report.json"

WGS84_EPSG = 4326
WG_UTM_EPSG = 32643
DEFAULT_GRID_SPACING_M = 5000.0
DEFAULT_COASTAL_TOLERANCE_M = 1000.0
WGS84_CLIP_BOUNDS = (71.0, 8.0, 79.0, 21.0)
NODATA_SENTINEL = -9999.0

VALIDATED_WG_TILE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "western_ghats": ("N09E076", "N11E076", "N19E072"),
    "western_ghats_tier1": (
        "N10E076",
        "N10E077",
        "N12E074",
        "N12E075",
        "N14E074",
        "N19E073",
    ),
    "western_maharashtra": (
        "N15E073",
        "N15E074",
        "N16E073",
        "N17E073",
        "N18E073",
    ),
}

EXPECTED_TILE_IDS: Tuple[str, ...] = tuple(
    sorted(tile_id for tiles in VALIDATED_WG_TILE_GROUPS.values() for tile_id in tiles)
)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False, float_format="%.8f")
    os.replace(tmp_path, path)


def parse_tile_bounds(tile_id: str) -> Tuple[int, int, int, int]:
    """Parse SRTM-style tile ID (e.g. N12E074) into (min_lon, min_lat, max_lon, max_lat)."""
    clean = tile_id.strip().upper()
    if len(clean) != 7 or clean[0] not in ("N", "S") or clean[3] not in ("E", "W"):
        raise ValueError(f"Unsupported tile ID format: {tile_id!r}")
    lat_val = int(clean[1:3]) * (1 if clean[0] == "N" else -1)
    lon_val = int(clean[4:7]) * (1 if clean[3] == "E" else -1)
    return lon_val, lat_val, lon_val + 1, lat_val + 1


def discover_validated_tile_paths(
    dem_root: Path = DEFAULT_DEM_ROOT,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve and validate the exact 14 Western Ghats Limited Pilot DEM tiles.

    Refuses missing tiles, missing metric rasters, or non-EPSG:32643 CRSs.
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    for group_name, tile_ids in VALIDATED_WG_TILE_GROUPS.items():
        for tile_id in tile_ids:
            metric_dir = dem_root / group_name / tile_id / "metric"
            elev_path = metric_dir / "elevation.tif"
            slope_path = metric_dir / "slope.tif"
            aspect_path = metric_dir / "aspect.tif"
            for required_path in (elev_path, slope_path, aspect_path):
                if not required_path.exists():
                    raise FileNotFoundError(
                        f"Missing required metric DEM raster for validated tile {tile_id}: {required_path}"
                    )
            with rasterio.open(elev_path) as src:
                epsg = src.crs.to_epsg() if src.crs is not None else None
                if epsg != WG_UTM_EPSG:
                    raise ValueError(
                        f"Tile {tile_id} elevation raster CRS EPSG={epsg}, expected EPSG:{WG_UTM_EPSG}"
                    )
            resolved[tile_id] = {
                "tile_id": tile_id,
                "tile_group": group_name,
                "metric_dir": metric_dir,
                "elevation_tif": elev_path,
                "slope_tif": slope_path,
                "aspect_tif": aspect_path,
                "epsg": WG_UTM_EPSG,
            }

    if len(resolved) != 14 or set(resolved.keys()) != set(EXPECTED_TILE_IDS):
        raise ValueError(
            f"Expected exactly 14 validated Western Ghats DEM tiles, got {len(resolved)}."
        )
    return resolved


def load_western_ghats_land_mask(
    land_mask_shp: Path = DEFAULT_LAND_MASK_SHP,
    coastal_tolerance_m: float = DEFAULT_COASTAL_TOLERANCE_M,
) -> Tuple[Any, Any]:
    """
    Load Natural Earth 10m land mask clipped safely in WGS84 and projected to EPSG:32643.

    Clipping before reprojection prevents Eurasia antimeridian TopologyExceptions.
    Returns (exact_land_geom_utm, buffered_land_geom_utm).
    """
    if not land_mask_shp.exists():
        raise FileNotFoundError(f"Natural Earth land shapefile not found: {land_mask_shp}")
    if coastal_tolerance_m < 0.0:
        raise ValueError(f"coastal_tolerance_m must be >= 0, got {coastal_tolerance_m}")

    gdf_world = gpd.read_file(land_mask_shp)
    if gdf_world.empty:
        raise ValueError(f"Loaded empty land mask shapefile: {land_mask_shp}")
    if gdf_world.crs is None:
        gdf_world = gdf_world.set_crs(epsg=WGS84_EPSG)

    clip_poly = box(*WGS84_CLIP_BOUNDS)
    gdf_clipped = gpd.clip(gdf_world.to_crs(epsg=WGS84_EPSG), clip_poly)
    if gdf_clipped.empty:
        raise ValueError("Clipped Natural Earth land geometry is empty for Western Ghats bounds.")

    gdf_utm = gdf_clipped.to_crs(epsg=WG_UTM_EPSG)
    land_geom_utm = unary_union(gdf_utm.geometry.values)
    buffer_geom_utm = (
        land_geom_utm.buffer(float(coastal_tolerance_m))
        if coastal_tolerance_m > 0.0
        else land_geom_utm
    )
    return land_geom_utm, buffer_geom_utm


def _compute_aspect_derivatives(aspect_deg: float) -> Tuple[float, float, int]:
    """Compute exact 5-feature RF contract derivatives from aspect degrees."""
    if not math.isfinite(aspect_deg) or aspect_deg < 0.0:
        return 0.0, 0.0, 1
    rad = math.radians(aspect_deg)
    return float(math.sin(rad)), float(math.cos(rad)), 0


def _sample_raster_values(
    raster_path: Path,
    coords: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, float]:
    """Sample single-band raster values at UTM (x, y) coordinates."""
    with rasterio.open(raster_path) as src:
        nodata = float(src.nodata) if src.nodata is not None else NODATA_SENTINEL
        sampled = np.array([val[0] for val in src.sample(coords)], dtype=np.float64)
    return sampled, nodata


def _is_valid_sample(val: float, nodata: float) -> bool:
    if not math.isfinite(val):
        return False
    if abs(val - nodata) <= 1e-6 or abs(val - NODATA_SENTINEL) <= 1e-6:
        return False
    return True


def _recover_seam_slope_aspect(
    slope_src: rasterio.DatasetReader,
    aspect_src: rasterio.DatasetReader,
    ux: float,
    uy: float,
    slope_nodata: float,
    aspect_nodata: float,
) -> Optional[Tuple[float, float]]:
    """
    Recover 3x3 pixel-border slope/aspect nodata along single-pixel DEM tile seams
    by picking the closest valid neighbor within +/- 1 pixel (~30 m) inside the same tile.
    """
    row, col = slope_src.index(ux, uy)
    candidates: List[Tuple[int, float, float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if not (0 <= r < slope_src.height and 0 <= c < slope_src.width):
                continue
            if not (0 <= r < aspect_src.height and 0 <= c < aspect_src.width):
                continue
            win = rasterio.windows.Window(c, r, 1, 1)
            s_val = float(slope_src.read(1, window=win)[0, 0])
            a_val = float(aspect_src.read(1, window=win)[0, 0])
            if (
                _is_valid_sample(s_val, slope_nodata)
                and _is_valid_sample(a_val, aspect_nodata)
                and 0.0 <= s_val <= 90.0
            ):
                dist2 = dr * dr + dc * dc
                candidates.append((dist2, s_val, a_val))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def generate_western_ghats_spatial_grid(
    dem_root: Path = DEFAULT_DEM_ROOT,
    land_mask_shp: Path = DEFAULT_LAND_MASK_SHP,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT,
    spacing_m: float = DEFAULT_GRID_SPACING_M,
    coastal_tolerance_m: float = DEFAULT_COASTAL_TOLERANCE_M,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate the validated 5 km EPSG:32643 spatial grid across the 14 Western Ghats DEM tiles."""
    if spacing_m <= 0.0:
        raise ValueError(f"spacing_m must be positive, got {spacing_m}")

    if output_csv.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists (pass overwrite=True to replace): {output_csv}"
            )
    existing_to_backup = [p for p in (output_csv, validation_report_path) if p.exists()]
    if existing_to_backup and overwrite:
        create_timestamped_backup(existing_to_backup, output_csv.parent / "backups")

    tile_registry = discover_validated_tile_paths(dem_root=dem_root)
    land_geom_utm, buffer_geom_utm = load_western_ghats_land_mask(
        land_mask_shp=land_mask_shp,
        coastal_tolerance_m=coastal_tolerance_m,
    )
    land_prep = prep(land_geom_utm)
    buffer_prep = prep(buffer_geom_utm)

    utm_to_wgs84 = Transformer.from_crs(
        f"EPSG:{WG_UTM_EPSG}", f"EPSG:{WGS84_EPSG}", always_xy=True
    )

    rows: List[Dict[str, Any]] = []
    seen_utm_cells = set()

    stats = {
        "candidate_points_in_tile_footprints": 0,
        "rejected_nodata_or_invalid_dem": 0,
        "recovered_tile_seam_slope_aspect": 0,
        "rejected_beyond_coastal_tolerance_ocean": 0,
        "rejected_buffer_nonpositive_ocean": 0,
        "accepted_exact_land": 0,
        "accepted_coastal_buffer_positive_elev": 0,
        "retained_below_sea_level_exact_land": 0,
    }
    tile_counts: Dict[str, int] = {tid: 0 for tid in EXPECTED_TILE_IDS}

    for tile_id in EXPECTED_TILE_IDS:
        tile_info = tile_registry[tile_id]
        min_lon, min_lat, max_lon, max_lat = parse_tile_bounds(tile_id)

        with rasterio.open(tile_info["elevation_tif"]) as elev_src:
            bounds = elev_src.bounds
            x_start = math.ceil(bounds.left / spacing_m) * spacing_m
            y_start = math.ceil(bounds.bottom / spacing_m) * spacing_m
            xs = np.arange(x_start, bounds.right + 1e-6, spacing_m, dtype=np.float64)
            ys = np.arange(y_start, bounds.top + 1e-6, spacing_m, dtype=np.float64)

        if len(xs) == 0 or len(ys) == 0:
            raise ValueError(f"No 5 km lattice candidates inside tile bounds for {tile_id}.")

        xx, yy = np.meshgrid(xs, ys)
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        lons, lats = utm_to_wgs84.transform(flat_x, flat_y)

        in_tile_mask = (
            (lons >= float(min_lon))
            & (lons < float(max_lon))
            & (lats >= float(min_lat))
            & (lats < float(max_lat))
        )
        cand_coords = [
            (float(flat_x[i]), float(flat_y[i]))
            for i in np.flatnonzero(in_tile_mask)
        ]
        cand_lons = lons[in_tile_mask]
        cand_lats = lats[in_tile_mask]
        stats["candidate_points_in_tile_footprints"] += len(cand_coords)

        if not cand_coords:
            raise ValueError(f"Zero candidates fell inside WGS84 bounds for {tile_id}.")

        elev_vals, elev_nodata = _sample_raster_values(tile_info["elevation_tif"], cand_coords)
        slope_vals, slope_nodata = _sample_raster_values(tile_info["slope_tif"], cand_coords)
        aspect_vals, aspect_nodata = _sample_raster_values(tile_info["aspect_tif"], cand_coords)

        with rasterio.open(tile_info["slope_tif"]) as slope_src, rasterio.open(
            tile_info["aspect_tif"]
        ) as aspect_src:
            for idx, (ux, uy) in enumerate(cand_coords):
                elev = float(elev_vals[idx])
                slope = float(slope_vals[idx])
                aspect = float(aspect_vals[idx])

                if not _is_valid_sample(elev, elev_nodata):
                    stats["rejected_nodata_or_invalid_dem"] += 1
                    continue

                if (
                    not _is_valid_sample(slope, slope_nodata)
                    or not _is_valid_sample(aspect, aspect_nodata)
                    or not (0.0 <= slope <= 90.0)
                ):
                    recovered = _recover_seam_slope_aspect(
                        slope_src=slope_src,
                        aspect_src=aspect_src,
                        ux=ux,
                        uy=uy,
                        slope_nodata=slope_nodata,
                        aspect_nodata=aspect_nodata,
                    )
                    if recovered is None:
                        stats["rejected_nodata_or_invalid_dem"] += 1
                        continue
                    slope, aspect = recovered
                    stats["recovered_tile_seam_slope_aspect"] += 1

                pt = Point(ux, uy)
                if not buffer_prep.covers(pt):
                    stats["rejected_beyond_coastal_tolerance_ocean"] += 1
                    continue

                in_exact = bool(land_prep.covers(pt))
                if not in_exact and elev <= 0.0:
                    stats["rejected_buffer_nonpositive_ocean"] += 1
                    continue

                cell_key = (int(round(ux)), int(round(uy)))
                if cell_key in seen_utm_cells:
                    raise RuntimeError(
                        f"Duplicate 5 km UTM coordinate detected at {cell_key} in tile {tile_id}."
                    )
                seen_utm_cells.add(cell_key)

                if in_exact:
                    stats["accepted_exact_land"] += 1
                    if elev < 0.0:
                        stats["retained_below_sea_level_exact_land"] += 1
                    coastal_class = "exact_land"
                    coast_dist_m = 0.0
                else:
                    stats["accepted_coastal_buffer_positive_elev"] += 1
                    coastal_class = "coastal_buffer_positive_elev"
                    coast_dist_m = float(pt.distance(land_geom_utm))

                a_sin, a_cos, a_flat = _compute_aspect_derivatives(aspect)
                tile_counts[tile_id] += 1
                rows.append(
                    {
                        "tile": tile_id,
                        "tile_group": tile_info["tile_group"],
                        "latitude": float(cand_lats[idx]),
                        "longitude": float(cand_lons[idx]),
                        "utm_x": float(ux),
                        "utm_y": float(uy),
                        "utm_epsg": int(WG_UTM_EPSG),
                        "elevation_m": elev,
                        "slope_degrees": slope,
                        "aspect_degrees": aspect,
                        "aspect_sin": a_sin,
                        "aspect_cos": a_cos,
                        "aspect_flat": int(a_flat),
                        "in_ne_exact_land": bool(in_exact),
                        "in_1km_coastal_buffer": True,
                        "coastal_rule_classification": coastal_class,
                        "distance_to_ne_coast_m": coast_dist_m,
                    }
                )

    if not rows:
        raise RuntimeError("Generated Western Ghats spatial grid is unexpectedly empty.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["tile", "utm_y", "utm_x"]).reset_index(drop=True)
    df.insert(0, "grid_id", [f"WG_GRID_{i + 1:05d}" for i in range(len(df))])

    validation_report = validate_western_ghats_spatial_grid(
        df=df,
        tile_counts=tile_counts,
        stats=stats,
        spacing_m=spacing_m,
        coastal_tolerance_m=coastal_tolerance_m,
        land_mask_shp=land_mask_shp,
        output_csv=output_csv,
    )

    _atomic_write_csv(output_csv, df)
    atomic_write_json(validation_report_path, validation_report)
    return df, validation_report


def validate_western_ghats_spatial_grid(
    df: pd.DataFrame,
    tile_counts: Dict[str, int],
    stats: Dict[str, int],
    spacing_m: float,
    coastal_tolerance_m: float,
    land_mask_shp: Path,
    output_csv: Path,
) -> Dict[str, Any]:
    """Run strict integrity checks on the generated Western Ghats 5 km grid."""
    required_cols = [
        "grid_id",
        "tile",
        "tile_group",
        "latitude",
        "longitude",
        "utm_x",
        "utm_y",
        "utm_epsg",
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "aspect_sin",
        "aspect_cos",
        "aspect_flat",
        "in_ne_exact_land",
        "in_1km_coastal_buffer",
        "coastal_rule_classification",
        "distance_to_ne_coast_m",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Generated Western Ghats grid missing columns: {missing_cols}")

    feature_cols = ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "aspect_flat"]
    nan_counts = {c: int(df[c].isna().sum()) for c in feature_cols + ["latitude", "longitude", "utm_x", "utm_y"]}
    if any(v > 0 for v in nan_counts.values()):
        raise ValueError(f"NaN values detected in generated grid: {nan_counts}")

    dup_utm = int(df.duplicated(subset=["utm_x", "utm_y"]).sum())
    dup_id = int(df.duplicated(subset=["grid_id"]).sum())
    if dup_utm > 0 or dup_id > 0:
        raise ValueError(f"Duplicate cells detected (dup_utm={dup_utm}, dup_id={dup_id}).")

    present_tiles = sorted(df["tile"].unique().tolist())
    missing_tiles = sorted(set(EXPECTED_TILE_IDS) - set(present_tiles))
    extra_tiles = sorted(set(present_tiles) - set(EXPECTED_TILE_IDS))
    if missing_tiles or extra_tiles:
        raise ValueError(f"Tile mismatch (missing={missing_tiles}, extra={extra_tiles}).")

    buffer_only_mask = ~df["in_ne_exact_land"].astype(bool)
    if bool((df.loc[buffer_only_mask, "elevation_m"] <= 0.0).any()):
        raise ValueError("Coastal buffer contains non-positive elevation pixel.")

    spacing_ok = bool(
        ((df["utm_x"] % spacing_m).abs() < 1e-6).all()
        and ((df["utm_y"] % spacing_m).abs() < 1e-6).all()
    )
    if not spacing_ok:
        raise ValueError(f"UTM coordinates are not aligned to {spacing_m} m lattice.")

    checks = {
        "all_14_validated_tiles_present": len(present_tiles) == 14 and not missing_tiles and not extra_tiles,
        "strictly_epsg_32643": bool((df["utm_epsg"] == WG_UTM_EPSG).all()),
        "exact_5km_utm_lattice": spacing_ok,
        "zero_duplicate_locations": dup_utm == 0 and dup_id == 0,
        "zero_nan_features": all(v == 0 for v in nan_counts.values()),
        "coastal_buffer_positive_elevation_only": not bool((df.loc[buffer_only_mask, "elevation_m"] <= 0.0).any()),
        "below_sea_level_land_retained": stats["retained_below_sea_level_exact_land"] > 0,
        "valid_slope_range_0_to_90": bool((df["slope_degrees"] >= 0.0).all() and (df["slope_degrees"] <= 90.0).all()),
        "flat_aspect_consistency": bool(
            (df.loc[df["aspect_flat"] == 1, "aspect_sin"].abs() < 1e-9).all()
            and (df.loc[df["aspect_flat"] == 1, "aspect_cos"].abs() < 1e-9).all()
        ),
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Spatial grid validation failed checks: {failed}")

    return {
        "stage": "Western Ghats Limited Pilot — Stage 6.1 Grid Generation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_disclosure": (
            "Western Ghats Limited Pilot: grid generated strictly across the 14 validated DEM tiles; "
            "does not imply complete Western Ghats coverage."
        ),
        "crs": f"EPSG:{WG_UTM_EPSG}",
        "grid_spacing_m": float(spacing_m),
        "validated_tiles_count": len(EXPECTED_TILE_IDS),
        "validated_tiles": list(EXPECTED_TILE_IDS),
        "tile_grid_counts": tile_counts,
        "total_grid_locations": int(len(df)),
        "land_mask_specification": {
            "source_shapefile": str(land_mask_shp),
            "coastal_tolerance_m": float(coastal_tolerance_m),
            "buffer_acceptance_rule": "exact Natural Earth land OR (within 1,000 m buffer AND elevation_m > 0.0)",
            "filtering_summary": stats,
        },
        "elevation_summary_m": {
            "min": float(df["elevation_m"].min()),
            "median": float(df["elevation_m"].median()),
            "max": float(df["elevation_m"].max()),
        },
        "slope_summary_degrees": {
            "min": float(df["slope_degrees"].min()),
            "median": float(df["slope_degrees"].median()),
            "max": float(df["slope_degrees"].max()),
        },
        "output_csv": str(output_csv),
        "checks": checks,
        "validation_passed": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Western Ghats Limited Pilot 5 km EPSG:32643 spatial grid."
    )
    parser.add_argument("--dem-root", type=Path, default=DEFAULT_DEM_ROOT)
    parser.add_argument("--land-mask-shp", type=Path, default=DEFAULT_LAND_MASK_SHP)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--spacing-m", type=float, default=DEFAULT_GRID_SPACING_M)
    parser.add_argument("--coastal-tolerance-m", type=float, default=DEFAULT_COASTAL_TOLERANCE_M)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    df, report = generate_western_ghats_spatial_grid(
        dem_root=args.dem_root,
        land_mask_shp=args.land_mask_shp,
        output_csv=args.output_csv,
        validation_report_path=args.report_json,
        spacing_m=args.spacing_m,
        coastal_tolerance_m=args.coastal_tolerance_m,
        overwrite=args.overwrite,
    )
    LOGGER.info(
        "Generated %d Western Ghats 5 km grid points across %d tiles -> %s",
        len(df),
        report["validated_tiles_count"],
        args.output_csv,
    )


if __name__ == "__main__":
    main()
