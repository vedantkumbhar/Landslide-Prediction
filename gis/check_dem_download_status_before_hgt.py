
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLAN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dem" / "dem_tile_plan.csv"
)
RAW_DEM_ROOT = PROJECT_ROOT / "data" / "raw" / "dem"
PROCESSED_DEM_ROOT = PROJECT_ROOT / "data" / "processed" / "dem"

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dem" / "dem_download_status.csv"
)


def find_tile_rasters(root: Path, tile: str, filename: str) -> list[Path]:
    """Find rasters for a tile, excluding backups and unrelated tiles."""
    matches = []

    if not root.exists():
        return matches

    for path in root.rglob(filename):
        if "backup_before_fix" in path.parts:
            continue

        # Tile-specific folders are authoritative when present.
        if tile in path.parts:
            matches.append(path)

    return sorted(matches)


def main():
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"DEM tile plan not found: {PLAN_PATH}")

    plan = pd.read_csv(PLAN_PATH, dtype={"tile": "string"})

    required = {"region", "tile", "event_count"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"Missing columns in tile plan: {sorted(missing)}")

    rows = []

    for _, item in plan.iterrows():
        tile = str(item["tile"]).strip()

        raw = find_tile_rasters(RAW_DEM_ROOT, tile, "output_SRTMGL1.tif")
        processed = find_tile_rasters(
            PROCESSED_DEM_ROOT, tile, "elevation.tif"
        )

        rows.append({
            "region": item["region"],
            "tile": tile,
            "event_count": int(item["event_count"]),
            "raw_dem_found": bool(raw),
            "raw_dem_files": len(raw),
            "raw_dem_paths": "; ".join(
                str(p.relative_to(PROJECT_ROOT)) for p in raw
            ),
            "processed_elevation_found": bool(processed),
            "processed_elevation_files": len(processed),
            "processed_elevation_paths": "; ".join(
                str(p.relative_to(PROJECT_ROOT)) for p in processed
            ),
        })

    status = pd.DataFrame(rows).sort_values(
        ["raw_dem_found", "event_count"],
        ascending=[False, False],
    )

    status.to_csv(OUTPUT_PATH, index=False)

    print("=" * 65)
    print("DEM TILE COVERAGE STATUS")
    print("=" * 65)
    print("Planned tile entries:", len(status))
    print("Tiles with raw DEM:", int(status["raw_dem_found"].sum()))
    print(
        "Tiles with processed elevation:",
        int(status["processed_elevation_found"].sum()),
    )

    print("\nCoverage by region:")
    print(
        status.groupby("region")[
            ["raw_dem_found", "processed_elevation_found"]
        ].sum().to_string()
    )

    print("\nHighest-priority tiles without processed elevation:")
    pending = status[~status["processed_elevation_found"]].head(15)
    if pending.empty:
        print("No pending tiles.")
    else:
        print(
            pending[["region", "tile", "event_count"]].to_string(index=False)
        )

    print("\nSaved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
