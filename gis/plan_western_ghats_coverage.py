from pathlib import Path
import math
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "india_landslide_events.csv"
RAW_DEM_ROOT = PROJECT_ROOT / "data" / "raw" / "dem"
PROCESSED_DEM_ROOT = PROJECT_ROOT / "data" / "processed" / "dem"
OUTPUT_PATH = PROCESSED_DEM_ROOT / "western_ghats_candidate_plan.csv"

# Broad screening envelope only; these are not official Western Ghats boundaries.
LAT_MIN, LAT_MAX = 8, 22
LON_MIN, LON_MAX = 72, 78


def tile_name(lat, lon):
    lat_degree = math.floor(lat)
    lon_degree = math.floor(lon)
    ns = "N" if lat_degree >= 0 else "S"
    ew = "E" if lon_degree >= 0 else "W"
    return f"{ns}{abs(lat_degree):02d}{ew}{abs(lon_degree):03d}"


def main():
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(EVENTS_PATH)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT_PATH}"
        )

    events = pd.read_csv(EVENTS_PATH)
    for col in ("latitude", "longitude"):
        events[col] = pd.to_numeric(events[col], errors="coerce")

    events = events.dropna(subset=["latitude", "longitude"]).copy()
    events = events[
        events["latitude"].between(LAT_MIN, LAT_MAX, inclusive="left")
        & events["longitude"].between(LON_MIN, LON_MAX, inclusive="left")
    ].copy()

    events["tile"] = events.apply(
        lambda row: tile_name(row["latitude"], row["longitude"]), axis=1
    )
    counts = events.groupby("tile").size().to_dict()

    candidates = {
        f"N{lat:02d}E{lon:03d}"
        for lat in range(LAT_MIN, LAT_MAX)
        for lon in range(LON_MIN, LON_MAX)
    }

    raw_tiles = {
        path.stem
        for path in RAW_DEM_ROOT.rglob("*.hgt")
        if re.fullmatch(r"[NS]\d{2}[EW]\d{3}", path.stem)
    }

    processed_tiles = set()
    for path in PROCESSED_DEM_ROOT.rglob("elevation.tif"):
        if path.parent.name == "metric":
            tile = path.parent.parent.name
        else:
            tile = path.parent.name

        if re.fullmatch(r"[NS]\d{2}[EW]\d{3}", tile):
            processed_tiles.add(tile)

    rows = []
    for tile in sorted(candidates):
        raw_present = tile in raw_tiles
        processed_present = tile in processed_tiles
        event_count = int(counts.get(tile, 0))

        if processed_present:
            status = "Processed DEM exists"
        elif raw_present:
            status = "Raw HGT exists; processed DEM not detected"
        else:
            status = "DEM not detected"

        rows.append({
            "region": "Western Ghats screening envelope",
            "tile": tile,
            "historical_event_count": event_count,
            "has_historical_events": event_count > 0,
            "raw_hgt_detected": raw_present,
            "processed_elevation_detected": processed_present,
            "dem_inventory_status": status,
            "inside_official_western_ghats_boundary": "Not assessed",
        })

    plan = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUTPUT_PATH, index=False)

    print("=" * 65)
    print("WESTERN GHATS PRELIMINARY TILE SCREENING")
    print("=" * 65)
    print(f"Candidate tiles: {len(plan)}")
    print(f"Historical events in envelope: {len(events)}")
    print(f"Tiles with historical events: {(plan['historical_event_count'] > 0).sum()}")
    print(f"Raw HGT detected: {int(plan['raw_hgt_detected'].sum())}")
    print(f"Processed elevation detected: {int(plan['processed_elevation_detected'].sum())}")
    print("\nTiles with historical events, highest count first:")
    print(
        plan[plan["historical_event_count"] > 0]
        .sort_values(["historical_event_count", "tile"], ascending=[False, True])
        [["tile", "historical_event_count", "dem_inventory_status"]]
        .to_string(index=False)
    )
    print(f"\nSaved separate report: {OUTPUT_PATH}")
    print("Existing DEM plan and raster files were not modified.")
    print("Geographic boundary membership remains unverified.")


if __name__ == "__main__":
    main()
