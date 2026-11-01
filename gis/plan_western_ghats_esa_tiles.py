
from pathlib import Path
import json
import math
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = (
    ROOT / "data" / "raw" / "boundaries"
    / "western_ghats_esa_outline.json"
)
EVENTS_PATH = (
    ROOT / "data" / "processed"
    / "india_landslide_events.csv"
)
RAW_ROOT = ROOT / "data" / "raw" / "dem"
DEM_ROOT = ROOT / "data" / "processed" / "dem"

# Keep the original report unchanged.
OUTPUT_PATH = DEM_ROOT / "western_ghats_esa_tile_plan_v2.csv"


def points_in_geometry(value):
    """Yield longitude/latitude pairs from nested GeoJSON coordinates."""
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield value[0], value[1]
    elif isinstance(value, list):
        for item in value:
            yield from points_in_geometry(item)


def point_on_segment(px, py, ax, ay, bx, by, eps=1e-10):
    """Return whether a point lies on a line segment."""
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)

    if abs(cross) > eps:
        return False

    return (
        min(ax, bx) - eps <= px <= max(ax, bx) + eps
        and min(ay, by) - eps <= py <= max(ay, by) + eps
    )


def point_in_ring(px, py, ring):
    """Ray casting; points on the ring count as inside."""
    if len(ring) < 4:
        return False

    inside = False

    for i in range(len(ring) - 1):
        ax, ay = ring[i]
        bx, by = ring[i + 1]

        if point_on_segment(px, py, ax, ay, bx, by):
            return True

        if (ay > py) != (by > py):
            crossing_x = (
                ax + (py - ay) * (bx - ax) / (by - ay)
            )
            if px < crossing_x:
                inside = not inside

    return inside


def point_in_polygon(px, py, rings):
    """The first ring is the shell; subsequent rings are holes."""
    if not rings or not point_in_ring(px, py, rings[0]):
        return False

    # A point inside a hole is not inside the filled polygon.
    return not any(
        point_in_ring(px, py, hole)
        for hole in rings[1:]
    )


def orientation(a, b, c):
    """Signed cross product used for segment intersection."""
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def segments_intersect(a, b, c, d):
    """Return whether two closed line segments intersect."""
    eps = 1e-10

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if (
        ((o1 > eps and o2 < -eps) or
         (o1 < -eps and o2 > eps))
        and
        ((o3 > eps and o4 < -eps) or
         (o3 < -eps and o4 > eps))
    ):
        return True

    return (
        (abs(o1) <= eps and point_on_segment(*c, *a, *b))
        or (abs(o2) <= eps and point_on_segment(*d, *a, *b))
        or (abs(o3) <= eps and point_on_segment(*a, *c, *d))
        or (abs(o4) <= eps and point_on_segment(*b, *c, *d))
    )


def ring_edges(ring):
    """Yield consecutive edges from a closed ring."""
    return zip(ring[:-1], ring[1:])


def polygon_intersects_tile(rings, west, south, east, north):
    """
    Test whether a polygon's filled area intersects a 1-degree tile.

    The first ring is the exterior shell. Remaining rings are holes.
    A hole boundary alone must not create a false-positive tile.
    """
    if not rings or len(rings[0]) < 4:
        return False

    shell = rings[0]

    corners = [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
    ]

    tile_edges = list(
        ring_edges(corners + [corners[0]])
    )

    # 1. Any tile corner lies in the filled polygon.
    if any(
        point_in_polygon(x, y, rings)
        for x, y in corners
    ):
        return True

    # 2. Any exterior-shell vertex lies strictly inside the tile.
    if any(
        west < x < east and south < y < north
        for x, y in shell
    ):
        return True

    def boundary_crosses_tile(ring):
        """Check whether a ring edge intersects a tile edge."""
        for a, b in ring_edges(ring):
            # Quickly skip edges whose bounding boxes miss the tile.
            if not (
                max(a[0], b[0]) >= west
                and min(a[0], b[0]) <= east
                and max(a[1], b[1]) >= south
                and min(a[1], b[1]) <= north
            ):
                continue

            if any(
                segments_intersect(a, b, c, d)
                for c, d in tile_edges
            ):
                return True

        return False

    # 3. An exterior-shell crossing establishes an intersection.
    if boundary_crosses_tile(shell):
        return True

    # The tile has no shell crossing and no shell vertex inside it.
    # Its center distinguishes an interior tile from an exterior tile.
    center_x = (west + east) / 2
    center_y = (south + north) / 2

    if not point_in_ring(center_x, center_y, shell):
        return False

    # 4. The tile is inside the exterior shell. If a hole boundary
    # crosses it, some filled polygon area may remain in the tile.
    # A tile entirely contained within a hole will return False here.
    return any(
        boundary_crosses_tile(hole)
        for hole in rings[1:]
    )


def tile_name(lat, lon):
    """Format a one-degree tile identifier."""
    prefix_lat = "N" if lat >= 0 else "S"
    prefix_lon = "E" if lon >= 0 else "W"

    return (
        f"{prefix_lat}{abs(lat):02d}"
        f"{prefix_lon}{abs(lon):03d}"
    )


def main():
    for path in (BOUNDARY_PATH, EVENTS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    # Never overwrite a previously generated report.
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite the existing v2 report:\n"
            f"{OUTPUT_PATH}\n"
            "Rename or archive it manually if you want to regenerate."
        )

    boundary = json.loads(
        BOUNDARY_PATH.read_text(encoding="utf-8")
    )
    features = boundary.get("features", [])

    if not features:
        raise ValueError(
            "Boundary file contains no GeoJSON features."
        )

    polygons = []

    for feature in features:
        geometry = feature.get("geometry") or {}

        if geometry.get("type") != "MultiPolygon":
            raise ValueError(
                "Unexpected geometry type: "
                f"{geometry.get('type')}"
            )

        state = (
            feature.get("properties", {})
            .get("gz_state", "Unknown")
        )

        for rings in geometry.get("coordinates", []):
            if rings and rings[0]:
                polygons.append((state, rings))

    if not polygons:
        raise ValueError(
            "No valid polygon parts were found in the boundary file."
        )

    all_points = [
        point
        for _, rings in polygons
        for point in points_in_geometry(rings)
    ]

    if not all_points:
        raise ValueError(
            "No coordinates were found in the boundary geometry."
        )

    min_lon = math.floor(min(x for x, _ in all_points))
    max_lon = math.floor(max(x for x, _ in all_points))
    min_lat = math.floor(min(y for _, y in all_points))
    max_lat = math.floor(max(y for _, y in all_points))

    # Load historical event coordinates.
    event_df = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(event_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing event columns: {sorted(missing_columns)}"
        )

    event_df["latitude"] = pd.to_numeric(
        event_df["latitude"],
        errors="coerce",
    )
    event_df["longitude"] = pd.to_numeric(
        event_df["longitude"],
        errors="coerce",
    )

    event_df = event_df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    event_df = event_df[
        event_df["latitude"].between(
            min_lat, max_lat + 1, inclusive="left"
        )
        & event_df["longitude"].between(
            min_lon, max_lon + 1, inclusive="left"
        )
    ].copy()

    event_df["tile"] = event_df.apply(
        lambda row: tile_name(
            math.floor(row["latitude"]),
            math.floor(row["longitude"]),
        ),
        axis=1,
    )

    event_counts = event_df.groupby("tile").size().to_dict()

    # Inventory existing raw SRTM HGT files without downloading anything.
    raw_tiles = {
        path.stem
        for path in RAW_ROOT.rglob("*.hgt")
        if re.fullmatch(
            r"[NS]\d{2}[EW]\d{3}",
            path.stem,
        )
    } if RAW_ROOT.exists() else set()

    # Inventory processed elevation rasters.
    processed_tiles = set()

    if DEM_ROOT.exists():
        for path in DEM_ROOT.rglob("elevation.tif"):
            tile = (
                path.parent.parent.name
                if path.parent.name == "metric"
                else path.parent.name
            )

            if re.fullmatch(r"[NS]\d{2}[EW]\d{3}", tile):
                processed_tiles.add(tile)

    rows = []

    for lat in range(min_lat, max_lat + 1):
        for lon in range(min_lon, max_lon + 1):
            tile = tile_name(lat, lon)

            overlapping_states = sorted({
                state
                for state, rings in polygons
                if polygon_intersects_tile(
                    rings,
                    lon,
                    lat,
                    lon + 1,
                    lat + 1,
                )
            })

            if not overlapping_states:
                continue

            raw_present = tile in raw_tiles
            processed_present = tile in processed_tiles
            event_count = int(event_counts.get(tile, 0))

            if processed_present:
                status = "Processed elevation detected"
            elif raw_present:
                status = (
                    "Raw HGT detected; processed elevation "
                    "not detected"
                )
            else:
                status = "DEM not detected"

            rows.append({
                "tile": tile,
                "esa_states": "; ".join(overlapping_states),
                "historical_event_count": event_count,
                "has_historical_events": event_count > 0,
                "raw_hgt_detected": raw_present,
                "processed_elevation_detected": processed_present,
                "dem_inventory_status": status,
                "boundary_source": (
                    "Western Ghats ESA reference; "
                    "not full mountain-range boundary"
                ),
            })

    if not rows:
        raise ValueError(
            "No intersecting tiles found. "
            "Review the GeoJSON coordinates and geometry."
        )

    plan = pd.DataFrame(rows).sort_values(
        ["historical_event_count", "tile"],
        ascending=[False, True],
    )

    DEM_ROOT.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUTPUT_PATH, index=False)

    print("=" * 65)
    print("WESTERN GHATS ESA TILE INTERSECTION PLAN — V2")
    print("=" * 65)
    print(f"Boundary features: {len(features)}")
    print(f"Polygon parts checked: {len(polygons)}")
    print(f"Intersecting 1-degree tiles: {len(plan)}")
    print(
        "Tiles with historical events: "
        f"{int(plan['has_historical_events'].sum())}"
    )
    print(
        "Historical events in candidate tiles: "
        f"{int(plan['historical_event_count'].sum())}"
    )
    print(
        "Raw HGT detected: "
        f"{int(plan['raw_hgt_detected'].sum())}"
    )
    print(
        "Processed elevation detected: "
        f"{int(plan['processed_elevation_detected'].sum())}"
    )

    print("\nHighest event counts:")
    print(
        plan[
            [
                "tile",
                "esa_states",
                "historical_event_count",
                "dem_inventory_status",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )

    print(f"\nSaved separate report: {OUTPUT_PATH}")
    print("No DEMs were downloaded.")
    print("The original tile plan and ESA report were not overwritten.")
    print(
        "Note: boundary overlap identifies candidate tiles; "
        "it does not mean an entire tile lies inside the boundary."
    )


if __name__ == "__main__":
    main()
