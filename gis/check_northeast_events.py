
from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "india_landslide_events.csv"
)

DEM_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N27E088"
    / "elevation.tif"
)


def main():
    print("=" * 65)
    print("NORTHEAST LANDSLIDE EVENT COVERAGE CHECK")
    print("=" * 65)

    # 1. Check that the required files exist
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Historical inventory not found: {EVENTS_FILE}"
        )

    if not DEM_FILE.exists():
        raise FileNotFoundError(
            f"DEM raster not found: {DEM_FILE}"
        )

    # 2. Load the historical landslide inventory
    events = pd.read_csv(EVENTS_FILE)

    print(f"\nTotal historical events: {len(events):,}")
    print(f"Inventory columns: {list(events.columns)}")

    # 3. Identify coordinate columns
    latitude_col = next(
        (c for c in ["latitude", "lat"] if c in events.columns),
        None,
    )
    longitude_col = next(
        (c for c in ["longitude", "lon", "lng"] if c in events.columns),
        None,
    )

    if latitude_col is None or longitude_col is None:
        raise ValueError(
            "Could not identify latitude/longitude columns. "
            "Check the inventory column names printed above."
        )

    # 4. Read the DEM's CRS and geographic coverage
    with rasterio.open(DEM_FILE) as dem:
        dem_crs = dem.crs
        dem_bounds = dem.bounds

        print("\nDEM INFORMATION")
        print("-" * 40)
        print(f"CRS: {dem_crs}")
        print(f"Dimensions: {dem.width} x {dem.height}")
        print(f"Bounds: {dem_bounds}")

        # 5. Convert event coordinates from WGS84 to DEM CRS
        transformer = Transformer.from_crs(
            "EPSG:4326",
            dem_crs,
            always_xy=True,
        )

        valid_coordinates = (
            events[longitude_col].notna()
            & events[latitude_col].notna()
        )

        events["dem_x"] = float("nan")
        events["dem_y"] = float("nan")

        x, y = transformer.transform(
            events.loc[valid_coordinates, longitude_col].to_numpy(),
            events.loc[valid_coordinates, latitude_col].to_numpy(),
        )

        events.loc[valid_coordinates, "dem_x"] = x
        events.loc[valid_coordinates, "dem_y"] = y

        # 6. Check whether each point lies inside the DEM bounds
        events["inside_dem"] = (
            valid_coordinates
            & events["dem_x"].between(
                dem_bounds.left, dem_bounds.right, inclusive="left"
            )
            & events["dem_y"].between(
                dem_bounds.bottom, dem_bounds.top, inclusive="left"
            )
        )

    inside = events[events["inside_dem"]].copy()
    outside_count = len(events) - len(inside)

    # 7. Print results
    print("\nEVENT COVERAGE RESULTS")
    print("-" * 40)
    print(f"Total inventory records: {len(events):,}")
    print(f"Events with valid coordinates: {valid_coordinates.sum():,}")
    print(f"Events inside DEM tile: {len(inside):,}")
    print(f"Events outside DEM tile: {outside_count:,}")

    if not inside.empty:
        print("\nSAMPLE EVENTS INSIDE THE DEM")
        print("-" * 40)

        display_columns = [
            c for c in [
                "event_id",
                "id",
                "date",
                "latitude",
                "longitude",
                "landslide_category",
                "landslide_trigger",
            ]
            if c in inside.columns
        ]

        print(inside[display_columns].head(10).to_string(index=False))

    # 8. Save the events that fall inside the tile
    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "northeast_N27E088_events.csv"
    )

    inside.to_csv(output_file, index=False)

    print(f"\nInside-tile events saved to:\n{output_file}")
    print("\nEvent coverage check completed.")


if __name__ == "__main__":
    main()
