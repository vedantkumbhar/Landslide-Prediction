
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


ROOT = Path(__file__).resolve().parent.parent

INVENTORY = ROOT / "data" / "processed" / "india_landslide_events.csv"
DEM_ROOT = ROOT / "data" / "processed" / "dem" / "western_ghats"

OUTPUT = (
    ROOT / "data" / "processed" / "western_ghats_background_samples.csv"
)

TILES = ["N09E076", "N19E072", "N11E076"]

SEED = 42
SAMPLES_PER_TILE = 200
MIN_EVENT_DISTANCE_KM = 1.0
MAX_ATTEMPTS_PER_TILE = SAMPLES_PER_TILE * 100

TERRAIN_FILES = {
    "elevation": "elevation.tif",
    "slope": "slope.tif",
    "aspect": "aspect.tif",
}


def main():
    print("=" * 65)
    print("WESTERN GHATS BACKGROUND SAMPLING")
    print("=" * 65)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite:\n{OUTPUT}"
        )

    if not INVENTORY.exists():
        raise FileNotFoundError(f"Event inventory not found: {INVENTORY}")

    inventory = pd.read_csv(INVENTORY)

    for col in ["latitude", "longitude"]:
        if col not in inventory.columns:
            raise ValueError(f"Missing inventory column: {col}")
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce")

    inventory = inventory.dropna(subset=["latitude", "longitude"])
    inventory = inventory[
        inventory["latitude"].between(-90, 90)
        & inventory["longitude"].between(-180, 180)
    ].copy()

    # Keep only historical events located within the three pilot tiles.
    inventory["tile"] = [
        f"N{int(np.floor(lat)):02d}E{int(np.floor(lon)):03d}"
        for lat, lon in zip(
            inventory["latitude"], inventory["longitude"]
        )
    ]
    inventory = inventory[inventory["tile"].isin(TILES)].copy()

    rng = np.random.default_rng(SEED)
    results = []

    for tile in TILES:
        print(f"\nSampling {tile}...")

        folder = DEM_ROOT / tile / "metric"
        paths = {
            name: folder / filename
            for name, filename in TERRAIN_FILES.items()
        }

        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{tile}: missing terrain raster(s):\n"
                + "\n".join(missing)
            )

        with (
            rasterio.open(paths["elevation"]) as elev,
            rasterio.open(paths["slope"]) as slope,
            rasterio.open(paths["aspect"]) as aspect,
        ):
            datasets = [elev, slope, aspect]

            for ds in datasets[1:]:
                if (
                    ds.crs != elev.crs
                    or ds.transform != elev.transform
                    or ds.width != elev.width
                    or ds.height != elev.height
                ):
                    raise ValueError(
                        f"{tile}: terrain rasters are not aligned."
                    )

            transformer = Transformer.from_crs(
                "EPSG:4326", elev.crs, always_xy=True
            )

            event_rows = inventory[inventory["tile"] == tile]
            event_x, event_y = transformer.transform(
                event_rows["longitude"].to_numpy(),
                event_rows["latitude"].to_numpy(),
            )
            event_x = np.asarray(event_x, dtype=float)
            event_y = np.asarray(event_y, dtype=float)

            accepted = []
            seen_cells = set()
            attempts = 0

            while (
                len(accepted) < SAMPLES_PER_TILE
                and attempts < MAX_ATTEMPTS_PER_TILE
            ):
                batch = min(
                    100,
                    MAX_ATTEMPTS_PER_TILE - attempts,
                )

                rows = rng.integers(0, elev.height, size=batch)
                cols = rng.integers(0, elev.width, size=batch)
                attempts += batch

                for row, col in zip(rows, cols):
                    row, col = int(row), int(col)
                    cell = (row, col)

                    if cell in seen_cells:
                        continue
                    seen_cells.add(cell)

                    values = []
                    valid = True

                    for ds in datasets:
                        value = ds.read(
                            1,
                            window=((row, row + 1), (col, col + 1)),
                            masked=True,
                        )[0, 0]

                        if np.ma.is_masked(value):
                            valid = False
                            break

                        value = float(value)

                        if not np.isfinite(value):
                            valid = False
                            break

                        if ds.nodata is not None and value == ds.nodata:
                            valid = False
                            break

                        values.append(value)

                    if not valid:
                        continue

                    # Aspect -1 is a valid flat-terrain sentinel.
                    x, y = rasterio.transform.xy(
                        elev.transform, row, col, offset="center"
                    )

                    if len(event_x):
                        distances = np.hypot(
                            event_x - x,
                            event_y - y,
                        )
                        nearest_km = float(np.min(distances)) / 1000.0
                    else:
                        nearest_km = np.inf

                    if nearest_km < MIN_EVENT_DISTANCE_KM:
                        continue

                    lon, lat = Transformer.from_crs(
                        elev.crs,
                        "EPSG:4326",
                        always_xy=True,
                    ).transform(x, y)

                    accepted.append({
                        "sample_id": f"{tile}_B{len(accepted) + 1:04d}",
                        "pilot_tile": tile,
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "elevation_m": values[0],
                        "slope_degrees": values[1],
                        "aspect_degrees": values[2],
                        "label": "background_unverified",
                        "nearest_recorded_event_km": nearest_km,
                        "sampling_method": "random_valid_terrain_cells",
                        "minimum_event_distance_km": MIN_EVENT_DISTANCE_KM,
                        "random_seed": SEED,
                    })

                    if len(accepted) >= SAMPLES_PER_TILE:
                        break

            results.extend(accepted)

            print(f"Candidate attempts: {attempts}")
            print(f"Accepted samples: {len(accepted)}/{SAMPLES_PER_TILE}")

            if len(accepted) < SAMPLES_PER_TILE:
                print(
                    "WARNING: target not reached. Consider reviewing "
                    "the distance threshold or attempt limit."
                )

    if not results:
        print("\nNo background samples generated; no output file created.")
        return

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 65)
    print("SAMPLING SUMMARY")
    print("=" * 65)
    print(f"Total samples: {len(output_df)}")
    print("\nSamples by tile:")
    print(output_df["pilot_tile"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT}")
    print(
        "\nCaution: background_unverified samples are not confirmed "
        "landslide-free locations."
    )


if __name__ == "__main__":
    main()
