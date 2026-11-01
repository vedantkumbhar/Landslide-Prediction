
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent

INVENTORY = ROOT / "data/processed/india_landslide_events.csv"
OUTPUT = ROOT / "data/processed/pilot_background_samples.csv"

SEED = 42
SAMPLES_PER_TILE = 200

TILES = {
    "N30E078": ROOT / "data/processed/dem/himalayas/N30E078/elevation.tif",
    "N27E088": ROOT / "data/processed/dem/northeast/N27E088/elevation.tif",
}


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT}"
        )

    inventory = pd.read_csv(INVENTORY)
    inventory["longitude"] = pd.to_numeric(
        inventory["longitude"], errors="coerce"
    )
    inventory["latitude"] = pd.to_numeric(
        inventory["latitude"], errors="coerce"
    )
    inventory = inventory.dropna(
        subset=["longitude", "latitude"]
    )

    rng = np.random.default_rng(SEED)
    results = []

    for tile, raster_path in TILES.items():
        print(f"Sampling {tile}...")

        with rasterio.open(raster_path) as src:
            # Randomly sample raster positions without loading the full raster.
            samples = []
            attempts = 0
            max_attempts = SAMPLES_PER_TILE * 20

            while len(samples) < SAMPLES_PER_TILE and attempts < max_attempts:
                batch_size = min(100, max_attempts - attempts)
                rows = rng.integers(0, src.height, size=batch_size)
                cols = rng.integers(0, src.width, size=batch_size)
                attempts += batch_size

                # Read only a 1x1 window at each sampled position.
                for row, col in zip(rows, cols):
                    value = src.read(
                        1,
                        window=((int(row), int(row) + 1),
                                (int(col), int(col) + 1)),
                        masked=True,
                    )
                    if (
                        not np.ma.is_masked(value[0, 0])
                        and np.isfinite(value[0, 0])
                    ):
                        samples.append((int(row), int(col)))
                        if len(samples) >= SAMPLES_PER_TILE:
                            break

            if not samples:
                print(f"No valid cells sampled for {tile}.")
                continue

            rows = np.array([p[0] for p in samples])
            cols = np.array([p[1] for p in samples])

            xs, ys = rasterio.transform.xy(
                src.transform, rows, cols, offset="center"
            )
            to_geo = Transformer.from_crs(
                src.crs, "EPSG:4326", always_xy=True
            )
            lon, lat = to_geo.transform(xs, ys)

            # Sampled points close to known historical event coordinates
            # are flagged for review, not labelled as verified negatives.
            inv_to_projected = Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            )
            inv_x, inv_y = inv_to_projected.transform(
                inventory["longitude"].to_numpy(),
                inventory["latitude"].to_numpy(),
            )

            for i, (x, y) in enumerate(zip(xs, ys)):
                distances = np.sqrt(
                    (np.asarray(inv_x) - x) ** 2
                    + (np.asarray(inv_y) - y) ** 2
                )
                nearest_km = float(np.nanmin(distances)) / 1000

                results.append({
                    "sample_id": f"{tile}_B{i + 1:04d}",
                    "pilot_tile": tile,
                    "longitude": float(lon[i]),
                    "latitude": float(lat[i]),
                    "label": "background_unverified",
                    "nearest_recorded_event_km": nearest_km,
                    "sampling_method": "random_valid_dem_cells",
                    "random_seed": SEED,
                })

        print(f"Sampled {len(samples)} cells for {tile}.")

    if not results:
        print("No samples generated. No output CSV created.")
        return

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT, index=False)

    print("\nBackground sampling complete.")
    print("Total samples:", len(result))
    print("Output:", OUTPUT)
    print("These are not confirmed landslide-free locations.")


if __name__ == "__main__":
    main()
