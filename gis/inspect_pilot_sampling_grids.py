
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_landslide_event_features.csv"
)

TILES = {
    "N30E078": (
        PROJECT_ROOT / "data" / "processed" / "dem"
        / "himalayas" / "N30E078"
    ),
    "N27E088": (
        PROJECT_ROOT / "data" / "processed" / "dem"
        / "northeast" / "N27E088"
    ),
}

df = pd.read_csv(DATASET)

print("=" * 65)
print("PILOT DEM GRID AND VALID-CELL INSPECTION")
print("=" * 65)

for tile, dem_dir in TILES.items():
    print(f"\n--- {tile} ---")

    paths = [
        dem_dir / "elevation.tif",
        dem_dir / "slope.tif",
        dem_dir / "aspect.tif",
    ]

    if not all(path.exists() for path in paths):
        for path in paths:
            if not path.exists():
                print(f"MISSING: {path}")
        continue

    with rasterio.open(paths[0]) as elevation:
        reference = elevation
        print("CRS:", reference.crs)
        print(
            "Dimensions:",
            f"{reference.width} x {reference.height}",
        )
        print("Bounds:", reference.bounds)
        print("Pixel size:", reference.res)

    datasets = [rasterio.open(path) for path in paths]

    try:
        reference = datasets[0]

        aligned = all(
            ds.crs == reference.crs
            and ds.width == reference.width
            and ds.height == reference.height
            and ds.transform == reference.transform
            for ds in datasets[1:]
        )

        print("All three rasters aligned:", aligned)

        for path, ds in zip(paths, datasets):
            print(
                f"{path.name}: "
                f"CRS={ds.crs}, "
                f"dimensions={ds.width}x{ds.height}, "
                f"NoData={ds.nodata}"
            )

        if not aligned:
            print(
                "WARNING: Grid mismatch. "
                "Do not sample these rasters together."
            )
            continue

        valid = np.ones(
            (reference.height, reference.width),
            dtype=bool,
        )

        for ds in datasets:
            band = ds.read(1, masked=True)
            values = band.filled(np.nan)

            valid &= ~np.ma.getmaskarray(band)
            valid &= np.isfinite(values)

        total = valid.size
        valid_count = int(valid.sum())

        print("Total raster cells:", total)
        print("Cells valid in all three rasters:", valid_count)
        print(
            "Valid-cell percentage:",
            round(100 * valid_count / total, 2),
        )

    finally:
        for ds in datasets:
            ds.close()

    events = df[df["pilot_tile"] == tile]
    print("Known event records:", len(events))

print("\nInspection complete. No files were modified.")
