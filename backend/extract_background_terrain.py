
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "pilot_background_samples.csv"
OUTPUT = ROOT / "data" / "processed" / "pilot_background_terrain_features.csv"

TILES = {
    "N30E078": ROOT / "data" / "processed" / "dem" / "himalayas" / "N30E078",
    "N27E088": ROOT / "data" / "processed" / "dem" / "northeast" / "N27E088",
}

RASTER_NAMES = {
    "elevation_m": "elevation.tif",
    "slope_degrees": "slope.tif",
    "aspect_degrees": "aspect.tif",
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    background = pd.read_csv(INPUT)

    required = {
        "sample_id", "pilot_tile", "longitude", "latitude", "label"
    }
    missing = required - set(background.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    if background["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found.")

    results = []

    for tile_name, group in background.groupby("pilot_tile"):
        if tile_name not in TILES:
            raise ValueError(f"No DEM configuration for tile {tile_name}")

        tile_dir = TILES[tile_name]
        paths = {
            feature: tile_dir / filename
            for feature, filename in RASTER_NAMES.items()
        }

        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

        with rasterio.open(paths["elevation_m"]) as elevation:
            crs = elevation.crs
            if crs is None:
                raise ValueError(f"Missing CRS in {paths['elevation_m']}")

            for feature, path in paths.items():
                with rasterio.open(path) as src:
                    if src.crs != crs:
                        raise ValueError(
                            f"CRS mismatch in {path}: {src.crs} vs {crs}"
                        )
                    if (
                        src.width != elevation.width
                        or src.height != elevation.height
                        or src.transform != elevation.transform
                    ):
                        raise ValueError(
                            f"Raster grid mismatch: {path}"
                        )

            transformer = Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True
            )

            x, y = transformer.transform(
                group["longitude"].to_numpy(),
                group["latitude"].to_numpy(),
            )

            coordinates = list(zip(x, y))
            tile_results = group[
                ["sample_id", "pilot_tile", "longitude", "latitude", "label"]
            ].copy()

            for feature, path in paths.items():
                with rasterio.open(path) as src:
                    values = np.array(
                        [sample[0] for sample in src.sample(coordinates)],
                        dtype="float64",
                    )

                    invalid = ~np.isfinite(values)
                    if src.nodata is not None:
                        invalid |= np.isclose(values, src.nodata)

                    values[invalid] = np.nan
                    tile_results[feature] = values

            feature_cols = list(paths.keys())
            tile_results["terrain_valid"] = (
                tile_results[feature_cols].notna().all(axis=1)
            )

            results.append(tile_results)

            print(
                f"{tile_name}: {len(tile_results)} samples; "
                f"{int(tile_results['terrain_valid'].sum())} "
                "with complete terrain"
            )

    output = pd.concat(results, ignore_index=True)
    output = output.sort_values("sample_id")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")

    try:
        output.to_csv(temporary, index=False)
        check = pd.read_csv(temporary)

        if len(check) != len(background):
            raise RuntimeError("Output row count does not match input.")

        if check["sample_id"].nunique() != len(background):
            raise RuntimeError("Output sample IDs are not unique.")

        temporary.replace(OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("\nBackground terrain extraction complete.")
    print("Output:", OUTPUT)
    print("Total samples:", len(output))
    print("Complete terrain:", int(output["terrain_valid"].sum()))
    print("Incomplete terrain:", int((~output["terrain_valid"]).sum()))
    print("\nMissing values:")
    print(output[feature_cols].isna().sum().to_string())
    print("\nTerrain summary:")
    print(output[feature_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
