
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem"
    / "northeast"
    / "N27E088"
)

RASTERS = {
    "ELEVATION": DEM_DIR / "elevation.tif",
    "SLOPE": DEM_DIR / "slope.tif",
    "ASPECT": DEM_DIR / "aspect.tif",
}


def main():
    print("=" * 65)
    print("NORTHEAST DEM TERRAIN VALIDATION")
    print("=" * 65)

    datasets = {}

    for name, path in RASTERS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required raster not found: {path}"
            )

        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()

            print(f"\n{name}")
            print("-" * 40)
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"CRS: {src.crs}")
            print(f"NoData: {src.nodata}")
            print(f"Pixel size: {src.res}")
            print(f"Bounds: {src.bounds}")
            print(f"Valid pixels: {valid.size:,}")
            print(f"NoData pixels: {int(data.mask.sum()):,}")

            if valid.size:
                print(
                    f"Non-finite valid values: "
                    f"{int((~np.isfinite(valid)).sum())}"
                )
                print(f"Minimum: {valid.min():.3f}")
                print(f"Maximum: {valid.max():.3f}")
                print(f"Mean: {valid.mean():.3f}")

            datasets[name] = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "data": data,
            }

    print("\n" + "=" * 65)
    print("CONSISTENCY CHECKS")
    print("=" * 65)

    reference = datasets["ELEVATION"]

    for name in ("SLOPE", "ASPECT"):
        current = datasets[name]

        checks = {
            "Same CRS": current["crs"] == reference["crs"],
            "Same dimensions": (
                current["width"] == reference["width"]
                and current["height"] == reference["height"]
            ),
            "Same grid transform": (
                current["transform"] == reference["transform"]
            ),
        }

        for description, passed in checks.items():
            print(
                f"{name} — {description}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

    slope = datasets["SLOPE"]["data"].compressed()
    aspect = datasets["ASPECT"]["data"].compressed()

    
    
    print("\nTerrain range checks:")

    slope_ok = (
        np.isfinite(slope).all()
        and (slope >= 0).all()
        and (slope <= 90).all()
    )

    flat_mask = aspect == -1
    directional_aspect = aspect[~flat_mask]

    aspect_ok = (
        np.isfinite(aspect).all()
        and (directional_aspect >= 0).all()
        and (directional_aspect < 360).all()
    )

    print(f"Slope between 0 and 90 degrees: {slope_ok}")
    print(f"Aspect follows the expected range/convention: {aspect_ok}")
    print(f"Flat-terrain pixels: {int(flat_mask.sum()):,}")

    if directional_aspect.size:
        print(
            "Minimum directional aspect: "
            f"{directional_aspect.min():.6f}"
        )
        print(
            "Maximum directional aspect: "
            f"{directional_aspect.max():.6f}"
        )


    print("\nValidation report completed.")


if __name__ == "__main__":
    main()
