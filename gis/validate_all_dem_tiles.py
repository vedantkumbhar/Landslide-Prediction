
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLAN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dem" / "dem_tile_plan.csv"
)
DEM_ROOT = PROJECT_ROOT / "data" / "processed" / "dem"
REPORT_PATH = DEM_ROOT / "dem_raster_validation.csv"

LAYERS = ("elevation", "slope", "aspect")
BLOCK_SIZE = 512


def find_layer_paths(tile_dir):
    """Support both metric/ and legacy tile-root raster layouts."""
    metric_dir = tile_dir / "metric"

    metric_paths = {
        layer: metric_dir / f"{layer}.tif"
        for layer in LAYERS
    }
    root_paths = {
        layer: tile_dir / f"{layer}.tif"
        for layer in LAYERS
    }

    # Prefer metric/ when all three layers are present.
    if all(path.is_file() for path in metric_paths.values()):
        return metric_paths

    # Otherwise use the legacy layout if all three layers exist there.
    if all(path.is_file() for path in root_paths.values()):
        return root_paths

    # Return the preferred layout so missing files are reported clearly.
    return metric_paths


def validate_raster(region, tile, layer, path):
    """Validate a single raster in bounded memory blocks."""
    errors = []

    result = {
        "region": region,
        "tile": tile,
        "layer": layer,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "status": "FAIL",
        "width": None,
        "height": None,
        "crs": None,
        "nodata": None,
        "valid_pixels": 0,
        "nodata_pixels": 0,
        "nonfinite_valid_pixels": 0,
        "minimum": None,
        "maximum": None,
        "range_error_pixels": 0,
        "details": "",
    }

    if not path.is_file():
        result["details"] = f"Missing file: {path}"
        return result, None

    try:
        with rasterio.open(path) as src:
            result.update({
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs) if src.crs else None,
                "nodata": src.nodata,
            })

            if src.count < 1:
                errors.append("Raster contains no bands")

            if src.crs is None:
                errors.append("Missing CRS")

            valid_count = 0
            nodata_count = 0
            nonfinite_count = 0
            range_error_count = 0
            min_value = float("inf")
            max_value = float("-inf")

            for row in range(0, src.height, BLOCK_SIZE):
                for col in range(0, src.width, BLOCK_SIZE):
                    height = min(BLOCK_SIZE, src.height - row)
                    width = min(BLOCK_SIZE, src.width - col)
                    window = Window(col, row, width, height)

                    data = src.read(1, window=window, masked=True)
                    nodata_count += int(np.ma.getmaskarray(data).sum())

                    # Count unmasked, non-finite values as errors.
                    raw = np.asarray(data.data)
                    mask = np.ma.getmaskarray(data)
                    finite = np.isfinite(raw)
                    nonfinite_count += int((~mask & ~finite).sum())

                    valid_mask = ~mask & finite
                    values = raw[valid_mask]

                    if values.size == 0:
                        continue

                    valid_count += int(values.size)
                    min_value = min(min_value, float(values.min()))
                    max_value = max(max_value, float(values.max()))

                    if layer == "slope":
                        range_error_count += int(
                            ((values < 0) | (values > 90)).sum()
                        )

                    elif layer == "aspect":
                        # The project's convention is -1 for flat terrain.
                        invalid_aspect = (
                            (values != -1)
                            & ((values < 0) | (values >= 360))
                        )
                        range_error_count += int(invalid_aspect.sum())

            if valid_count == 0:
                errors.append("No valid finite pixels")

            if nonfinite_count:
                errors.append(
                    f"{nonfinite_count} unmasked non-finite pixels"
                )

            if range_error_count:
                errors.append(
                    f"{range_error_count} pixels outside expected range"
                )

            result.update({
                "valid_pixels": valid_count,
                "nodata_pixels": nodata_count,
                "nonfinite_valid_pixels": nonfinite_count,
                "minimum": min_value if valid_count else None,
                "maximum": max_value if valid_count else None,
                "range_error_pixels": range_error_count,
            })

            metadata = {
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "transform": src.transform,
                "bounds": tuple(src.bounds),
            }

    except Exception as exc:
        result["details"] = f"Raster read failed: {exc}"
        return result, None

    result["status"] = "FAIL" if errors else "PASS"
    result["details"] = "; ".join(errors) if errors else "OK"

    return result, metadata


def main():
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(f"Tile plan not found: {PLAN_PATH}")

    plan = pd.read_csv(PLAN_PATH, dtype={"tile": "string"})

    required = {"region", "tile"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(
            f"Tile plan missing required columns: {sorted(missing)}"
        )

    region_dirs = {
        "Himalayas": "himalayas",
        "Northeast India": "northeast",
        "Western Maharashtra": "western_maharashtra",
    }

    results = []

    for _, item in plan.iterrows():
        region = str(item["region"]).strip()
        tile = str(item["tile"]).strip()

        print(f"Validating {tile} | {region}")

        region_dir = region_dirs.get(region)

        if region_dir is None:
            results.append({
                "region": region,
                "tile": tile,
                "layer": "ALL",
                "status": "FAIL",
                "details": f"Unknown region: {region}",
            })
            continue

        tile_dir = DEM_ROOT / region_dir / tile
        paths = find_layer_paths(tile_dir)
        metadata = {}

        for layer in LAYERS:
            result, info = validate_raster(
                region, tile, layer, paths[layer]
            )
            results.append(result)

            if info is not None:
                metadata[layer] = info

        # Compare slope and aspect grids with elevation.
        if "elevation" in metadata:
            reference = metadata["elevation"]

            for layer in ("slope", "aspect"):
                if layer not in metadata:
                    results.append({
                        "region": region,
                        "tile": tile,
                        "layer": f"{layer}_alignment",
                        "status": "FAIL",
                        "details": f"{layer} could not be read",
                    })
                    continue

                current = metadata[layer]
                failed = []

                if current["crs"] != reference["crs"]:
                    failed.append("CRS mismatch")

                if (
                    current["width"] != reference["width"]
                    or current["height"] != reference["height"]
                ):
                    failed.append("Dimensions mismatch")

                if current["transform"] != reference["transform"]:
                    failed.append("Grid transform mismatch")

                results.append({
                    "region": region,
                    "tile": tile,
                    "layer": f"{layer}_alignment",
                    "status": "FAIL" if failed else "PASS",
                    "details": "; ".join(failed)
                    if failed else "Matches elevation grid",
                })

    report = pd.DataFrame(results)
    report.to_csv(REPORT_PATH, index=False)

    passed = int((report["status"] == "PASS").sum())
    failed = int((report["status"] == "FAIL").sum())

    print("\n" + "=" * 65)
    print("ALL DEM TILE VALIDATION SUMMARY")
    print("=" * 65)
    print("Planned tiles:", len(plan))
    print("Validation checks:", len(report))
    print("PASS:", passed)
    print("FAIL:", failed)
    print("Report:", REPORT_PATH)

    failures = report[report["status"] == "FAIL"]

    if not failures.empty:
        print("\nFailures (first 40):")
        columns = [
            col for col in ("region", "tile", "layer", "details")
            if col in failures.columns
        ]
        print(failures[columns].head(40).to_string(index=False))
    else:
        print("\nAll recorded checks passed.")


if __name__ == "__main__":
    main()
