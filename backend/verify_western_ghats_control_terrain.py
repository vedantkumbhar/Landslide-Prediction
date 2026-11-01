
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAINING_FILE = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

VALIDATION_DIR = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "historical_validation"
)

CONTROLS_FILE = VALIDATION_DIR / "matched_rainfall_controls.csv"

AUDIT_FILE = (
    VALIDATION_DIR / "western_ghats_experimental_terrain_audit.csv"
)

DEM_DIR = ROOT / "data" / "processed" / "dem" / "western_ghats"

OUTPUT_FILE = (
    VALIDATION_DIR / "western_ghats_control_dem_verification.csv"
)

TERRAIN_FEATURES = {
    "elevation_m": "elevation",
    "slope_degrees": "slope",
    "aspect_degrees": "aspect",
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description):
    """Return the first available column from a list of candidates."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Missing {description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def numeric_value(value):
    """Convert a value to float, or NaN if unavailable."""
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def sample_dem(raster_path, longitude, latitude):
    """
    Sample a raster at a WGS84 longitude/latitude coordinate.

    Returns a dictionary containing the value, status and CRS.
    """
    result = {
        "value": np.nan,
        "status": "not_sampled",
        "raster_crs": None,
    }

    with rasterio.open(raster_path) as src:
        result["raster_crs"] = str(src.crs)

        if src.crs is None:
            result["status"] = "missing_raster_crs"
            return result

        if src.count < 1:
            result["status"] = "no_raster_band"
            return result

        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            [float(longitude)],
            [float(latitude)],
        )

        x, y = xs[0], ys[0]

        if not (
            src.bounds.left <= x <= src.bounds.right
            and src.bounds.bottom <= y <= src.bounds.top
        ):
            result["status"] = "outside_raster_bounds"
            return result

        row, col = src.index(x, y)

        if not (
            0 <= row < src.height
            and 0 <= col < src.width
        ):
            result["status"] = "outside_raster_grid"
            return result

        value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]

        if not np.isfinite(value):
            result["status"] = "nonfinite_pixel"
            return result

        if src.nodata is not None and np.isclose(
            value, src.nodata, rtol=0, atol=1e-8
        ):
            result["status"] = "nodata_pixel"
            return result

        result["value"] = float(value)
        result["status"] = "sampled"

    return result


# ============================================================
# 3. MAIN VERIFICATION
# ============================================================

def main():
    print("=" * 70)
    print("WESTERN GHATS CONTROL DEM VERIFICATION")
    print("=" * 70)

    # Validate required files.
    required_files = [
        TRAINING_FILE,
        CONTROLS_FILE,
        AUDIT_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # Never overwrite an existing report.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "The output report already exists. "
            "Rename or move it manually before rerunning:\n"
            f"{OUTPUT_FILE}"
        )

    training = pd.read_csv(TRAINING_FILE)
    controls = pd.read_csv(CONTROLS_FILE)
    audit = pd.read_csv(AUDIT_FILE)

    print(f"\nTraining rows: {len(training)}")
    print(f"Matched controls: {len(controls)}")
    print(f"Audit rows: {len(audit)}")

    # --------------------------------------------------------
    # A. Identify flagged control IDs.
    # --------------------------------------------------------

    audit_key_col = find_column(
        audit,
        ["sample_key"],
        "audit sample key",
    )

    audit_tile_col = find_column(
        audit,
        ["pilot_tile"],
        "audit pilot tile",
    )

    flagged = audit.copy()

    # Example: CONTROL_CTRL_99 -> CTRL_99
    flagged["control_id"] = (
        flagged[audit_key_col]
        .astype(str)
        .str.replace(r"^CONTROL_", "", regex=True)
    )

    controls_id_col = find_column(
        controls,
        ["control_id"],
        "control identifier",
    )

    background_id_col = find_column(
        controls,
        ["background_sample_id"],
        "background sample identifier",
    )

    latitude_col = find_column(
        controls,
        ["control_latitude", "latitude", "lat"],
        "control latitude",
    )

    longitude_col = find_column(
        controls,
        ["control_longitude", "longitude", "lon", "long"],
        "control longitude",
    )

    # Keep only the columns needed for coordinate lookup.
    control_lookup = controls[
        [
            controls_id_col,
            background_id_col,
            latitude_col,
            longitude_col,
        ]
    ].copy()

    control_lookup = control_lookup.rename(
        columns={
            controls_id_col: "control_id",
            background_id_col: "background_sample_id",
            latitude_col: "control_latitude",
            longitude_col: "control_longitude",
        }
    )

    # Ignore missing IDs in this temporary lookup.
    control_lookup = control_lookup.loc[
        control_lookup["control_id"].notna()
    ].copy()

    control_lookup["control_id"] = (
        control_lookup["control_id"].astype(str).str.strip()
    )

    duplicate_control_ids = control_lookup.loc[
        control_lookup["control_id"].duplicated(keep=False),
        "control_id",
    ].unique()

    if len(duplicate_control_ids):
        raise ValueError(
            "Duplicate control IDs detected:\n"
            f"{duplicate_control_ids[:10].tolist()}"
        )

    flagged["control_id"] = flagged["control_id"].astype(str).str.strip()

    check = flagged.merge(
        control_lookup,
        on="control_id",
        how="left",
        validate="many_to_one",
        indicator="control_match_status",
    )

    print("\nControl lookup status:")
    print(check["control_match_status"].value_counts(dropna=False))

    # --------------------------------------------------------
    # B. Match the control to its original training row.
    # --------------------------------------------------------

    training_id_col = find_column(
        training,
        ["sample_id", "background_sample_id", "sample_key"],
        "training sample identifier",
    )

    training_tile_col = find_column(
        training,
        ["pilot_tile"],
        "training pilot tile",
    )

    training_subset = training.copy()

    # Remove missing IDs from the temporary lookup only.
    training_subset = training_subset.loc[
        training_subset[training_id_col].notna()
    ].copy()

    training_subset["_training_id"] = (
        training_subset[training_id_col].astype(str).str.strip()
    )

    duplicate_training_ids = training_subset.loc[
        training_subset["_training_id"].duplicated(keep=False),
        "_training_id",
    ].unique()

    if len(duplicate_training_ids):
        raise ValueError(
            "Duplicate training sample IDs detected. "
            "Cannot safely select terrain records.\n"
            f"Examples: {duplicate_training_ids[:10].tolist()}"
        )

    check["_background_id"] = (
        check["background_sample_id"].astype("string").str.strip()
    )

    check = check.merge(
        training_subset,
        left_on="_background_id",
        right_on="_training_id",
        how="left",
        suffixes=("", "_training"),
        validate="many_to_one",
        indicator="training_match_status",
    )

    print("\nTraining sample lookup status:")
    print(check["training_match_status"].value_counts(dropna=False))

    unmatched = check.loc[
        check["training_match_status"] != "both",
        [audit_key_col, "background_sample_id"],
    ]

    if not unmatched.empty:
        print("\nControls without a matching training row:")
        print(unmatched.to_string(index=False))

    # --------------------------------------------------------
    # C. Sample the original DEMs.
    # --------------------------------------------------------

    results = []

    for _, row in check.iterrows():
        sample_key = row.get(audit_key_col)
        tile = row.get(audit_tile_col)

        result = {
            "sample_key": sample_key,
            "control_id": row.get("control_id"),
            "background_sample_id": row.get("background_sample_id"),
            "pilot_tile": tile,
            "control_latitude": numeric_value(
                row.get("control_latitude")
            ),
            "control_longitude": numeric_value(
                row.get("control_longitude")
            ),
            "control_match_status": row.get("control_match_status"),
            "training_match_status": row.get("training_match_status"),
        }

        # Record the terrain values stored in the original training row.
        for feature in TERRAIN_FEATURES:
            result[f"stored_{feature}"] = numeric_value(row.get(feature))

        if pd.isna(tile):
            result["verification_status"] = "missing_tile"
            results.append(result)
            continue

        if row.get("control_match_status") != "both":
            result["verification_status"] = "control_lookup_failed"
            results.append(result)
            continue

        if row.get("training_match_status") != "both":
            result["verification_status"] = "training_lookup_failed"
            results.append(result)
            continue

        latitude = result["control_latitude"]
        longitude = result["control_longitude"]

        if not np.isfinite(latitude) or not np.isfinite(longitude):
            result["verification_status"] = "missing_coordinates"
            results.append(result)
            continue

        metric_dir = DEM_DIR / str(tile) / "metric"

        raster_paths = {
            "elevation": metric_dir / "elevation.tif",
            "slope": metric_dir / "slope.tif",
            "aspect": metric_dir / "aspect.tif",
        }

        missing_rasters = [
            name
            for name, path in raster_paths.items()
            if not path.exists()
        ]

        if missing_rasters:
            result["verification_status"] = "missing_dem_raster"
            result["missing_rasters"] = ";".join(missing_rasters)
            results.append(result)
            continue

        statuses = []

        for raster_name, raster_path in raster_paths.items():
            sampled = sample_dem(
                raster_path,
                longitude,
                latitude,
            )

            result[f"dem_{raster_name}"] = sampled["value"]
            result[f"dem_{raster_name}_status"] = sampled["status"]
            result[f"dem_{raster_name}_crs"] = sampled["raster_crs"]

            statuses.append(sampled["status"])

        # Compare stored values with the newly sampled DEM values.
        for feature, raster_name in TERRAIN_FEATURES.items():
            stored = result[f"stored_{feature}"]
            sampled_value = result.get(f"dem_{raster_name}", np.nan)

            if np.isfinite(stored) and np.isfinite(sampled_value):
                # A small tolerance accommodates floating-point rounding.
                result[f"matches_{feature}"] = bool(
                    np.isclose(stored, sampled_value, atol=0.05, rtol=0)
                )
            else:
                result[f"matches_{feature}"] = None

        if all(status == "sampled" for status in statuses):
            result["verification_status"] = "all_rasters_sampled"
        elif any(status == "sampled" for status in statuses):
            result["verification_status"] = "partially_sampled"
        else:
            result["verification_status"] = "no_valid_dem_samples"

        results.append(result)

    # --------------------------------------------------------
    # D. Save the new report and print the results.
    # --------------------------------------------------------

    output = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Recheck immediately before writing to prevent overwriting.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report:\n{OUTPUT_FILE}"
        )

    output.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print("\nOverall status:")
    print(output["verification_status"].value_counts(dropna=False))

    for feature in TERRAIN_FEATURES:
        column = f"matches_{feature}"
        if column in output.columns:
            print(f"\nStored vs DEM: {feature}")
            print(output[column].value_counts(dropna=False))

    # Show the original and resampled elevation for easy inspection.
    display_columns = [
        "sample_key",
        "pilot_tile",
        "control_latitude",
        "control_longitude",
        "stored_elevation_m",
        "dem_elevation",
        "dem_elevation_status",
        "stored_slope_degrees",
        "dem_slope",
        "stored_aspect_degrees",
        "dem_aspect",
        "verification_status",
    ]

    display_columns = [
        column for column in display_columns
        if column in output.columns
    ]

    print("\nControl terrain comparison:")
    print(output[display_columns].to_string(index=False))

    print("\nNew report saved to:")
    print(OUTPUT_FILE)

    print(
        "\nNo original training data, control data, or DEM raster "
        "was modified."
    )


if __name__ == "__main__":
    main()
