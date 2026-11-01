
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union
except ImportError as exc:
    raise SystemExit(
        "Missing GIS dependencies. Install them inside your .venv with:\n"
        "python -m pip install geopandas shapely\n"
        f"Original error: {exc}"
    )


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONTROLS_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "historical_validation"
    / "matched_rainfall_controls.csv"
)

TRAINING_FILE = (
    ROOT / "data" / "processed"
    / "western_ghats_pilot_training_dataset.csv"
)

REFERENCE_DIR = (
    ROOT / "data" / "raw" / "reference" / "naturalearth_land"
)

LAND_ZIP = REFERENCE_DIR / "ne_10m_land.zip"
LAND_SHP = REFERENCE_DIR / "ne_10m_land.shp"

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "historical_validation"
    / "western_ghats_control_land_validity_audit.csv"
)

LAND_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"

# Controls this close to a land boundary require extra inspection.
COASTAL_REVIEW_METRES = 200.0


# ============================================================
# 2. DOWNLOAD AND PREPARE FREE LAND DATA
# ============================================================

def prepare_land_dataset():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    if not LAND_SHP.exists():
        if not LAND_ZIP.exists():
            print("Downloading free Natural Earth land polygons...")
            print(LAND_URL)

            try:
                urlretrieve(LAND_URL, LAND_ZIP)
            except Exception as exc:
                raise RuntimeError(
                    "Could not download Natural Earth land polygons. "
                    "Check your internet connection and rerun. "
                    "No project datasets were changed."
                ) from exc

        print("Extracting Natural Earth land polygons...")

        with ZipFile(LAND_ZIP, "r") as archive:
            # Only extract files from the expected Natural Earth archive.
            expected = [
                name for name in archive.namelist()
                if Path(name).name.startswith("ne_10m_land.")
            ]

            if not any(name.endswith(".shp") for name in expected):
                raise RuntimeError(
                    "The downloaded archive does not contain "
                    "ne_10m_land.shp."
                )

            for name in expected:
                destination = REFERENCE_DIR / Path(name).name

                # Never overwrite an existing extracted file.
                if not destination.exists():
                    with archive.open(name) as source, destination.open("wb") as target:
                        target.write(source.read())

    if not LAND_SHP.exists():
        raise FileNotFoundError(
            f"Land polygon shapefile not found: {LAND_SHP}"
        )

    land = gpd.read_file(LAND_SHP)

    if land.empty:
        raise RuntimeError("Natural Earth land dataset is empty.")

    if land.crs is None:
        raise RuntimeError("Land dataset has no defined coordinate system.")

    # Point coordinates are longitude/latitude (EPSG:4326).
    land = land.to_crs("EPSG:4326")
    land_geometry = unary_union(land.geometry)

    print(f"Land polygon features loaded: {len(land)}")
    return land_geometry


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Missing {description}.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def number_or_nan(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def approximate_distance_metres(point, boundary):
    """
    Approximate distance in metres for a small distance in geographic
    coordinates. This is used only to flag points near a coastline.
    """
    distance_degrees = point.distance(boundary)

    latitude_radians = np.radians(point.y)
    metres_per_degree = 111320.0 * np.sqrt(
        (np.cos(latitude_radians) ** 2 + 1.0) / 2.0
    )

    return float(distance_degrees * metres_per_degree)


# ============================================================
# 4. AUDIT ALL 56 CONTROLS
# ============================================================

def main():
    print("=" * 70)
    print("WESTERN GHATS CONTROL LAND-VALIDITY AUDIT")
    print("=" * 70)

    for path in (CONTROLS_FILE, TRAINING_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found:\n{path}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing audit report:\n"
            f"{OUTPUT_FILE}\n"
            "Rename the old report manually if you want a new run."
        )

    controls = pd.read_csv(CONTROLS_FILE)
    training = pd.read_csv(TRAINING_FILE)

    print(f"\nControls loaded: {len(controls)}")

    control_id_col = find_column(
        controls, ["control_id"], "control ID"
    )
    sample_id_col = find_column(
        controls, ["background_sample_id"], "background sample ID"
    )
    lat_col = find_column(
        controls,
        ["control_latitude", "latitude", "lat"],
        "control latitude",
    )
    lon_col = find_column(
        controls,
        ["control_longitude", "longitude", "lon", "long"],
        "control longitude",
    )

    training_id_col = find_column(
        training,
        ["sample_id", "background_sample_id", "sample_key"],
        "training sample ID",
    )

    # Prepare a temporary training lookup without missing IDs.
    terrain_columns = [
        column for column in (
            "pilot_tile",
            "terrain_source_tile",
            "elevation_m",
            "slope_degrees",
            "aspect_degrees",
            "terrain_valid",
            "terrain_status",
        )
        if column in training.columns
    ]

    training_lookup = training[
        [training_id_col] + terrain_columns
    ].copy()

    training_lookup = training_lookup.loc[
        training_lookup[training_id_col].notna()
    ].copy()

    training_lookup["_sample_key"] = (
        training_lookup[training_id_col].astype(str).str.strip()
    )

    duplicates = training_lookup.loc[
        training_lookup["_sample_key"].duplicated(keep=False),
        "_sample_key",
    ].unique()

    if len(duplicates):
        raise ValueError(
            "Duplicate training sample IDs prevent a safe match. "
            f"Examples: {duplicates[:10].tolist()}"
        )

    controls["_sample_key"] = (
        controls[sample_id_col].astype("string").str.strip()
    )

    controls = controls.merge(
        training_lookup.drop(columns=[training_id_col]),
        left_on="_sample_key",
        right_on="_sample_key",
        how="left",
        validate="many_to_one",
        indicator="training_match_status",
        suffixes=("", "_training"),
    )

    land_geometry = prepare_land_dataset()
    land_boundary = land_geometry.boundary

    results = []

    for _, row in controls.iterrows():
        latitude = number_or_nan(row.get(lat_col))
        longitude = number_or_nan(row.get(lon_col))

        result = {
            "control_id": row.get(control_id_col),
            "background_sample_id": row.get(sample_id_col),
            "latitude": latitude,
            "longitude": longitude,
            "control_status": row.get("control_status"),
            "background_label": row.get("background_label"),
            "pilot_tile": row.get("pilot_tile"),
            "terrain_source_tile": row.get("terrain_source_tile"),
            "elevation_m": number_or_nan(row.get("elevation_m")),
            "slope_degrees": number_or_nan(row.get("slope_degrees")),
            "aspect_degrees": number_or_nan(row.get("aspect_degrees")),
            "training_match_status": row.get("training_match_status"),
        }

        if not np.isfinite(latitude) or not np.isfinite(longitude):
            result["land_status"] = "unverified_missing_coordinates"
            result["distance_to_land_boundary_m_approx"] = np.nan
            results.append(result)
            continue

        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            result["land_status"] = "unverified_invalid_coordinates"
            result["distance_to_land_boundary_m_approx"] = np.nan
            results.append(result)
            continue

        point = Point(longitude, latitude)

        if not land_geometry.is_valid:
            result["land_status"] = "unverified_invalid_land_geometry"
            result["distance_to_land_boundary_m_approx"] = np.nan
            results.append(result)
            continue

        inside_land = land_geometry.covers(point)
        distance_m = approximate_distance_metres(point, land_boundary)

        result["inside_land_polygon"] = bool(inside_land)
        result["distance_to_land_boundary_m_approx"] = round(distance_m, 2)

        if distance_m <= COASTAL_REVIEW_METRES:
            result["land_status"] = "coastal_boundary_review"
        elif inside_land:
            result["land_status"] = "land"
        else:
            result["land_status"] = "outside_land_polygon"

        # This is a warning, not a decision to exclude the control.
        elevation = result["elevation_m"]

        if np.isfinite(elevation) and elevation < 0:
            result["negative_elevation_review"] = True
        else:
            result["negative_elevation_review"] = False

        if result["training_match_status"] != "both":
            result["data_match_warning"] = "training_sample_not_matched"
        else:
            result["data_match_warning"] = ""

        results.append(result)

    output = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite report:\n{OUTPUT_FILE}"
        )

    output.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print("\nLand classification:")
    print(output["land_status"].value_counts(dropna=False))

    print("\nNegative elevation review:")
    print(
        output["negative_elevation_review"]
        .value_counts(dropna=False)
    )

    print("\nTraining sample matching:")
    print(
        output["training_match_status"]
        .value_counts(dropna=False)
    )

    print("\nControls requiring review:")
    review = output.loc[
        output["land_status"] != "land"
    ]

    if review.empty:
        print("No controls were flagged by the land-polygon check.")
    else:
        columns = [
            "control_id",
            "background_sample_id",
            "latitude",
            "longitude",
            "pilot_tile",
            "elevation_m",
            "land_status",
            "distance_to_land_boundary_m_approx",
        ]
        print(review[columns].to_string(index=False))

    print("\nSaved new audit report:")
    print(OUTPUT_FILE)

    print(
        "\nIMPORTANT: Natural Earth is a generalized land-boundary dataset. "
        "Coastal points and points near small islands or narrow landforms "
        "may need higher-resolution inspection. No controls were deleted, "
        "and no model was retrained."
    )


if __name__ == "__main__":
    main()
