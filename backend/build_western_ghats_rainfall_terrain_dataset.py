
from pathlib import Path
import pandas as pd

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAINING_FILE = (
    BASE_DIR / "data/processed/western_ghats_pilot_training_dataset.csv"
)

VALIDATION_DIR = (
    BASE_DIR
    / "data/processed/rainfall/western_ghats/historical_validation"
)

PAIRS_FILE = VALIDATION_DIR / "paired_event_control_rainfall_analysis.csv"
CONTROLS_FILE = VALIDATION_DIR / "matched_rainfall_controls.csv"

OUTPUT_FILE = (
    VALIDATION_DIR
    / "western_ghats_rainfall_terrain_experimental_dataset.csv"
)

PILOT_TILES = {"N09E076", "N11E076", "N19E072"}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def require_columns(frame, required, name):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def normalize_numeric_id(series):
    """Normalize numeric identifiers such as 99 and 99.0."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype("string")


def require_unique(frame, column, name):
    if frame[column].isna().any():
        raise ValueError(f"{name} contains missing values in {column}.")

    duplicates = frame.loc[
        frame[column].duplicated(keep=False), column
    ].dropna().unique().tolist()

    if duplicates:
        raise ValueError(
            f"{name} has duplicate {column} values: {duplicates[:10]}"
        )


def normalize_sample_id(series):
    return series.astype("string").str.strip()


# ============================================================
# 1. INPUT VALIDATION
# ============================================================

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Output already exists. Refusing to overwrite:\n{OUTPUT_FILE}\n"
        "Inspect or rename that file before running this script again."
    )

for path in [TRAINING_FILE, PAIRS_FILE, CONTROLS_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

training = pd.read_csv(TRAINING_FILE)
pairs = pd.read_csv(PAIRS_FILE)
controls = pd.read_csv(CONTROLS_FILE)

require_columns(
    training,
    [
        "event_id",
        "sample_id",
        "sample_source",
        "label",
        "pilot_tile",
        "terrain_source_tile",
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "longitude",
        "latitude",
    ],
    "Training dataset",
)

require_columns(
    pairs,
    [
        "event_id",
        "control_id",
        "background_sample_id",
        "event_time_status",
        "event_pre_event_24h_mm",
        "control_pre_event_24h_mm",
        "event_pre_event_72h_mm",
        "control_pre_event_72h_mm",
        "event_pre_event_24h_valid_hours",
        "control_pre_event_24h_valid_hours",
        "event_pre_event_72h_valid_hours",
        "control_pre_event_72h_valid_hours",
    ],
    "Rainfall pairs",
)

require_columns(
    controls,
    [
        "control_id",
        "matched_event_id",
        "background_sample_id",
        "control_status",
        "background_label",
    ],
    "Matched controls",
)

# Only use records in the three selected pilot tiles.
training = training[
    training["pilot_tile"].isin(PILOT_TILES)
].copy()

events = training[
    training["sample_source"].eq("historical_inventory")
].copy()

background = training[
    training["sample_source"].eq("random_background_sampling")
].copy()

if events.empty:
    raise ValueError("No historical event records found in the pilot tiles.")

if background.empty:
    raise ValueError("No background samples found in the pilot tiles.")

# Create normalized join keys.
events["_event_key"] = normalize_numeric_id(events["event_id"])
pairs["_event_key"] = normalize_numeric_id(pairs["event_id"])
controls["_event_key"] = normalize_numeric_id(controls["matched_event_id"])

background["_sample_key"] = normalize_sample_id(background["sample_id"])
pairs["_sample_key"] = normalize_sample_id(pairs["background_sample_id"])
controls["_sample_key"] = normalize_sample_id(controls["background_sample_id"])

# Validate identifiers.
require_unique(events, "_event_key", "Historical event records")
require_unique(pairs, "_event_key", "Rainfall pairs")
require_unique(pairs, "control_id", "Rainfall pairs")
require_unique(controls, "control_id", "Matched controls")
require_unique(background, "_sample_key", "Background samples")

print("=" * 78)
print("WESTERN GHATS RAINFALL + TERRAIN DATASET BUILDER")
print("=" * 78)
print(f"Training rows in pilot tiles: {len(training)}")
print(f"Historical event records available: {len(events)}")
print(f"Background records available: {len(background)}")
print(f"Rainfall pairs: {len(pairs)}")


# ============================================================
# 2. VALIDATE EVENT-CONTROL MAPPING
# ============================================================

mapping = pairs[
    ["control_id", "_event_key", "_sample_key"]
].merge(
    controls[
        [
            "control_id",
            "_event_key",
            "_sample_key",
            "control_status",
            "background_label",
        ]
    ],
    on="control_id",
    how="left",
    suffixes=("_pair", "_mapping"),
    validate="one_to_one",
    indicator="_mapping_merge",
)

if not mapping["_mapping_merge"].eq("both").all():
    raise ValueError("Some rainfall pairs have no matching control mapping.")

event_mismatch = (
    mapping["_event_key_pair"].astype("string")
    != mapping["_event_key_mapping"].astype("string")
)

sample_mismatch = (
    mapping["_sample_key_pair"].astype("string")
    != mapping["_sample_key_mapping"].astype("string")
)

if event_mismatch.any():
    raise ValueError("Event IDs disagree between rainfall and control files.")

if sample_mismatch.any():
    raise ValueError(
        "Background sample IDs disagree between rainfall and control files."
    )

# Confirm the labels in the control mapping are background labels,
# not verified evidence that a location is landslide-free.
if not pd.to_numeric(
    mapping["background_label"], errors="coerce"
).eq(0).all():
    raise ValueError("Unexpected background_label values in control mapping.")


# ============================================================
# 3. PREPARE TERRAIN TABLES WITH UNIQUE COLUMN NAMES
# ============================================================

terrain_columns = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "longitude",
    "latitude",
    "pilot_tile",
    "terrain_source_tile",
]

event_terrain = events[
    ["_event_key"] + terrain_columns
].copy()

event_terrain = event_terrain.rename(
    columns={
        "elevation_m": "terrain_elevation_m",
        "slope_degrees": "terrain_slope_degrees",
        "aspect_degrees": "terrain_aspect_degrees",
        "longitude": "terrain_longitude",
        "latitude": "terrain_latitude",
        "pilot_tile": "terrain_pilot_tile",
        "terrain_source_tile": "terrain_source_tile_matched",
    }
)

background_terrain = background[
    ["_sample_key"] + terrain_columns
].copy()

background_terrain = background_terrain.rename(
    columns={
        "elevation_m": "terrain_elevation_m",
        "slope_degrees": "terrain_slope_degrees",
        "aspect_degrees": "terrain_aspect_degrees",
        "longitude": "terrain_longitude",
        "latitude": "terrain_latitude",
        "pilot_tile": "terrain_pilot_tile",
        "terrain_source_tile": "terrain_source_tile_matched",
    }
)


# ============================================================
# 4. MATCH RAINFALL PAIRS TO HISTORICAL EVENT TERRAIN
# ============================================================

event_pairs = pairs.merge(
    event_terrain,
    on="_event_key",
    how="left",
    validate="many_to_one",
    indicator="_event_merge",
)

if not event_pairs["_event_merge"].eq("both").all():
    unmatched = event_pairs.loc[
        event_pairs["_event_merge"].ne("both"), "event_id"
    ].tolist()

    raise ValueError(
        f"Events without terrain matches: {unmatched[:10]}. "
        "Verify the pilot-tile coverage and event identifiers."
    )

event_pairs = event_pairs.drop(columns=["_event_merge"])


# ============================================================
# 5. MATCH BACKGROUND CONTROLS TO TERRAIN
# ============================================================

# Keep the authoritative control metadata from matched_rainfall_controls.csv.
# The rainfall-pairs file already has a control_status column, so avoid
# merging that duplicate column a second time.
control_metadata = controls[
    [
        "control_id",
        "_event_key",
        "_sample_key",
        "control_status",
        "background_label",
    ]
].copy()

control_pairs = pairs.drop(
    columns=["control_status"],
    errors="ignore",
).merge(
    control_metadata,
    on=["control_id", "_event_key", "_sample_key"],
    how="left",
    validate="one_to_one",
    indicator="_control_merge",
)

if not control_pairs["_control_merge"].eq("both").all():
    raise ValueError(
        "Some rainfall pairs failed control mapping validation."
    )

control_rows = control_pairs.merge(
    background_terrain,
    on="_sample_key",
    how="left",
    validate="many_to_one",
    indicator="_terrain_merge",
)

if not control_rows["_terrain_merge"].eq("both").all():
    unmatched = control_rows.loc[
        control_rows["_terrain_merge"].ne("both"),
        "background_sample_id",
    ].tolist()

    raise ValueError(
        f"Controls without terrain matches: {unmatched[:10]}. "
        "Check sample_id against background_sample_id."
    )

control_rows = control_rows.drop(
    columns=["_control_merge", "_terrain_merge"],
    errors="ignore",
)



# ============================================================
# 6. BUILD HISTORICAL EVENT ROWS
# ============================================================

event_output = pd.DataFrame({
    "sample_key": "EVENT_" + event_pairs["_event_key"].astype(str),
    "matched_event_id": event_pairs["event_id"],
    "control_id": event_pairs["control_id"],
    "background_sample_id": event_pairs["background_sample_id"],
    "sample_role": "historical_event",
    "target_label": 1,
    "label_interpretation": "recorded_historical_landslide",
    "control_status": "not_applicable",
    "event_time_status": event_pairs["event_time_status"],

    "elevation_m": event_pairs["terrain_elevation_m"],
    "slope_degrees": event_pairs["terrain_slope_degrees"],
    "aspect_degrees": event_pairs["terrain_aspect_degrees"],
    "longitude": event_pairs["terrain_longitude"],
    "latitude": event_pairs["terrain_latitude"],
    "pilot_tile": event_pairs["terrain_pilot_tile"],
    "terrain_source_tile": event_pairs["terrain_source_tile_matched"],

    "pre_event_24h_rainfall_mm": event_pairs["event_pre_event_24h_mm"],
    "pre_event_24h_valid_hours": event_pairs["event_pre_event_24h_valid_hours"],
    "pre_event_72h_rainfall_mm": event_pairs["event_pre_event_72h_mm"],
    "pre_event_72h_valid_hours": event_pairs["event_pre_event_72h_valid_hours"],
})


# ============================================================
# 7. BUILD UNVERIFIED BACKGROUND CONTROL ROWS
# ============================================================

control_output = pd.DataFrame({
    "sample_key": "CONTROL_" + control_rows["control_id"].astype(str),
    "matched_event_id": control_rows["event_id"],
    "control_id": control_rows["control_id"],
    "background_sample_id": control_rows["background_sample_id"],
    "sample_role": "background_control",
    "target_label": 0,
    "label_interpretation": (
        "unverified_background_not_confirmed_negative"
    ),
    "control_status": control_rows["control_status"],
    "event_time_status": control_rows["event_time_status"],

    "elevation_m": control_rows["terrain_elevation_m"],
    "slope_degrees": control_rows["terrain_slope_degrees"],
    "aspect_degrees": control_rows["terrain_aspect_degrees"],
    "longitude": control_rows["terrain_longitude"],
    "latitude": control_rows["terrain_latitude"],
    "pilot_tile": control_rows["terrain_pilot_tile"],
    "terrain_source_tile": control_rows["terrain_source_tile_matched"],

    "pre_event_24h_rainfall_mm": control_rows["control_pre_event_24h_mm"],
    "pre_event_24h_valid_hours": control_rows["control_pre_event_24h_valid_hours"],
    "pre_event_72h_rainfall_mm": control_rows["control_pre_event_72h_mm"],
    "pre_event_72h_valid_hours": control_rows["control_pre_event_72h_valid_hours"],
})


# ============================================================
# 8. VALIDATE RAINFALL WINDOW COVERAGE
# ============================================================

for frame in [event_output, control_output]:
    numeric_columns = [
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "longitude",
        "latitude",
        "pre_event_24h_rainfall_mm",
        "pre_event_24h_valid_hours",
        "pre_event_72h_rainfall_mm",
        "pre_event_72h_valid_hours",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # A complete 24-hour window requires 24 valid hourly records.
    valid_24h = frame["pre_event_24h_valid_hours"].eq(24)

    ambiguous_time = frame["event_time_status"].astype(str).str.contains(
        "ambiguous",
        case=False,
        na=False,
    )

    frame.loc[
        (~valid_24h) | ambiguous_time,
        "pre_event_24h_rainfall_mm",
    ] = pd.NA

    # A complete 72-hour window requires 72 valid hourly records.
    valid_72h = frame["pre_event_72h_valid_hours"].eq(72)

    frame.loc[
        ~valid_72h,
        "pre_event_72h_rainfall_mm",
    ] = pd.NA

    # Event-day precipitation is deliberately not included.
    frame["event_day_rainfall_included"] = False
    frame["dataset_status"] = "experimental_not_validated_for_prediction"


# ============================================================
# 9. FINAL VALIDATION AND SAVE
# ============================================================

output = pd.concat(
    [event_output, control_output],
    ignore_index=True,
)

if output["sample_key"].isna().any():
    raise ValueError("Final output contains missing sample keys.")

if output["sample_key"].duplicated().any():
    raise ValueError("Final output contains duplicate sample keys.")

if len(event_output) != len(pairs):
    raise ValueError(
        f"Expected {len(pairs)} event rows; got {len(event_output)}."
    )

if len(control_output) != len(pairs):
    raise ValueError(
        f"Expected {len(pairs)} control rows; got {len(control_output)}."
    )

if len(output) != 2 * len(pairs):
    raise ValueError("Unexpected number of rows in final output.")

# Check that every row has a valid pilot tile.
if not output["pilot_tile"].isin(PILOT_TILES).all():
    raise ValueError("Unexpected or missing pilot tile in output.")

output.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# 10. REPORT
# ============================================================

print("\n" + "=" * 78)
print("WESTERN GHATS RAINFALL + TERRAIN DATASET CREATED")
print("=" * 78)

print(f"Saved file: {OUTPUT_FILE}")
print(f"Input rainfall pairs: {len(pairs)}")
print(f"Historical event rows: {len(event_output)}")
print(f"Unverified control rows: {len(control_output)}")
print(f"Total rows: {len(output)}")

print("\nRows by role and tile:")
print(
    output.groupby(["sample_role", "pilot_tile"])
    .size()
    .to_string()
)

print("\nAvailable rainfall feature counts:")
print(
    output.groupby("sample_role")[
        [
            "pre_event_24h_rainfall_mm",
            "pre_event_72h_rainfall_mm",
        ]
    ].count().to_string()
)

print("\nMissing feature counts:")
print(
    output[
        [
            "elevation_m",
            "slope_degrees",
            "aspect_degrees",
            "pre_event_24h_rainfall_mm",
            "pre_event_72h_rainfall_mm",
        ]
    ].isna().sum().to_string()
)

print("\nImportant interpretation notes:")
print("- Background controls are unverified, not confirmed negatives.")
print("- Event-day rainfall is excluded to prevent temporal leakage.")
print("- Missing rainfall values are not replaced with zero.")
print("- This dataset is experimental and not yet validated for prediction.")
