
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TERRAIN_FILE = (
    ROOT / "data" / "processed"
    / "regional_training_dataset_30oct.csv"
)

RAINFALL_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "regional_rainfall_features_30oct_v3.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rainfall"
    / "regional_terrain_rainfall_integrated_30oct.csv"
)


def main():

    print("\nREGIONAL TERRAIN + RAINFALL INTEGRATION")
    print("=" * 55)

    if not TERRAIN_FILE.exists():
        raise FileNotFoundError(TERRAIN_FILE)

    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(RAINFALL_FILE)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {OUTPUT_FILE}"
        )

    terrain = pd.read_csv(TERRAIN_FILE)
    rainfall = pd.read_csv(RAINFALL_FILE)

    print("\nInput datasets:")
    print("Terrain records:", len(terrain))
    print("Rainfall records:", len(rainfall))

    terrain_required = {
        "latitude", "longitude", "elevation_m",
        "slope_degrees", "aspect_degrees",
        "dem_region", "tile", "label"
    }

    rainfall_required = {
        "region", "tile", "timestamp_local",
        "record_type", "precipitation_mm",
        "recent_rainfall_1h_mm",
        "recent_rainfall_3h_mm",
        "recent_rainfall_6h_mm",
        "recent_rainfall_12h_mm",
        "recent_rainfall_24h_mm",
        "forecast_rainfall_3h_mm",
        "forecast_rainfall_6h_mm",
        "forecast_rainfall_24h_mm"
    }

    if not terrain_required.issubset(terrain.columns):
        raise ValueError(
            f"Missing terrain columns: "
            f"{terrain_required - set(terrain.columns)}"
        )

    if not rainfall_required.issubset(rainfall.columns):
        raise ValueError(
            f"Missing rainfall columns: "
            f"{rainfall_required - set(rainfall.columns)}"
        )

    terrain["tile"] = terrain["tile"].astype(str)
    rainfall["tile"] = rainfall["tile"].astype(str)

    terrain["dem_region"] = terrain["dem_region"].astype(str)
    rainfall["region"] = rainfall["region"].astype(str)

    rainfall["timestamp_local"] = pd.to_datetime(
        rainfall["timestamp_local"],
        errors="coerce"
    )

    if rainfall["timestamp_local"].isna().any():
        raise ValueError("Invalid rainfall timestamps found.")

    # Verify that each tile belongs to the same region
    # in both datasets.
    terrain_mapping = terrain[
        ["tile", "dem_region"]
    ].drop_duplicates()

    rainfall_mapping = rainfall[
        ["tile", "region"]
    ].drop_duplicates()

    mapping = terrain_mapping.merge(
        rainfall_mapping,
        on="tile",
        how="outer",
        indicator=True
    )

    if not (mapping["_merge"] == "both").all():
        raise ValueError(
            "Terrain and rainfall tile coverage does not match."
        )

    mismatches = mapping[
        mapping["dem_region"] != mapping["region"]
    ]

    if not mismatches.empty:
        raise ValueError(
            f"Region mismatches found:\n{mismatches}"
        )

    # Each tile must have only one rainfall record
    # for each timestamp.
    duplicates = rainfall.duplicated(
        ["region", "tile", "timestamp_local"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate rainfall timestamps."
        )

    # Avoid duplicate region columns after merging.
    rainfall = rainfall.drop(columns=["region"])

    print("\nJoining terrain samples with rainfall timelines...")

    # Many terrain samples belong to each DEM tile.
    # Each sample receives that tile's hourly rainfall timeline.
    integrated = terrain.merge(
        rainfall,
        on="tile",
        how="inner",
        validate="many_to_many"
    )

    # Verify that all terrain records were retained.
    if integrated["event_id"].nunique() != terrain["event_id"].nunique():
        raise ValueError(
            "Some terrain event IDs were lost during integration."
        )

    expected_rows = sum(
        len(group) * rainfall[
            rainfall["tile"] == tile
        ].shape[0]
        for tile, group in terrain.groupby("tile")
    )

    if len(integrated) != expected_rows:
        raise ValueError(
            f"Unexpected output row count: "
            f"{len(integrated)}; expected {expected_rows}"
        )

    integrated = integrated.sort_values(
        ["dem_region", "tile", "timestamp_local", "event_id"],
        na_position="last"
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    integrated.to_csv(
        temporary_file,
        index=False
    )

    temporary_file.replace(OUTPUT_FILE)

    print("\nINTEGRATION SUCCESSFUL")
    print("=" * 55)

    print("Output:", OUTPUT_FILE)
    print("Integrated records:", len(integrated))
    print("Unique terrain samples:", integrated["event_id"].nunique())
    print("Unique DEM tiles:", integrated["tile"].nunique())

    print("\nRecords by region:")
    print(integrated.groupby("dem_region").size())

    print("\nRecords by rainfall type:")
    print(integrated.groupby("record_type").size())

    print("\nRainfall feature availability:")

    features = [
        "recent_rainfall_1h_mm",
        "recent_rainfall_3h_mm",
        "recent_rainfall_6h_mm",
        "recent_rainfall_12h_mm",
        "recent_rainfall_24h_mm",
        "forecast_rainfall_3h_mm",
        "forecast_rainfall_6h_mm",
        "forecast_rainfall_24h_mm"
    ]

    print(integrated[features].count())

    print("\nSample integrated records:")
    print(
        integrated[
            [
                "dem_region",
                "tile",
                "latitude",
                "longitude",
                "elevation_m",
                "slope_degrees",
                "timestamp_local",
                "record_type",
                "precipitation_mm",
                "recent_rainfall_3h_mm",
                "forecast_rainfall_3h_mm"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
