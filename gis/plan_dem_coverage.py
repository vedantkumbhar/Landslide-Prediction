from pathlib import Path
import math
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "india_landslide_events.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "dem" / "dem_tile_plan.csv"
)


def tile_name(latitude, longitude):
    lat_degree = math.floor(latitude)
    lon_degree = math.floor(longitude)

    ns = "N" if lat_degree >= 0 else "S"
    ew = "E" if lon_degree >= 0 else "W"

    return (
        f"{ns}{abs(lat_degree):02d}"
        f"{ew}{abs(lon_degree):03d}"
    )


def assign_region(row):
    lat = row["latitude"]
    lon = row["longitude"]

    # Approximate planning regions; not official boundaries.
    if 20 <= lat < 30 and 88 <= lon < 98:
        return "Northeast India"

    if 15 <= lat < 19 and 73 <= lon < 75:
        return "Western Maharashtra"

    if 27 <= lat < 36 and 72 <= lon < 88:
        return "Himalayas"

    return None


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    df["region"] = df.apply(assign_region, axis=1)
    selected = df[df["region"].notna()].copy()

    if selected.empty:
        raise ValueError("No events matched the planning regions.")

    selected["tile"] = selected.apply(
        lambda r: tile_name(r["latitude"], r["longitude"]),
        axis=1,
    )

    plan = (
        selected.groupby(["region", "tile"])
        .agg(
            event_count=("event_id", "size"),
            min_latitude=("latitude", "min"),
            max_latitude=("latitude", "max"),
            min_longitude=("longitude", "min"),
            max_longitude=("longitude", "max"),
        )
        .reset_index()
        .sort_values(["region", "tile"])
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUTPUT_PATH, index=False)

    print("=" * 55)
    print("DEM TILE COVERAGE PLAN")
    print("=" * 55)
    print(f"All valid India events: {len(df):,}")
    print(f"Events in selected planning regions: {len(selected):,}")
    print(f"Unique region/tile combinations: {len(plan):,}")
    print(f"Unique geographic tiles: {plan['tile'].nunique():,}")

    print("\nEvents by planning region:")
    print(selected["region"].value_counts().to_string())

    print("\nTiles by region:")
    print(plan.groupby("region")["tile"].nunique().to_string())

    print("\nFirst 20 planned tiles:")
    print(plan.head(20).to_string(index=False))

    print(f"\nSaved plan to: {OUTPUT_PATH}")
    print("No DEM files were downloaded or modified.")


if __name__ == "__main__":
    main()
