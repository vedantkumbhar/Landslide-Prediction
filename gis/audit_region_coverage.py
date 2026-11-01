
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "india_landslide_events.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regional_event_coverage.csv"
)


REGIONS = {
    "Himalayas": {
        "lat_min": 27,
        "lat_max": 36,
        "lon_min": 72,
        "lon_max": 88,
    },
    "Northeast India": {
        "lat_min": 20,
        "lat_max": 30,
        "lon_min": 88,
        "lon_max": 98,
    },
    "Western Maharashtra": {
        "lat_min": 15,
        "lat_max": 19,
        "lon_min": 73,
        "lon_max": 75,
    },
}


def main():
    df = pd.read_csv(EVENTS_PATH)

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    assigned = pd.Series(
        "Outside planned regions",
        index=df.index,
        dtype="object",
    )

    print("=" * 65)
    print("REGIONAL HISTORICAL EVENT COVERAGE AUDIT")
    print("=" * 65)

    results = []

    for region_name, bounds in REGIONS.items():
        inside = (
            df["latitude"].between(
                bounds["lat_min"], bounds["lat_max"],
                inclusive="left",
            )
            & df["longitude"].between(
                bounds["lon_min"], bounds["lon_max"],
                inclusive="left",
            )
        )

        # Report overlaps instead of silently reassigning events.
        already_assigned = assigned != "Outside planned regions"
        overlap_count = int((inside & already_assigned).sum())

        newly_assigned = inside & ~already_assigned
        assigned.loc[newly_assigned] = region_name

        results.append({
            "region": region_name,
            "events_in_bounds": int(inside.sum()),
            "overlap_with_prior_regions": overlap_count,
            "newly_assigned_events": int(newly_assigned.sum()),
        })

    for item in results:
        print(f"\n{item['region']}")
        print(f"  Events inside bounds: {item['events_in_bounds']}")
        print(f"  Overlap with prior regions: {item['overlap_with_prior_regions']}")
        print(f"  Newly assigned events: {item['newly_assigned_events']}")

    outside_count = int(
        (assigned == "Outside planned regions").sum()
    )

    print(f"\nTotal valid India events: {len(df)}")
    print(f"Events outside planned regions: {outside_count}")

    df["planned_region"] = assigned
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved regional audit: {OUTPUT_PATH}")
    print("The original inventory was not modified.")


if __name__ == "__main__":
    main()
