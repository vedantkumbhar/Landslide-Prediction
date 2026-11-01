
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation"
)

INPUT = BASE / "pilot_historical_rainfall_summary.csv"
TILE_OUTPUT = BASE / "historical_rainfall_tile_analysis.csv"
STATUS_OUTPUT = BASE / "historical_rainfall_time_status_analysis.csv"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    # Never overwrite previous analysis.
    for path in (TILE_OUTPUT, STATUS_OUTPUT):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    df = pd.read_csv(INPUT)

    required = [
        "event_id",
        "terrain_source_tile",
        "event_time_status",
        "pre_event_24h_mm",
        "pre_event_72h_mm",
        "event_day_precipitation_mm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_cols = [
        "pre_event_24h_mm",
        "pre_event_72h_mm",
        "event_day_precipitation_mm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("=" * 65)
    print("WESTERN GHATS HISTORICAL RAINFALL ANALYSIS")
    print("=" * 65)

    print("\nEvents by pilot tile:")
    print(df["terrain_source_tile"].value_counts().to_string())

    print("\n--- Overall rainfall statistics (mm) ---")
    for col in numeric_cols:
        print(f"\n{col}")
        values = df[col].dropna()
        if values.empty:
            print("No complete values")
        else:
            print(values.describe().round(2).to_string())

    # Descriptive summaries by tile. No predictive claims.
    tile_rows = []
    for tile, group in df.groupby("terrain_source_tile"):
        for col in numeric_cols:
            values = group[col].dropna()
            tile_rows.append({
                "terrain_source_tile": tile,
                "rainfall_metric": col,
                "events_total": len(group),
                "events_with_value": len(values),
                "mean_mm": round(values.mean(), 3)
                    if len(values) else None,
                "median_mm": round(values.median(), 3)
                    if len(values) else None,
                "minimum_mm": round(values.min(), 3)
                    if len(values) else None,
                "maximum_mm": round(values.max(), 3)
                    if len(values) else None,
            })

    tile_df = pd.DataFrame(tile_rows)
    tile_df.to_csv(TILE_OUTPUT, index=False)

    # Check whether timestamp status affects metric availability.
    status_rows = []
    for status, group in df.groupby(
        "event_time_status", dropna=False
    ):
        for col in numeric_cols:
            values = group[col].dropna()
            status_rows.append({
                "event_time_status": status,
                "rainfall_metric": col,
                "events_total": len(group),
                "events_with_value": len(values),
                "mean_mm": round(values.mean(), 3)
                    if len(values) else None,
                "median_mm": round(values.median(), 3)
                    if len(values) else None,
            })

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(STATUS_OUTPUT, index=False)

    print("\n--- By pilot tile ---")
    print(tile_df.to_string(index=False))

    print("\n--- By event-time status ---")
    print(status_df.to_string(index=False))

    print("\nSaved new analysis files:")
    print(TILE_OUTPUT)
    print(STATUS_OUTPUT)

    print(
        "\nInterpretation limits: event-only descriptive analysis; "
        "no non-landslide control comparison. Whole event-day rainfall "
        "may include precipitation after the event. Do not use it as "
        "a pre-event predictor."
    )

if __name__ == "__main__":
    main()
