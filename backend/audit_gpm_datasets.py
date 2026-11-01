
import pandas as pd
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "rainfall"
)

FILES = {
    "3h": (
        "gpm_regional_timeseries_3h_01nov2026.csv",
        "gpm_3h_accumulation_mm"
    ),
    "6h": (
        "gpm_regional_timeseries_6h_01nov2026.csv",
        "gpm_6h_accumulation_mm"
    ),
    "12h": (
        "gpm_regional_timeseries_12h_01nov2026.csv",
        "gpm_12h_accumulation_mm"
    ),
    "24h": (
        "gpm_regional_timeseries_24h_01nov2026.csv",
        "gpm_24h_accumulation_mm"
    )
}


def main():

    print("\nGPM DATASET AUDIT")
    print("=" * 65)

    for duration, (filename, rainfall_column) in FILES.items():

        filepath = DATA_DIR / filename

        print(f"\n{duration.upper()} DATASET")
        print("-" * 65)

        if not filepath.exists():
            print("FILE MISSING:", filepath)
            continue

        df = pd.read_csv(filepath)

        print("Filename:", filename)
        print("Rows:", len(df))
        print("Rainfall column exists:",
              rainfall_column in df.columns)

        if rainfall_column in df.columns:

            rainfall = pd.to_numeric(
                df[rainfall_column],
                errors="coerce"
            )

            print("Valid rainfall values:", rainfall.notna().sum())
            print("Missing rainfall values:", rainfall.isna().sum())
            print("Mean rainfall (mm):", round(rainfall.mean(), 4))
            print("Maximum rainfall (mm):", round(rainfall.max(), 4))

        if "dem_region" in df.columns:

            print("\nRegional sample counts:")
            print(df["dem_region"].value_counts().to_string())

        else:
            print("WARNING: dem_region column missing")

        if {"latitude", "longitude"}.issubset(df.columns):

            duplicates = df.duplicated(
                subset=["latitude", "longitude"]
            ).sum()

            print("\nDuplicate coordinates:", duplicates)

        else:
            print("WARNING: Coordinate columns missing")

    print("\n" + "=" * 65)
    print("GPM DATASET AUDIT FINISHED")


if __name__ == "__main__":
    main()
