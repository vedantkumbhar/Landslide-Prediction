from pathlib import Path
import pandas as pd

FILES = [
    Path("data/processed/pilot_landslide_event_features.csv"),
    Path("data/processed/india_landslide_events.csv"),
    Path("data/raw/global_landslide_catalog.csv"),
]

print("\nNON-MIDNIGHT TIMESTAMP SOURCE INSPECTION")
print("=" * 75)

for file_path in FILES:
    print(f"\nFILE: {file_path}")
    print("-" * 75)

    if not file_path.exists():
        print("File not found. Skipping.")
        continue

    df = pd.read_csv(file_path, low_memory=False)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    relevant_columns = [
        col for col in df.columns
        if any(term in col.lower()
               for term in ["date", "time", "event_id", "source", "id"])
    ]

    print("\nRelevant columns:")
    print(relevant_columns)

    date_columns = [
        col for col in df.columns
        if "date" in col.lower() or "time" in col.lower()
    ]

    for col in date_columns:
        parsed = pd.to_datetime(df[col], errors="coerce")

        non_midnight = (
            parsed.notna()
            & (
                parsed.dt.hour.ne(0)
                | parsed.dt.minute.ne(0)
                | parsed.dt.second.ne(0)
            )
        )

        print(f"\nColumn: {col}")
        print("Parseable:", parsed.notna().sum())
        print("Non-midnight:", non_midnight.sum())

        if non_midnight.any():
            display_columns = relevant_columns[:]
            if col not in display_columns:
                display_columns.append(col)

            print("\nNon-midnight examples:")
            print(
                df.loc[non_midnight, display_columns]
                .head(10)
                .to_string(index=False)
            )

print("\nINSPECTION COMPLETE")
print("Read-only. No files modified.")