
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Pilot events": ROOT / "data/processed/pilot_landslide_event_features.csv",
    "India inventory": ROOT / "data/processed/india_landslide_events.csv",
}

print("\nPILOT EVENT TIMESTAMP PRECISION AUDIT")
print("=" * 65)

for name, path in FILES.items():

    if not path.exists():
        print(f"\n{name}: FILE NOT FOUND")
        continue

    df = pd.read_csv(path, low_memory=False)

    print(f"\n{name}")
    print("-" * 65)
    print("Total records:", len(df))

    if "event_date_parsed" not in df.columns:
        print("event_date_parsed not available")
        continue

    dates = pd.to_datetime(
        df["event_date_parsed"],
        errors="coerce"
    )

    valid = dates.notna()

    # Midnight timestamps may represent date-only records.
    # They are classified as ambiguous, not assumed to be exact.
    midnight = (
        valid
        & (dates.dt.hour == 0)
        & (dates.dt.minute == 0)
        & (dates.dt.second == 0)
    )

    non_midnight = valid & ~midnight

    print("Parseable timestamps:", int(valid.sum()))
    print("Unparseable timestamps:", int((~valid).sum()))
    print("Midnight / date-only candidates:", int(midnight.sum()))
    print("Non-midnight timestamps:", int(non_midnight.sum()))

    if non_midnight.any():
        print("\nNon-midnight examples:")
        print(
            dates[non_midnight]
            .head(10)
            .to_string(index=False)
        )

    if "event_time" in df.columns:
        event_time = df["event_time"].dropna()

        print("\nSeparate event_time field:")
        print("Non-null:", len(event_time))
        print("Unique values:", event_time.nunique())

    if "event_id" in df.columns:
        print("\nUnique event IDs:", df["event_id"].nunique())

print("\nAUDIT COMPLETE")
print("Read-only. No files modified.")
