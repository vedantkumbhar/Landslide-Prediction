from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_experimental_risk_scores.csv"
)

df = pd.read_csv(INPUT)

df["timestamp_local"] = pd.to_datetime(
    df["timestamp_local"], errors="coerce"
)

df["retrieved_at_local"] = pd.to_datetime(
    df["retrieved_at_local"], errors="coerce"
)

print("\n========== DATA FRESHNESS AUDIT ==========")

print("\nTimestamp range:")
print("Earliest:", df["timestamp_local"].min())
print("Latest:", df["timestamp_local"].max())

print("\nRetrieval timestamps:")
print("Earliest:", df["retrieved_at_local"].min())
print("Latest:", df["retrieved_at_local"].max())

print("\nBy record type:")
for record_type, group in df.groupby("record_type"):
    print(f"\n{record_type}")
    print("Rows:", len(group))
    print("Timestamp range:",
          group["timestamp_local"].min(),
          "to",
          group["timestamp_local"].max())

print("\nBy temporal role:")
print(
    df.groupby("temporal_role")["timestamp_local"]
    .agg(["min", "max", "count"])
)

print("\n========== AUDIT COMPLETE ==========")