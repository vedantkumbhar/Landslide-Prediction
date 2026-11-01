
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "initial_training_dataset.csv"

df = pd.read_csv(DATA_PATH)

regions = ["Himalayas", "Northeast India", "Western Maharashtra"]

print("\nREGIONAL SPATIAL VALIDATION FEASIBILITY")
print("=" * 75)

for region in regions:
    regional = df[df["dem_region"] == region].copy()

    print(f"\nREGION: {region}")
    print("-" * 75)

    if regional.empty:
        print("No records found.")
        continue

    print("Total records:", len(regional))
    print("Positive events:", int((regional["label"] == 1).sum()))
    print("Unverified backgrounds:", int((regional["label"] == 0).sum()))

    tile_table = pd.crosstab(
        regional["tile"],
        regional["label"]
    ).reindex(columns=[0, 1], fill_value=0)

    tile_table.columns = [
        "background_unverified",
        "historical_events"
    ]

    tile_table["total"] = tile_table.sum(axis=1)

    print("\nRecords by tile:")
    print(tile_table.to_string())

    eligible = tile_table[
        (tile_table["background_unverified"] > 0)
        & (tile_table["historical_events"] > 0)
    ]

    print("\nTiles containing both classes:", len(eligible))

    if len(eligible) >= 2:
        print("Potential for leave-one-tile-out validation: YES")
    else:
        print("Potential for leave-one-tile-out validation: LIMITED")

    print(
        "Note: Both classes must also remain in the training "
        "data after each tile is held out."
    )

print("\n" + "=" * 75)
print("INSPECTION COMPLETE - NO FILES MODIFIED")
