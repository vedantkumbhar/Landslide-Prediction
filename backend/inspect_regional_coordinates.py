from pathlib import Path
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    ROOT / "data" / "processed"
    / "regional_training_dataset_30oct.csv"
)

HDF5_FILE = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_test"
    / "3B-HHR-E.MS.MRG.3IMERG."
    "20261101-S103000-E105959.0630.V07C.HDF5"
)


def main():
    print("INSPECTING REGIONAL COORDINATES")
    print("=" * 55)

    df = pd.read_csv(CSV_FILE)

    print("\nCSV SHAPE:", df.shape)
    print("\nCSV COLUMNS:")
    for column in df.columns:
        print("-", column)

    print("\nFIRST 3 RECORDS:")
    print(df.head(3).to_string(index=False))

    print("\nREGION COUNTS:")
    print(df["dem_region"].value_counts())

    with h5py.File(HDF5_FILE, "r") as hdf:
        lat = hdf["Grid/lat"][:]
        lon = hdf["Grid/lon"][:]
        rainfall = hdf["Grid/precipitation"][0]

        print("\nGPM GRID INFORMATION:")
        print("Latitude range:", lat.min(), "to", lat.max())
        print("Longitude range:", lon.min(), "to", lon.max())
        print("Rainfall array shape:", rainfall.shape)

        print("Valid rainfall cells:",
              (rainfall != -9999.9).sum())


if __name__ == "__main__":
    main()