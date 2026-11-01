
from pathlib import Path
import h5py

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "data" / "raw" / "satellite"
    / "gpm_imerge_test"
    / "3B-HHR-E.MS.MRG.3IMERG."
      "20261101-S103000-E105959.0630.V07C.HDF5"
)


def inspect_group(name, obj):

    if isinstance(obj, h5py.Dataset):
        print(f"\nDATASET: {name}")
        print("Shape:", obj.shape)
        print("Data type:", obj.dtype)

        for key, value in obj.attrs.items():
            print(f"Attribute - {key}: {value}")


def main():

    if not FILE.exists():
        raise FileNotFoundError(FILE)

    print("INSPECTING NASA GPM IMERG HDF5")
    print("=" * 50)

    with h5py.File(FILE, "r") as hdf:

        print("\nROOT GROUPS:")
        for key in hdf.keys():
            print("-", key)

        print("\nDATASETS AND METADATA:")
        hdf.visititems(inspect_group)

        print("\nGLOBAL ATTRIBUTES:")
        for key, value in hdf.attrs.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
