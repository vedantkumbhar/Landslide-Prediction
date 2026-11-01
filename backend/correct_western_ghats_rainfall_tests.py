
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data/processed/rainfall/western_ghats/historical_validation"
    / "rainfall_pair_statistical_tests.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data/processed/rainfall/western_ghats/historical_validation"
    / "rainfall_tile_tests_multiple_comparison_corrected.csv"
)


def holm_correction(p_values):
    """Return Holm-adjusted p-values in their original order."""
    values = [float(p) for p in p_values]

    if any(not 0 <= p <= 1 for p in values):
        raise ValueError("All p-values must be between 0 and 1.")

    count = len(values)
    order = sorted(range(count), key=lambda i: values[i])

    adjusted = [0.0] * count
    running_max = 0.0

    for rank, original_index in enumerate(order):
        candidate = (count - rank) * values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)

    return adjusted


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Output already exists; refusing to overwrite: {OUTPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required = {
    "group",
    "rainfall_metric",
    "complete_pairs",
    "mean_difference_mm",
    "wilcoxon_p_value",
    "sign_test_p_value",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Select the three pilot tiles only.
tile_mask = df["group"].astype(str).str.fullmatch(r"N\d{2}E\d{3}")
tile_tests = df.loc[tile_mask].copy()

if len(tile_tests) != 9:
    raise ValueError(
        f"Expected 9 tile-level comparisons, found {len(tile_tests)}. "
        "Check the input file before proceeding."
    )

for column in ["wilcoxon_p_value", "sign_test_p_value"]:
    p_values = pd.to_numeric(tile_tests[column], errors="coerce")

    if p_values.isna().any() or not p_values.between(0, 1).all():
        raise ValueError(f"Invalid p-values in {column}")

    tile_tests[f"{column}_holm"] = holm_correction(
        p_values.tolist()
    )

tile_tests["wilcoxon_significant_holm_05"] = (
    tile_tests["wilcoxon_p_value_holm"] < 0.05
)

tile_tests["sign_test_significant_holm_05"] = (
    tile_tests["sign_test_p_value_holm"] < 0.05
)

tile_tests.to_csv(OUTPUT_FILE, index=False)

columns = [
    "group",
    "rainfall_metric",
    "complete_pairs",
    "mean_difference_mm",
    "wilcoxon_p_value",
    "wilcoxon_p_value_holm",
    "wilcoxon_significant_holm_05",
    "sign_test_p_value",
    "sign_test_p_value_holm",
    "sign_test_significant_holm_05",
]

print("=" * 75)
print("WESTERN GHATS RAINFALL: HOLM-CORRECTED TILE TESTS")
print("=" * 75)
print(
    tile_tests[columns]
    .sort_values("wilcoxon_p_value_holm")
    .to_string(index=False)
)

print(f"\nSaved: {OUTPUT_FILE}")
print("\nInterpretation cautions:")
print("- These are exploratory comparisons, not proof of causality.")
print("- Holm correction addresses multiple testing only.")
print("- It does not correct uncertain control labels or event timing.")
print("- Event-day rainfall may include precipitation after a landslide.")
