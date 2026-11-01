
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "data" / "processed" / "rainfall" / "western_ghats"
    / "historical_validation"
)

INPUT_FILE = BASE / "paired_event_control_rainfall_analysis.csv"
OUTPUT_FILE = BASE / "rainfall_pair_statistical_tests.csv"

SEED = 42
BOOTSTRAP_REPLICATES = 10000

METRICS = {
    "pre_event_24h_mm": "Preceding 24-hour rainfall",
    "pre_event_72h_mm": "Preceding 72-hour rainfall",
    "event_day_precipitation_mm": "Event calendar-day rainfall",
}


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return None, None

    sampled_indices = rng.integers(
        0, len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    means = values[sampled_indices].mean(axis=1)

    lower, upper = np.percentile(means, [2.5, 97.5])
    return round(float(lower), 3), round(float(upper), 3)


def analyse_group(group, metric, label, rng):
    event_col = f"event_{metric}"
    control_col = f"control_{metric}"
    diff_col = f"event_minus_control_{metric}"

    valid = group[[event_col, control_col, diff_col]].dropna()
    differences = valid[diff_col].to_numpy(dtype=float)
    n = len(differences)

    result = {
        "group": label,
        "rainfall_metric": metric,
        "matched_pairs_available": len(group),
        "complete_pairs": n,
        "event_mean_mm": None,
        "control_mean_mm": None,
        "mean_difference_mm": None,
        "median_difference_mm": None,
        "bootstrap_95ci_lower_mm": None,
        "bootstrap_95ci_upper_mm": None,
        "wilcoxon_p_value": None,
        "sign_test_p_value": None,
        "pairs_event_higher": 0,
        "pairs_equal": 0,
        "pairs_control_higher": 0,
    }

    if n == 0:
        return result

    result["event_mean_mm"] = round(
        float(valid[event_col].mean()), 3
    )
    result["control_mean_mm"] = round(
        float(valid[control_col].mean()), 3
    )
    result["mean_difference_mm"] = round(
        float(differences.mean()), 3
    )
    result["median_difference_mm"] = round(
        float(np.median(differences)), 3
    )

    lower, upper = bootstrap_mean_ci(differences, rng)
    result["bootstrap_95ci_lower_mm"] = lower
    result["bootstrap_95ci_upper_mm"] = upper

    positive = int((differences > 0).sum())
    negative = int((differences < 0).sum())
    equal = int((differences == 0).sum())

    result["pairs_event_higher"] = positive
    result["pairs_equal"] = equal
    result["pairs_control_higher"] = negative

    nonzero = differences[differences != 0]

    if len(nonzero) > 0:
        try:
            test = wilcoxon(
                nonzero,
                alternative="two-sided",
                method="auto",
            )
            result["wilcoxon_p_value"] = round(
                float(test.pvalue), 6
            )
        except ValueError:
            result["wilcoxon_p_value"] = None

        sign_result = binomtest(
            positive,
            n=positive + negative,
            p=0.5,
            alternative="two-sided",
        )
        result["sign_test_p_value"] = round(
            float(sign_result.pvalue), 6
        )

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "event_id",
        "terrain_source_tile",
        "event_time_status",
    ]

    for metric in METRICS:
        required.extend([
            f"event_{metric}",
            f"control_{metric}",
            f"event_minus_control_{metric}",
        ])

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs found.")

    rng = np.random.default_rng(SEED)
    results = []

    for metric in METRICS:
        results.append(
            analyse_group(df, metric, "ALL_TILES", rng)
        )

        for tile, group in df.groupby("terrain_source_tile"):
            results.append(
                analyse_group(group, metric, tile, rng)
            )

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 75)
    print("PAIRED RAINFALL STATISTICAL EVALUATION")
    print("=" * 75)

    overall = result_df[result_df["group"] == "ALL_TILES"]

    display_columns = [
        "rainfall_metric",
        "complete_pairs",
        "event_mean_mm",
        "control_mean_mm",
        "mean_difference_mm",
        "median_difference_mm",
        "bootstrap_95ci_lower_mm",
        "bootstrap_95ci_upper_mm",
        "wilcoxon_p_value",
        "sign_test_p_value",
        "pairs_event_higher",
        "pairs_control_higher",
    ]

    print("\n--- Overall paired comparisons ---")
    print(
        overall[display_columns]
        .to_string(index=False)
    )

    print("\n--- Results by tile ---")
    print(
        result_df[result_df["group"] != "ALL_TILES"][
            ["group"] + display_columns
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_FILE)

    print(
        "\nInterpretation cautions:\n"
        "- These are exploratory tests, not proof of causality.\n"
        "- The 24-hour result uses only events with recorded non-midnight times.\n"
        "- The event-day metric may include rain after the event.\n"
        "- Background locations are unverified and may have experienced "
        "unrecorded landslides.\n"
        "- Confidence intervals and p-values do not correct for these "
        "sampling limitations or establish model performance."
    )


if __name__ == "__main__":
    main()
