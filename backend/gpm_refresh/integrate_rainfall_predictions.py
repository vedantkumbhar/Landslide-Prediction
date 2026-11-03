from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from backend.gpm_refresh.refresh_config import (
    PROJECT_ROOT,
    get_run_directory,
)


BASE_INPUT = (
    PROJECT_ROOT
    / "data/processed/rainfall/refresh_runs/20261102_152742/"
      "predictions/regional_gpm_risk_predictions_02nov2026_with_3h.csv"
)


def read_csv(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    df = pd.read_csv(path)

    required = {"record_id", "latitude", "longitude"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")

    if df["record_id"].isna().any():
        raise ValueError(f"{name} contains missing record IDs.")

    if df["record_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate record IDs.")

    return df


def validate_source(base, source, name):
    if len(base) != len(source):
        raise ValueError(
            f"{name}: record count mismatch. "
            f"Base={len(base)}, source={len(source)}"
        )

    if set(base["record_id"]) != set(source["record_id"]):
        raise ValueError(f"{name}: record IDs do not match.")

    source = (
        source.set_index("record_id")
        .loc[base["record_id"]]
        .reset_index()
    )

    for column in ["latitude", "longitude"]:
        left = pd.to_numeric(base[column], errors="coerce").to_numpy()
        right = pd.to_numeric(source[column], errors="coerce").to_numpy()

        if (
            not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or not np.allclose(left, right, atol=1e-6, rtol=0)
        ):
            raise ValueError(
                f"{name}: {column} values do not match the base."
            )

    return source


def replace_columns(base, source, columns, name):
    missing = [column for column in columns if column not in source.columns]

    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")

    for column in columns:
        base[column] = source[column].to_numpy()


def validate_rainfall(df):
    rainfall_columns = [
        "gpm_latest_rate_mm_hr",
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_24h_accumulation_mm",
    ]

    for column in rainfall_columns:
        values = pd.to_numeric(df[column], errors="coerce")

        if (
            values.isna().any()
            or not np.isfinite(values.to_numpy()).all()
            or (values < 0).any()
        ):
            raise ValueError(
                f"Invalid rainfall values in {column}."
            )

        df[column] = values

    nested = [
        "gpm_1h_accumulation_mm",
        "gpm_3h_accumulation_mm",
        "gpm_6h_accumulation_mm",
        "gpm_12h_accumulation_mm",
        "gpm_24h_accumulation_mm",
    ]

    for previous, current in zip(nested, nested[1:]):
        invalid = df[current] + 1e-6 < df[previous]

        if invalid.any():
            raise ValueError(
                f"Accumulation consistency failed: "
                f"{current} is lower than {previous} "
                f"for {int(invalid.sum())} records."
            )


def validate_window_metadata(df):
    windows = [
        ("gpm_3h_series_start_utc", "gpm_3h_series_end_utc", 3),
        ("gpm_6h_series_start_utc", "gpm_6h_series_end_utc", 6),
        ("gpm_12h_series_start_utc", "gpm_12h_series_end_utc", 12),
        ("gpm_24h_series_start_utc", "gpm_24h_series_end_utc", 24),
    ]

    endpoints = []

    for start_col, end_col, hours in windows:
        for column in [start_col, end_col]:
            if column not in df.columns:
                raise ValueError(f"Missing metadata column: {column}")

        starts = pd.to_datetime(df[start_col], utc=True, errors="coerce")
        ends = pd.to_datetime(df[end_col], utc=True, errors="coerce")

        if starts.isna().any() or ends.isna().any():
            raise ValueError(f"Invalid timestamps in {start_col}/{end_col}.")

        if starts.nunique() != 1 or ends.nunique() != 1:
            raise ValueError(
                f"Inconsistent timestamps in {start_col}/{end_col}."
            )

        if ends.iloc[0] - starts.iloc[0] != pd.Timedelta(hours=hours):
            raise ValueError(f"Invalid {hours}-hour observation duration.")

        endpoints.append(ends.iloc[0])

    if len(set(endpoints)) != 1:
        raise ValueError(
            "Rainfall windows do not share the same UTC endpoint."
        )


def main(args):
    print("\nGPM MULTI-WINDOW PREDICTION INTEGRATION")
    print("=" * 65)

    base = read_csv(args.base_csv, "Base prediction dataset")

    print("Base records:", len(base))

    if "terrain_model_score" not in base.columns:
        raise ValueError(
            "Base prediction dataset is missing terrain_model_score."
        )

    terrain_scores = pd.to_numeric(
        base["terrain_model_score"],
        errors="coerce",
    )

    if (
        terrain_scores.isna().any()
        or not np.isfinite(terrain_scores.to_numpy()).all()
    ):
        raise ValueError("Invalid terrain model scores.")

    sources = [
        (
            "3-hour extraction",
            args.input_3h,
            [
                "gpm_latest_rate_mm_hr",
                "gpm_1h_accumulation_mm",
                "gpm_3h_accumulation_mm",
                "gpm_3h_valid_30min_intervals",
                "gpm_3h_latest_quality_index",
                "gpm_3h_latest_random_error_mm_hr",
                "gpm_3h_grid_match_distance_km",
                "gpm_3h_series_start_utc",
                "gpm_3h_series_end_utc",
            ],
        ),
        (
            "6-hour extraction",
            args.input_6h,
            [
                "gpm_6h_accumulation_mm",
                "gpm_6h_valid_30min_intervals",
                "gpm_6h_latest_quality_index",
                "gpm_6h_latest_random_error_mm_hr",
                "gpm_6h_grid_match_distance_km",
                "gpm_6h_series_start_utc",
                "gpm_6h_series_end_utc",
            ],
        ),
        (
            "12-hour extraction",
            args.input_12h,
            [
                "gpm_12h_accumulation_mm",
                "gpm_12h_valid_30min_intervals",
                "gpm_12h_series_start_utc",
                "gpm_12h_series_end_utc",
            ],
        ),
        (
            "24-hour combination",
            args.input_24h,
            [
                "gpm_24h_accumulation_mm",
                "gpm_24h_valid_30min_intervals",
                "gpm_24h_series_start_utc",
                "gpm_24h_series_end_utc",
            ],
        ),
    ]

    for name, path, columns in sources:
        source = read_csv(path, name)
        source = validate_source(base, source, name)
        replace_columns(base, source, columns, name)

        print(f"{name}: validated and integrated.")

    validate_rainfall(base)
    validate_window_metadata(base)

    output = args.output_csv

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output, index=False)

    print("\n========== INTEGRATION SUMMARY ==========")
    print("Records:", len(base))
    print("Rainfall windows: 1h, 3h, 6h, 12h, 24h")
    print("Terrain scores preserved:", True)
    print("Terrain score column: terrain_model_score")
    print(
        "Common UTC endpoint:",
        base["gpm_3h_series_end_utc"].iloc[0],
    )
    print("Output:", output)
    print("\nINTEGRATION COMPLETED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)

    parser.add_argument(
        "--base-csv",
        type=Path,
        default=BASE_INPUT,
    )

    parser.add_argument("--input-3h", type=Path, required=True)
    parser.add_argument("--input-6h", type=Path, required=True)
    parser.add_argument("--input-12h", type=Path, required=True)
    parser.add_argument("--input-24h", type=Path, required=True)

    parser.add_argument("--output-csv", type=Path)

    args = parser.parse_args()

    run_dir = get_run_directory(args.run_id)

    if args.output_csv is None:
        args.output_csv = (
            run_dir
            / "predictions"
            / "regional_gpm_risk_predictions_integrated.csv"
        )

    main(args)