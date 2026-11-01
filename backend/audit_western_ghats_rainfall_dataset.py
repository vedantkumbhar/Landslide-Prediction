
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data/processed/rainfall/western_ghats/historical_validation"
)

INPUT_FILE = DATA_DIR / "paired_event_control_rainfall_analysis.csv"
OUTPUT_FILE = DATA_DIR / "rainfall_dataset_quality_audit.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Output already exists; refusing to overwrite: {OUTPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "event_id",
    "control_id",
    "background_sample_id",
    "terrain_source_tile",
    "event_datetime",
    "event_time_status",
    "control_status",
    "event_pre_event_24h_mm",
    "control_pre_event_24h_mm",
    "event_pre_event_72h_mm",
    "control_pre_event_72h_mm",
    "event_event_day_precipitation_mm",
    "control_event_day_precipitation_mm",
    "event_pre_event_72h_valid_hours",
    "control_pre_event_72h_valid_hours",
    "event_pre_event_24h_valid_hours",
    "control_pre_event_24h_valid_hours",
]

missing_columns = sorted(set(required_columns) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

issues = []


def add_issue(check, count, severity, details):
    issues.append({
        "check": check,
        "issue_count": int(count),
        "severity": severity,
        "details": details,
    })


print("=" * 78)
print("WESTERN GHATS HISTORICAL RAINFALL DATA QUALITY AUDIT")
print("=" * 78)

print(f"\nInput rows: {len(df)}")
print(f"Unique event IDs: {df['event_id'].nunique(dropna=True)}")
print(f"Unique control IDs: {df['control_id'].nunique(dropna=True)}")

# 1. Identifiers
for column in ["event_id", "control_id", "background_sample_id"]:
    missing_count = int(df[column].isna().sum())
    add_issue(
        f"{column}: missing values",
        missing_count,
        "HIGH" if missing_count else "PASS",
        "Identifiers should be present for reliable pair tracking.",
    )

add_issue(
    "Duplicate event-control pairs",
    df.duplicated(["event_id", "control_id"]).sum(),
    "HIGH" if df.duplicated(["event_id", "control_id"]).any() else "PASS",
    "Each event-control pair should appear once.",
)

add_issue(
    "Events paired with multiple controls",
    (df.groupby("event_id")["control_id"].nunique() > 1).sum(),
    "REVIEW",
    "Review whether the design intentionally allows multiple controls.",
)

add_issue(
    "Controls reused across events",
    (df.groupby("control_id")["event_id"].nunique() > 1).sum(),
    "HIGH",
    "Reused controls can invalidate independent-pair assumptions.",
)

# 2. Dates and time status
parsed_dates = pd.to_datetime(df["event_datetime"], errors="coerce")
invalid_dates = int(parsed_dates.isna().sum())

add_issue(
    "Invalid or missing event datetimes",
    invalid_dates,
    "HIGH" if invalid_dates else "PASS",
    "Datetime quality determines whether pre-event windows are trustworthy.",
)

midnight_mask = (
    parsed_dates.notna()
    & parsed_dates.dt.hour.eq(0)
    & parsed_dates.dt.minute.eq(0)
    & parsed_dates.dt.second.eq(0)
)

add_issue(
    "Midnight event timestamps",
    midnight_mask.sum(),
    "REVIEW",
    "Midnight may represent an unknown time rather than a confirmed event time.",
)

ambiguous_mask = (
    df["event_time_status"]
    .astype(str)
    .str.contains("ambiguous", case=False, na=False)
)

add_issue(
    "Events marked time-ambiguous",
    ambiguous_mask.sum(),
    "REVIEW",
    "Do not treat ambiguous event times as precise timestamps.",
)

# 3. Rainfall validity and missing values
rainfall_columns = [
    "event_pre_event_24h_mm",
    "control_pre_event_24h_mm",
    "event_pre_event_72h_mm",
    "control_pre_event_72h_mm",
    "event_event_day_precipitation_mm",
    "control_event_day_precipitation_mm",
]

for column in rainfall_columns:
    values = pd.to_numeric(df[column], errors="coerce")
    missing_count = int(values.isna().sum())
    negative_count = int((values.dropna() < 0).sum())

    add_issue(
        f"{column}: missing or nonnumeric",
        missing_count,
        "REVIEW" if missing_count else "PASS",
        "Check whether missingness follows event-time limitations.",
    )

    add_issue(
        f"{column}: negative rainfall",
        negative_count,
        "HIGH" if negative_count else "PASS",
        "Precipitation totals should not be negative.",
    )

for column in [
    "event_pre_event_24h_valid_hours",
    "control_pre_event_24h_valid_hours",
    "event_pre_event_72h_valid_hours",
    "control_pre_event_72h_valid_hours",
]:
    values = pd.to_numeric(df[column], errors="coerce")
    invalid_count = int(
        (values.isna() | (values < 0)).sum()
    )

    add_issue(
        f"{column}: missing or negative",
        invalid_count,
        "HIGH" if invalid_count else "PASS",
        "Validity-hour counts must be present and nonnegative.",
    )

# 4. Verify the pre-event features have complete windows.
# A 24-hour total requires all 24 hourly observations.
# A 72-hour total requires all 72 hourly observations.
for group in ["event", "control"]:
    for window in [24, 72]:
        rain_col = f"{group}_pre_event_{window}h_mm"
        hours_col = f"{group}_pre_event_{window}h_valid_hours"

        rain = pd.to_numeric(df[rain_col], errors="coerce")
        hours = pd.to_numeric(df[hours_col], errors="coerce")

        incomplete_with_value = (
            rain.notna() & hours.notna() & (hours < window)
        ).sum()

        add_issue(
            f"{group} {window}h totals with incomplete hourly coverage",
            incomplete_with_value,
            "HIGH" if incomplete_with_value else "PASS",
            "Do not assume a partial rainfall window is a complete total.",
        )

# 5. Check stored differences against component values.
difference_checks = [
    (
        "event_minus_control_pre_event_24h_mm",
        "event_pre_event_24h_mm",
        "control_pre_event_24h_mm",
    ),
    (
        "event_minus_control_pre_event_72h_mm",
        "event_pre_event_72h_mm",
        "control_pre_event_72h_mm",
    ),
    (
        "event_minus_control_event_day_precipitation_mm",
        "event_event_day_precipitation_mm",
        "control_event_day_precipitation_mm",
    ),
]

for diff_col, event_col, control_col in difference_checks:
    stored = pd.to_numeric(df[diff_col], errors="coerce")
    event_values = pd.to_numeric(df[event_col], errors="coerce")
    control_values = pd.to_numeric(df[control_col], errors="coerce")

    valid = stored.notna() & event_values.notna() & control_values.notna()
    mismatch = valid & (
        (stored - (event_values - control_values)).abs() > 0.011
    )

    add_issue(
        f"{diff_col}: arithmetic mismatch",
        mismatch.sum(),
        "HIGH" if mismatch.any() else "PASS",
        "Stored difference should equal event rainfall minus control rainfall.",
    )

# 6. Explicitly flag event-day precipitation as unsuitable for
# prediction before the event.
add_issue(
    "Event-day rainfall feature leakage risk",
    len(df),
    "REVIEW",
    "Event-day rainfall may include precipitation after the landslide. "
    "Exclude it from pre-event prediction features.",
)

audit = pd.DataFrame(issues)
audit.to_csv(OUTPUT_FILE, index=False)

print("\n--- Audit findings ---")
print(audit.to_string(index=False))

print("\n--- Event-time status counts ---")
print(df["event_time_status"].value_counts(dropna=False).to_string())

print("\n--- Control status counts ---")
print(df["control_status"].value_counts(dropna=False).to_string())

print(f"\nSaved audit: {OUTPUT_FILE}")
print("\nThis is a quality audit, not proof of predictive model performance.")
