from pathlib import Path
import pandas as pd

INDIA_FILE = Path("data/processed/india_landslide_events.csv")
PILOT_FILE = Path("data/processed/pilot_landslide_event_features.csv")

print("\nHISTORICAL RAINFALL ELIGIBILITY AUDIT")
print("=" * 65)

india = pd.read_csv(INDIA_FILE, low_memory=False)
pilot = pd.read_csv(PILOT_FILE, low_memory=False)

# Identify coordinate columns
lat_col = next(
    (c for c in ["latitude", "lat", "Latitude"] if c in india.columns),
    None
)
lon_col = next(
    (c for c in ["longitude", "lon", "long", "Longitude"] if c in india.columns),
    None
)

if not lat_col or not lon_col:
    raise ValueError("Latitude or longitude column not found.")

# Parse dates and coordinates
india["audit_date"] = pd.to_datetime(
    india["event_date_parsed"] if "event_date_parsed" in india.columns
    else india["event_date"],
    errors="coerce"
)

india["audit_lat"] = pd.to_numeric(india[lat_col], errors="coerce")
india["audit_lon"] = pd.to_numeric(india[lon_col], errors="coerce")

valid_date = india["audit_date"].notna()

valid_coordinates = (
    india["audit_lat"].between(-90, 90)
    & india["audit_lon"].between(-180, 180)
)

valid_location = (
    valid_coordinates
    & india["audit_lat"].notna()
    & india["audit_lon"].notna()
)

eligible = valid_date & valid_location

# Compare pilot IDs with India inventory IDs
india_ids = set(india["event_id"].dropna().astype(str))
pilot_ids = set(pilot["event_id"].dropna().astype(str))

overlap_ids = india_ids.intersection(pilot_ids)
remaining_ids = india_ids - pilot_ids

print("\n1. INDIA INVENTORY")
print("-" * 65)
print("Total events:", len(india))
print("Unique event IDs:", india["event_id"].nunique())
print("Valid dates:", valid_date.sum())
print("Valid coordinates:", valid_location.sum())
print("Both date and coordinates valid:", eligible.sum())

print("\n2. MISSING OR INVALID DATA")
print("-" * 65)
print("Missing/invalid dates:", (~valid_date).sum())
print("Missing/invalid coordinates:", (~valid_location).sum())
print("Invalid latitude/longitude range:",
      (~valid_coordinates & india["audit_lat"].notna()
       & india["audit_lon"].notna()).sum())

print("\n3. PILOT OVERLAP")
print("-" * 65)
print("Pilot unique IDs:", len(pilot_ids))
print("IDs present in both datasets:", len(overlap_ids))
print("India IDs not in pilot:", len(remaining_ids))

remaining = india[india["event_id"].astype(str).isin(remaining_ids)]
remaining_eligible = (
    remaining["audit_date"].notna()
    & remaining["audit_lat"].between(-90, 90)
    & remaining["audit_lon"].between(-180, 180)
    & remaining["audit_lat"].notna()
    & remaining["audit_lon"].notna()
)

print("Remaining eligible events:", remaining_eligible.sum())

print("\n4. REGIONAL DISTRIBUTION")
print("-" * 65)

if "region" in india.columns:
    print(
        india.loc[eligible, "region"]
        .value_counts(dropna=False)
        .to_string()
    )
else:
    print("Region column not found.")

print("\nAUDIT COMPLETE")
print("Read-only. No files modified.")