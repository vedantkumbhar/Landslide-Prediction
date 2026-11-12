# Standard Refresh Run Summary — `std_refresh_20261112T080206Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261112T080206Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-12T08:02:06Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-11T03:00:00Z` to `2026-11-12T03:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `5.035 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `0`, Cached: `48`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `31,421` |
| 9 | **Assessed vs Unassessed** | Assessed: `31,421` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=13.42`, `P80=30.55`, `P95=41.14`
- **Category Counts**: Low=`6,026`, Moderate=`3,615`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`9,133,045` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,337,813` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `17,009,617` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`2,056` features, types=`['Polygon']`, `5,132,306` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=24.57`, `P80=36.87`, `P95=43.86`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`12,624,580` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,794,509` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,597,196` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,794,509` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,597,196` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`3,289` features, types=`['Polygon']`, `7,595,545` bytes)

### Western Ghats (Limited Pilot) — `SUCCESS`

- **Coverage Scope**: Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.
- **Updated Locations**: `4,819` across `14` tiles (`Assessed=4,819`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=8.98`, `P80=19.75`, `P95=30.75`
- **Category Counts**: Low=`2,409`, Moderate=`1,446`, High=`723`, Very High=`241`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`4,654,801` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_grid_dynamic_risk\western_ghats_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`6,401` features, types=`['Point']`, `5,963,415` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_5km_spatial_cells.geojson` (`4,819` features, types=`['Polygon']`, `6,906,807` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_regions.geojson` (`1,472` features, types=`['Polygon']`, `3,565,696` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`17/17` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`16/16` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (2.66s) — Ran 9 tests in 1.152s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (1.68s) — Ran 11 tests in 0.294s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (12.89s) — Ran 28 tests in 11.198s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (3.46s) — Ran 9 tests in 1.474s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (1.45s) — Ran 11 tests in 0.221s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (76.21s) — Ran 12 tests in 74.413s | OK
  - `backend.test_western_ghats_spatial_grid`: passed=`True` (4.23s) — Ran 6 tests in 2.346s | OK
  - `backend.test_western_ghats_terrain_susceptibility`: passed=`True` (1.74s) — Ran 4 tests in 0.128s | OK
  - `backend.test_western_ghats_gpm_grid_rainfall`: passed=`True` (4.27s) — Ran 5 tests in 2.595s | OK
  - `backend.test_western_ghats_grid_dynamic_risk`: passed=`True` (2.57s) — Ran 3 tests in 0.784s | OK
  - `backend.test_western_ghats_polygon_risk_map`: passed=`True` (7.21s) — Ran 11 tests in 5.139s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (7.57s) — Ran 12 tests in 6.238s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (8.61s) — Ran 10 tests in 7.340s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (1.56s) — Ran 20 tests in 0.394s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (5.55s) — Ran 14 tests in 3.645s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (4.28s) — Ran 10 tests in 2.068s | OK
