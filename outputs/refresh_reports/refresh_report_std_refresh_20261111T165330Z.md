# Standard Refresh Run Summary — `std_refresh_20261111T165330Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261111T165330Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-11T16:53:30Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-10T12:00:00Z` to `2026-11-11T12:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `4.892 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `0`, Cached: `48`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `28,184` |
| 9 | **Assessed vs Unassessed** | Assessed: `28,184` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=13.39`, `P80=30.37`, `P95=40.88`
- **Category Counts**: Low=`6,025`, Moderate=`3,616`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`8,546,115` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,333,270` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `15,060,658` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`2,106` features, types=`['Polygon']`, `4,834,500` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=25.46`, `P80=38.38`, `P95=46.64`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`11,436,149` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,966,623` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `18,233,227` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,966,623` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `18,233,227` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`2,978` features, types=`['Polygon']`, `6,517,601` bytes)

### Western Ghats — `SUCCESS`

- **Updated Locations**: `1,582` across `14` tiles (`Assessed=1,582`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=8.43`, `P80=18.77`, `P95=45.12`
- **Category Counts**: Low=`791`, Moderate=`474`, High=`237`, Very High=`80`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`489,981` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_dynamic_risk\western_ghats_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_dynamic_risk\western_ghats_dynamic_hazard_risk.geojson` (`1,582` features, types=`['Point']`, `3,612,546` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`1,582` features, types=`['Point']`, `3,612,546` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`15/15` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`11/11` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (2.14s) — Ran 9 tests in 0.953s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (1.5s) — Ran 11 tests in 0.212s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (13.25s) — Ran 27 tests in 11.885s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (3.02s) — Ran 9 tests in 1.643s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (1.37s) — Ran 11 tests in 0.203s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (202.3s) — Ran 12 tests in 200.745s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (9.59s) — Ran 12 tests in 8.443s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (13.21s) — Ran 10 tests in 12.080s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (1.1s) — Ran 19 tests in 0.226s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (4.78s) — Ran 14 tests in 3.497s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (5.49s) — Ran 10 tests in 3.173s | OK
