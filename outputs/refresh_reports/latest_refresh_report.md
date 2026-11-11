# Standard Refresh Run Summary — `std_refresh_20261111T183803Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261111T183803Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-11T18:38:03Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-10T14:00:00Z` to `2026-11-11T14:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `4.634 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `2`, Cached: `46`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `31,421` |
| 9 | **Assessed vs Unassessed** | Assessed: `31,421` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=13.37`, `P80=30.36`, `P95=40.84`
- **Category Counts**: Low=`6,026`, Moderate=`3,615`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`9,175,019` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,333,185` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `17,003,052` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`2,116` features, types=`['Polygon']`, `5,233,592` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=26.04`, `P80=38.67`, `P95=47.35`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`12,309,158` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `32,040,845` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,640,142` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `32,040,845` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,640,142` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`2,898` features, types=`['Polygon']`, `6,952,958` bytes)

### Western Ghats (Limited Pilot) — `SUCCESS`

- **Coverage Scope**: Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.
- **Updated Locations**: `4,819` across `14` tiles (`Assessed=4,819`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=8.84`, `P80=19.56`, `P95=30.71`
- **Category Counts**: Low=`2,409`, Moderate=`1,446`, High=`723`, Very High=`241`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`4,152,240` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_grid_dynamic_risk\western_ghats_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`6,401` features, types=`['Point']`, `5,960,547` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_5km_spatial_cells.geojson` (`4,819` features, types=`['Polygon']`, `6,901,744` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_regions.geojson` (`1,478` features, types=`['Polygon']`, `3,575,722` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`17/17` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`16/16` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (4.73s) — Ran 9 tests in 1.994s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (2.38s) — Ran 11 tests in 0.396s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (10.9s) — Ran 28 tests in 8.858s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (2.73s) — Ran 9 tests in 1.481s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (1.94s) — Ran 11 tests in 0.179s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (58.11s) — Ran 12 tests in 56.224s | OK
  - `backend.test_western_ghats_spatial_grid`: passed=`True` (4.06s) — Ran 6 tests in 2.166s | OK
  - `backend.test_western_ghats_terrain_susceptibility`: passed=`True` (1.94s) — Ran 4 tests in 0.163s | OK
  - `backend.test_western_ghats_gpm_grid_rainfall`: passed=`True` (3.92s) — Ran 5 tests in 1.982s | OK
  - `backend.test_western_ghats_grid_dynamic_risk`: passed=`True` (2.39s) — Ran 3 tests in 0.570s | OK
  - `backend.test_western_ghats_polygon_risk_map`: passed=`True` (5.27s) — Ran 8 tests in 2.820s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (8.13s) — Ran 12 tests in 6.299s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (7.26s) — Ran 10 tests in 5.896s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (1.7s) — Ran 19 tests in 0.282s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (4.32s) — Ran 14 tests in 2.576s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (4.9s) — Ran 10 tests in 2.232s | OK
