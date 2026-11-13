# Standard Refresh Run Summary — `std_refresh_20261113T080519Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261113T080519Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-13T08:05:19Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-12T03:00:00Z` to `2026-11-13T03:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `5.089 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `26`, Cached: `22`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `31,421` |
| 9 | **Assessed vs Unassessed** | Assessed: `31,421` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=14.27`, `P80=31.82`, `P95=42.43`
- **Category Counts**: Low=`6,026`, Moderate=`3,615`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`9,006,437` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,367,741` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `17,044,385` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`1,894` features, types=`['Polygon']`, `4,840,283` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=22.73`, `P80=34.71`, `P95=41.35`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`12,507,551` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,250,952` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,524,991` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,250,952` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,524,991` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`3,499` features, types=`['Polygon']`, `7,958,894` bytes)

### Western Ghats (Limited Pilot) — `SUCCESS`

- **Coverage Scope**: Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.
- **Updated Locations**: `4,819` across `14` tiles (`Assessed=4,819`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=8.89`, `P80=19.49`, `P95=30.61`
- **Category Counts**: Low=`2,409`, Moderate=`1,446`, High=`723`, Very High=`241`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`4,656,307` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_grid_dynamic_risk\western_ghats_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`6,401` features, types=`['Point']`, `5,963,709` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_5km_spatial_cells.geojson` (`4,819` features, types=`['Polygon']`, `6,901,159` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_regions.geojson` (`1,478` features, types=`['Polygon']`, `3,577,156` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`17/17` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`16/16` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (3.39s) — Ran 9 tests in 1.768s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (2.08s) — Ran 11 tests in 0.261s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (13.94s) — Ran 28 tests in 11.867s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (5.27s) — Ran 9 tests in 2.677s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (2.6s) — Ran 11 tests in 0.287s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (138.33s) — Ran 12 tests in 135.676s | OK
  - `backend.test_western_ghats_spatial_grid`: passed=`True` (8.27s) — Ran 6 tests in 5.484s | OK
  - `backend.test_western_ghats_terrain_susceptibility`: passed=`True` (2.49s) — Ran 4 tests in 0.196s | OK
  - `backend.test_western_ghats_gpm_grid_rainfall`: passed=`True` (5.7s) — Ran 5 tests in 3.634s | OK
  - `backend.test_western_ghats_grid_dynamic_risk`: passed=`True` (3.27s) — Ran 3 tests in 0.982s | OK
  - `backend.test_western_ghats_polygon_risk_map`: passed=`True` (17.81s) — Ran 11 tests in 14.863s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (17.21s) — Ran 12 tests in 15.077s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (28.54s) — Ran 10 tests in 26.364s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (2.45s) — Ran 20 tests in 0.869s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (9.25s) — Ran 14 tests in 7.160s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (5.14s) — Ran 10 tests in 2.595s | OK

## Unified India Landslide Dynamic Risk Map

- **Status**: `SUCCESS`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\india_unified_dynamic_risk_map.html` (`6,304,157` bytes, `6.01` MB)
- **Observation Window (UTC)**: `2026-11-12T03:00:00Z` to `2026-11-13T03:00:00Z` (Latency: `5.0888 h`)
- **Generation Timestamp (UTC)**: `2026-11-13T08:05:19Z`
- **Validation Passed**: `True`
- **Feature Counts**: Total Cells=`31,421`, Events=`1,179`, Background=`2,306`
