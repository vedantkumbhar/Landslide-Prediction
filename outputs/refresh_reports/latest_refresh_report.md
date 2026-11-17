# Standard Refresh Run Summary — `std_refresh_20261116T041418Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261116T041418Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-16T04:14:18Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-14T23:00:00Z` to `2026-11-15T23:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `5.238 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `30`, Cached: `18`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `31,421` |
| 9 | **Assessed vs Unassessed** | Assessed: `31,421` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=13.38`, `P80=30.52`, `P95=41.03`
- **Category Counts**: Low=`6,026`, Moderate=`3,615`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`9,254,179` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,348,666` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `17,018,584` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`2,089` features, types=`['Polygon']`, `5,189,024` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=22.85`, `P80=34.78`, `P95=41.43`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`12,496,181` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,316,312` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,535,554` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,316,312` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,535,554` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`3,474` features, types=`['Polygon']`, `7,917,095` bytes)

### Western Ghats (Limited Pilot) — `SUCCESS`

- **Coverage Scope**: Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.
- **Updated Locations**: `4,819` across `14` tiles (`Assessed=4,819`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=9.68`, `P80=20.12`, `P95=31.04`
- **Category Counts**: Low=`2,409`, Moderate=`1,446`, High=`723`, Very High=`241`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`4,722,016` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_grid_dynamic_risk\western_ghats_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`6,401` features, types=`['Point']`, `5,976,051` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_5km_spatial_cells.geojson` (`4,819` features, types=`['Polygon']`, `6,925,074` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_regions.geojson` (`1,494` features, types=`['Polygon']`, `3,611,349` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`17/17` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`16/16` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (1.91s) — Ran 9 tests in 0.898s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (0.9s) — Ran 11 tests in 0.140s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (7.73s) — Ran 28 tests in 6.785s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (1.91s) — Ran 9 tests in 1.055s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (0.94s) — Ran 11 tests in 0.138s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (108.78s) — Ran 12 tests in 107.764s | OK
  - `backend.test_western_ghats_spatial_grid`: passed=`True` (2.78s) — Ran 6 tests in 1.746s | OK
  - `backend.test_western_ghats_terrain_susceptibility`: passed=`True` (1.16s) — Ran 4 tests in 0.127s | OK
  - `backend.test_western_ghats_gpm_grid_rainfall`: passed=`True` (2.69s) — Ran 5 tests in 1.733s | OK
  - `backend.test_western_ghats_grid_dynamic_risk`: passed=`True` (1.56s) — Ran 3 tests in 0.506s | OK
  - `backend.test_western_ghats_polygon_risk_map`: passed=`True` (5.71s) — Ran 11 tests in 4.392s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (8.05s) — Ran 12 tests in 7.105s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (9.74s) — Ran 10 tests in 8.854s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (1.21s) — Ran 20 tests in 0.424s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (5.8s) — Ran 14 tests in 4.750s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (2.76s) — Ran 10 tests in 1.466s | OK

## Unified India Landslide Dynamic Risk Map

- **Status**: `SUCCESS`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\india_unified_dynamic_risk_map.html` (`6,316,596` bytes, `6.02` MB)
- **Observation Window (UTC)**: `2026-11-14T23:00:00Z` to `2026-11-15T23:00:00Z` (Latency: `5.2384 h`)
- **Generation Timestamp (UTC)**: `2026-11-16T04:14:18Z`
- **Validation Passed**: `True`
- **Feature Counts**: Total Cells=`31,421`, Events=`1,179`, Background=`2,306`
