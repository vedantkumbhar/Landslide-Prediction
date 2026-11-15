# Standard Refresh Run Summary — `std_refresh_20261115T130831Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261115T130831Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-15T13:08:31Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-14T08:00:00Z` to `2026-11-15T08:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `5.142 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `48`, Cached: `0`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `31,421` |
| 9 | **Assessed vs Unassessed** | Assessed: `31,421` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=13.46`, `P80=30.65`, `P95=41.09`
- **Category Counts**: Low=`6,026`, Moderate=`3,615`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`9,322,531` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,364,020` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `17,039,316` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`2,093` features, types=`['Polygon']`, `5,201,183` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=22.66`, `P80=34.62`, `P95=41.23`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`12,454,479` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,177,636` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,517,731` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,177,636` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,517,731` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`3,505` features, types=`['Polygon']`, `7,964,258` bytes)

### Western Ghats (Limited Pilot) — `SUCCESS`

- **Coverage Scope**: Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.
- **Updated Locations**: `4,819` across `14` tiles (`Assessed=4,819`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=9.53`, `P80=20.05`, `P95=31.43`
- **Category Counts**: Low=`2,409`, Moderate=`1,446`, High=`723`, Very High=`241`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`4,612,530` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_grid_dynamic_risk\western_ghats_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`6,401` features, types=`['Point']`, `5,970,434` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_5km_spatial_cells.geojson` (`4,819` features, types=`['Polygon']`, `6,918,772` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_regions.geojson` (`1,421` features, types=`['Polygon']`, `3,472,275` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`17/17` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`16/16` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (6.93s) — Ran 9 tests in 1.094s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (1.18s) — Ran 11 tests in 0.221s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (26.95s) — Ran 28 tests in 25.324s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (3.76s) — Ran 9 tests in 2.135s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (1.31s) — Ran 11 tests in 0.192s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (240.46s) — Ran 12 tests in 238.601s | OK
  - `backend.test_western_ghats_spatial_grid`: passed=`True` (22.76s) — Ran 6 tests in 19.997s | OK
  - `backend.test_western_ghats_terrain_susceptibility`: passed=`True` (1.75s) — Ran 4 tests in 0.135s | OK
  - `backend.test_western_ghats_gpm_grid_rainfall`: passed=`True` (7.92s) — Ran 5 tests in 6.062s | OK
  - `backend.test_western_ghats_grid_dynamic_risk`: passed=`True` (2.21s) — Ran 3 tests in 0.677s | OK
  - `backend.test_western_ghats_polygon_risk_map`: passed=`True` (15.05s) — Ran 11 tests in 13.105s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (13.53s) — Ran 12 tests in 12.126s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (19.21s) — Ran 10 tests in 17.649s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (2.52s) — Ran 20 tests in 0.607s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (13.86s) — Ran 14 tests in 11.875s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (7.1s) — Ran 10 tests in 4.538s | OK

## Unified India Landslide Dynamic Risk Map

- **Status**: `SUCCESS`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\india_unified_dynamic_risk_map.html` (`6,319,480` bytes, `6.03` MB)
- **Observation Window (UTC)**: `2026-11-14T08:00:00Z` to `2026-11-15T08:00:00Z` (Latency: `5.1421 h`)
- **Generation Timestamp (UTC)**: `2026-11-15T13:08:31Z`
- **Validation Passed**: `True`
- **Feature Counts**: Total Cells=`31,421`, Events=`1,179`, Background=`2,306`
