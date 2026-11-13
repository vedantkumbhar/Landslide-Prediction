# Standard Refresh Run Summary — `std_refresh_20261113T102314Z`

## 1–6. Execution, Window, Latency & Granule Telemetry

| Item | Metric | Value |
|---|---|---|
| 1 | **Run ID** | `std_refresh_20261113T102314Z` |
| 2 | **Execution Timestamp (UTC)** | `2026-11-13T10:23:14Z` |
| 3 | **Final Run Status** | **SUCCESS** |
| 4 | **Observation Window (UTC)** | `2026-11-12T05:00:00Z` to `2026-11-13T05:00:00Z` (`NASA_CMR_ONLINE`) |
| 5 | **Observed Data Latency** | `5.387 h` (Limit: `12.0 h`, Fresh: `True`) |
| 6 | **GPM Granules (24h Window)** | Discovered: `150`, Downloaded: `4`, Cached: `44`, Validated: `48/48` |
| 7 | **5-Window Rainfall Status** | `VERIFIED_MONOTONIC` (`1h, 3h, 6h, 12h, 24h`) |
| 8 | **Updated Locations Total** | `31,421` |
| 9 | **Assessed vs Unassessed** | Assessed: `31,421` / Unassessed: `0` |

## 8–11. Per-Region Breakdown, Categories & Regenerated GIS Paths

### Himalayas — `SUCCESS`

- **Updated Locations**: `12,052` across `29` tiles (`Assessed=12,052`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=14.22`, `P80=31.61`, `P95=42.20`
- **Category Counts**: Low=`6,026`, Moderate=`3,615`, High=`1,808`, Very High=`603`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.html` (`9,055,947` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\himalayas_dynamic_risk\himalayas_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_map.geojson` (`13,037` features, types=`['Point']`, `13,366,072` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_5km_spatial_cells.geojson` (`12,052` features, types=`['Polygon']`, `17,041,485` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\himalayas_dynamic_risk_regions.geojson` (`1,933` features, types=`['Polygon']`, `4,902,818` bytes)

### Northeast India — `SUCCESS`

- **Updated Locations**: `14,550` across `33` tiles (`Assessed=14,550`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=22.67`, `P80=34.69`, `P95=41.30`
- **Category Counts**: Low=`7,275`, Moderate=`4,365`, High=`2,182`, Very High=`728`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_map.html` (`12,489,966` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk\northeast_india_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,246,295` bytes)
  - `D:\Vedant\Codes\Landslide-India\data\processed\northeast_dynamic_risk_map\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,524,022` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_hazard_risk.geojson` (`14,550` features, types=`['Point']`, `31,246,295` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_5km_spatial_cells.geojson` (`14,550` features, types=`['Polygon']`, `20,524,022` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\northeast_india_dynamic_risk_regions.geojson` (`3,492` features, types=`['Polygon']`, `7,947,795` bytes)

### Western Ghats (Limited Pilot) — `SUCCESS`

- **Coverage Scope**: Limited Western Ghats pilot only: 14 validated DEM tiles, 4,819 5 km grid cells, 182 historical landslide events, and 1,400 background samples; does not imply complete Western Ghats coverage.
- **Updated Locations**: `4,819` across `14` tiles (`Assessed=4,819`, `Unassessed=0`)
- **Dynamic Cutoffs**: `P50=8.89`, `P80=19.49`, `P95=30.61`
- **Category Counts**: Low=`2,409`, Moderate=`1,446`, High=`723`, Very High=`241`, Unassessed=`0`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_map.html` (`4,654,754` bytes)
- **Risk CSV**: `D:\Vedant\Codes\Landslide-India\data\processed\western_ghats_grid_dynamic_risk\western_ghats_5km_dynamic_hazard_risk.csv`
- **GeoJSON Outputs**:
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_hazard_risk.geojson` (`6,401` features, types=`['Point']`, `5,964,023` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_5km_spatial_cells.geojson` (`4,819` features, types=`['Polygon']`, `6,901,865` bytes)
  - `D:\Vedant\Codes\Landslide-India\outputs\western_ghats_dynamic_risk_regions.geojson` (`1,478` features, types=`['Polygon']`, `3,577,194` bytes)

## 12. Validation, Test Results, Errors & Skipped Regions

- **Pre-flight Check Status**: `PASSED` (`17/17` checks)
- **Cross-Region Consistency Passed**: `True`
- **Unit Tests Executed**: `True` | **All Passed**: `True` (`16/16` suites)
- **Skipped Regions**: `[]`
- **Errors**: `[]`
- **Warnings**: `[]`
  - `backend.test_himalayas_gpm_rainfall`: passed=`True` (19.0s) — Ran 9 tests in 13.109s | OK
  - `backend.test_himalayas_dynamic_risk`: passed=`True` (5.32s) — Ran 11 tests in 0.884s | OK
  - `backend.test_himalayas_dynamic_risk_map`: passed=`True` (53.55s) — Ran 28 tests in 46.807s | OK
  - `backend.test_northeast_gpm_rainfall`: passed=`True` (14.02s) — Ran 9 tests in 7.612s | OK
  - `backend.test_northeast_dynamic_risk`: passed=`True` (6.47s) — Ran 11 tests in 1.444s | OK
  - `backend.test_northeast_risk_map`: passed=`True` (394.32s) — Ran 12 tests in 388.671s | OK
  - `backend.test_western_ghats_spatial_grid`: passed=`True` (24.87s) — Ran 6 tests in 17.445s | OK
  - `backend.test_western_ghats_terrain_susceptibility`: passed=`True` (8.19s) — Ran 4 tests in 0.607s | OK
  - `backend.test_western_ghats_gpm_grid_rainfall`: passed=`True` (24.97s) — Ran 5 tests in 17.290s | OK
  - `backend.test_western_ghats_grid_dynamic_risk`: passed=`True` (14.96s) — Ran 3 tests in 3.868s | OK
  - `backend.test_western_ghats_polygon_risk_map`: passed=`True` (26.2s) — Ran 11 tests in 19.725s | OK
  - `backend.test_western_ghats_gpm_refresh`: passed=`True` (28.76s) — Ran 12 tests in 24.076s | OK
  - `backend.test_western_ghats_rainfall_terrain_integration`: passed=`True` (50.27s) — Ran 10 tests in 46.079s | OK
  - `backend.test_western_ghats_dynamic_risk`: passed=`True` (7.23s) — Ran 20 tests in 1.999s | OK
  - `backend.test_western_ghats_risk_map`: passed=`True` (27.85s) — Ran 14 tests in 22.042s | OK
  - `backend.test_standard_refresh_procedure`: passed=`True` (21.84s) — Ran 10 tests in 12.708s | OK

## Unified India Landslide Dynamic Risk Map

- **Status**: `SUCCESS`
- **HTML Map**: `D:\Vedant\Codes\Landslide-India\outputs\india_unified_dynamic_risk_map.html` (`6,318,400` bytes, `6.03` MB)
- **Observation Window (UTC)**: `2026-11-12T05:00:00Z` to `2026-11-13T05:00:00Z` (Latency: `5.3874 h`)
- **Generation Timestamp (UTC)**: `2026-11-13T10:23:14Z`
- **Validation Passed**: `True`
- **Feature Counts**: Total Cells=`31,421`, Events=`1,179`, Background=`2,306`
