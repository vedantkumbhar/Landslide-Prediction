# 🌍 India Landslide Prediction & Dynamic Risk Monitoring System

An AI/ML-powered, GIS-based landslide risk assessment and monitoring system for selected landslide-prone regions of India, integrating terrain characteristics, historical landslide records, and NASA GPM IMERG satellite rainfall observations.

**Project:** Landslide Prediction Mega Project

**Status:** Research Prototype — Core Monitoring Pipeline Implemented

**Repository:** [Landslide-Prediction](https://github.com/vedantkumbhar/Landslide-Prediction)

---

## 1. Project Overview

Landslides are a major natural hazard in several parts of India. Their occurrence is influenced by multiple factors, including terrain characteristics, slope conditions, historical landslide activity, and rainfall.

This project aims to develop a regional AI/ML-based system for landslide susceptibility assessment and dynamic rainfall-responsive risk monitoring.

The system combines:

- Historical landslide event records.
- Digital Elevation Models (DEM).
- Terrain-derived features such as elevation, slope, and aspect.
- Regional machine learning models.
- NASA GPM IMERG satellite rainfall observations.
- GIS-based spatial analysis and visualization.
- Automated rainfall refresh and risk-map generation.

The objective is to move beyond static historical landslide maps and provide regularly updated screening-level risk information for monitored grid locations.

## 2. Target Regions

The system focuses on three geographical regions:

| Region | Implementation Status |
|---|---|
| Himalayas | Regional monitoring pipeline implemented |
| Northeast India | Regional monitoring pipeline implemented |
| Western Ghats | Limited pilot implementation |

**Important — Western Ghats limitation:** The current Western Ghats implementation is a limited pilot based on available Western Maharashtra data and selected pilot areas. It is not representative of the entire Western Ghats region.

## 3. System Architecture

The project follows a multi-stage processing workflow:

1. **Data acquisition:** Collect historical landslide records, terrain datasets, and satellite rainfall observations.
2. **Data preprocessing:** Clean, validate, and prepare geospatial and environmental features.
3. **Terrain analysis:** Extract terrain characteristics from DEM datasets.
4. **Regional ML assessment:** Apply region-specific trained models.
5. **Rainfall integration:** Retrieve and process NASA GPM IMERG rainfall observations.
6. **Dynamic risk assessment:** Combine terrain susceptibility and rainfall-related indicators.
7. **GIS visualization:** Generate regional and unified India risk maps.
8. **Automated refresh:** Periodically refresh rainfall inputs and regenerate validated outputs.

## 4. Satellite Rainfall Integration

### NASA GPM IMERG

The system integrates NASA Global Precipitation Measurement (GPM) Integrated Multi-satellitE Retrievals for GPM (IMERG), using the Early Run V07C product.

Implemented capabilities include:

- NASA Common Metadata Repository (CMR) granule discovery.
- Retrieval and reuse of satellite rainfall granules.
- Contiguous 24-hour observation-window processing.
- Half-hourly rainfall granule integration.
- Rainfall feature generation for 1-hour, 3-hour, 6-hour, 12-hour, and 24-hour windows.
- Regional rainfall-to-risk processing.
- Observation freshness and completeness validation.

**Data freshness safeguard:** Rainfall observations older than the configured 12-hour threshold are not silently treated as current. Affected rainfall-driven risk assessments are marked `UNASSESSED` until valid observations become available.

**Spatial resolution note:** GPM IMERG has an approximately 0.1° native spatial resolution. A finer model or GIS grid does not increase the native resolution of the satellite rainfall observations.

### ISRO INSAT

ISRO INSAT satellite integration is planned as a future enhancement. It is not part of the currently implemented production rainfall pipeline.

## 5. Machine Learning and Risk Assessment

The project uses regional machine learning models and terrain-related features to support landslide susceptibility assessment.

The current implementation includes:

- Regional model workflows for the target regions.
- Frozen trained models used during production refreshes.
- Terrain susceptibility assessment.
- Rainfall-responsive risk calculations.
- Spatial grid-based monitoring.
- Historical landslide and background sample layers.

**Important:** The scheduled production refresh uses the existing trained models. It does not retrain models during routine rainfall updates.

Risk categories are experimental screening indicators. They should not be interpreted as calibrated probabilities, official warnings, or confirmed landslide predictions.

## 6. Dynamic GIS Dashboard

The project generates a unified India dynamic landslide risk map:

`outputs/india_unified_dynamic_risk_map.html`

The dashboard provides:

- Interactive Leaflet-based GIS visualization.
- Regional navigation and monitoring.
- Independently toggleable map layers.
- Rainfall-responsive risk information.
- Historical landslide event visualization.
- Background sample visualization.
- Regional telemetry and freshness information.
- Live UTC clock.
- Risk assessment and data-quality disclosures.

Historical landslide locations are maintained separately from currently monitored grid-cell risk information.

The dashboard is regenerated through the standard refresh workflow after successful regional processing and validation.

## 7. Automated Monitoring and Refresh

The project includes an automated NASA GPM rainfall refresh workflow using Windows Task Scheduler.

The standard production refresh can be executed manually using:

```powershell
python -X utf8 -B -m backend.run_standard_refresh
```

The workflow includes:

- Pre-flight validation.
- NASA GPM observation discovery.
- Rainfall granule retrieval and reuse.
- Regional processing.
- Dynamic risk-map generation.
- GIS output generation.
- Automated test verification.
- JSON and Markdown refresh reports.
- Process locking to prevent overlapping executions.
- Backup and rollback safeguards.
- Data freshness and completeness checks.

The configured Windows scheduled task is:

`LandslidePrediction_GPM_Refresh`

The local scheduled task and its configuration are not automatically installed when cloning the GitHub repository.

## 8. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn, Random Forest and regional ML workflows |
| Data Processing | Pandas, NumPy |
| Geospatial Processing | Rasterio, GeoPandas, PyProj |
| Remote Sensing | NASA GPM IMERG |
| GIS Visualization | Folium, Leaflet, GeoJSON |
| Data Storage | CSV, JSON, geospatial datasets |
| Automation | Windows Task Scheduler, Python |
| Version Control | Git and GitHub |

## 9. Repository Structure

```text
Landslide-Prediction/
│
├── backend/
│   ├── Regional processing pipelines
│   ├── GPM rainfall integration
│   ├── Standard refresh orchestration
│   ├── Unified GIS dashboard generation
│   └── Automated validation and tests
│
├── gis/
│   └── GIS and terrain data processing scripts
│
├── ml/
│   └── Machine learning workflows
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── Regional GIS maps
│   ├── Unified dynamic risk dashboard
│   └── Refresh reports
│
├── scripts/
│   └── Automation and utility scripts
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Data files, trained model artifacts, and some local operational files may be excluded from GitHub. The directory overview is illustrative rather than an exhaustive file listing.*

## 10. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Windows PowerShell (for the documented Windows automation workflow)
- NASA Earthdata credentials for online GPM data retrieval
- Required project datasets and trained model artifacts

### Step 1: Clone the repository

```powershell
git clone https://github.com/vedantkumbhar/Landslide-Prediction.git
cd Landslide-Prediction
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3: Activate the environment

```powershell
.venv\Scripts\activate
```

### Step 4: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 5: Configure NASA Earthdata credentials

Configure the required `NASA_EARTHDATA_TOKEN` in the project's supported environment configuration.

Do not commit authentication tokens, passwords, or other credentials to GitHub.

### Step 6: Prepare project data

Obtain the required datasets, regional inputs, and trained model artifacts. These are not necessarily included in the public repository.

### Step 7: Run the standard refresh

```powershell
python -X utf8 -B -m backend.run_standard_refresh
```

A successful run performs the configured validation and processing workflow and generates updated regional outputs, the unified dashboard, and refresh reports.

## 11. Testing and Validation

The project includes automated tests covering:

- Regional rainfall integration.
- Terrain susceptibility processing.
- Dynamic risk assessment.
- Spatial grid generation.
- GIS map generation.
- Unified dashboard generation.
- Standard refresh orchestration.
- Data freshness safeguards.
- Atomic output generation.
- Backup and rollback safety.
- Scheduled refresh behavior.

**Latest recorded verification:** 263 automated tests passed across the tested suites, with zero test failures.

## 12. Current Limitations

The project is a research and development prototype.

Important limitations include:

- Western Ghats coverage is currently limited to pilot areas.
- GPM rainfall observations have approximately 0.1° native spatial resolution.
- Risk categories are experimental and are not calibrated probabilities.
- Rainfall freshness depends on satellite data availability and retrieval.
- ISRO INSAT integration is not yet implemented.
- The system is not an officially certified early-warning platform.
- Availability of the required datasets and model artifacts is necessary to reproduce the complete pipeline.

The system must not be used as the sole basis for evacuation, emergency response, or disaster-management decisions.

## 13. Future Development

Planned enhancements include:

1. ISRO INSAT satellite rainfall integration.
2. Expanded Western Ghats geographical coverage.
3. Additional satellite-based environmental monitoring.
4. Improved regional model validation and calibration.
5. Earthquake-triggered landslide monitoring.
6. Enhanced early-warning and alert mechanisms.
7. Further field validation and operational evaluation.

## 14. Project Status

The project has progressed from historical landslide data preparation and terrain analysis to a regional rainfall-responsive monitoring prototype with automated refresh, GIS visualization, and validation safeguards.

The core monitoring pipeline and unified dashboard are implemented. Further development will focus on improving geographical coverage, validation, and satellite data integration.

---

**Project:** Landslide Prediction Mega Project
**Domain:** Artificial Intelligence, Machine Learning, GIS and Remote Sensing
**Repository:** [github.com/vedantkumbhar/Landslide-Prediction](https://github.com/vedantkumbhar/Landslide-Prediction)

*Developed as an academic research and engineering project.*