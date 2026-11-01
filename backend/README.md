# 🌍 Landslide Prediction Mega Project

### AI/ML-Based Dynamic Landslide Prediction and Monitoring System Using Satellite Data

## 📌 Project Overview

The Landslide Prediction Mega Project aims to develop an AI/ML-based system for landslide risk assessment, prediction and monitoring across vulnerable regions of India.

The system combines historical landslide records, geographical information systems (GIS), Digital Elevation Models (DEM), rainfall analysis and satellite-based environmental monitoring.

Our long-term objective is to develop a dynamic GIS risk map that updates landslide risk according to changing rainfall and terrain-related conditions.

## 🎯 Target Regions

The project focuses on three major geographical regions:

1. **Himalayas**
2. **Northeast India**
3. **Western Ghats**

Western Maharashtra is being used as a limited pilot area and is not considered representative of the entire Western Ghats.

## 🛠️ Technology Stack

- Python
- Machine Learning
- Scikit-learn
- TensorFlow
- GIS and Remote Sensing
- Rasterio
- GDAL
- Pandas and NumPy
- HTML, CSS and JavaScript
- Satellite Data Integration (planned)

## 📂 Project Structure

```text
Landslide-Prediction/
│
├── backend/
│   ├── app/
│   └── rainfall processing scripts
│
├── frontend/
│
├── gis/
│   ├── DEM processing
│   ├── terrain extraction
│   ├── feature preparation
│   └── regional validation
│
├── ml/
│   ├── train_baseline.py
│   └── model validation
│
├── reports/
│   ├── regional_baseline_evaluation/
│   └── terrain_rainfall_baseline/
│
├── tests/
│
├── data/                 # Excluded from GitHub
├── requirements.txt
├── .gitignore
└── README.md
```

## ✅ Development Progress

### Completed / Implemented Development Work

- Historical landslide dataset collection and preprocessing.
- India-specific landslide event extraction.
- GIS-based landslide location inspection.
- DEM downloading and processing workflows.
- Elevation, slope and aspect feature extraction.
- Regional terrain data processing workflows.
- Pilot background sample generation.
- Initial training dataset preparation.
- Rainfall feature extraction and processing workflows.
- Western Ghats rainfall and terrain analysis.
- Regional baseline model training and evaluation workflows.
- Preliminary regional risk scoring and GIS visualization workflows.

### 🚧 Ongoing Development

- Regional dataset refinement and validation.
- Improving rainfall and terrain feature reliability.
- Regional model development for all three target regions.
- Satellite rainfall data integration.
- Dynamic landslide risk assessment.

### 🔜 Planned Development

- NASA GPM IMERG satellite rainfall integration.
- ISRO INSAT data integration.
- Satellite-based terrain and environmental monitoring.
- Improved regional ML prediction models.
- Dynamic GIS-based landslide risk map.
- Real-time or regularly refreshed risk monitoring.
- Alert and notification system.
- Complete frontend and backend integration.

## 🛰️ Satellite Data Integration

Planned satellite data sources:

- NASA GPM IMERG
- ISRO INSAT

The objective is to incorporate satellite-derived environmental information into the landslide risk assessment workflow.

## 📊 Current Project Status

**Development Stage:** Research, Data Processing, GIS and ML Development

**Project Status:** Work in Progress

This repository contains the current development code. The complete prediction and monitoring system is still under development. Experimental results and pilot datasets should not be interpreted as validated nationwide or Western Ghats-wide prediction performance.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/vedantkumbhar/Landslide-Prediction.git
```

Navigate to the project:

```bash
cd Landslide-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 👥 Project Collaboration

This repository is used to maintain source code, development history, project documentation and collaborative contributions.

Project members can contribute through GitHub commits and pull requests.

## ⚠️ Disclaimer

This project is an academic research and development initiative. Predictions and risk assessments are experimental and should not be used as the sole basis for real-world disaster management or evacuation decisions.

---

**Project:** Landslide Prediction Mega Project

**Repository:** https://github.com/vedantkumbhar/Landslide-Prediction

**Status:** Under Development
