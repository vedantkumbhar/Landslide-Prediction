from pathlib import Path
import subprocess
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Landslide Intelligence Platform",
    description="Landslide monitoring and environmental data API.",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAINFALL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall"
    / "pilot_hourly_rainfall.csv"
)

POINT_COLUMNS = [
    "region",
    "requested_latitude",
    "requested_longitude",
]

# PROVISIONAL DEMONSTRATION BANDS ONLY.
# These are not scientifically calibrated landslide thresholds.
def rainfall_monitoring_indicator(rainfall_3h, rainfall_24h):
    if rainfall_3h is None or rainfall_24h is None:
        return {
            "level": "Unavailable",
            "reason": "Insufficient rainfall data",
        }

    if rainfall_3h >= 30 or rainfall_24h >= 100:
        level = "Elevated rainfall"
    elif rainfall_3h >= 15 or rainfall_24h >= 50:
        level = "Increased rainfall"
    else:
        level = "Lower rainfall accumulation"

    return {
        "level": level,
        "reason": "Provisional rainfall bands; not a landslide-risk score",
    }


@app.get("/")
def root():
    return {
        "project": "Landslide Intelligence Platform",
        "status": "operational",
        "version": "0.4.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "backend"}


@app.get("/rainfall/latest")
def latest_rainfall():
    if not RAINFALL_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Rainfall data is not available. Fetch it first.",
        )

    try:
        df = pd.read_csv(RAINFALL_FILE)

        required = set(
            POINT_COLUMNS
            + [
                "timestamp_local",
                "precipitation_mm",
                "record_type",
                "source",
                "retrieved_at_local",
                "api_latitude",
                "api_longitude",
            ]
        )

        if not required.issubset(df.columns):
            raise HTTPException(
                status_code=500,
                detail="Rainfall CSV is missing required columns.",
            )

        df["timestamp_local"] = pd.to_datetime(
            df["timestamp_local"], errors="coerce"
        )
        df["precipitation_mm"] = pd.to_numeric(
            df["precipitation_mm"], errors="coerce"
        )

        df = df.dropna(
            subset=POINT_COLUMNS
            + ["timestamp_local", "precipitation_mm"]
        )

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="No valid rainfall records are available.",
            )

        df = df.sort_values("timestamp_local")
        regions = []

        for _, point in df.groupby(POINT_COLUMNS, sort=False):
            point = point.sort_values("timestamp_local")
            latest = point.iloc[-1]
            latest_time = latest["timestamp_local"]

            last_3h = point[
                (point["timestamp_local"] > latest_time - pd.Timedelta(hours=3))
                & (point["timestamp_local"] <= latest_time)
            ]

            last_24h = point[
                (point["timestamp_local"] > latest_time - pd.Timedelta(hours=24))
                & (point["timestamp_local"] <= latest_time)
            ]

            rainfall_3h = round(
                float(last_3h["precipitation_mm"].sum()), 2
            )
            rainfall_24h = round(
                float(last_24h["precipitation_mm"].sum()), 2
            )

            indicator = rainfall_monitoring_indicator(
                rainfall_3h, rainfall_24h
            )

            regions.append({
                "region": str(latest["region"]),
                "requested_latitude": float(
                    latest["requested_latitude"]
                ),
                "requested_longitude": float(
                    latest["requested_longitude"]
                ),
                "api_latitude": float(latest["api_latitude"]),
                "api_longitude": float(latest["api_longitude"]),
                "timestamp_local": latest_time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "precipitation_mm": float(
                    latest["precipitation_mm"]
                ),
                "rainfall_3h_mm": rainfall_3h,
                "rainfall_24h_mm": rainfall_24h,
                "records_3h": int(len(last_3h)),
                "records_24h": int(len(last_24h)),
                "forecast_records_3h": int(
                    (last_3h["record_type"] == "forecast").sum()
                ),
                "forecast_records_24h": int(
                    (last_24h["record_type"] == "forecast").sum()
                ),
                "record_type": str(latest["record_type"]),
                "rainfall_monitoring": indicator,
            })

        retrieved = df["retrieved_at_local"].dropna()

        return {
            "source": "Open-Meteo",
            "units": {
                "precipitation": "mm/hour",
                "accumulation": "mm",
            },
            "data_retrieved_at": (
                str(retrieved.iloc[-1]) if not retrieved.empty else None
            ),
            "total_points": len(regions),
            "regions": regions,
            "note": (
                "Accumulations use available hourly records and may "
                "include forecast data. Record counts and forecast "
                "counts are provided for transparency. Rainfall "
                "monitoring bands are provisional and are not validated "
                "landslide-risk scores or official warnings."
            ),
        }

    except HTTPException:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read rainfall data: {exc}",
        ) from exc


@app.post("/rainfall/refresh")
def refresh_rainfall():
    script = PROJECT_ROOT / "backend" / "fetch_rainfall.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Rainfall refresh timed out.",
        ) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail=f"Rainfall refresh failed: {result.stderr[-1500:]}",
        )

    return {
        "status": "success",
        "message": "Rainfall data refreshed.",
        "output": result.stdout,
    }
