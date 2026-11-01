
# -*- coding: cp1252 -*-

from pathlib import Path
import html

import pandas as pd
import folium
from branca.colormap import LinearColormap

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data/processed/rainfall/regional_gpm_risk_predictions_01nov2026.csv"

OUTPUT = BASE / "outputs/regional_landslide_risk_map_esri_01nov2026.html"

RAIN_COLUMNS = {
    "3-hour rainfall": "gpm_3h_accumulation_mm",
    "6-hour rainfall": "gpm_6h_accumulation_mm",
    "12-hour rainfall": "gpm_12h_accumulation_mm",
    "24-hour rainfall": "gpm_24h_accumulation_mm",
}

REGIONS = [
    "Himalayas",
    "Northeast India",
    "Western Maharashtra",
]


def safe(value):
    return html.escape(str(value))


def main():

    if OUTPUT.exists():
        raise FileExistsError(
            f"Map already exists; refusing to overwrite: {OUTPUT}"
        )

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = [
        "latitude",
        "longitude",
        "dem_region",
        "terrain_model_score",
        *RAIN_COLUMNS.values(),
        "elevation_m",
        "slope_degrees",
        "rainfall_observation_end_utc",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in map fields.")

    if not df["dem_region"].isin(REGIONS).all():
        raise ValueError("Unexpected region values found.")

    if not df["terrain_model_score"].between(0, 1).all():
        raise ValueError("Terrain model scores must be between 0 and 1.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    center = [
        df["latitude"].mean(),
        df["longitude"].mean(),
    ]

    # Create map without the blocked OpenStreetMap tiles.
    map_object = folium.Map(
        location=center,
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )

    # Esri topographic basemap.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, HERE, Garmin, USGS, EPA, NPS",
        name="Esri World Topographic Map",
        overlay=False,
        control=True,
        show=True,
    ).add_to(map_object)

    # Terrain model score legend.
    colormap = LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=0,
        vmax=1,
        caption=(
            "Terrain model score (0 to 1; "
            "not calibrated probability)"
        ),
    )

    colormap.add_to(map_object)

    # Regional layers.
    for region in REGIONS:

        region_data = df[df["dem_region"] == region]

        layer = folium.FeatureGroup(
            name=f"{region} ({len(region_data)} records)",
            show=True,
        )

        for _, row in region_data.iterrows():

            score = float(row["terrain_model_score"])
            color = colormap(score)

            rainfall_html = "".join(
                f"<tr><td>{safe(label)}</td>"
                f"<td>{float(row[column]):.2f} mm</td></tr>"
                for label, column in RAIN_COLUMNS.items()
            )

            popup_html = f"""
            <div style="width:260px">
                <h4>{safe(region)}</h4>

                <b>Terrain model score:</b> {score:.3f}<br>
                <b>Latitude:</b> {float(row['latitude']):.5f}<br>
                <b>Longitude:</b> {float(row['longitude']):.5f}<br>
                <b>Elevation:</b> {float(row['elevation_m']):.1f} m<br>
                <b>Slope:</b> {float(row['slope_degrees']):.2f} degrees<br>

                <hr>

                <b>GPM rainfall accumulation</b>
                <table>{rainfall_html}</table>

                <hr>

                <b>Observation end (UTC):</b><br>
                {safe(row['rainfall_observation_end_utc'])}

                <hr>

                <small>
                Terrain-only model score. Rainfall is displayed
                separately and is not yet incorporated into the model.
                </small>
            </div>
            """

            folium.CircleMarker(
                location=[
                    float(row["latitude"]),
                    float(row["longitude"]),
                ],
                radius=4,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(
                    popup_html,
                    max_width=320,
                ),
                tooltip=f"{region} | Score: {score:.3f}",
            ).add_to(layer)

        layer.add_to(map_object)

        print(
            f"Added {len(region_data)} points: {region}"
        )

    # Regional layer selector.
    folium.LayerControl(
        collapsed=False
    ).add_to(map_object)

    # Map title.
    title_html = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 10px 14px;
        border: 1px solid #777;
        border-radius: 5px;">

        <b>Regional Landslide Prediction – GPM Integration</b><br>
        <small>Snapshot: 01 November 2026 | NASA GPM IMERG</small>

    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(title_html)
    )

    # Save map.
    map_object.save(str(OUTPUT))

    print("\nMAP CREATED SUCCESSFULLY")
    print("Records:", len(df))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
