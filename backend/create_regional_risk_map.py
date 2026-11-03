
from pathlib import Path
import argparse
import html

import pandas as pd
import folium
from branca.colormap import LinearColormap


BASE = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    BASE
    / "data/processed/rainfall/refresh_runs/20261102_152742"
    / "predictions"
    / "regional_gpm_risk_predictions_02nov2026_with_3h.csv"
)

DEFAULT_OUTPUT = (
    BASE
    / "outputs/regional_landslide_risk_map_esri_02nov2026.html"
)

RAIN_COLUMNS = {
    "1-hour rainfall": "gpm_1h_accumulation_mm",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the regional GPM landslide monitoring map."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-map",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    input_csv = args.input_csv
    output_map = args.output_map

    print("\nREGIONAL LANDSLIDE RISK MAP")
    print("=" * 60)
    print("Input:", input_csv)
    print("Output:", output_map)

    if output_map.exists():
        raise FileExistsError(
            f"Map already exists; refusing to overwrite: {output_map}"
        )

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    required = [
        "record_id",
        "latitude",
        "longitude",
        "dem_region",
        "terrain_model_score",
        "elevation_m",
        "slope_degrees",
        "gpm_24h_series_start_utc",
        "gpm_24h_series_end_utc",
        *RAIN_COLUMNS.values(),
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in required map fields.")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs found.")

    if not df["dem_region"].isin(REGIONS).all():
        raise ValueError("Unexpected region values found.")

    if not df["terrain_model_score"].between(0, 1).all():
        raise ValueError("Terrain scores must be between 0 and 1.")

    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values.")

    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values.")

    for column in RAIN_COLUMNS.values():
        if (df[column] < 0).any():
            raise ValueError(f"Negative rainfall values in {column}")

    accumulation_pairs = [
        ("gpm_1h_accumulation_mm", "gpm_3h_accumulation_mm"),
        ("gpm_3h_accumulation_mm", "gpm_6h_accumulation_mm"),
        ("gpm_6h_accumulation_mm", "gpm_12h_accumulation_mm"),
        ("gpm_12h_accumulation_mm", "gpm_24h_accumulation_mm"),
    ]

    for shorter, longer in accumulation_pairs:
        if (df[shorter] > df[longer] + 1e-5).any():
            raise ValueError(
                f"Rainfall accumulation inconsistency: "
                f"{shorter} exceeds {longer}"
            )

    start_times = df["gpm_24h_series_start_utc"].unique()
    end_times = df["gpm_24h_series_end_utc"].unique()

    if len(start_times) != 1 or len(end_times) != 1:
        raise ValueError("Inconsistent 24-hour observation timestamps.")

    observation_start = start_times[0]
    observation_end = end_times[0]

    output_map.parent.mkdir(parents=True, exist_ok=True)

    center = [
        df["latitude"].mean(),
        df["longitude"].mean(),
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )

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

    for region in REGIONS:

        region_data = df[df["dem_region"] == region]

        layer_name = (
            "Western Maharashtra pilot"
            if region == "Western Maharashtra"
            else region
        )

        layer = folium.FeatureGroup(
            name=f"{layer_name} ({len(region_data)} records)",
            show=True,
        )

        for _, row in region_data.iterrows():

            score = float(row["terrain_model_score"])
            color = colormap(score)

            rainfall_html = "".join(
                f"<tr>"
                f"<td>{safe(label)}</td>"
                f"<td>{float(row[column]):.3f} mm</td>"
                f"</tr>"
                for label, column in RAIN_COLUMNS.items()
            )

            popup_html = f"""
            <div style="width:280px; font-family:Arial">

                <h4>{safe(layer_name)}</h4>

                <b>Record ID:</b> {safe(row['record_id'])}<br>

                <b>Terrain model score:</b> {score:.3f}<br>

                <b>Latitude:</b> {float(row['latitude']):.5f}<br>

                <b>Longitude:</b> {float(row['longitude']):.5f}<br>

                <b>Elevation:</b> {float(row['elevation_m']):.1f} m<br>

                <b>Slope:</b> {float(row['slope_degrees']):.2f} degrees

                <hr>

                <b>NASA GPM IMERG rainfall</b>

                <table style="width:100%">
                    {rainfall_html}
                </table>

                <hr>

                <b>24-hour observation window (UTC)</b><br>

                {safe(observation_start)}<br>
                to<br>
                {safe(observation_end)}

                <hr>

                <small>
                The terrain model score is terrain-only and is not
                a calibrated landslide probability. Rainfall values
                are displayed separately and are not yet incorporated
                into the model.
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
                    max_width=340,
                ),
                tooltip=(
                    f"{layer_name} | "
                    f"Terrain score: {score:.3f} | "
                    f"24h rainfall: "
                    f"{float(row['gpm_24h_accumulation_mm']):.2f} mm"
                ),
            ).add_to(layer)

        layer.add_to(map_object)

        print(f"Added {len(region_data)} points: {layer_name}")

    folium.LayerControl(collapsed=False).add_to(map_object)

    title_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 12px 15px;
        border: 1px solid #777;
        border-radius: 5px;
        max-width: 380px;">

        <b>Regional Landslide Monitoring – NASA GPM IMERG</b><br>

        <small>
        Total records: {len(df):,}<br>
        Observation: {safe(observation_start)} to
        {safe(observation_end)} UTC<br>
        Western Maharashtra data represents a limited pilot.
        </small>

    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(title_html)
    )

    map_object.save(str(output_map))

    print("\nMAP CREATED SUCCESSFULLY")
    print("Total records:", len(df))
    print("Rainfall windows:", len(RAIN_COLUMNS))
    print("Observation start:", observation_start)
    print("Observation end:", observation_end)
    print("Output:", output_map)


if __name__ == "__main__":
    main()
