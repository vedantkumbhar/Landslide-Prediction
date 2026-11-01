
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSV = (
    ROOT / "data" / "processed" / "rainfall"
    / "western_ghats" / "coordinate_level"
    / "western_ghats_experimental_risk_scores.csv"
)

HTML = ROOT / "western_ghats_pilot_map.html"

REQUIRED = [
    "sample_id", "pilot_tile",
    "terrain_latitude", "terrain_longitude",
    "timestamp_local", "temporal_role",
    "elevation_m", "slope_degrees",
    "recent_rain_6h_mm", "recent_rain_24h_mm",
    "forecast_rain_6h_mm", "forecast_rain_24h_mm",
    "recent_risk_index", "forecast_risk_index",
]


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"CSV not found: {CSV}")

    df = pd.read_csv(CSV)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        df["temporal_role"] == "current_or_forecast_candidate"
    ].copy()

    if df.empty:
        raise ValueError("No current/forecast candidate records found.")

    numeric_cols = [
        "terrain_latitude", "terrain_longitude",
        "elevation_m", "slope_degrees",
        "recent_rain_6h_mm", "recent_rain_24h_mm",
        "forecast_rain_6h_mm", "forecast_rain_24h_mm",
        "recent_risk_index", "forecast_risk_index",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["terrain_latitude", "terrain_longitude"]
    )
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"]
    ).dt.strftime("%Y-%m-%d %H:%M")

    if df.empty:
        raise ValueError("No valid coordinates found.")

    records = []
    for row in df.itertuples(index=False):
        records.append({
            "id": str(row.sample_id),
            "tile": str(row.pilot_tile),
            "time": str(row.timestamp_local),
            "lat": float(row.terrain_latitude),
            "lon": float(row.terrain_longitude),
            "elevation": (
                float(row.elevation_m)
                if pd.notna(row.elevation_m) else None
            ),
            "slope": (
                float(row.slope_degrees)
                if pd.notna(row.slope_degrees) else None
            ),
            "recent6": (
                float(row.recent_rain_6h_mm)
                if pd.notna(row.recent_rain_6h_mm) else None
            ),
            "recent24": (
                float(row.recent_rain_24h_mm)
                if pd.notna(row.recent_rain_24h_mm) else None
            ),
            "forecast6": (
                float(row.forecast_rain_6h_mm)
                if pd.notna(row.forecast_rain_6h_mm) else None
            ),
            "forecast24": (
                float(row.forecast_rain_24h_mm)
                if pd.notna(row.forecast_rain_24h_mm) else None
            ),
            "recent_index": (
                float(row.recent_risk_index)
                if pd.notna(row.recent_risk_index) else None
            ),
            "forecast_index": (
                float(row.forecast_risk_index)
                if pd.notna(row.forecast_risk_index) else None
            ),
        })

    data_json = json.dumps(
        records, ensure_ascii=False, allow_nan=False
    ).replace("</", "<\\/")

    page = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Western Ghats Experimental Risk Map</title>
<link rel="stylesheet"
 href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body{margin:0;height:100%;font-family:Arial,sans-serif}
#toolbar{padding:12px;background:white;display:flex;flex-wrap:wrap;
gap:12px;align-items:center;border-bottom:1px solid #ccc}
#map{height:calc(100% - 165px);min-height:350px}
label{font-size:13px;font-weight:bold}
select{padding:6px}
#status{font-size:12px;color:#444}
.note{font-size:12px;color:#444;padding:7px 12px}
.legend{background:white;padding:10px;line-height:1.6;
border:1px solid #aaa;border-radius:5px}
</style>
</head>
<body>
<div id="toolbar">
<label>Pilot tile: <select id="tile"></select></label>
<label>Timestamp: <select id="time"></select></label>
<label>Display:
<select id="metric">
<option value="forecast_index">Forecast screening index</option>
<option value="recent_index">Recent-rainfall screening index</option>
<option value="recent6">Recent rainfall (6h, mm)</option>
<option value="recent24">Recent rainfall (24h, mm)</option>
<option value="forecast6">Forecast rainfall (6h, mm)</option>
<option value="forecast24">Forecast rainfall (24h, mm)</option>
</select></label>
<span id="status"></span>
</div>
<div id="map"></div>
<div class="note">
Pilot sample points only. Screening indices are illustrative and
uncalibrated—not landslide probabilities or official warnings.
Gray markers mean the selected value is unavailable.
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const records = DATA_PLACEHOLDER;
const map = L.map("map", {preferCanvas:true});
L.tileLayer(
"https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
{attribution:"Tiles © Esri and its data providers",maxZoom:16}
).addTo(map);

const layer=L.layerGroup().addTo(map);
const tileSelect=document.getElementById("tile");
const timeSelect=document.getElementById("time");
const metricSelect=document.getElementById("metric");
const status=document.getElementById("status");

[...new Set(records.map(r=>r.tile))].sort().forEach(tile=>{
  const o=document.createElement("option");
  o.value=tile;o.textContent=tile;tileSelect.appendChild(o);
});

function fillTimes(){
  const tile=tileSelect.value;
  const tileRecords=records.filter(r=>r.tile===tile);
  const times=[...new Set(tileRecords.map(r=>r.time))].sort();

  timeSelect.replaceChildren();
  times.forEach(time=>{
    const o=document.createElement("option");
    o.value=time;o.textContent=time;timeSelect.appendChild(o);
  });

  // Pick the latest timestamp with at least one available
  // forecast index for this tile, not simply the last hour.
  const validTimes=times.filter(time=>
    tileRecords.some(r=>
      r.time===time && r.forecast_index!==null
    )
  );

  if(validTimes.length){
    timeSelect.value=validTimes[validTimes.length-1];
  }else if(times.length){
    timeSelect.value=times[times.length-1];
  }
}

function color(value,metric){
  if(value===null || !Number.isFinite(value)) return "#777777";
  if(metric.includes("index")){
    if(value>=30)return "#d7191c";
    if(value>=15)return "#fdae61";
    return "#2c7bb6";
  }
  if(value>=50)return "#7f0000";
  if(value>=25)return "#d73027";
  if(value>=10)return "#fc8d59";
  if(value>0)return "#91bfdb";
  return "#4575b4";
}

function fmt(value,unit=""){
  return value===null ? "Incomplete" : value.toFixed(2)+unit;
}

function renderMap(){
  layer.clearLayers();
  const tile=tileSelect.value;
  const time=timeSelect.value;
  const metric=metricSelect.value;

  const selected=records.filter(r=>r.tile===tile && r.time===time);
  const bounds=[];
  let available=0;

  selected.forEach(r=>{
    const value=r[metric];
    const valid=value!==null && Number.isFinite(value);
    if(valid)available++;

    const popup=`
    <b>Sample:</b> ${r.id}<br>
    <b>Tile:</b> ${r.tile}<br>
    <b>Timestamp:</b> ${r.time}<br>
    <b>Elevation:</b> ${fmt(r.elevation," m")}<br>
    <b>Slope:</b> ${fmt(r.slope,"°")}<br>
    <b>Recent rainfall 6h:</b> ${fmt(r.recent6," mm")}<br>
    <b>Recent rainfall 24h:</b> ${fmt(r.recent24," mm")}<br>
    <b>Forecast rainfall 6h:</b> ${fmt(r.forecast6," mm")}<br>
    <b>Forecast rainfall 24h:</b> ${fmt(r.forecast24," mm")}<br>
    <b>Recent index:</b> ${fmt(r.recent_index)}<br>
    <b>Forecast index:</b> ${fmt(r.forecast_index)}<br>
    <i>Experimental screening only; not a validated warning.</i>`;

    L.circleMarker([r.lat,r.lon],{
      radius:5,color:color(value,metric),
      fillColor:color(value,metric),fillOpacity:0.85,weight:1
    }).bindPopup(popup).bindTooltip(
      `${r.id}: ${valid?value.toFixed(2):"Incomplete"}`
    ).addTo(layer);

    bounds.push([r.lat,r.lon]);
  });

  if(bounds.length)map.fitBounds(bounds,{padding:[25,25],maxZoom:11});
  status.textContent=`${selected.length} locations · ${available} with data`;
}

tileSelect.addEventListener("change",()=>{fillTimes();renderMap()});
timeSelect.addEventListener("change",renderMap);
metricSelect.addEventListener("change",renderMap);

const legend=L.control({position:"bottomright"});
legend.onAdd=function(){
  const d=L.DomUtil.create("div","legend");
  d.innerHTML="<b>Screening index</b><br>"+
  '<span style="color:#2c7bb6">●</span> Below 15<br>'+
  '<span style="color:#fdae61">●</span> 15 to below 30<br>'+
  '<span style="color:#d7191c">●</span> 30 or above<br>'+
  '<span style="color:#777">●</span> Missing value';
  return d;
};
legend.addTo(map);

fillTimes();
renderMap();
</script>
</body>
</html>
""".replace("DATA_PLACEHOLDER", data_json)

    if HTML.exists():
        raise FileExistsError(
            f"Map already exists: {HTML}\n"
            "Close it, remove the old HTML, and rerun."
        )

    HTML.write_text(page, encoding="utf-8")
    print("Interactive map created.")
    print("Records:", len(records))
    print("Tiles:", sorted(df["pilot_tile"].unique()))
    print("Output:", HTML)


if __name__ == "__main__":
    main()
