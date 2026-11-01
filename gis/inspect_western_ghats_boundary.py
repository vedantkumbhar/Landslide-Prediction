import json
from pathlib import Path

path = Path("data/raw/boundaries/western_ghats_esa_outline.json")
data = json.loads(path.read_text(encoding="utf-8"))


def coordinates(value):
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield value[0], value[1]
    elif isinstance(value, list):
        for item in value:
            yield from coordinates(item)


for feature in data["features"]:
    points = list(coordinates(feature["geometry"]["coordinates"]))
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]

    print(
        feature["properties"]["gz_state"],
        "| longitude:", round(min(longitudes), 4), "to", round(max(longitudes), 4),
        "| latitude:", round(min(latitudes), 4), "to", round(max(latitudes), 4),
        "| points:", len(points),
    )
