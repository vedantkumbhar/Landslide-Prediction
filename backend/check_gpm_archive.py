
import requests

URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

params = {
    "collection_concept_id": "C2723758340-GES_DISC",
    "page_size": 5,
    "sort_key": "-start_date"
}

print("Searching NASA GPM IMERG Early V07 granules...")

response = requests.get(URL, params=params, timeout=60)
response.raise_for_status()

data = response.json()
granules = data.get("feed", {}).get("entry", [])

print("HTTP STATUS:", response.status_code)
print("GRANULES FOUND:", len(granules))

for i, granule in enumerate(granules, 1):
    print(f"\nGRANULE {i}")
    print("Title:", granule.get("title"))
    print("Start:", granule.get("time_start"))
    print("End:", granule.get("time_end"))

    for link in granule.get("links", []):
        if link.get("rel", "").endswith("/data#"):
            print("Data URL:", link.get("href"))
            break

if not granules:
    print("No granules returned. Check collection availability.")
