
import requests
from datetime import datetime, timedelta, timezone

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_ID = "C2723758340-GES_DISC"

# Reference endpoint from the failed refresh
END_TIME = datetime(
    2026, 11, 3, 9, 0, tzinfo=timezone.utc
)

WINDOW_HOURS = 3
WINDOW_COUNT = 8
EXPECTED_GRANULES = 6


def search_granules(start_time, end_time):
    params = {
        "collection_concept_id": COLLECTION_ID,
        "temporal": (
            f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')},"
            f"{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        ),
        "page_size": 100,
        "sort_key": "start_date",
    }

    response = requests.get(
        CMR_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("feed", {}).get("entry", [])


def inspect_window(start_time, end_time):
    print("\n" + "=" * 65)
    print("WINDOW")
    print("=" * 65)

    print("Start:", start_time.isoformat())
    print("End:  ", end_time.isoformat())

    entries = search_granules(start_time, end_time)

    # Keep only granules whose start timestamps belong
    # to this half-open interval.
    entries = [
        item
        for item in entries
        if item.get("time_start")
        and start_time
        <= datetime.fromisoformat(
            item["time_start"].replace("Z", "+00:00")
        )
        < end_time
    ]

    entries.sort(key=lambda item: item["time_start"])

    timestamps = [
        datetime.fromisoformat(
            item["time_start"].replace("Z", "+00:00")
        )
        for item in entries
    ]

    print(f"\nGranules found: {len(entries)}")
    print(f"Expected: {EXPECTED_GRANULES}")

    for item in entries:
        print(
            item.get("time_start"),
            "|",
            item.get("producer_granule_id"),
        )

    complete = len(timestamps) == EXPECTED_GRANULES

    if complete:
        for index in range(1, len(timestamps)):
            difference = timestamps[index] - timestamps[index - 1]

            if difference != timedelta(minutes=30):
                complete = False
                break

    if complete:
        print("\nSTATUS: COMPLETE 3-HOUR WINDOW")
    else:
        print("\nSTATUS: INCOMPLETE WINDOW")

    return complete, entries


def main():
    print("=" * 65)
    print("NASA GPM IMERG EARLY RUN - CMR AVAILABILITY CHECK")
    print("=" * 65)

    print("Collection:", COLLECTION_ID)
    print("Reference endpoint:", END_TIME.isoformat())

    end_time = END_TIME
    complete_windows = []

    for index in range(WINDOW_COUNT):
        start_time = end_time - timedelta(hours=WINDOW_HOURS)

        complete, entries = inspect_window(
            start_time,
            end_time,
        )

        if complete:
            complete_windows.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "granules": len(entries),
                }
            )

        end_time = start_time

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    if complete_windows:
        print("\nComplete windows found:")

        for window in complete_windows:
            print(
                window["start"].isoformat(),
                "to",
                window["end"].isoformat(),
                "|",
                window["granules"],
                "granules",
            )

        latest = complete_windows[0]

        print("\nLatest complete 3-hour window:")
        print("Start:", latest["start"].isoformat())
        print("End:  ", latest["end"].isoformat())

    else:
        print("\nNo complete 3-hour window found.")

    print("\nAvailability check finished.")


if __name__ == "__main__":
    main()
