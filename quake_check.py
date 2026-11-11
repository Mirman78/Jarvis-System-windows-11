import hashlib
import json
import os
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

# Where we remember which quake IDs have already been reported, so a
# 15-minute cron on a short lookback window doesn't repeat the same
# events over and over in the chat log. Shared across both data sources
# below, with a prefix per source so their IDs can never collide.
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".quake_check_state.json")
MAX_REMEMBERED_IDS = 1000

# How far back to look each run. USGS's global catalog and MetMalaysia's
# local network don't always publish on the same delay, and this script
# only runs when the app is actually open - if the machine was off or the
# app wasn't running when an event happened, a 6-hour window won't reach
# back far enough to ever catch it once that window has passed. Widening
# this doesn't fix "the app wasn't running," but it gives more slack for
# gaps between runs (missed cron ticks, restarts, etc).
DEFAULT_HOURS_BACK = 12


def load_seen_ids() -> set:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("seen_ids", []))
    except (OSError, json.JSONDecodeError):
        return set()


def save_seen_ids(ids: set) -> None:
    try:
        trimmed = list(ids)[-MAX_REMEMBERED_IDS:]
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"seen_ids": trimmed}, f)
    except OSError as e:
        print(f"[WARN] Could not save state file: {e}")


# Southeast Asia Bounding Box (Approximate)
# Lat: -11°N to 25°N, Lon: 90°E to 150°E
# (Indonesia spans roughly -11 to 6 degrees latitude, so this has to
# reach well south of the equator to cover it, and it also comfortably
# covers all of Malaysia, Philippines, Thailand, Vietnam, etc.)
REGION = {
    "min_lat": -11.0,
    "max_lat": 25.0,
    "min_lon": 90.0,
    "max_lon": 150.0
}


class USGSEarthquakeMonitor:
    """
    Fetches recent seismic activity in the region from the USGS
    Earthquake API. USGS has good global coverage but tends to only
    reliably carry moderate-to-large events (roughly M4.5+) - smaller
    quakes detected by local/national networks often never appear here.
    """

    def __init__(self):
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def fetch(self, hours_back: int) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        start_time = (now - timedelta(hours=hours_back)).strftime('%Y-%m-%dT%H:%M:%S')

        params = {
            "format": "geojson",
            "starttime": start_time,
            "minlatitude": REGION["min_lat"],
            "maxlatitude": REGION["max_lat"],
            "minlongitude": REGION["min_lon"],
            "maxlongitude": REGION["max_lon"]
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            events = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                time_ms = props.get("time")
                if time_ms:
                    when = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)
                    date_str = when.strftime('%Y-%m-%d %H:%M:%S UTC')
                else:
                    date_str = "Unknown time"
                events.append({
                    "source": "USGS",
                    "fingerprint": "usgs:" + str(feature.get("id", "")),
                    "magnitude": props.get("mag"),
                    "place": props.get("place") or "Unknown location",
                    "date_str": date_str,
                })
            return events
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Could not reach USGS API: {e}")
            return []


class MetMalaysiaEarthquakeMonitor:
    """
    Fetches recent earthquake warnings from MetMalaysia via Malaysia's
    official open data API (data.gov.my). USGS frequently misses
    smaller local-network detections in Malaysia (e.g. low-to-mid M3
    quakes), so this fills that gap specifically for the region.
    """

    def __init__(self):
        self.base_url = "https://api.data.gov.my/weather/warning/earthquake"

    def fetch(self, hours_back: int) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        start_time = (now - timedelta(hours=hours_back)).strftime('%Y-%m-%d %H:%M:%S')

        params = {
            "timestamp_start": f"{start_time}@utcdatetime",
            "sort": "-utcdatetime",
            "limit": 50,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list):
                # Some data.gov.my endpoints wrap results in {"data": [...]}
                records = records.get("data", []) if isinstance(records, dict) else []

            events = []
            for rec in records:
                lat = rec.get("lat")
                lon = rec.get("lon")
                # Filter to the same regional bounding box as USGS, since
                # this endpoint isn't limited to Malaysia's own borders
                # (it also reports large regional quakes felt in Malaysia).
                if lat is None or lon is None:
                    continue
                if not (REGION["min_lat"] <= lat <= REGION["max_lat"] and
                        REGION["min_lon"] <= lon <= REGION["max_lon"]):
                    continue

                utc_str = rec.get("utcdatetime", "")
                local_str = rec.get("localdatetime", "")
                place = rec.get("location_original") or rec.get("location") or "Unknown location"
                fingerprint_raw = f"{utc_str}|{lat}|{lon}"
                events.append({
                    "source": "MetMalaysia",
                    "fingerprint": "metmy:" + hashlib.sha1(fingerprint_raw.encode("utf-8")).hexdigest(),
                    "magnitude": rec.get("magdefault"),
                    "place": place,
                    "date_str": f"{local_str} local ({utc_str} UTC)" if local_str else (utc_str or "Unknown time"),
                })
            return events
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Could not reach MetMalaysia (data.gov.my) API: {e}")
            return []


def display_report(events: List[Dict[str, Any]], hours_back: int) -> None:
    """
    Formats and prints the combined seismic report, showing only events
    not already reported in a previous run.
    """
    seen_ids = load_seen_ids()
    new_events = [e for e in events if e["fingerprint"] not in seen_ids]

    print(f"--- Seismic Activity Report (Past {hours_back} Hours) ---")
    print("Region: Southeast Asia | Sources: USGS, MetMalaysia (data.gov.my)")
    print("-" * 50)

    if not new_events:
        if events:
            print("No new activity since the last check (all events already reported).")
        else:
            print(f"No significant activity reported in Southeast Asia in the last {hours_back} hours.")
    else:
        for i, event in enumerate(new_events, 1):
            print(f"{i}. Magnitude: {event['magnitude']} | Location: {event['place']} | Source: {event['source']}")
            print(f"   Time: {event['date_str']}")
            print("-" * 50)

    # Remember every event fingerprint seen this run (not just the new
    # ones) so nothing gets reported twice on the next pass.
    seen_ids.update(e["fingerprint"] for e in events)
    save_seen_ids(seen_ids)


if __name__ == "__main__":
    # Requirements: pip install requests
    hours_back = DEFAULT_HOURS_BACK
    all_events: List[Dict[str, Any]] = []
    all_events.extend(USGSEarthquakeMonitor().fetch(hours_back))
    all_events.extend(MetMalaysiaEarthquakeMonitor().fetch(hours_back))
    display_report(all_events, hours_back)
