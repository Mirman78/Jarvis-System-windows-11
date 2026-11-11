import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

RATE_API = "https://open.er-api.com/v6/latest/MYR"
CACHE_FILE = "rates_cache.json"
CACHE_TTL = 60 * 60 * 6  # 6 hours; source updates ~daily
TARGETS = ("USD", "GBP", "IDR")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if time.time() - cached["fetched_at"] < CACHE_TTL:
            return cached["data"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    return None


def _save_cache(data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.time(), "data": data}, f)


def get_exchange_rates(use_cache=True):
    """Return {'base', 'date', 'rates'} or None on failure."""
    if use_cache:
        cached = _load_cache()
        if cached:
            return cached

    try:
        resp = requests.get(RATE_API, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"Network error: {e}")
        return None
    except ValueError:
        print("Response was not valid JSON.")
        return None

    if data.get("result") != "success" or not isinstance(data.get("rates"), dict):
        print(f"API returned an error: {data.get('error-type', 'unknown')}")
        return None

    rates = {code: data["rates"][code] for code in TARGETS if code in data["rates"]}

    result = {
        "base": data.get("base_code", "MYR"),
        "date": data.get("time_last_update_utc", "unknown"),
        "rates": rates,
    }
    _save_cache(result)
    return result


def format_output(rates, reverse=False):
    if not rates:
        return

    base = rates["base"]
    print("=" * 50)
    print(f"Exchange rates — updated {rates['date']}")
    print("=" * 50)

    for code, rate in rates["rates"].items():
        if reverse:
            print(f"1 {code} = {1 / rate:.4f} {base}")
        else:
            print(f"1 {base} = {rate:.4f} {code}")

    print("=" * 50)
    print(f"Local time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")


def main():
    parser = argparse.ArgumentParser(description="Show MYR exchange rates.")
    parser.add_argument("--reverse", action="store_true",
                        help="Show 1 USD/GBP/IDR = X MYR instead")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force a fresh fetch")
    args = parser.parse_args()

    rates = get_exchange_rates(use_cache=not args.no_cache)
    format_output(rates, reverse=args.reverse)


if __name__ == "__main__":
    main()
