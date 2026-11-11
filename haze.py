#!/usr/bin/env python3
"""
Haze / air-quality monitor for Malaysia.

Uses Open-Meteo's free air-quality API — no API key required, no signup,
reliable enough to run on a 15-minute cron.

Output is a plain-text report printed to stdout; the J.A.R.V.I.S. workflow
scheduler captures stdout and sends it to the chat panel.
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ------------------------------------------------------------------
# Force UTF-8 on stdout/stderr.
#
# Windows consoles and pipes (including the one main.js gives this script
# when it spawns it) often default to a legacy codepage (cp1252), which
# cannot encode the emoji used throughout this file's output (🌫️ ⚠️ ❌ 🟢🟡🟠🔴).
# Without this, the FIRST emoji print raises UnicodeEncodeError. Worse, that
# includes the "Monitor crashed: ..." message in main()'s except block —
# which is exactly how the previous failure showed only the crash header
# and lost the actual error message (the traceback went to stderr, and
# main.js's old `stdout || stderr` logic silently dropped stderr whenever
# stdout had any content).
#
# Same pattern already used at the top of ai_server.py.
# ------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Pin the DB to this script's own folder rather than the current working
# directory. When spawned from main.js, cwd is the Electron app folder,
# not the script's folder — so the DB was being created in two different
# places depending on how the script was launched ("works from terminal,
# fails on cron", or vice versa).
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haze_monitor.db")

# Cities to monitor, with lat/lon for Open-Meteo.
# Coverage: peninsular Malaysia (KL, Penang, Sungai Petani, JB),
# East Malaysia (Kuching, Miri in Sarawak; Kota Kinabalu in Sabah).
CITIES = [
    {"name": "Kuala Lumpur",  "lat":  3.1390, "lng": 101.6869},
    {"name": "Penang",        "lat":  5.4141, "lng": 100.3288},
    {"name": "Sungai Petani", "lat":  5.6470, "lng": 100.4877},
    {"name": "Johor Bahru",   "lat":  1.4927, "lng": 103.7414},
    {"name": "Kuching",       "lat":  1.5533, "lng": 110.3592},
    {"name": "Miri",          "lat":  4.3995, "lng": 113.9914},
    {"name": "Kota Kinabalu", "lat":  5.9804, "lng": 116.0735},
]

# WHO short-term (24h) guideline for PM2.5 is 15 µg/m³.
# Malaysia's DOE "unhealthy" API reading corresponds to roughly
# PM2.5 > 55 µg/m³, so we warn before that.
WHO_24H_GUIDELINE  = 15   # µg/m³
MALAYSIA_WARN      = 35   # µg/m³  — moderate
MALAYSIA_UNHEALTHY = 55   # µg/m³  — unhealthy

HTTP_TIMEOUT_SEC = 15


# ------------------------------------------------------------------
# Data fetch
# ------------------------------------------------------------------
def fetch_city(city):
    """Fetch current air quality for one city. Returns dict or None."""
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={city['lat']}&longitude={city['lng']}"
        "&current=pm2_5,pm10,european_aqi"
        "&timezone=Asia%2FKuala_Lumpur"
    )
    try:
        req = Request(url, headers={"User-Agent": "JARVIS-haze-monitor/1.0"})
        with urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"  ⚠️ {city['name']}: fetch failed ({e})")
        return None
    except json.JSONDecodeError as e:
        print(f"  ⚠️ {city['name']}: invalid JSON ({e})")
        return None

    current = payload.get("current") or {}
    pm25 = current.get("pm2_5")
    pm10 = current.get("pm10")
    aqi  = current.get("european_aqi")

    if pm25 is None:
        print(f"  ⚠️ {city['name']}: no PM2.5 value in response")
        return None

    return {
        "city": city["name"],
        "pm25": float(pm25),
        "pm10": float(pm10) if pm10 is not None else None,
        "aqi":  float(aqi)  if aqi  is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def fetch_all():
    """Fetch every city. Never raises — returns whatever succeeded."""
    print(f"🌫️ Fetching air quality for {len(CITIES)} cities...")
    results = []
    for city in CITIES:
        entry = fetch_city(city)
        if entry:
            results.append(entry)
    return results


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------
def save_to_database(readings):
    """Persist to SQLite. Index on (city, timestamp) keeps queries fast."""
    if not readings:
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS haze_readings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                city       TEXT NOT NULL,
                pm25       REAL NOT NULL,
                pm10       REAL,
                aqi        REAL,
                timestamp  TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_city_ts
            ON haze_readings (city, timestamp)
        """)
        cursor.executemany(
            "INSERT INTO haze_readings (city, pm25, pm10, aqi, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            [(r["city"], r["pm25"], r["pm10"], r["aqi"], r["timestamp"])
             for r in readings],
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"  ⚠️ DB write failed: {e}")
    finally:
        if conn:
            conn.close()


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------
def classify(pm25):
    """Return (label, emoji) for a PM2.5 level."""
    if pm25 <= WHO_24H_GUIDELINE:
        return "GOOD",      "🟢"
    if pm25 <= MALAYSIA_WARN:
        return "MODERATE",  "🟡"
    if pm25 <= MALAYSIA_UNHEALTHY:
        return "UNHEALTHY", "🟠"
    return "HAZARDOUS",     "🔴"


def build_report(readings):
    """Produce the human-readable report string that the scheduler shows."""
    lines = []
    lines.append("=" * 60)
    lines.append("MALAYSIA AIR QUALITY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    if not readings:
        lines.append("")
        lines.append("❌ No data could be retrieved from the air-quality API.")
        lines.append("   Check that this machine can reach air-quality-api.open-meteo.com")
        return "\n".join(lines)

    # Sort worst first so the most urgent city is at the top
    ordered = sorted(readings, key=lambda r: r["pm25"], reverse=True)

    alerts = []
    for r in ordered:
        label, emoji = classify(r["pm25"])
        pm10_part = (
            f", PM10: {r['pm10']:.0f} µg/m³"
            if r["pm10"] is not None else ""
        )
        lines.append(
            f"  {emoji} {r['city']:<16} PM2.5: {r['pm25']:>5.1f} µg/m³"
            f"{pm10_part}   [{label}]"
        )
        if r["pm25"] > MALAYSIA_WARN:
            alerts.append(
                f"⚠️ {r['city']} is {label} (PM2.5 {r['pm25']:.1f} µg/m³)"
            )

    lines.append("")
    lines.append(f"  Threshold: WHO 24h guideline = {WHO_24H_GUIDELINE} µg/m³")
    lines.append(
        f"  Highest PM2.5: {ordered[0]['city']} ({ordered[0]['pm25']:.1f} µg/m³)"
    )

    if alerts:
        lines.append("")
        lines.append("ALERTS:")
        lines.extend("  " + a for a in alerts)
    else:
        lines.append("")
        lines.append("✅ All monitored cities are below the warning threshold.")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    try:
        readings = fetch_all()
        save_to_database(readings)
        print(build_report(readings))
        return 0
    except Exception as e:
        # Any uncaught exception becomes a clean report line rather than
        # a traceback, so the workflow scheduler shows something useful.
        # The `repr(e)` also gives us the exception class, which is more
        # useful than str(e) alone when the message is empty (e.g. bare
        # "URLError: <urlopen error ...>" cases).
        print("=" * 60)
        print("MALAYSIA AIR QUALITY REPORT")
        print("=" * 60)
        print(f"❌ Monitor crashed: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())