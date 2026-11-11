import re
import json
from playwright.sync_api import sync_playwright

MAIN_ROW = re.compile(
    r"^(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s+[AP]M)\s+"
    r"(.+?)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)\s*$"
)

BARAJ_ROW = re.compile(
    r"^(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s+[AP]M)\s+"
    r"([\d.]+)\s+"
    r"([\d,]+\.?\d*)\s+"
    r"([\d.]+)\s*$"
)


def scrape():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://iwrims.luas.gov.my",
                  wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(5000)
        text = page.inner_text("body")
        browser.close()

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    dams = []
    baraj = []
    section = None

    for line in lines:
        if "MAKLUMAT EMPANGAN" in line:
            section = "dams"
            continue
        if "MAKLUMAT STESEN PARAS AIR DAN HUJAN" in line:
            section = "baraj"
            continue

        if section == "dams":
            m = MAIN_ROW.match(line)
            if m:
                dt, station, level, vol, pct, rain = m.groups()
                dams.append({
                    "date_time": dt,
                    "station": station.strip(),
                    "water_level_m": float(level),
                    "volume_mm3": float(vol),
                    "percentage": float(pct),
                    "rainfall_mm": float(rain),
                })
        elif section == "baraj":
            m = BARAJ_ROW.match(line)
            if m:
                dt, level, flow, rain = m.groups()
                baraj.append({
                    "date_time": dt,
                    "water_level_m": float(level),
                    "flow_jlh": float(flow.replace(",", "")),
                    "rainfall_mm": float(rain),
                })

    return dams, baraj


def main():
    dams, baraj = scrape()

    print("--- Dam Water Levels (IWRIMS) ---")
    for d in dams:
        print(f"[{d['date_time']}] {d['station']}: "
              f"Level {d['water_level_m']}m, "
              f"Volume {d['volume_mm3']}Mm3, "
              f"{d['percentage']}%, "
              f"Rain {d['rainfall_mm']}mm")
    
if __name__ == "__main__":
    main()