"""
Download daily historical weather data for Visakhapatnam spots.
Source: Open-Meteo Historical Weather API (archive-api.open-meteo.com)
"""

import os
import json
import time
import socket
import requests
import pandas as pd

# Force IPv4 to prevent unreachable NAT64 IPv6 connection timeouts on Windows
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in res if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else res
socket.getaddrinfo = _ipv4_getaddrinfo

SPOTS = [
    {"city": "Visakhapatnam", "spot": "Madhurawada", "lat": 17.80, "lon": 83.37},
    {"city": "Visakhapatnam", "spot": "Pendurthi", "lat": 17.82, "lon": 83.20},
    {"city": "Visakhapatnam", "spot": "Gajuwaka", "lat": 17.69, "lon": 83.21},
    {"city": "Visakhapatnam", "spot": "Maddilapalem", "lat": 17.74, "lon": 83.32},
]

START_YEAR = 2000
END_YEAR = 2025

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
CLEAN_FILE = os.path.join(CLEAN_DIR, "vizag_daily.csv")

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_spot_year(lat, lon, year):
    """Fetch 1 year of daily data for a spot with retries."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": [
            "precipitation_sum",
            "temperature_2m_max",
            "wind_speed_10m_max",
            "relative_humidity_2m_mean",
        ],
        "timezone": "Asia/Kolkata",
    }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=(10, 35))
            resp.raise_for_status()
            data = resp.json()
            if "daily" in data and "time" in data["daily"]:
                return data["daily"]
            raise ValueError(f"Malformed response for year {year}")
        except Exception as err:
            print(f"    [Year {year}] Attempt {attempt}/{max_retries} failed: {err}", flush=True)
            if attempt == max_retries:
                raise err
            time.sleep(3 * attempt)


def fetch_spot_weather(spot_info):
    """Fetch full weather series (2000-2025) for one spot or load from cache."""
    city = spot_info["city"]
    spot = spot_info["spot"]
    lat = spot_info["lat"]
    lon = spot_info["lon"]

    safe_name = f"vizag_{spot}".lower().replace(" ", "_")
    raw_file = os.path.join(RAW_DIR, f"{safe_name}.json")

    # Check if already cached
    if os.path.exists(raw_file):
        print(f"Loading cached raw data for {city} - {spot} from {raw_file}", flush=True)
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    print(f"\nDownloading 2000-2025 data for {city} - {spot} ({lat}, {lon})...", flush=True)

    combined_daily = {
        "time": [],
        "precipitation_sum": [],
        "temperature_2m_max": [],
        "wind_speed_10m_max": [],
        "relative_humidity_2m_mean": [],
    }

    for year in range(START_YEAR, END_YEAR + 1):
        t0 = time.time()
        year_daily = fetch_spot_year(lat, lon, year)
        for key in combined_daily:
            combined_daily[key].extend(year_daily.get(key, []))
        print(f"  {spot} {year}: fetched {len(year_daily.get('time', []))} days in {time.time()-t0:.1f}s", flush=True)
        time.sleep(0.3)  # Gentle delay between requests

    data = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Kolkata",
        "daily": combined_daily,
    }

    # Save to data/raw
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(combined_daily['time'])} total days to {raw_file}", flush=True)

    return data


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    all_rows = []

    for spot_info in SPOTS:
        city = spot_info["city"]
        spot = spot_info["spot"]
        data = fetch_spot_weather(spot_info)

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        rain = daily.get("precipitation_sum", [])
        temp_max = daily.get("temperature_2m_max", [])
        wind_max = daily.get("wind_speed_10m_max", [])
        humidity = daily.get("relative_humidity_2m_mean", [])

        for i in range(len(dates)):
            all_rows.append({
                "city": city,
                "spot": spot,
                "date": dates[i],
                "rain_mm": rain[i] if i < len(rain) else None,
                "temp_max": temp_max[i] if i < len(temp_max) else None,
                "wind_max": wind_max[i] if i < len(wind_max) else None,
                "humidity_mean": humidity[i] if i < len(humidity) else None,
            })

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSuccessfully created {CLEAN_FILE} with {len(df)} total rows.", flush=True)
    print(df.head(), flush=True)


if __name__ == "__main__":
    main()
