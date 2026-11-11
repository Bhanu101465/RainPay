"""
Download daily historical weather data for Bengaluru and Chennai spots.
Source: Open-Meteo Historical Weather API (archive-api.open-meteo.com)
"""

import os
import json
import time
import requests
import pandas as pd

SPOTS = [
    # Bengaluru
    {"city": "Bengaluru", "spot": "Koramangala", "lat": 12.935, "lon": 77.625},
    {"city": "Bengaluru", "spot": "Silk Board", "lat": 12.918, "lon": 77.623},
    {"city": "Bengaluru", "spot": "Whitefield", "lat": 12.970, "lon": 77.750},
    {"city": "Bengaluru", "spot": "Yelahanka", "lat": 13.101, "lon": 77.596},
    # Chennai
    {"city": "Chennai", "spot": "Velachery", "lat": 12.982, "lon": 80.218},
    {"city": "Chennai", "spot": "Adyar", "lat": 13.007, "lon": 80.257},
    {"city": "Chennai", "spot": "Anna Nagar", "lat": 13.085, "lon": 80.210},
    {"city": "Chennai", "spot": "Tambaram", "lat": 12.925, "lon": 80.100},
]

START_DATE = "2000-01-01"
END_DATE = "2025-12-31"

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
CLEAN_FILE = os.path.join(CLEAN_DIR, "weather_daily.csv")

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_spot_weather(spot_info):
    """Fetch weather for one spot or load from raw file if already saved."""
    city = spot_info["city"]
    spot = spot_info["spot"]
    lat = spot_info["lat"]
    lon = spot_info["lon"]

    safe_name = f"{city}_{spot}".lower().replace(" ", "_")
    raw_file = os.path.join(RAW_DIR, f"{safe_name}.json")

    # Check if already downloaded
    if os.path.exists(raw_file):
        print(f"Loading cached raw data for {city} - {spot} from {raw_file}")
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    print(f"Downloading data for {city} - {spot} ({lat}, {lon})...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
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
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as err:
            print(f"  Attempt {attempt}/{max_retries} failed for {spot}: {err}")
            if attempt == max_retries:
                raise err
            time.sleep(3 * attempt)

    # Save to data/raw
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw data to {raw_file}")

    # Friendly pause between requests to respect API rate limits
    time.sleep(2)
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
    # Ensure correct data types
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSuccessfully created {CLEAN_FILE} with {len(df)} total rows.")
    print(df.head())


if __name__ == "__main__":
    main()
