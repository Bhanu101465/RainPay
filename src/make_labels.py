"""
Create labels based on official IMD (India Meteorological Department) rules.
Input: data/clean/weather_daily.csv
Output: data/clean/weather_labeled.csv
"""

import os
import pandas as pd

INPUT_FILE = os.path.join("data", "clean", "weather_daily.csv")
OUTPUT_FILE = os.path.join("data", "clean", "weather_labeled.csv")

# IMD official daily rainfall classification thresholds (mm):
# Heavy rain: 64.5 to 115.5 mm
# Very heavy rain: 115.6 to 204.4 mm
# Extremely heavy rain: >= 204.5 mm
HEAVY_THRESHOLD = 64.5
VERY_HEAVY_THRESHOLD = 115.6
EXTREMELY_HEAVY_THRESHOLD = 204.5


def make_labels():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run src/download_weather.py first.")

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Ensure sorted by city, spot, date
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    # 1. Rain intensity flags
    df["heavy_today"] = (df["rain_mm"] >= HEAVY_THRESHOLD).astype(int)
    df["very_heavy_today"] = (df["rain_mm"] >= VERY_HEAVY_THRESHOLD).astype(int)
    df["extremely_heavy_today"] = (df["rain_mm"] >= EXTREMELY_HEAVY_THRESHOLD).astype(int)

    # 2. Heavy tomorrow (target to predict): shifted by 1 day forward within each spot
    df["heavy_tomorrow"] = (
        df.groupby(["city", "spot"])["heavy_today"]
        .shift(-1)
        .astype("Int64")
    )

    # 3. Heat flag: 1 if temp_max is 42°C or more (simple version of the real rule)
    df["heat_flag"] = (df["temp_max"] >= 42.0).astype(int)

    # 4. Month (1 to 12)
    df["month"] = df["date"].dt.month

    # Format date back to string YYYY-MM-DD
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Save to clean directory
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully labeled data and saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    make_labels()

