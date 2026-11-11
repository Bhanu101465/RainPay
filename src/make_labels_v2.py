"""
Combine Chennai and Visakhapatnam daily weather data and make labels.
Input: data/clean/weather_daily.csv (Chennai rows) & data/clean/vizag_daily.csv
Output: data/clean/weather_labeled_v2.csv
"""

import os
import pandas as pd

WEATHER_FILE = os.path.join("data", "clean", "weather_daily.csv")
VIZAG_FILE = os.path.join("data", "clean", "vizag_daily.csv")
OUTPUT_FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")

# Thresholds
HEAVY_THRESHOLD = 64.5        # IMD heavy rainfall threshold (mm)
WORK_STOP_THRESHOLD = 35.0    # Lower work-stopping rain threshold (mm)
HEAT_THRESHOLD = 42.0         # Heat threshold (deg C)


def make_labels_v2():
    if not os.path.exists(WEATHER_FILE):
        raise FileNotFoundError(f"Missing {WEATHER_FILE}.")
    if not os.path.exists(VIZAG_FILE):
        raise FileNotFoundError(f"Missing {VIZAG_FILE}.")

    # Load Chennai from weather_daily.csv (exclude Bengaluru)
    df_weather = pd.read_csv(WEATHER_FILE)
    df_chennai = df_weather[df_weather["city"] == "Chennai"].copy()

    # Load Visakhapatnam
    df_vizag = pd.read_csv(VIZAG_FILE).copy()

    # Combine both cities
    df = pd.concat([df_chennai, df_vizag], ignore_index=True)

    # Sort strictly by city, spot, and date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    # Make labels
    df["heavy_today"] = (df["rain_mm"] >= HEAVY_THRESHOLD).astype(int)
    df["work_stop_today"] = (df["rain_mm"] >= WORK_STOP_THRESHOLD).astype(int)
    df["very_heavy_today"] = (df["rain_mm"] >= 115.6).astype(int)
    df["extremely_heavy_today"] = (df["rain_mm"] >= 204.5).astype(int)
    df["heat_flag"] = (df["temp_max"] >= HEAT_THRESHOLD).astype(int)
    df["month"] = df["date"].dt.month

    # Shift heavy_today by 1 day forward within the SAME spot only.
    # The last day of each spot will have NaN (no made-up label).
    df["heavy_tomorrow"] = (
        df.groupby(["city", "spot"])["heavy_today"]
        .shift(-1)
        .astype("Int64")
    )

    # =========================================================================
    # STEP 2: SAFETY CHECKS
    # =========================================================================
    num_spots = df.groupby(["city", "spot"]).ngroups
    assert num_spots == 8, f"Expected 8 spots (4 Chennai + 4 Vizag), got {num_spots}"

    # Check 1: Exactly 1 NaN per spot for heavy_tomorrow (the last date)
    nan_count = int(df["heavy_tomorrow"].isna().sum())
    assert nan_count == num_spots, f"Expected {num_spots} NaNs, found {nan_count}"

    # Check 2: The NaNs occur strictly at the last day of each spot
    last_indices = df.groupby(["city", "spot"]).tail(1).index
    assert df.loc[last_indices, "heavy_tomorrow"].isna().all(), "Last day of each spot must be NaN"

    # Check 3: heavy_tomorrow never bleeds or jumps across spots
    same_spot = (df["spot"] == df["spot"].shift(-1)) & (df["city"] == df["city"].shift(-1))
    matches = (df.loc[same_spot, "heavy_tomorrow"] == df.loc[same_spot, "heavy_today"].shift(-1)).all()
    assert matches, "heavy_tomorrow must strictly match next day inside the same spot"

    # Save to clean CSV
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {OUTPUT_FILE} with {len(df)} total rows.")
    print(f"Safety check verified: 8 spots, no label jumping, final day of each spot is NA.")


if __name__ == "__main__":
    make_labels_v2()

