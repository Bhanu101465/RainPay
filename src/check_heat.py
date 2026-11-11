"""
Check heat levels in Chennai and Visakhapatnam:
1. Counts for temp >= 37°C, >= 40°C, >= 42°C, and feels-like (Heat Index) >= 41°C.
2. Counts per spot and per city.
3. Hot days by month (heat seasonality).
4. Train (2000-2018) vs Test (2019-2025) distributions.
"""

import os
import numpy as np
import pandas as pd

FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")


def compute_heat_index(temp_c, rel_humidity):
    """
    Calculate NOAA/NWS Steadman-Rothfusz Heat Index (Feels-Like Temperature).
    Input: temp_c in Celsius, rel_humidity in %.
    Output: Feels-like temperature in Celsius.
    """
    tf = temp_c * 9.0 / 5.0 + 32.0
    rh = rel_humidity

    # Rothfusz regression equation
    hi_f = (
        -42.379
        + 2.04901523 * tf
        + 10.14333127 * rh
        - 0.22475541 * tf * rh
        - 0.00683783 * tf * tf
        - 0.05481717 * rh * rh
        + 0.00122874 * tf * tf * rh
        + 0.00085282 * tf * rh * rh
        - 0.00000199 * tf * tf * rh * rh
    )

    # Steadman simple formula for cooler temperatures (tf < 80°F)
    simple = 0.5 * (tf + 61.0 + ((tf - 68.0) * 1.2) + (rh * 0.094))
    hi_final = np.where(tf < 80.0, simple, hi_f)
    return (hi_final - 32.0) * 5.0 / 9.0


def run_heat_checks():
    if not os.path.exists(FILE):
        raise FileNotFoundError(f"Missing {FILE}.")

    df = pd.read_csv(FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Compute feels-like temperature using NOAA Heat Index formula
    df["feels_like"] = compute_heat_index(df["temp_max"], df["humidity_mean"])

    # Threshold flags
    df["heat_37"] = (df["temp_max"] >= 37.0).astype(int)
    df["heat_40"] = (df["temp_max"] >= 40.0).astype(int)
    df["heat_42"] = (df["temp_max"] >= 42.0).astype(int)
    df["feels_41"] = (df["feels_like"] >= 41.0).astype(int)

    # =========================================================================
    # 1. COUNTS PER SPOT AND PER CITY
    # =========================================================================
    print("=" * 85)
    print("1. HEAT COUNTS PER SPOT (2000 - 2025: 9,497 DAYS PER SPOT)")
    print("=" * 85)

    spot_rows = []
    for (city, spot), group in df.groupby(["city", "spot"]):
        n = len(group)
        c37 = int(group["heat_37"].sum())
        c40 = int(group["heat_40"].sum())
        c42 = int(group["heat_42"].sum())
        fl41 = int(group["feels_41"].sum())

        spot_rows.append({
            "City": city,
            "Spot": spot,
            "Total Days": n,
            ">= 37°C": f"{c37} ({c37/n*100:.1f}%)",
            ">= 40°C": f"{c40} ({c40/n*100:.2f}%)",
            ">= 42°C": f"{c42} ({c42/n*100:.2f}%)",
            "Feels >= 41°C": f"{fl41} ({fl41/n*100:.1f}%)",
        })

    print(pd.DataFrame(spot_rows).to_string(index=False))

    print("\n" + "-" * 85)
    print("CITY TOTALS")
    print("-" * 85)
    city_rows = []
    for city in sorted(df["city"].unique()):
        sub = df[df["city"] == city]
        n = len(sub)
        c37 = int(sub["heat_37"].sum())
        c40 = int(sub["heat_40"].sum())
        c42 = int(sub["heat_42"].sum())
        fl41 = int(sub["feels_41"].sum())
        city_rows.append({
            "City": city,
            "Total Spot-Days": n,
            ">= 37°C": f"{c37} ({c37/n*100:.2f}%)",
            ">= 40°C": f"{c40} ({c40/n*100:.2f}%)",
            ">= 42°C": f"{c42} ({c42/n*100:.2f}%)",
            "Feels >= 41°C": f"{fl41} ({fl41/n*100:.1f}%)",
        })
    print(pd.DataFrame(city_rows).to_string(index=False))

    # =========================================================================
    # 2. HOT DAYS BY MONTH (SEASONALITY)
    # =========================================================================
    print("\n" + "=" * 85)
    print("2. HOT DAYS BY MONTH (SEASONALITY ACROSS BOTH CITIES)")
    print("=" * 85)

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    monthly_list = []
    for m in range(1, 13):
        m_df = df[df["month"] == m]
        monthly_list.append({
            "Month": month_names[m],
            "Chennai >=37°C": int(m_df[m_df["city"] == "Chennai"]["heat_37"].sum()),
            "Chennai >=40°C": int(m_df[m_df["city"] == "Chennai"]["heat_40"].sum()),
            "Vizag >=37°C": int(m_df[m_df["city"] == "Visakhapatnam"]["heat_37"].sum()),
            "Vizag >=40°C": int(m_df[m_df["city"] == "Visakhapatnam"]["heat_40"].sum()),
            "Both >=40°C": int(m_df["heat_40"].sum()),
            "Both >=42°C": int(m_df["heat_42"].sum()),
        })

    print(pd.DataFrame(monthly_list).to_string(index=False))

    # =========================================================================
    # 3. TRAIN (2000-2018) vs TEST (2019-2025) COUNTS
    # =========================================================================
    print("\n" + "=" * 85)
    print("3. TIME-BASED SPLIT (TRAIN: 2000-2018 vs TEST: 2019-2025)")
    print("=" * 85)

    train_df = df[df["date"].dt.year <= 2018]
    test_df = df[df["date"].dt.year >= 2019]

    for city in sorted(df["city"].unique()):
        tr = train_df[train_df["city"] == city]
        te = test_df[test_df["city"] == city]

        print(f"\n{city}:")
        print(f"  Train (2000-2018: {len(tr)} spot-days):")
        print(f"    - Days >= 37°C: {int(tr['heat_37'].sum())} ({(tr['heat_37'].mean()*100):.2f}%)")
        print(f"    - Days >= 40°C: {int(tr['heat_40'].sum())} ({(tr['heat_40'].mean()*100):.2f}%)")
        print(f"    - Days >= 42°C: {int(tr['heat_42'].sum())} ({(tr['heat_42'].mean()*100):.2f}%)")
        print(f"    - Feels >= 41°C: {int(tr['feels_41'].sum())} ({(tr['feels_41'].mean()*100):.1f}%)")

        print(f"  Test  (2019-2025: {len(te)} spot-days):")
        print(f"    - Days >= 37°C: {int(te['heat_37'].sum())} ({(te['heat_37'].mean()*100):.2f}%)")
        print(f"    - Days >= 40°C: {int(te['heat_40'].sum())} ({(te['heat_40'].mean()*100):.2f}%)")
        print(f"    - Days >= 42°C: {int(te['heat_42'].sum())} ({(te['heat_42'].mean()*100):.2f}%)")
        print(f"    - Feels >= 41°C: {int(te['feels_41'].sum())} ({(te['feels_41'].mean()*100):.1f}%)")


if __name__ == "__main__":
    run_heat_checks()

