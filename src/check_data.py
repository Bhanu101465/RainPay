"""
Check labeled weather data:
1. Rain frequency per spot (heavy, very heavy, extremely heavy).
2. Rain seasonality by month for each city.
3. Train/test split counts (Train: <= 2018, Test: 2019-2025).
"""

import os
import pandas as pd

LABELED_FILE = os.path.join("data", "clean", "weather_labeled.csv")


def run_checks():
    if not os.path.exists(LABELED_FILE):
        raise FileNotFoundError(f"Missing {LABELED_FILE}. Run src/make_labels.py first.")

    df = pd.read_csv(LABELED_FILE)
    df["date"] = pd.to_datetime(df["date"])

    print("=" * 80)
    print("1. SUMMARY TABLE BY SPOT")
    print("=" * 80)

    spot_summary = []
    for (city, spot), group in df.groupby(["city", "spot"]):
        total_days = len(group)
        heavy_days = int(group["heavy_today"].sum())
        heavy_pct = (heavy_days / total_days) * 100

        very_heavy_days = int(group["very_heavy_today"].sum())
        very_heavy_pct = (very_heavy_days / total_days) * 100

        extreme_days = int(group["extremely_heavy_today"].sum())
        extreme_pct = (extreme_days / total_days) * 100

        spot_summary.append({
            "City": city,
            "Spot": spot,
            "Total Days": total_days,
            "Heavy Days (>=64.5mm)": heavy_days,
            "Heavy %": f"{heavy_pct:.2f}%",
            "Very Heavy (>=115.6mm)": very_heavy_days,
            "Very Heavy %": f"{very_heavy_pct:.2f}%",
            "Extremely Heavy (>=204.5mm)": extreme_days,
            "Extreme %": f"{extreme_pct:.2f}%",
        })

    summary_df = pd.DataFrame(spot_summary)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. HEAVY RAIN DAYS BY MONTH (RAIN SEASONS)")
    print("=" * 80)

    # Month names for readable display
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df["month_name"] = df["month"].map(month_names)

    monthly = df.groupby(["month", "city"])["heavy_today"].sum().unstack(fill_value=0)
    monthly.index = [month_names[m] for m in monthly.index]
    print(monthly)

    print("\n" + "=" * 80)
    print("3. TIME-BASED TRAIN / TEST SPLIT (Train <= 2018, Test: 2019-2025)")
    print("=" * 80)

    train_df = df[df["date"].dt.year <= 2018]
    test_df = df[(df["date"].dt.year >= 2019) & (df["date"].dt.year <= 2025)]

    for city in sorted(df["city"].unique()):
        train_city = train_df[train_df["city"] == city]
        test_city = test_df[test_df["city"] == city]

        train_heavy = int(train_city["heavy_today"].sum())
        train_total = len(train_city)
        test_heavy = int(test_city["heavy_today"].sum())
        test_total = len(test_city)

        print(f"\nCity: {city}")
        print(f"  Train (2000-2018): {train_heavy} heavy days out of {train_total} total days ({(train_heavy/train_total)*100:.2f}%)")
        print(f"  Test  (2019-2025): {test_heavy} heavy days out of {test_total} total days ({(test_heavy/test_total)*100:.2f}%)")

    print("\n" + "=" * 80)
    print("4. SANITY / ANOMALY CHECKS")
    print("=" * 80)
    # Check for missing values
    null_counts = df[["rain_mm", "temp_max", "wind_max", "humidity_mean"]].isnull().sum()
    print("Missing values in weather columns:")
    for col, count in null_counts.items():
        print(f"  - {col}: {count} missing")

    # Check heat flag counts
    heat_days = df.groupby(["city", "spot"])["heat_flag"].sum()
    print("\nHeat flag days (temp_max >= 42 C):")
    print(heat_days.to_string())


if __name__ == "__main__":
    run_checks()

