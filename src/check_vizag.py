"""
Check Visakhapatnam weather data:
1. Count days with rain >= 64.5 mm (IMD heavy) and rain >= 35.0 mm (work-stopping).
2. Count heavy days by month (64.5 mm rule).
3. Train/Test split counts (Train: 2000-2018, Test: 2019-2025).
"""

import os
import pandas as pd

VIZAG_FILE = os.path.join("data", "clean", "vizag_daily.csv")


def run_checks():
    if not os.path.exists(VIZAG_FILE):
        raise FileNotFoundError(f"Missing {VIZAG_FILE}. Run src/download_vizag.py first.")

    df = pd.read_csv(VIZAG_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["heavy_64_5"] = (df["rain_mm"] >= 64.5).astype(int)
    df["heavy_35_0"] = (df["rain_mm"] >= 35.0).astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    print("=" * 80)
    print("1. VISAKHAPATNAM SUMMARY TABLE BY SPOT (2000 - 2025)")
    print("=" * 80)

    spot_rows = []
    for spot, group in df.groupby("spot"):
        total_days = len(group)
        h64 = int(group["heavy_64_5"].sum())
        h64_pct = (h64 / total_days) * 100

        h35 = int(group["heavy_35_0"].sum())
        h35_pct = (h35 / total_days) * 100

        spot_rows.append({
            "Spot": spot,
            "Total Days": total_days,
            "Heavy Days (>= 64.5mm)": h64,
            "Heavy %": f"{h64_pct:.2f}%",
            "Days >= 35mm": h35,
            "35mm %": f"{h35_pct:.2f}%",
        })

    summary_df = pd.DataFrame(spot_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. HEAVY RAIN DAYS (>= 64.5mm) BY MONTH")
    print("=" * 80)

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    monthly = df.groupby("month")["heavy_64_5"].sum().reset_index()
    monthly["Month Name"] = monthly["month"].map(month_names)
    monthly = monthly.rename(columns={"heavy_64_5": "Heavy Days (>= 64.5mm)"})
    print(monthly[["month", "Month Name", "Heavy Days (>= 64.5mm)"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("3. TIME-BASED TRAIN / TEST SPLIT (Train <= 2018, Test: 2019-2025)")
    print("=" * 80)

    train_df = df[df["year"] <= 2018]
    test_df = df[(df["year"] >= 2019) & (df["year"] <= 2025)]

    train_total = len(train_df)
    train_h64 = int(train_df["heavy_64_5"].sum())
    train_h35 = int(train_df["heavy_35_0"].sum())

    test_total = len(test_df)
    test_h64 = int(test_df["heavy_64_5"].sum())
    test_h35 = int(test_df["heavy_35_0"].sum())

    print(f"Train (2000-2018):")
    print(f"  Total spot-days: {train_total}")
    print(f"  Days >= 64.5mm: {train_h64} ({(train_h64/train_total)*100:.2f}%)")
    print(f"  Days >= 35.0mm: {train_h35} ({(train_h35/train_total)*100:.2f}%)")

    print(f"\nTest (2019-2025):")
    print(f"  Total spot-days: {test_total}")
    print(f"  Days >= 64.5mm: {test_h64} ({(test_h64/test_total)*100:.2f}%)")
    print(f"  Days >= 35.0mm: {test_h35} ({(test_h35/test_total)*100:.2f}%)")


if __name__ == "__main__":
    run_checks()

