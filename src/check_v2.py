"""
Check combined Chennai and Visakhapatnam labeled dataset (weather_labeled_v2.csv).
1. Table of spots with heavy days and work-stop days.
2. Train (2000-2018) vs Test (2019-2025) heavy counts per city.
"""

import os
import pandas as pd

FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")


def run_checks():
    if not os.path.exists(FILE):
        raise FileNotFoundError(f"Missing {FILE}. Run src/make_labels_v2.py first.")

    df = pd.read_csv(FILE)
    df["date"] = pd.to_datetime(df["date"])

    print("=" * 80)
    print("1. SUMMARY TABLE (CHENNAI + VISAKHAPATNAM)")
    print("=" * 80)

    rows = []
    for (city, spot), group in df.groupby(["city", "spot"]):
        total_days = len(group)
        heavy = int(group["heavy_today"].sum())
        work_stop = int(group["work_stop_today"].sum())

        rows.append({
            "City": city,
            "Spot": spot,
            "Total Days": total_days,
            "Heavy Days (>=64.5mm)": heavy,
            "Heavy %": f"{(heavy / total_days) * 100:.2f}%",
            "Work-Stop (>=35mm)": work_stop,
            "Work-Stop %": f"{(work_stop / total_days) * 100:.2f}%",
        })

    table_df = pd.DataFrame(rows)
    print(table_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("2. TRAIN (2000-2018) vs TEST (2019-2025) COUNTS")
    print("=" * 80)

    train_df = df[df["date"].dt.year <= 2018]
    test_df = df[(df["date"].dt.year >= 2019) & (df["date"].dt.year <= 2025)]

    for city in sorted(df["city"].unique()):
        tr = train_df[train_df["city"] == city]
        te = test_df[test_df["city"] == city]

        tr_heavy = int(tr["heavy_today"].sum())
        tr_stop = int(tr["work_stop_today"].sum())
        te_heavy = int(te["heavy_today"].sum())
        te_stop = int(te["work_stop_today"].sum())

        print(f"\n{city}:")
        print(f"  Train (2000-2018): {tr_heavy} heavy days, {tr_stop} work-stop days (out of {len(tr)} spot-days)")
        print(f"  Test  (2019-2025): {te_heavy} heavy days, {te_stop} work-stop days (out of {len(te)} spot-days)")

    print("\n" + "=" * 80)
    print("3. SAFETY VERIFICATION")
    print("=" * 80)
    nan_count = df["heavy_tomorrow"].isna().sum()
    print(f"Missing values in heavy_tomorrow: {nan_count} (expected 8, exactly one for each spot's last day)")


if __name__ == "__main__":
    run_checks()

