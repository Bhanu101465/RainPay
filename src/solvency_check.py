"""
GigShield - Historical Solvency & Loss Ratio Simulation (2019 - 2025)
Replays 7 real years of weekly premiums and work-stop payouts for 1,000 workers per city.
"""

import os
import sys

# Ensure local imports resolve whether called from project root or src
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from zone_risk import get_weekly_premium

DATA_FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")

# =============================================================================
# STEP 1: CONFIGURATION (Easy to change payout amount in one place)
# =============================================================================
PAYOUT_PER_DAY = 400.0          # Base payout in Rupees per work-stop day (>= 35 mm)
MAX_PAYOUTS_PER_WEEK = 3        # Maximum allowable claim days per worker per week
WORKERS_PER_CITY = 1000         # Active worker cohort per city
SPOTS_PER_CITY = 4              # 4 micro-locations per city
WORKERS_PER_SPOT = WORKERS_PER_CITY // SPOTS_PER_CITY  # 250 workers per spot

START_YEAR = 2019
END_YEAR = 2025


def run_simulation(payout_amount=PAYOUT_PER_DAY):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE}.")

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Filter strictly to test years
    test_df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    # Define calendar weeks (Monday to Sunday)
    test_df["week"] = test_df["date"].dt.to_period("W")

    annual_records = []

    for (city, year), cy_df in test_df.groupby(["city", "year"]):
        city_prem = 0.0
        city_payout = 0.0
        total_ws_days_spot = 0
        total_capped_claims = 0

        for (spot, week), w_df in cy_df.groupby(["spot", "week"]):
            # Use the representative month of this week
            m = w_df["month"].iloc[0]

            # 1. Collect weekly premium (assuming 0 claim-free weeks)
            prem_info = get_weekly_premium(spot, m, claim_free_weeks=0)
            weekly_rate = prem_info["final_premium"]
            city_prem += WORKERS_PER_SPOT * weekly_rate

            # 2. Work-stop days in this week (>= 35 mm)
            raw_ws_days = int(w_df["work_stop_today"].sum())
            total_ws_days_spot += raw_ws_days

            # 3. Apply 3-day cap per worker per week
            paid_days = min(MAX_PAYOUTS_PER_WEEK, raw_ws_days)
            total_capped_claims += paid_days
            city_payout += WORKERS_PER_SPOT * paid_days * payout_amount

        loss_ratio = (city_payout / city_prem * 100.0) if city_prem > 0 else 0.0

        annual_records.append({
            "city": city,
            "year": year,
            "premiums_collected": city_prem,
            "payouts_paid": city_payout,
            "raw_work_stop_days": total_ws_days_spot,
            "paid_worker_days": total_capped_claims * WORKERS_PER_SPOT,
            "loss_ratio": loss_ratio,
        })

    return pd.DataFrame(annual_records)


def main():
    print("=" * 85)
    print(f"GIGSHIELD 7-YEAR SOLVENCY REPLAY (2019-2025) @ Rs. {PAYOUT_PER_DAY:.0f} Payout / Day")
    print(f"Cohort: {WORKERS_PER_CITY:,} workers per city (250 / spot) | Cap: {MAX_PAYOUTS_PER_WEEK} days / worker / week")
    print("=" * 85)

    sim_df = run_simulation(PAYOUT_PER_DAY)

    # Format table for display
    display_df = sim_df.copy()
    display_df["Premiums Collected"] = display_df["premiums_collected"].apply(lambda v: f"Rs. {v:,.0f}")
    display_df["Payouts Paid"] = display_df["payouts_paid"].apply(lambda v: f"Rs. {v:,.0f}")
    display_df["Loss Ratio"] = display_df["loss_ratio"].apply(lambda v: f"{v:.1f}%")
    display_df["Surplus (Reserve)"] = (display_df["premiums_collected"] - display_df["payouts_paid"]).apply(lambda v: f"Rs. {v:,.0f}")

    table_cols = ["city", "year", "Premiums Collected", "Payouts Paid", "Loss Ratio", "Surplus (Reserve)"]
    print(display_df[table_cols].to_string(index=False))

    # City & Total Summaries
    print("\n" + "=" * 85)
    print("ALL-TIME FINANCIAL SUMMARY (2019 - 2025)")
    print("=" * 85)

    summary_rows = []
    for city in ["Chennai", "Visakhapatnam", "Combined Fund"]:
        if city == "Combined Fund":
            c_prem = sim_df["premiums_collected"].sum()
            c_pay = sim_df["payouts_paid"].sum()
        else:
            sub = sim_df[sim_df["city"] == city]
            c_prem = sub["premiums_collected"].sum()
            c_pay = sub["payouts_paid"].sum()

        c_surplus = c_prem - c_pay
        c_lr = (c_pay / c_prem * 100.0) if c_prem > 0 else 0.0

        summary_rows.append({
            "Entity": city,
            "Total Premiums": f"Rs. {c_prem:,.0f}",
            "Total Payouts": f"Rs. {c_pay:,.0f}",
            "Total Reserve Surplus": f"Rs. {c_surplus:,.0f}",
            "7-Year Loss Ratio": f"{c_lr:.1f}%",
        })

    print(pd.DataFrame(summary_rows).to_string(index=False))

    # Calculate optimal payout for 70% loss ratio
    total_prem = sim_df["premiums_collected"].sum()
    total_paid_days = sim_df["paid_worker_days"].sum()
    target_loss_ratio = 0.70
    optimal_payout = (total_prem * target_loss_ratio) / total_paid_days

    print("\n" + "=" * 85)
    print("TARGET 70% LOSS RATIO CALCULATION")
    print("=" * 85)
    print(f"Total worker claim-days paid: {int(total_paid_days):,}")
    print(f"Optimal payout amount for ~70% loss ratio: Rs. {optimal_payout:.2f} (approx Rs. 470 - Rs. 475 / day)")


if __name__ == "__main__":
    main()
