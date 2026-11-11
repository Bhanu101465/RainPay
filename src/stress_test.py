"""
GigShield - 26-Year Historical Stress Test (2000 - 2025)
Replays 26 real years of weekly premiums and work-stop payouts
for payouts of Rs. 400, Rs. 450, and Rs. 470.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from zone_risk import get_weekly_premium, ZONE_RISK_FILE

DATA_FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")
WORKERS_PER_CITY = 1000
SPOTS_PER_CITY = 4
WORKERS_PER_SPOT = WORKERS_PER_CITY // SPOTS_PER_CITY
MAX_PAYOUTS_PER_WEEK = 3


def get_rates_map():
    zone_df = pd.read_csv(ZONE_RISK_FILE)
    rates = {}
    for spot in zone_df["spot"].unique():
        rates[spot] = {}
        for m in range(1, 13):
            rates[spot][m] = get_weekly_premium(spot, m, claim_free_weeks=0, zone_df=zone_df)["final_premium"]
    return rates


def run_full_simulation(payout_amount, rates_map):
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.to_period("W")

    annual_records = []

    for (city, year), cy_df in df.groupby(["city", "year"]):
        city_prem = 0.0
        city_payout = 0.0

        for (spot, week), w_df in cy_df.groupby(["spot", "week"]):
            m = w_df["month"].iloc[0]
            weekly_rate = rates_map[spot][m]
            city_prem += WORKERS_PER_SPOT * weekly_rate

            raw_ws = int(w_df["work_stop_today"].sum())
            paid_days = min(MAX_PAYOUTS_PER_WEEK, raw_ws)
            city_payout += WORKERS_PER_SPOT * paid_days * payout_amount

        loss_ratio = (city_payout / city_prem * 100.0) if city_prem > 0 else 0.0
        annual_records.append({
            "city": city,
            "year": year,
            "premiums": city_prem,
            "payouts": city_payout,
            "net": city_prem - city_payout,
            "loss_ratio": loss_ratio,
        })

    ann_df = pd.DataFrame(annual_records)
    return ann_df


def compute_running_balance_and_drawdown(payout_amount, rates_map):
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.to_period("W")

    results = {}
    for scope in ["Chennai", "Visakhapatnam", "Both"]:
        sub_df = df if scope == "Both" else df[df["city"] == scope]
        
        # Aggregate weekly cash flow in chronological order
        weeks_data = []
        for week, w_df in sub_df.groupby("week"):
            tot_prem = 0.0
            tot_payout = 0.0
            for spot, s_df in w_df.groupby("spot"):
                m = s_df["month"].iloc[0]
                weekly_rate = rates_map[spot][m]
                tot_prem += WORKERS_PER_SPOT * weekly_rate
                
                ws_days = int(s_df["work_stop_today"].sum())
                paid_days = min(MAX_PAYOUTS_PER_WEEK, ws_days)
                tot_payout += WORKERS_PER_SPOT * paid_days * payout_amount

            weeks_data.append({
                "week": week,
                "premiums": tot_prem,
                "payouts": tot_payout,
                "net": tot_prem - tot_payout
            })

        w_df = pd.DataFrame(weeks_data).sort_values("week").reset_index(drop=True)
        w_df["cum_balance"] = w_df["net"].cumsum()
        w_df["peak_balance"] = w_df["cum_balance"].cummax()
        w_df["drawdown"] = w_df["peak_balance"] - w_df["cum_balance"]

        results[scope] = {
            "lowest_running_balance": w_df["cum_balance"].min(),
            "final_balance": w_df["cum_balance"].iloc[-1],
            "max_drawdown": w_df["drawdown"].max(),
        }

    return results


def main():
    payouts = [400, 450, 470]
    rates_map = get_rates_map()
    
    for p in payouts:
        print(f"\n{'='*75}\nPAYOUT: Rs. {p} PER WORK-STOP DAY\n{'='*75}")
        ann_df = run_full_simulation(p, rates_map)

        # 1. Top 5 worst years per city
        for city in ["Chennai", "Visakhapatnam"]:
            c_data = ann_df[ann_df["city"] == city].sort_values("loss_ratio", ascending=False)
            print(f"\nTop 5 Worst Years for {city} (Rs. {p}):")
            formatted = c_data.head(5).copy()
            formatted["premiums"] = formatted["premiums"].apply(lambda v: f"Rs. {v:,.0f}")
            formatted["payouts"] = formatted["payouts"].apply(lambda v: f"Rs. {v:,.0f}")
            formatted["loss_ratio"] = formatted["loss_ratio"].apply(lambda v: f"{v:.1f}%")
            formatted["net"] = formatted["net"].apply(lambda v: f"Rs. {v:,.0f}")
            print(formatted[["year", "premiums", "payouts", "loss_ratio", "net"]].to_string(index=False))

        # Combined top 5 worst years
        comb = ann_df.groupby("year")[["premiums", "payouts"]].sum().reset_index()
        comb["loss_ratio"] = (comb["payouts"] / comb["premiums"]) * 100.0
        comb["net"] = comb["premiums"] - comb["payouts"]
        print(f"\nTop 5 Worst Years for Combined Fund (Rs. {p}):")
        comb_formatted = comb.sort_values("loss_ratio", ascending=False).head(5).copy()
        comb_formatted["premiums"] = comb_formatted["premiums"].apply(lambda v: f"Rs. {v:,.0f}")
        comb_formatted["payouts"] = comb_formatted["payouts"].apply(lambda v: f"Rs. {v:,.0f}")
        comb_formatted["loss_ratio"] = comb_formatted["loss_ratio"].apply(lambda v: f"{v:.1f}%")
        comb_formatted["net"] = comb_formatted["net"].apply(lambda v: f"Rs. {v:,.0f}")
        print(comb_formatted[["year", "premiums", "payouts", "loss_ratio", "net"]].to_string(index=False))

        # 2. Years with loss ratio > 100%
        gt_100_chennai = ann_df[(ann_df["city"] == "Chennai") & (ann_df["loss_ratio"] > 100.0)]
        gt_100_vizag = ann_df[(ann_df["city"] == "Visakhapatnam") & (ann_df["loss_ratio"] > 100.0)]
        gt_100_comb = comb[comb["loss_ratio"] > 100.0]

        print(f"\nYears with Loss Ratio > 100%:")
        print(f"  Chennai:       {len(gt_100_chennai)} years {gt_100_chennai['year'].tolist()}")
        print(f"  Visakhapatnam: {len(gt_100_vizag)} years {gt_100_vizag['year'].tolist()}")
        print(f"  Combined Fund: {len(gt_100_comb)} years {gt_100_comb['year'].tolist()}")

        # 3. Running balance and drawdown
        res = compute_running_balance_and_drawdown(p, rates_map)
        print(f"\nBalance & Reserve Analysis (2000 - 2025):")
        for scope, vals in res.items():
            print(f"  {scope:15s}: Lowest Balance: Rs. {vals['lowest_running_balance']:,.0f} | "
                  f"Max Drawdown (Worst Dip): Rs. {vals['max_drawdown']:,.0f} | "
                  f"Final 26-Yr Surplus: Rs. {vals['final_balance']:,.0f}")


if __name__ == "__main__":
    main()
