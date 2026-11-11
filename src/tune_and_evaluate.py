"""
Analysis and Tuning Script:
Step 1: Breakdown of false alarms by next-day rain amount.
Step 2: Threshold tuning across cut-offs [0.1, 0.2, 0.3, 0.5, 0.7].
Step 3: Train XGBoost on easier target (work_stop_tomorrow >= 35mm).
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb

DATA_FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")


def load_and_prep_data():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    # Lags for rain
    grouped_rain = df.groupby(["city", "spot"])["rain_mm"]
    df["rain_lag_1"] = grouped_rain.shift(1)
    df["rain_lag_2"] = grouped_rain.shift(2)
    df["rain_lag_3"] = grouped_rain.shift(3)
    df["rain_lag_7"] = grouped_rain.shift(7)

    df["rain_sum_3d"] = grouped_rain.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_sum_7d"] = grouped_rain.transform(lambda s: s.rolling(7, min_periods=1).sum())

    # Next-day rain and next-day work stop target
    df["rain_tomorrow"] = grouped_rain.shift(-1)
    df["work_stop_tomorrow"] = (df["rain_tomorrow"] >= 35.0).astype("Int64")

    # City code
    df["city_code"] = (df["city"] == "Visakhapatnam").astype(int)

    # Clean missing boundary rows
    df = df.dropna(subset=["heavy_tomorrow", "rain_lag_7", "rain_tomorrow"]).copy()
    df["heavy_tomorrow"] = df["heavy_tomorrow"].astype(int)
    df["work_stop_tomorrow"] = df["work_stop_tomorrow"].astype(int)

    return df


def calculate_metrics(y_true, y_pred):
    alarms = int(np.sum(y_pred == 1))
    real_pos = int(np.sum(y_true == 1))
    caught = int(np.sum((y_pred == 1) & (y_true == 1)))

    recall = (caught / real_pos * 100) if real_pos > 0 else 0.0
    precision = (caught / alarms * 100) if alarms > 0 else 0.0

    return alarms, real_pos, caught, recall, precision


def main():
    df = load_and_prep_data()

    feature_cols = [
        "city_code", "month", "rain_mm", "temp_max", "wind_max",
        "humidity_mean", "heavy_today", "work_stop_today", "heat_flag",
        "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_lag_7",
        "rain_sum_3d", "rain_sum_7d"
    ]

    train_df = df[df["date"].dt.year <= 2018].copy()
    test_df = df[(df["date"].dt.year >= 2019) & (df["date"].dt.year <= 2025)].copy()

    X_train = train_df[feature_cols]
    y_train_heavy = train_df["heavy_tomorrow"]

    X_test = test_df[feature_cols]
    y_test_heavy = test_df["heavy_tomorrow"]

    # =========================================================================
    # Model B for heavy_tomorrow (scale_pos_weight ~ 288)
    # =========================================================================
    scale_pos_heavy = (y_train_heavy == 0).sum() / (y_train_heavy == 1).sum()
    model_heavy = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_heavy,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model_heavy.fit(X_train, y_train_heavy)

    train_probs_heavy = model_heavy.predict_proba(X_train)[:, 1]
    test_probs_heavy = model_heavy.predict_proba(X_test)[:, 1]

    test_df["prob_heavy"] = test_probs_heavy
    test_df["pred_heavy_05"] = (test_probs_heavy >= 0.5).astype(int)

    # =========================================================================
    # STEP 1: Check the false alarms
    # =========================================================================
    print("=" * 80)
    print("STEP 1: BREAKDOWN OF FALSE ALARMS (Model B cut-off 0.5, Test 2019-2025)")
    print("=" * 80)

    # False alarms: model predicted alarm (pred=1), but heavy_tomorrow was 0 (< 64.5mm)
    fa_df = test_df[(test_df["pred_heavy_05"] == 1) & (test_df["heavy_tomorrow"] == 0)].copy()

    fa_summary = []
    for city in ["Chennai", "Visakhapatnam", "Both Cities"]:
        sub = fa_df if city == "Both Cities" else fa_df[fa_df["city"] == city]
        tot_fa = len(sub)
        rain_ge_35 = int((sub["rain_tomorrow"] >= 35.0).sum())
        rain_15_35 = int(((sub["rain_tomorrow"] >= 15.0) & (sub["rain_tomorrow"] < 35.0)).sum())
        rain_lt_15 = int((sub["rain_tomorrow"] < 15.0).sum())

        pct_ge_35 = (rain_ge_35 / tot_fa * 100) if tot_fa > 0 else 0
        pct_15_35 = (rain_15_35 / tot_fa * 100) if tot_fa > 0 else 0
        pct_lt_15 = (rain_lt_15 / tot_fa * 100) if tot_fa > 0 else 0
        pct_wet = ((rain_ge_35 + rain_15_35) / tot_fa * 100) if tot_fa > 0 else 0

        fa_summary.append({
            "City": city,
            "Total False Alarms": tot_fa,
            ">= 35 mm (Work-Stopping)": f"{rain_ge_35} ({pct_ge_35:.1f}%)",
            "15 to 35 mm (Moderate)": f"{rain_15_35} ({pct_15_35:.1f}%)",
            "< 15 mm (Light/Dry)": f"{rain_lt_15} ({pct_lt_15:.1f}%)",
            "Noticeable Rain (>= 15mm)": f"{rain_ge_35 + rain_15_35} ({pct_wet:.1f}%)"
        })

    print(pd.DataFrame(fa_summary).to_string(index=False))

    # =========================================================================
    # STEP 2: Try different alarm levels (cut-offs)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: DIFFERENT ALARM CUT-OFFS (TRAIN vs TEST)")
    print("=" * 80)

    cutoffs = [0.1, 0.2, 0.3, 0.5, 0.7]

    print("\nTRAIN DATA SCORES (Used to decide the best balance):")
    train_rows = []
    for c in cutoffs:
        preds = (train_probs_heavy >= c).astype(int)
        alarms, real, caught, rec, prec = calculate_metrics(y_train_heavy, preds)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        train_rows.append({
            "Cut-off": c,
            "Total Alarms": alarms,
            "Caught (Recall)": f"{caught}/{real} ({rec:.1f}%)",
            "Alarms Right (Precision)": f"{caught}/{alarms} ({prec:.1f}%)",
            "F1 Score": f"{f1:.1f}%"
        })
    print(pd.DataFrame(train_rows).to_string(index=False))

    print("\nTEST DATA SCORES (Reporting per city):")
    test_rows = []
    for city in ["Chennai", "Visakhapatnam", "Both Cities"]:
        sub_test = test_df if city == "Both Cities" else test_df[test_df["city"] == city]
        y_true = sub_test["heavy_tomorrow"]
        p_prob = sub_test["prob_heavy"]

        for c in cutoffs:
            p_pred = (p_prob >= c).astype(int)
            alarms, real, caught, rec, prec = calculate_metrics(y_true, p_pred)
            test_rows.append({
                "City": city,
                "Cut-off": c,
                "Real Heavy": real,
                "Total Alarms": alarms,
                "Caught (Recall)": f"{caught} ({rec:.1f}%)",
                "Right (Precision)": f"{caught} ({prec:.1f}%)",
            })
    print(pd.DataFrame(test_rows).to_string(index=False))

    # =========================================================================
    # STEP 3: Easier target (work_stop_tomorrow >= 35mm, scale_pos_weight ~ 20)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EASIER TARGET -> work_stop_tomorrow (>= 35 mm)")
    print("=" * 80)

    y_train_ws = train_df["work_stop_tomorrow"]
    y_test_ws = test_df["work_stop_tomorrow"]

    scale_pos_ws = 20.0  # As instructed (~20)

    model_ws = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_ws,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model_ws.fit(X_train, y_train_ws)

    test_df["pred_ws_05"] = model_ws.predict(X_test)

    ws_rows = []
    for city in ["Chennai", "Visakhapatnam", "Both Cities"]:
        sub_test = test_df if city == "Both Cities" else test_df[test_df["city"] == city]
        y_true = sub_test["work_stop_tomorrow"]
        p_pred = sub_test["pred_ws_05"]

        alarms, real, caught, rec, prec = calculate_metrics(y_true, p_pred)
        ws_rows.append({
            "City": city,
            "Real Work-Stop Days": real,
            "Total Alarms": alarms,
            "Caught (Recall)": f"{caught} ({rec:.1f}%)",
            "Right (Precision)": f"{caught} ({prec:.1f}%)",
        })

    print(pd.DataFrame(ws_rows).to_string(index=False))


if __name__ == "__main__":
    main()

