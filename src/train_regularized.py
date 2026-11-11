"""
Train regularized XGBoost on work_stop_tomorrow (>= 35 mm):
1. Three-way split: Train (2000-2014), Validation (2015-2018), Test (2019-2025).
2. Less memorizing: max_depth=3, min_child_weight, regularization, early stopping.
3. Choose cut-off on validation set aiming for ~60% recall.
4. Report once on test set with day-level and consecutive-storm metrics.
"""

import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

DATA_FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")
MODEL_JSON = os.path.join("models", "xgb_work_stop_final.json")
MODEL_PKL = os.path.join("models", "xgb_work_stop_final.pkl")


def prepare_data():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    # Lags and rolling rain within each spot
    grouped_rain = df.groupby(["city", "spot"])["rain_mm"]
    df["rain_lag_1"] = grouped_rain.shift(1)
    df["rain_lag_2"] = grouped_rain.shift(2)
    df["rain_lag_3"] = grouped_rain.shift(3)
    df["rain_lag_7"] = grouped_rain.shift(7)

    df["rain_sum_3d"] = grouped_rain.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_sum_7d"] = grouped_rain.transform(lambda s: s.rolling(7, min_periods=1).sum())

    # Target: tomorrow's rain >= 35mm
    df["rain_tomorrow"] = grouped_rain.shift(-1)
    df["work_stop_tomorrow"] = (df["rain_tomorrow"] >= 35.0).astype("Int64")

    # City code
    df["city_code"] = (df["city"] == "Visakhapatnam").astype(int)

    # Clean boundaries
    df = df.dropna(subset=["work_stop_tomorrow", "rain_lag_7"]).copy()
    df["work_stop_tomorrow"] = df["work_stop_tomorrow"].astype(int)

    return df


def calculate_metrics(y_true, y_pred):
    alarms = int(np.sum(y_pred == 1))
    real_pos = int(np.sum(y_true == 1))
    caught = int(np.sum((y_pred == 1) & (y_true == 1)))

    recall = (caught / real_pos * 100) if real_pos > 0 else 0.0
    precision = (caught / alarms * 100) if alarms > 0 else 0.0

    return alarms, real_pos, caught, recall, precision


def calculate_storm_metrics(sub_df, pred_col):
    """
    Count consecutive work-stop days at the same spot as ONE storm.
    A storm is caught if at least one day in the storm has an alarm.
    Also count consecutive alarm days as ONE alarm event.
    """
    sub = sub_df.sort_values(by=["spot", "date"]).copy()

    # Identify distinct storms (consecutive days where work_stop_tomorrow == 1)
    sub["is_work_stop"] = sub["work_stop_tomorrow"] == 1
    sub["prev_ws"] = sub.groupby("spot")["is_work_stop"].shift(1) == True
    sub["new_storm"] = sub["is_work_stop"] & (~sub["prev_ws"])
    sub["storm_id"] = sub.groupby("spot")["new_storm"].cumsum()
    # Only keep storm_id for work stop rows
    sub.loc[~sub["is_work_stop"], "storm_id"] = np.nan

    # Identify distinct alarm events (consecutive alarm days)
    sub["is_alarm"] = sub[pred_col] == 1
    sub["prev_alarm"] = sub.groupby("spot")["is_alarm"].shift(1) == True
    sub["new_alarm_event"] = sub["is_alarm"] & (~sub["prev_alarm"])
    total_alarm_events = int(sub["new_alarm_event"].sum())

    # Count storms
    valid_storms = sub.dropna(subset=["storm_id"])
    if len(valid_storms) == 0:
        return 0, 0, 0, 0.0, total_alarm_events

    storm_groups = valid_storms.groupby(["spot", "storm_id"])
    total_storms = storm_groups.ngroups

    # A storm is caught if any day in the storm group had an alarm
    caught_storms = int(storm_groups[pred_col].apply(lambda s: (s == 1).any()).sum())
    missed_storms = total_storms - caught_storms
    storm_recall = (caught_storms / total_storms * 100) if total_storms > 0 else 0.0

    return total_storms, caught_storms, missed_storms, storm_recall, total_alarm_events


def main():
    df = prepare_data()

    feature_cols = [
        "city_code", "month", "rain_mm", "temp_max", "wind_max",
        "humidity_mean", "heavy_today", "work_stop_today", "heat_flag",
        "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_lag_7",
        "rain_sum_3d", "rain_sum_7d"
    ]

    # Step 1: Three-way split
    train_df = df[df["date"].dt.year <= 2014].copy()
    val_df = df[(df["date"].dt.year >= 2015) & (df["date"].dt.year <= 2018)].copy()
    test_df = df[df["date"].dt.year >= 2019].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["work_stop_tomorrow"]

    X_val = val_df[feature_cols]
    y_val = val_df["work_stop_tomorrow"]

    X_test = test_df[feature_cols]
    y_test = test_df["work_stop_tomorrow"]

    print("=" * 80)
    print("STEP 1: THREE-WAY SPLIT")
    print("=" * 80)
    print(f"Train (2000-2014):        {len(train_df):5d} rows, {int(y_train.sum()):3d} work-stop days ({y_train.mean()*100:.2f}%)")
    print(f"Practice Exam (2015-2018): {len(val_df):5d} rows, {int(y_val.sum()):3d} work-stop days ({y_val.mean()*100:.2f}%)")
    print(f"Test (2019-2025):         {len(test_df):5d} rows, {int(y_test.sum()):3d} work-stop days ({y_test.mean()*100:.2f}%)")

    # Step 2: Less memorizing XGBoost
    # Smaller depth (3), early stopping on practice exam, moderate pos weight (20)
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,             # Shallow trees prevent memorization
        learning_rate=0.04,
        scale_pos_weight=20.0,   # As requested (~20)
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,      # Requires at least 5 samples in leaves
        reg_lambda=2.0,          # L2 regularization
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=15
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"\nModel trained. Stopped at best iteration: {model.best_iteration}")

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    val_df["prob"] = val_probs
    test_df["prob"] = test_probs

    # Step 2 Check: Train Recall vs Practice Exam Recall side-by-side
    print("\n" + "=" * 80)
    print("STEP 2: CHECK OVERFITTING (Train vs Practice-Exam Recall & Precision)")
    print("=" * 80)
    check_cutoffs = [0.3, 0.4, 0.5, 0.6, 0.7]
    overfit_rows = []
    for c in check_cutoffs:
        tr_pred = (train_probs >= c).astype(int)
        va_pred = (val_probs >= c).astype(int)
        _, _, tr_caught, tr_rec, tr_prec = calculate_metrics(y_train, tr_pred)
        _, _, va_caught, va_rec, va_prec = calculate_metrics(y_val, va_pred)

        overfit_rows.append({
            "Cut-off": c,
            "Train Recall": f"{tr_rec:.1f}%",
            "Practice Recall": f"{va_rec:.1f}%",
            "Recall Gap": f"{abs(tr_rec - va_rec):.1f}%",
            "Train Precision": f"{tr_prec:.1f}%",
            "Practice Precision": f"{va_prec:.1f}%",
        })
    print(pd.DataFrame(overfit_rows).to_string(index=False))

    # Step 3: Choose cut-off on Practice Exam years only
    print("\n" + "=" * 80)
    print("STEP 3: CHOOSE CUT-OFF (Practice Exam Years 2015-2018 Only)")
    print("=" * 80)
    val_table = []
    all_cutoffs = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
    for c in all_cutoffs:
        va_pred = (val_probs >= c).astype(int)
        alarms, real, caught, rec, prec = calculate_metrics(y_val, va_pred)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        val_table.append({
            "Cut-off": c,
            "Real Days": real,
            "Total Alarms": alarms,
            "Caught (Recall)": f"{caught} ({rec:.1f}%)",
            "Right (Precision)": f"{caught} ({prec:.1f}%)",
            "F1 Score": f"{f1:.1f}%",
        })
    print(pd.DataFrame(val_table).to_string(index=False))

    # Pick cut-off with about 60% recall and best precision on val
    # Looking at the candidates with recall ~55% to ~65%
    chosen_cutoff = 0.5  # We will also dynamically verify from table
    for row in val_table:
        rec_val = float(row["Caught (Recall)"].split("(")[1].replace("%)", ""))
        if 55.0 <= rec_val <= 65.0:
            chosen_cutoff = row["Cut-off"]

    print(f"\n--> CHOSEN CUT-OFF: {chosen_cutoff} (gives ~60% recall with highest precision on Practice Exam)")

    # Step 4: Report once on test years
    test_df["pred"] = (test_probs >= chosen_cutoff).astype(int)

    print("\n" + "=" * 80)
    print(f"STEP 4A: DAY-LEVEL PERFORMANCE ON TEST YEARS (2019-2025, Cut-off = {chosen_cutoff})")
    print("=" * 80)
    test_day_rows = []
    for city in ["Chennai", "Visakhapatnam", "Both Cities"]:
        sub = test_df if city == "Both Cities" else test_df[test_df["city"] == city]
        alarms, real, caught, rec, prec = calculate_metrics(sub["work_stop_tomorrow"], sub["pred"])
        test_day_rows.append({
            "City": city,
            "Real Work-Stop Days": real,
            "Total Alarms": alarms,
            "Caught (Recall)": f"{caught} ({rec:.1f}%)",
            "Right (Precision)": f"{caught} ({prec:.1f}%)",
        })
    print(pd.DataFrame(test_day_rows).to_string(index=False))

    print("\n" + "=" * 80)
    print(f"STEP 4B: STORM-LEVEL PERFORMANCE ON TEST YEARS (2019-2025, Cut-off = {chosen_cutoff})")
    print("=" * 80)
    test_storm_rows = []
    for city in ["Chennai", "Visakhapatnam", "Both Cities"]:
        sub = test_df if city == "Both Cities" else test_df[test_df["city"] == city]
        tot_storms, caught_s, missed_s, s_recall, tot_alarm_events = calculate_storm_metrics(sub, "pred")
        test_storm_rows.append({
            "City": city,
            "Total Storm Events": tot_storms,
            "Storms Caught": f"{caught_s} ({s_recall:.1f}%)",
            "Storms Missed": missed_s,
            "Alarm Episodes": tot_alarm_events,
        })
    print(pd.DataFrame(test_storm_rows).to_string(index=False))

    # Step 5: Save model
    model.save_model(MODEL_JSON)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved final model to {MODEL_JSON} and {MODEL_PKL}")


if __name__ == "__main__":
    main()
