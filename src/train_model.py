"""
Train weather prediction models:
Model A: Simple rule ("tomorrow is heavy if today is heavy")
Model B: XGBoost with scale_pos_weight for class imbalance

Inputs use strictly today's and past days' weather. Never tomorrow's weather.
Strict temporal split: Train (2000-2018), Test (2019-2025).
"""

import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

DATA_FILE = os.path.join("data", "clean", "weather_labeled_v2.csv")
MODEL_DIR = os.path.join("models")
MODEL_FILE_PKL = os.path.join(MODEL_DIR, "xgb_model.pkl")
MODEL_FILE_JSON = os.path.join(MODEL_DIR, "xgb_model.json")


def prepare_features():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run make_labels_v2.py first.")

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Ensure strictly sorted by city, spot, date
    df = df.sort_values(by=["city", "spot", "date"]).reset_index(drop=True)

    # 1. Add rain lags per spot (last 1, 2, 3, and 7 days)
    # strictly from past days inside the same spot
    grouped_rain = df.groupby(["city", "spot"])["rain_mm"]
    df["rain_lag_1"] = grouped_rain.shift(1)
    df["rain_lag_2"] = grouped_rain.shift(2)
    df["rain_lag_3"] = grouped_rain.shift(3)
    df["rain_lag_7"] = grouped_rain.shift(7)

    # Rolling sum of recent rainfall (past 3 and 7 days)
    df["rain_sum_3d"] = grouped_rain.transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["rain_sum_7d"] = grouped_rain.transform(lambda s: s.rolling(7, min_periods=1).sum())

    # 2. Add city column (0 for Chennai, 1 for Visakhapatnam)
    df["city_code"] = (df["city"] == "Visakhapatnam").astype(int)

    # 3. Remove rows with empty heavy_tomorrow (last day of each spot)
    df = df.dropna(subset=["heavy_tomorrow"]).copy()

    # Also drop the initial rows with NaN from the 7-day lag
    df = df.dropna(subset=["rain_lag_7"]).copy()

    df["heavy_tomorrow"] = df["heavy_tomorrow"].astype(int)

    return df


def calculate_metrics(y_true, y_pred):
    """Calculate alarms, caught days (recall), and right alarms (precision)."""
    alarms = int(np.sum(y_pred == 1))
    real_heavy = int(np.sum(y_true == 1))
    caught = int(np.sum((y_pred == 1) & (y_true == 1)))  # True Positives

    recall = (caught / real_heavy * 100) if real_heavy > 0 else 0.0
    precision = (caught / alarms * 100) if alarms > 0 else 0.0

    return {
        "Total Real Heavy": real_heavy,
        "Total Alarms": alarms,
        "Caught (Recall Count)": caught,
        "Recall %": f"{recall:.1f}%",
        "Right (Precision Count)": caught,
        "Precision %": f"{precision:.1f}%",
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = prepare_features()

    # Step 2: Time split (Train <= 2018, Test: 2019-2025)
    train_df = df[df["date"].dt.year <= 2018].copy()
    test_df = df[(df["date"].dt.year >= 2019) & (df["date"].dt.year <= 2025)].copy()

    # Input feature list: only today's and past days' weather
    feature_cols = [
        "city_code",
        "month",
        "rain_mm",           # Today's rain
        "temp_max",          # Today's max temperature
        "wind_max",          # Today's max wind
        "humidity_mean",     # Today's humidity
        "heavy_today",       # Today's heavy rain flag
        "work_stop_today",   # Today's work-stopping flag
        "heat_flag",         # Today's heat flag
        "rain_lag_1",        # Rain 1 day ago
        "rain_lag_2",        # Rain 2 days ago
        "rain_lag_3",        # Rain 3 days ago
        "rain_lag_7",        # Rain 7 days ago
        "rain_sum_3d",       # 3-day rain sum
        "rain_sum_7d",       # 7-day rain sum
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["heavy_tomorrow"]

    X_test = test_df[feature_cols]
    y_test = test_df["heavy_tomorrow"]

    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Train period (2000-2018): {len(train_df)} rows, {int(y_train.sum())} heavy_tomorrow days ({y_train.mean()*100:.2f}%)")
    print(f"Test period  (2019-2025): {len(test_df)} rows, {int(y_test.sum())} heavy_tomorrow days ({y_test.mean()*100:.2f}%)")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # =========================================================================
    # Step 3A: Rule A ("tomorrow is heavy if today is heavy")
    # =========================================================================
    test_df["pred_rule_a"] = test_df["heavy_today"]

    # =========================================================================
    # Step 3B: Model B (XGBoost with scale_pos_weight)
    # =========================================================================
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"\nTraining XGBoost with scale_pos_weight = {scale_pos:.2f}...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    # Predictions
    test_df["pred_xgb"] = xgb_model.predict(X_test)
    test_df["prob_xgb"] = xgb_model.predict_proba(X_test)[:, 1]

    # Save model
    xgb_model.save_model(MODEL_FILE_JSON)
    with open(MODEL_FILE_PKL, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"Saved trained XGBoost model to {MODEL_FILE_JSON} and {MODEL_FILE_PKL}")

    # =========================================================================
    # Step 4: Scoring on Test Years (2019 - 2025)
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL COMPARISON ON TEST YEARS (2019 - 2025)")
    print("=" * 80)

    results = []

    for city in ["Chennai", "Visakhapatnam", "Both Cities"]:
        if city == "Both Cities":
            sub_test = test_df
        else:
            sub_test = test_df[test_df["city"] == city]

        y_true = sub_test["heavy_tomorrow"]
        m_a = calculate_metrics(y_true, sub_test["pred_rule_a"])
        m_b = calculate_metrics(y_true, sub_test["pred_xgb"])

        results.append({
            "City": city,
            "Model": "Rule A (Today = Heavy)",
            "Real Heavy Days": m_a["Total Real Heavy"],
            "Total Alarms": m_a["Total Alarms"],
            "Caught (Recall)": f"{m_a['Caught (Recall Count)']} ({m_a['Recall %']})",
            "Right (Precision)": f"{m_a['Right (Precision Count)']} ({m_a['Precision %']})",
        })
        results.append({
            "City": city,
            "Model": "Model B (XGBoost)",
            "Real Heavy Days": m_b["Total Real Heavy"],
            "Total Alarms": m_b["Total Alarms"],
            "Caught (Recall)": f"{m_b['Caught (Recall Count)']} ({m_b['Recall %']})",
            "Right (Precision)": f"{m_b['Right (Precision Count)']} ({m_b['Precision %']})",
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Feature Importances
    print("\n" + "=" * 80)
    print("TOP PREDICTIVE FEATURES (XGBoost)")
    print("=" * 80)
    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, val in importances.items():
        print(f"  - {feat:16s}: {val*100:.1f}%")


if __name__ == "__main__":
    main()

