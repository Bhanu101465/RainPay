"""
GigShield - Parametric Weather Protection FastAPI Backend
Real weather data, XGBoost model forecast, and PayPal Sandbox integration.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import pandas as pd
import xgboost as xgb

# -----------------------------------------------------------------------------
# 1. SETUP & ENVIRONMENT
# -----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.zone_risk import get_weekly_premium

# Load .env from root, then src/ (root takes priority if both exist)
load_dotenv(os.path.join(BASE_DIR, "src", ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_SECRET    = os.getenv("PAYPAL_SECRET")
WORKER_EMAIL     = os.getenv("WORKER_EMAIL", "worker@example.com")
PAYPAL_BASE_URL  = os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")

CURRENCY      = os.getenv("CURRENCY", "USD")
EXCHANGE_RATE = float(os.getenv("EXCHANGE_RATE", "83.0"))

PAYOUT_INR          = 400.0
MAX_PAYOUTS_PER_WEEK = 3

DATA_FILE       = os.path.join(BASE_DIR, "data", "clean", "weather_labeled_v2.csv")
MODEL_FILE      = os.path.join(BASE_DIR, "models", "xgb_work_stop_final.json")
PAYOUT_LOG_FILE = os.path.join(BASE_DIR, "data", "clean", "payout_log.json")

# 6. Print which credentials are in use at startup
print("=" * 60)
print("GigShield FastAPI starting")
print(f"  PayPal Client ID : {(PAYPAL_CLIENT_ID or 'NOT SET')[:6]}...")
print(f"  Worker email     : {WORKER_EMAIL}")
print(f"  PayPal base URL  : {PAYPAL_BASE_URL}")
print(f"  Currency         : {CURRENCY}  (Rs. {EXCHANGE_RATE:.0f} = 1 USD)")
print("=" * 60)

app = FastAPI(title="GigShield Parametric Weather API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 2. LOAD REAL WEATHER DATA & MODEL AT IMPORT TIME
# -----------------------------------------------------------------------------
print("Loading real weather dataset and computing lag features...")
df_weather = pd.read_csv(DATA_FILE)
df_weather["date_str"] = df_weather["date"]
df_weather["date_dt"]  = pd.to_datetime(df_weather["date"])
df_weather = df_weather.sort_values(by=["city", "spot", "date_dt"]).reset_index(drop=True)

grouped_rain = df_weather.groupby(["city", "spot"])["rain_mm"]
df_weather["rain_lag_1"]  = grouped_rain.shift(1)
df_weather["rain_lag_2"]  = grouped_rain.shift(2)
df_weather["rain_lag_3"]  = grouped_rain.shift(3)
df_weather["rain_lag_7"]  = grouped_rain.shift(7)
df_weather["rain_sum_3d"] = grouped_rain.transform(lambda s: s.rolling(3, min_periods=1).sum())
df_weather["rain_sum_7d"] = grouped_rain.transform(lambda s: s.rolling(7, min_periods=1).sum())
df_weather["city_code"]   = (df_weather["city"] == "Visakhapatnam").astype(int)

# O(1) lookup dict: (spot, date_str) -> record
WEATHER_DB = df_weather.set_index(["spot", "date_str"]).to_dict(orient="index")

print("Loading trained XGBoost model...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(MODEL_FILE)

FEATURE_COLS = [
    "city_code", "month", "rain_mm", "temp_max", "wind_max",
    "humidity_mean", "heavy_today", "work_stop_today", "heat_flag",
    "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_lag_7",
    "rain_sum_3d", "rain_sum_7d"
]

AVAILABLE_SPOTS = sorted(df_weather["spot"].unique().tolist())
print(f"Ready. {len(WEATHER_DB):,} weather records loaded for spots: {AVAILABLE_SPOTS}")
print("=" * 60)

# -----------------------------------------------------------------------------
# 3. HELPERS
# -----------------------------------------------------------------------------
def inr_to_usd(inr: float) -> float:
    return round(inr / EXCHANGE_RATE, 2)


def get_paypal_token() -> str:
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
        raise HTTPException(
            status_code=500,
            detail="PayPal credentials missing in .env"
        )
    res = requests.post(
        f"{PAYPAL_BASE_URL}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
        data={"grant_type": "client_credentials"},
        headers={"Accept": "application/json", "Accept-Language": "en_US"},
        timeout=15,
    )
    if res.status_code != 200:
        raise HTTPException(status_code=502, detail=f"PayPal auth failed: {res.text}")
    return res.json()["access_token"]


def load_payout_logs() -> list:
    if os.path.exists(PAYOUT_LOG_FILE):
        try:
            with open(PAYOUT_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_payout_entry(entry: dict):
    os.makedirs(os.path.dirname(PAYOUT_LOG_FILE), exist_ok=True)
    logs = load_payout_logs()
    logs.append(entry)
    with open(PAYOUT_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


def count_weekly_paid_claims(email: str, week_str: str) -> int:
    """
    4. Count only payouts where PayPal returned HTTP 201 (payout_paypal_success=True).
    """
    logs = load_payout_logs()
    return sum(
        1 for item in logs
        if item.get("worker_email") == email
        and item.get("week") == week_str
        and item.get("payout_paypal_success") is True   # only real PayPal successes
    )


# -----------------------------------------------------------------------------
# 4. API ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/premium")
def get_premium(
    spot: str = Query("Velachery", description="Delivery zone spot name"),
    week_start: str = Query("2024-11-25", description="Week start date (YYYY-MM-DD)")
):
    """GET /premium — Weekly premium from zone_risk.py, shown in Rs. and USD."""
    if spot not in AVAILABLE_SPOTS:
        raise HTTPException(400, detail=f"Unknown spot '{spot}'. Available: {AVAILABLE_SPOTS}")
    try:
        month = pd.to_datetime(week_start).month
    except Exception:
        raise HTTPException(400, detail="Invalid week_start. Use YYYY-MM-DD.")

    info = get_weekly_premium(spot, month, 0)
    final_inr = float(info["final_premium"])
    return {
        "spot": spot,
        "city": info["city"],
        "week_start": week_start,
        "month": month,
        "risk_score": info["risk_score"],
        "is_monsoon": info["is_monsoon"],
        "premium_inr": final_inr,
        "premium_usd": inr_to_usd(final_inr),
        "currency": CURRENCY,
        "exchange_rate": f"Rs. {EXCHANGE_RATE:.0f} = 1 {CURRENCY}",
        "breakdown": {
            "base_inr": info["base"],
            "flood_addon_inr": info["flood_addon"],
            "monsoon_addon_inr": info["monsoon_addon"],
            "discount_inr": info["discount"]
        }
    }


@app.get("/forecast")
def get_forecast(
    spot: str = Query("Velachery", description="Delivery spot name"),
    date: str = Query("2023-12-03", description="Weather date (YYYY-MM-DD)")
):
    """GET /forecast — XGBoost probability of work-stopping rain tomorrow."""
    if spot not in AVAILABLE_SPOTS:
        raise HTTPException(400, detail=f"Unknown spot '{spot}'. Available: {AVAILABLE_SPOTS}")

    record = WEATHER_DB.get((spot, date))
    if record is None:
        raise HTTPException(404, detail=f"No weather data for {spot} on {date}. Range: 2000-01-08 to 2025-12-31.")

    X = pd.DataFrame([[record[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    prob = float(xgb_model.predict_proba(X)[0][1])
    prob_pct = round(prob * 100.0, 1)

    next_date_str = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_rec = WEATHER_DB.get((spot, next_date_str))

    return {
        "spot": spot,
        "city": record["city"],
        "date": date,
        "work_stop_chance_percent": prob_pct,
        "alarm_triggered": bool(prob >= 0.25),
        "alarm_cutoff_percent": 25.0,
        "today_weather": {
            "rain_mm": float(record["rain_mm"]),
            "temp_max_c": float(record["temp_max"]),
            "wind_max_kmh": float(record["wind_max"]),
            "humidity_mean": int(record["humidity_mean"]),
            "work_stop_today": int(record["work_stop_today"])
        },
        "next_day_real_outcome": {
            "next_date": next_date_str,
            "rain_mm": float(next_rec["rain_mm"]),
            "work_stop_tomorrow": int(next_rec["work_stop_today"])
        } if next_rec else None
    }


@app.post("/pay-premium")
def pay_premium(
    spot: str = Query("Velachery", description="Delivery zone spot name"),
    week_start: str = Query("2024-11-25", description="Week start date (YYYY-MM-DD)")
):
    """POST /pay-premium — Create a PayPal Sandbox order for the weekly premium."""
    if spot not in AVAILABLE_SPOTS:
        raise HTTPException(400, detail=f"Unknown spot '{spot}'. Available: {AVAILABLE_SPOTS}")
    try:
        month = pd.to_datetime(week_start).month
    except Exception:
        raise HTTPException(400, detail="Invalid week_start.")

    info = get_weekly_premium(spot, month, 0)
    inr_amount = float(info["final_premium"])
    usd_amount = inr_to_usd(inr_amount)

    token = get_paypal_token()
    res = requests.post(
        f"{PAYPAL_BASE_URL}/v2/checkout/orders",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        json={
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": f"gigshield_prem_{int(time.time())}",
                "description": f"GigShield Weekly Coverage ({spot}, {week_start})",
                "amount": {"currency_code": CURRENCY, "value": f"{usd_amount:.2f}"}
            }],
            "application_context": {
                "return_url": "http://localhost:8000/capture-order",
                "cancel_url": "http://localhost:8000/"
            }
        },
        timeout=15
    )

    if res.status_code != 201:
        raise HTTPException(502, detail=f"PayPal order creation failed: {res.text}")

    order_data  = res.json()
    order_id    = order_data["id"]
    approval_url = next(
        (lnk["href"] for lnk in order_data.get("links", []) if lnk.get("rel") == "approve"),
        None
    )

    return {
        "status": "ORDER_CREATED",
        "order_id": order_id,
        "approval_url": approval_url,
        "spot": spot,
        "week_start": week_start,
        "premium_inr": inr_amount,
        "premium_usd": usd_amount,
        "currency": CURRENCY,
        "exchange_rate": f"Rs. {EXCHANGE_RATE:.0f} = 1 {CURRENCY}",
        "instructions": "Open approval_url in a browser to pay, then call POST /capture-order."
    }


@app.post("/capture-order")
@app.get("/capture-order")
def capture_order(
    order_id: Optional[str] = Query(None),
    token: Optional[str] = Query(None)
):
    """Capture an approved PayPal order. PayPal redirects here with ?token=<id>."""
    oid = order_id or token
    if not oid:
        raise HTTPException(400, detail="Missing order_id or token.")

    access_token = get_paypal_token()
    res = requests.post(
        f"{PAYPAL_BASE_URL}/v2/checkout/orders/{oid}/capture",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"},
        timeout=15
    )
    cap = res.json()
    if res.status_code in [200, 201]:
        return {"status": "COMPLETED", "order_id": oid, "capture_details": cap}
    return {
        "status": "NOT_COMPLETED",
        "order_id": oid,
        "paypal_status_code": res.status_code,
        "paypal_response": cap,
        "message": "Order not approved yet or capture failed. Approve on PayPal Sandbox first."
    }


@app.post("/check-claim")
def check_claim(
    spot: str = Query("Velachery", description="Delivery zone spot name"),
    date: str = Query("2023-12-04", description="Storm date (YYYY-MM-DD)")
):
    """
    POST /check-claim — Parametric payout trigger.
    Reads REAL rainfall. Only if PayPal Payouts returns HTTP 201 is the claim
    counted as paid. Any other response is returned as PAYOUT_FAILED with the
    full PayPal error.
    """
    if spot not in AVAILABLE_SPOTS:
        raise HTTPException(400, detail=f"Unknown spot '{spot}'. Available: {AVAILABLE_SPOTS}")

    record = WEATHER_DB.get((spot, date))
    if record is None:
        raise HTTPException(404, detail=f"No weather data for {spot} on {date}.")

    rain_mm         = float(record["rain_mm"])
    work_stop_today = int(record["work_stop_today"])
    city            = record["city"]
    week_str        = str(pd.to_datetime(date).to_period("W"))

    # --- Check 1: Rainfall below trigger threshold ---
    if work_stop_today == 0:
        return {
            "claim_status": "NO_CLAIM",
            "payout_approved": False,
            "spot": spot, "city": city, "date": date,
            "rain_mm": rain_mm,
            "work_stop_threshold_mm": 35.0,
            "message": f"Rain was {rain_mm:.1f} mm — below 35.0 mm threshold. No disruption recorded."
        }

    # --- Check 2: Weekly cap (only count PayPal-confirmed payouts) ---
    existing_claims = count_weekly_paid_claims(WORKER_EMAIL, week_str)
    if existing_claims >= MAX_PAYOUTS_PER_WEEK:
        return {
            "claim_status": "WEEKLY_CAP_REACHED",
            "payout_approved": False,
            "spot": spot, "city": city, "date": date,
            "week": week_str,
            "rain_mm": rain_mm,
            "weekly_paid_claims": existing_claims,
            "max_allowed_per_week": MAX_PAYOUTS_PER_WEEK,
            "message": f"Worker already received {existing_claims} PayPal-confirmed payouts in week {week_str}."
        }

    # --- Attempt PayPal Payout ---
    payout_inr = PAYOUT_INR
    payout_usd = inr_to_usd(payout_inr)
    paypal_token = get_paypal_token()
    batch_id     = f"gigshield_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    payout_res = requests.post(
        f"{PAYPAL_BASE_URL}/v1/payments/payouts",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {paypal_token}"},
        json={
            "sender_batch_header": {
                "sender_batch_id": batch_id,
                "email_subject": "GigShield Parametric Rain Relief",
                "email_message": (
                    f"Severe rain ({rain_mm:.1f} mm) in {spot} on {date}. "
                    f"Your income protection payout of ${payout_usd:.2f} is on its way!"
                )
            },
            "items": [{
                "recipient_type": "EMAIL",
                "amount": {"value": f"{payout_usd:.2f}", "currency": CURRENCY},
                "note": f"GigShield work-stop rain relief ({spot}, {date})",
                "sender_item_id": f"claim_{spot}_{date}_{uuid.uuid4().hex[:6]}",
                "receiver": WORKER_EMAIL
            }]
        },
        timeout=15
    )

    paypal_status_code = payout_res.status_code
    try:
        paypal_json = payout_res.json()
    except Exception:
        paypal_json = {}

    # 2. ONLY mark as success if PayPal returns HTTP 201
    if paypal_status_code == 201:
        real_batch_id = paypal_json.get("batch_header", {}).get("payout_batch_id", batch_id)

        # Save to log only on real PayPal success
        save_payout_entry({
            "timestamp":          datetime.now().isoformat(),
            "spot":               spot,
            "city":               city,
            "date":               date,
            "week":               week_str,
            "rain_mm":            rain_mm,
            "work_stop_today":    1,
            "worker_email":       WORKER_EMAIL,
            "payout_inr":         payout_inr,
            "payout_usd":         payout_usd,
            "currency":           CURRENCY,
            "payout_paypal_success": True,          # used by weekly cap counter
            "weekly_claim_index": existing_claims + 1,
            "paypal_batch_id":    real_batch_id,
            "paypal_status_code": 201,
        })

        return {
            "claim_status":    "APPROVED",
            "payout_approved": True,
            "spot": spot, "city": city, "date": date,
            "week": week_str,
            "rain_mm":         rain_mm,
            "work_stop_today": 1,
            "payout_inr":      payout_inr,
            "payout_usd":      payout_usd,
            "currency":        CURRENCY,
            "exchange_rate":   f"Rs. {EXCHANGE_RATE:.0f} = 1 {CURRENCY}",
            "worker_email":    WORKER_EMAIL,
            "weekly_claims_count": existing_claims + 1,
            "weekly_cap":      MAX_PAYOUTS_PER_WEEK,
            "paypal_batch_id": real_batch_id,
            "paypal_status_code": 201,
            "message": (
                f"Rain reached {rain_mm:.1f} mm. "
                f"PayPal payout of Rs. {payout_inr:.0f} (${payout_usd:.2f} USD) "
                f"sent to {WORKER_EMAIL}!"
            )
        }

    # 3. Any other PayPal response → PAYOUT_FAILED with full error text + debug id
    paypal_error_name    = paypal_json.get("name", "UNKNOWN")
    paypal_error_message = paypal_json.get("message", payout_res.text)
    # debug_id is in the JSON body; fall back to the Paypal-Debug-Id response header
    paypal_debug_id = (
        paypal_json.get("debug_id")
        or payout_res.headers.get("Paypal-Debug-Id")
        or "n/a"
    )

    return {
        "claim_status":    "PAYOUT_FAILED",
        "payout_approved": False,
        "spot": spot, "city": city, "date": date,
        "rain_mm":         rain_mm,
        "work_stop_today": 1,
        "payout_inr":      payout_inr,
        "payout_usd":      payout_usd,
        "worker_email":    WORKER_EMAIL,
        "paypal_status_code":    paypal_status_code,
        "paypal_error_name":     paypal_error_name,
        "paypal_error_message":  paypal_error_message,
        "paypal_debug_id":       paypal_debug_id,
        "paypal_full_response":  paypal_json,
        "message": (
            f"Rain qualified ({rain_mm:.1f} mm >= 35 mm) but PayPal Payouts refused: "
            f"[{paypal_status_code}] {paypal_error_name} — {paypal_error_message} "
            f"(debug_id: {paypal_debug_id})"
        )
    }


@app.get("/payouts")
def list_payouts():
    """Returns the last 20 logged payouts (PayPal-confirmed only)."""
    logs = load_payout_logs()
    return {"total_payouts": len(logs), "payouts": logs[-20:]}


@app.delete("/payouts/reset")
def reset_payout_log():
    """5. Clear the payout log so the weekly cap starts from zero."""
    if os.path.exists(PAYOUT_LOG_FILE):
        with open(PAYOUT_LOG_FILE, "w") as f:
            json.dump([], f)
    return {"status": "OK", "message": "Payout log cleared. Weekly cap reset to 0."}


# -----------------------------------------------------------------------------
# 5. FRONTEND
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    template_path = os.path.join(BASE_DIR, "app", "templates", "index.html")
    if os.path.exists(template_path):
        return FileResponse(template_path)
    return HTMLResponse("<h2>GigShield API running. (app/templates/index.html not found)</h2>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
