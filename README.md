# GigShield — Parametric Weather Insurance for Gig Workers

GigShield is a weather insurance for gig delivery workers.
An AI model (XGBoost) predicts work-stopping rain one day ahead.
PayPal collects the weekly premium and pays the claim automatically.

> Built for the PayPal AI Hackathon 2026.

---

## How it works

```
Worker pays the weekly premium  →  PayPal Orders API
         ↓
App checks real weather for the spot and date  →  weather_labeled_v2.csv
         ↓
If rain is 35 mm or more, the claim is paid  →  PayPal Payouts API
         ↓
Maximum 3 paid days per worker per week (cap enforced)
```

The **Check Forecast** button runs the XGBoost model on real weather
to give tomorrow's work-stopping rain risk (%), one day before it arrives.

---

## PayPal tools used

| Tool | Purpose |
| :--- | :--- |
| **PayPal Orders API** (`/v2/checkout/orders`) | Worker pays the weekly premium |
| **PayPal Payouts API** (`/v1/payments/payouts`) | App sends the claim payout to the worker |

Sandbox only. No real money moves.

---

## Run it yourself

### Requirements
- Python 3.10 or later
- A PayPal Developer account with a Sandbox app (see setup below)

### Install
```bash
pip install fastapi uvicorn requests python-dotenv pandas xgboost scikit-learn
```

### Create your .env file
```bash
# Copy the example, then fill in your credentials
cp .env.example .env
```

Open `.env` and fill in the values:
```env
PAYPAL_CLIENT_ID=your_sandbox_client_id
PAYPAL_SECRET=your_sandbox_secret
WORKER_EMAIL=your_sandbox_personal_email@example.com
CURRENCY=USD
EXCHANGE_RATE=83.0
```

### Start the server
```bash
uvicorn app.main:app --reload --port 8000
```

The terminal will print the first 6 letters of your Client ID and
the worker email so you can confirm the right credentials loaded.

### Open in browser
- **Web page (3 buttons):** http://localhost:8000/
- **API docs (Swagger):** http://localhost:8000/docs

---

## PayPal sandbox setup

1. Go to [developer.paypal.com](https://developer.paypal.com) and log in.
2. Under **Sandbox → Accounts**, create two test accounts:
   - **Business account** — country: United States (this is your app/merchant).
   - **Personal account** — country: United States (this is your test worker/recipient).
3. Under **Apps & Credentials**, click **Create App** linked to the Business account.
4. In the app settings, turn on **Payouts** (it is off by default).
5. Copy the **Client ID** and **Secret** into your `.env` file.
6. Use the Personal account's email as `WORKER_EMAIL` in `.env`.

> **Why US accounts?** PayPal Payouts is not available from Indian merchant accounts
> in the Sandbox. Creating US test accounts removes this restriction.

---

## Honest notes

- **Weather data is real.** Downloaded from Open-Meteo (2000 to 2025), 8 spots
  across Chennai and Visakhapatnam.
- **Storm days are replayed from real history.** No invented events.
- **PayPal Payouts from India is blocked in Sandbox.** Use US test accounts
  as described above. The API call is real; only the merchant country matters.
- **Amounts are in Rs.** Converted to USD at a fixed rate of Rs. 83 = 1 USD
  for PayPal. Both amounts are shown in every response.
- **Sandbox only.** The app points to `api-m.sandbox.paypal.com`. No real
  money is charged or sent.
- **Not a licensed insurance product.** This is a hackathon prototype.

---

## Screenshots

| Pay Premium | Check Forecast | Storm Day Replay |
| :---------: | :------------: | :--------------: |
| ![Pay Premium](docs/premium.png) | ![Check Forecast](docs/forecast.png) | ![Storm Replay](docs/payout.png) |

*(Replace the placeholder images with real screenshots after running the app.)*

---

## Target cities

- **Chennai** (4 spots: Adyar, Anna Nagar, Tambaram, Velachery)
- **Visakhapatnam** (4 spots: Gajuwaka, Maddilapalem, Madhurawada, Pendurthi)

### Why Bengaluru was dropped
Bengaluru had only 3–5 heavy rain days per spot over 26 real years (15 total).
Global weather grids smooth out localised cloudbursts, giving too few positive
cases to train a reliable model. Visakhapatnam was selected instead, with
173 heavy rain days and 617 work-stopping days across 4 spots.

---

## Rain & weather thresholds

| Label | Rule |
| :--- | :--- |
| **Heavy rain (IMD standard)** | ≥ 64.5 mm per day |
| **Work-stop rain (GigShield trigger)** | ≥ 35.0 mm per day |
| **Heat wave (IMD coastal rule)** | ≥ 40.0 °C max temperature |

---

## Zone risk scoring & weekly premium

Risk scores (0–100) are based on historical work-stop days per spot in
the train years 2000–2014, saved in `data/clean/zone_risk.csv`.

Annual forecasting test: a simple 5-year rolling average beat a Random Forest
regressor (MAE: 2.02 days vs 2.69 days). The simple average is used in pricing.

**Weekly premium formula** (`src/zone_risk.py`):

| Component | Amount |
| :--- | :--- |
| Base premium | Rs. 70 / week |
| Flood risk add-on | Rs. 5 to Rs. 15 (by risk score) |
| Monsoon season add-on | +Rs. 8 (Chennai: Oct–Dec; Vizag: Jun–Oct) |
| Loyalty discount | −Rs. 5 (4+ claim-free weeks); −Rs. 10 (12+ weeks) |
| Floor / cap | Rs. 60 to Rs. 100 / week |

---

## AI prediction model

**File:** `models/xgb_work_stop_final.json`

| Setting | Value |
| :--- | :--- |
| Algorithm | XGBoost classifier |
| `max_depth` | 3 (shallow, to prevent memorising) |
| `scale_pos_weight` | 20 (heavy rain days are rare) |
| `reg_lambda` | 2 (L2 regularisation) |
| Train years | 2000–2014 |
| Validation years | 2015–2018 (used for early stopping & cutoff) |
| Test years | 2019–2025 (reported once, never used for tuning) |
| Target | `work_stop_tomorrow` (≥ 35 mm next day, same spot only) |

**Performance on test years:**
- Training recall and validation recall match closely (~60%) — no overfitting.
- Alarm cutoff 0.20–0.30 catches ~60% of real work-stopping days one day ahead.
- Storm-level precision: 40–50% (most false alarms are on adjacent rainy days or rain of 15–35 mm).

---

## Solvency check (test years 2019–2025)

*1,000 workers per city (250 per spot), max 3 paid days per week.*

| Entity | 7-year premiums | 7-year payouts (Rs. 400/day) | Net surplus | Loss ratio |
| :--- | :---: | :---: | :---: | :---: |
| Chennai | Rs. 30,19,468 | Rs. 16,00,000 | +Rs. 14,19,468 | 53.0% |
| Visakhapatnam | Rs. 32,76,700 | Rs. 21,40,000 | +Rs. 11,36,700 | 65.3% |
| **Combined fund** | **Rs. 62,96,168** | **Rs. 37,40,000** | **+Rs. 25,56,168** | **59.4%** |

Optimal payout for a 70% target loss ratio: ~Rs. 470 per work-stop day.

---

## 26-year historical stress test (2000–2025)

### Years with loss ratio above 100% (payouts exceeded premiums)

| Payout | Chennai (26 yrs) | Visakhapatnam (26 yrs) | Combined fund |
| :---: | :---: | :---: | :---: |
| Rs. 400 | 0 years | 1 year (2010: 119.8%) | 0 years (worst: 2010 at 82.3%) |
| Rs. 450 | 0 years | 1 year (2010: 134.8%) | 0 years (worst: 2010 at 92.6%) |
| Rs. 470 | 0 years | 3 years (2003, 2010, 2024) | 0 years (worst: 2010 at 96.8%) |

### Worst 5 years for the combined fund

- **Rs. 400:** 2010 (82.3%), 2024 (78.9%), 2021 (65.7%), 2025 (65.6%), 2020 (61.1%)
- **Rs. 450:** 2010 (92.6%), 2024 (88.8%), 2021 (73.9%), 2025 (73.8%), 2020 (68.8%)
- **Rs. 470:** 2010 (96.8%), 2024 (92.8%), 2021 (77.1%), 2025 (77.1%), 2020 (71.8%)

### Emergency reserve needed

By pooling Chennai (northeast monsoon) and Visakhapatnam (southwest monsoon),
risk is naturally spread across different seasons.

| Payout | Reserve needed | Per worker (2,000 workers) |
| :---: | :---: | :---: |
| Rs. 400 | Rs. 30 Lakh | ~Rs. 1,500 |
| Rs. 450 | Rs. 37 Lakh | ~Rs. 1,850 |
| Rs. 470 | Rs. 40 Lakh | ~Rs. 2,000 |

---

## FastAPI web app (`app/`)

### API endpoints

| Method | Endpoint | What it does |
| :--- | :--- | :--- |
| `GET` | `/premium?spot=...&week_start=...` | Returns the weekly premium (Rs. and USD) |
| `GET` | `/forecast?spot=...&date=...` | XGBoost storm risk % for tomorrow |
| `POST` | `/pay-premium?spot=...&week_start=...` | Creates PayPal Sandbox order for the premium |
| `GET/POST` | `/capture-order?order_id=...` | Captures an approved PayPal order |
| `POST` | `/check-claim?spot=...&date=...` | Checks real rain; pays via PayPal Payouts if ≥ 35 mm |
| `GET` | `/payouts` | Lists PayPal-confirmed payouts from the log |
| `DELETE` | `/payouts/reset` | Clears the log and resets the weekly cap |
| `GET` | `/` | Interactive web page with 3 buttons |

---

## Project files

### Source (`src/`)
| File | Purpose |
| :--- | :--- |
| `download_weather.py` | Downloads real weather for Chennai from Open-Meteo |
| `download_vizag.py` | Downloads real weather for Visakhapatnam from Open-Meteo |
| `make_labels_v2.py` | Combines cities, applies IMD labels, creates safe shifted targets |
| `check_v2.py` | Verifies label safety and shows train/test distributions |
| `train_regularized.py` | Trains regularised XGBoost with 3-way time split |
| `zone_risk.py` | Zone risk scores and weekly premium calculator |
| `check_heat.py` | IMD coastal heat wave analysis |
| `solvency_check.py` | 2019–2025 solvency replay |
| `stress_test.py` | 26-year stress test across Rs. 400, 450, 470 payouts |

### Clean datasets (`data/clean/`)
| File | Contents |
| :--- | :--- |
| `weather_labeled_v2.csv` | 75,976 rows, Chennai + Visakhapatnam, 2000–2025 |
| `zone_risk.csv` | Risk scores (0–100) per spot |
| `vizag_daily.csv` | Daily weather for Visakhapatnam |
| `weather_daily.csv` | Daily weather for Chennai & Bengaluru |
| `payout_log.json` | PayPal-confirmed claim payouts |

### Models (`models/` — ignored by Git)
| File | Contents |
| :--- | :--- |
| `xgb_work_stop_final.json` | Final XGBoost classifier (load with `load_model`) |
| `xgb_work_stop_final.pkl` | Same model, pickle format |
| `rf_zone_risk.pkl` | Random Forest for annual forecasting (baseline lost) |

---

## Data credit

> Weather data from [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0).
> Free tier is for non-commercial use only.

---

## License

MIT © 2026 Bhanu Prakash Akkireddi — see [LICENSE](LICENSE).
