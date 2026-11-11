# RainPay

Instant PayPal payouts for gig workers when heavy rain stops their work.

## The problem
Delivery and gig workers lose income on heavy-rain days. They have no safety net.

## What it does
- Worker pays a small weekly premium through PayPal.
- The app checks tomorrow's rain risk.
- When a heavy-rain day stops work, the worker gets a payout through PayPal.

## How PayPal is used
- Premium payment (Orders / Checkout)
- Worker payout (Payouts API)
- Tested in the PayPal sandbox

## How AI is used
- XGBoost model warns when heavy rain (35 mm or more) will stop work tomorrow.
- Random Forest zone risk score for each area.
- Weekly premium formula based on zone risk, monsoon season, and loyalty.
- AI tool used for building: Antigravity.

## Data (all real, no fake data)
- Daily weather from Open-Meteo, 2000-2025
- 8 spots: 4 in Chennai, 4 in Visakhapatnam
- Heavy-rain rules follow IMD

## Results
- Test years 2019-2025: model catches about 50% of work-stop days, with 22% precision.
- Money check (Rs. 400 payout per day, max 3 per week): loss ratio 59.4% over 2019-2025.
- 26-year stress test: fund stayed under 82.3% loss ratio overall.

## What's new in this hackathon
This started from my earlier project, GigShield (a fraud and risk pipeline). For this hackathon I built a new, focused version:
- Real weather data and a new rain-warning model
- Zone risk score and weekly premium
- Money checks on 26 years of history
- A working PayPal app with premium payment and payout
The old fraud-detection parts were left out on purpose.

## How to run
1. git clone https://github.com/Bhanu101465/RainPay.git
2. cd RainPay
3. pip install -r requirements.txt
4. cp .env.example .env  (then fill in your PayPal sandbox credentials)
5. uvicorn app.main:app --reload --port 8000
6. Open http://localhost:8000 in your browser

> **PayPal sandbox note:** Create a US Business account and a US Personal account
> in the PayPal Developer dashboard. Turn on Payouts for your app.
> Use the Business credentials in `.env` and the Personal email as `WORKER_EMAIL`.
