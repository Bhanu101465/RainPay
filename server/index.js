/**
 * RainPay Node proxy server
 * Calls the existing RainPay FastAPI endpoints.
 * All PayPal keys live in .env — never in this file.
 */
import "dotenv/config";
import express from "express";
import cors from "cors";

const app = express();
app.use(express.json());
app.use(cors({ origin: process.env.WEB_ORIGIN || "*" }));

// Base URL of the running FastAPI app (local or Render)
const API = process.env.FASTAPI_URL || "http://localhost:8000";

// ── helpers ──────────────────────────────────────────────────────────────────
async function apiFetch(path, method = "GET") {
  const { default: fetch } = await import("node-fetch");
  const res = await fetch(`${API}${path}`, { method });
  const json = await res.json();
  return { status: res.status, json };
}

// ── routes ───────────────────────────────────────────────────────────────────

/**
 * GET /api/premium?spot=...&week_start=...
 * Returns the weekly premium (Rs. and USD) for a spot.
 */
app.get("/api/premium", async (req, res) => {
  const { spot = "Velachery", week_start = "2024-11-25" } = req.query;
  const { status, json } = await apiFetch(
    `/premium?spot=${encodeURIComponent(spot)}&week_start=${encodeURIComponent(week_start)}`
  );
  res.status(status).json(json);
});

/**
 * POST /api/pay-premium?spot=...&week_start=...
 * Creates a PayPal Sandbox order and returns the approval link.
 */
app.post("/api/pay-premium", async (req, res) => {
  const { spot = "Velachery", week_start = "2024-11-25" } = req.query;
  const { status, json } = await apiFetch(
    `/pay-premium?spot=${encodeURIComponent(spot)}&week_start=${encodeURIComponent(week_start)}`,
    "POST"
  );
  res.status(status).json(json);
});

/**
 * GET /api/forecast?spot=...&date=...
 * Returns tomorrow's work-stop rain risk from the XGBoost model.
 */
app.get("/api/forecast", async (req, res) => {
  const { spot = "Velachery", date } = req.query;
  if (!date) return res.status(400).json({ error: "date is required (YYYY-MM-DD)" });
  const { status, json } = await apiFetch(
    `/forecast?spot=${encodeURIComponent(spot)}&date=${encodeURIComponent(date)}`
  );
  res.status(status).json(json);
});

/**
 * POST /api/check-claim?spot=...&date=...
 * Checks real rainfall and triggers PayPal payout if >= 35 mm.
 */
app.post("/api/check-claim", async (req, res) => {
  const { spot = "Velachery", date } = req.query;
  if (!date) return res.status(400).json({ error: "date is required (YYYY-MM-DD)" });
  const { status, json } = await apiFetch(
    `/check-claim?spot=${encodeURIComponent(spot)}&date=${encodeURIComponent(date)}`,
    "POST"
  );
  res.status(status).json(json);
});

/**
 * GET /api/payouts
 * Returns the last 20 PayPal-confirmed payouts.
 */
app.get("/api/payouts", async (req, res) => {
  const { status, json } = await apiFetch("/payouts");
  res.status(status).json(json);
});

/**
 * GET /api/health
 * Quick health check for Render uptime monitoring.
 */
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", fastapi: API });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () =>
  console.log(`RainPay proxy server running on http://localhost:${PORT}`)
);

