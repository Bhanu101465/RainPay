/**
 * Screen 3 — Payout status.
 * Checks real historical rainfall; if >= 35 mm, triggers PayPal Payouts.
 * Also shows the last 20 PayPal-confirmed payouts from the log.
 */
import { useState, useEffect } from 'react'
import { API } from '../config'

export default function ScreenPayout() {
  const [spot, setSpot]   = useState('Velachery')
  const [date, setDate]   = useState('2023-12-04')
  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [logs, setLogs]       = useState([])

  useEffect(() => { loadLogs() }, [])

  async function loadLogs() {
    try {
      const res = await fetch(`${API}/api/payouts`)
      const data = await res.json()
      setLogs(data.payouts?.reverse() || [])
    } catch { /* ignore */ }
  }

  async function checkClaim() {
    setLoading(true); setError(''); setClaim(null)
    try {
      const res = await fetch(
        `${API}/api/check-claim?spot=${encodeURIComponent(spot)}&date=${encodeURIComponent(date)}`,
        { method: 'POST' }
      )
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || data.error || 'Failed')
      setClaim(data)
      if (data.claim_status === 'APPROVED') loadLogs()
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const statusStyle = {
    'APPROVED':           { cls: 'alert-success', icon: '✅ PAID via PayPal' },
    'PAYOUT_FAILED':      { cls: 'alert-fail',    icon: '❌ PayPal refused' },
    'WEEKLY_CAP_REACHED': { cls: 'alert-fail',    icon: '🚫 Weekly cap reached' },
    'NO_CLAIM':           { cls: 'alert-info',    icon: '☀ No disruption' },
  }

  const ss = claim ? (statusStyle[claim.claim_status] || { cls:'alert-info', icon: claim.claim_status }) : null

  return (
    <div className="page">
      <div className="card">
        <h2>Check Payout Status</h2>
        <p style={{ fontSize: 13, color: '#697386', marginBottom: 14 }}>
          Pick a storm day. If real rain was ≥ 35 mm, Rs. 400 is sent
          to the worker via PayPal Payouts (max 3 days per week).
        </p>

        <label>Delivery spot</label>
        <select value={spot} onChange={e => setSpot(e.target.value)}>
          <optgroup label="Chennai">
            {['Adyar','Anna Nagar','Tambaram','Velachery'].map(s =>
              <option key={s}>{s}</option>
            )}
          </optgroup>
          <optgroup label="Visakhapatnam">
            {['Gajuwaka','Maddilapalem','Madhurawada','Pendurthi'].map(s =>
              <option key={s}>{s}</option>
            )}
          </optgroup>
        </select>

        <label>Storm date (use a real heavy-rain day)</label>
        <input
          type="date"
          value={date}
          onChange={e => setDate(e.target.value)}
        />

        {/* Famous storm presets */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, color: '#697386', marginBottom: 6 }}>Real storm days:</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {[
              { label: 'Michaung 2023 (191.8 mm)', spot: 'Velachery',   date: '2023-12-04' },
              { label: 'Chennai Nov 2021 (207.6 mm)', spot: 'Adyar',    date: '2021-11-11' },
              { label: 'Vizag 2015 (285.3 mm)',        spot: 'Pendurthi', date: '2015-11-01' },
              { label: 'Dry day (0 mm)',                spot: 'Velachery', date: '2024-04-10' },
            ].map(p => (
              <button
                key={p.label}
                onClick={() => { setSpot(p.spot); setDate(p.date) }}
                style={{
                  background: '#eef2f6', border: 'none', borderRadius: 6,
                  padding: '4px 10px', fontSize: 12, cursor: 'pointer'
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <button className="btn btn-green" onClick={checkClaim} disabled={loading}>
          {loading ? 'Checking…' : 'Check & Pay Claim'}
        </button>
      </div>

      {error && <div className="alert alert-fail">{error}</div>}

      {claim && (
        <div className="card">
          <h2>{claim.spot} · {date}</h2>
          <div className={`alert ${ss.cls}`}>{ss.icon}</div>
          <p style={{ marginTop: 10, fontSize: 14 }}>{claim.message}</p>

          <div className="stats" style={{ marginTop: 14 }}>
            <div className="stat">
              <div className="stat-label">Actual Rain</div>
              <div className="stat-value">{claim.rain_mm?.toFixed(1)} mm</div>
            </div>
            <div className="stat">
              <div className="stat-label">Payout</div>
              <div className="stat-value">
                {claim.payout_approved ? `Rs. ${claim.payout_inr} ($${claim.payout_usd})` : 'Rs. 0'}
              </div>
            </div>
          </div>

          {claim.claim_status === 'PAYOUT_FAILED' && (
            <div className="alert alert-fail" style={{ marginTop: 12, fontSize: 13 }}>
              <strong>PayPal error:</strong> [{claim.paypal_status_code}] {claim.paypal_error_name}<br />
              {claim.paypal_error_message}<br />
              <strong>Debug ID:</strong> <code>{claim.paypal_debug_id}</code>
            </div>
          )}

          {claim.claim_status === 'APPROVED' && (
            <div className="alert alert-info" style={{ marginTop: 12, fontSize: 13 }}>
              <strong>PayPal Batch ID:</strong> <code>{claim.paypal_batch_id}</code><br />
              <strong>Sent to:</strong> {claim.worker_email}<br />
              <strong>Week cap used:</strong> {claim.weekly_claims_count} / {claim.weekly_cap}
            </div>
          )}
        </div>
      )}

      {/* Payout log */}
      {logs.length > 0 && (
        <div className="card">
          <h2>Recent PayPal Payouts</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc', color: '#697386' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Date</th>
                  <th>Spot</th>
                  <th>Rain</th>
                  <th>Payout</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((p, i) => (
                  <tr key={i} style={{ borderTop: '1px solid #e3e8ee' }}>
                    <td style={{ padding: '8px 10px' }}><strong>{p.date}</strong></td>
                    <td style={{ textAlign: 'center' }}>{p.spot}</td>
                    <td style={{ textAlign: 'center' }}>{p.rain_mm?.toFixed(1)} mm</td>
                    <td style={{ textAlign: 'center' }}>Rs.{p.payout_inr} / ${p.payout_usd}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

