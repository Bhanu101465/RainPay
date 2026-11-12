/**
 * Screen 1 — Sign up and pay the weekly premium through PayPal Sandbox.
 */
import { useState } from 'react'
import { API, SPOTS, TODAY } from '../config'

export default function ScreenPremium() {
  const [spot, setSpot]           = useState('Velachery')
  const [weekStart, setWeekStart] = useState(TODAY)
  const [info, setInfo]           = useState(null)   // premium info
  const [order, setOrder]         = useState(null)   // PayPal order
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState('')

  async function fetchPremium() {
    setLoading(true); setError(''); setInfo(null); setOrder(null)
    try {
      const res = await fetch(
        `${API}/api/premium?spot=${encodeURIComponent(spot)}&week_start=${encodeURIComponent(weekStart)}`
      )
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || data.error || 'Failed')
      setInfo(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  async function createOrder() {
    setLoading(true); setError(''); setOrder(null)
    try {
      const res = await fetch(
        `${API}/api/pay-premium?spot=${encodeURIComponent(spot)}&week_start=${encodeURIComponent(weekStart)}`,
        { method: 'POST' }
      )
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || data.error || 'PayPal order failed')
      setOrder(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="page">
      <div className="card">
        <h2>Pay Weekly Premium</h2>
        <p style={{ fontSize: 13, color: '#697386', marginBottom: 16 }}>
          Choose your delivery zone and week. The premium is based on the zone's
          historical flood risk and the current monsoon season.
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

        <label>Week starting</label>
        <input
          type="date"
          value={weekStart}
          onChange={e => setWeekStart(e.target.value)}
        />

        <button className="btn btn-blue" onClick={fetchPremium} disabled={loading}>
          {loading ? 'Loading…' : 'Check Premium'}
        </button>
      </div>

      {error && <div className="alert alert-fail">{error}</div>}

      {info && (
        <div className="card">
          <h2>{info.spot} · {info.city}</h2>
          <div className="stats">
            <div className="stat">
              <div className="stat-label">Weekly Premium (Rs.)</div>
              <div className="stat-value">Rs. {info.premium_inr}</div>
            </div>
            <div className="stat">
              <div className="stat-label">In USD</div>
              <div className="stat-value">${info.premium_usd}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Zone Risk Score</div>
              <div className="stat-value">{info.risk_score}/100</div>
            </div>
            <div className="stat">
              <div className="stat-label">Monsoon Month?</div>
              <div className="stat-value">{info.is_monsoon ? '☔ Yes' : '☀ No'}</div>
            </div>
          </div>

          {!order && (
            <button
              className="btn btn-blue"
              style={{ marginTop: 16 }}
              onClick={createOrder}
              disabled={loading}
            >
              {loading ? 'Creating order…' : '→ Pay on PayPal Sandbox'}
            </button>
          )}

          {order && (
            <div className="alert alert-info" style={{ marginTop: 16 }}>
              <strong>PayPal Order created!</strong>
              <br />
              Order ID: <code>{order.order_id}</code>
              <br /><br />
              <a
                href={order.approval_url}
                target="_blank"
                rel="noreferrer"
                style={{
                  display: 'inline-block', background: '#0070ba', color: '#fff',
                  padding: '8px 14px', borderRadius: 6, fontWeight: 700,
                  textDecoration: 'none'
                }}
              >
                Approve on PayPal Sandbox →
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

