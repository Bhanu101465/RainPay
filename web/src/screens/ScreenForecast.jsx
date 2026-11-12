/**
 * Screen 2 — Tomorrow's warning.
 * Shows rain risk for the worker's area using the XGBoost model.
 */
import { useState } from 'react'
import { API, TODAY } from '../config'

// Famous real storm days for quick testing
const PRESETS = [
  { label: 'Michaung (191.8 mm) — Day Before', spot: 'Velachery',   date: '2023-12-03' },
  { label: 'Chennai Nov 2021 Flood — Day Before', spot: 'Adyar',    date: '2021-11-10' },
  { label: 'Vizag Downpour — Day Before',          spot: 'Pendurthi', date: '2015-10-31' },
  { label: 'Dry summer day',                        spot: 'Velachery', date: '2024-04-10' },
]

function riskColor(pct) {
  if (pct >= 60) return '#d83b01'
  if (pct >= 30) return '#f7a800'
  return '#107c41'
}

export default function ScreenForecast() {
  const [spot, setSpot]   = useState('Velachery')
  const [date, setDate]   = useState('2023-12-03')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  async function checkForecast() {
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await fetch(
        `${API}/api/forecast?spot=${encodeURIComponent(spot)}&date=${encodeURIComponent(date)}`
      )
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || data.error || 'Failed')
      setResult(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  function applyPreset(p) { setSpot(p.spot); setDate(p.date) }

  const pct = result?.work_stop_chance_percent ?? 0

  return (
    <div className="page">
      <div className="card">
        <h2>Tomorrow's Rain Warning</h2>
        <p style={{ fontSize: 13, color: '#697386', marginBottom: 14 }}>
          The XGBoost model predicts the chance of work-stopping rain
          (≥ 35 mm) tomorrow, based on today's real weather.
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

        <label>Date (checks forecast for the next day)</label>
        <input
          type="date"
          value={date}
          onChange={e => setDate(e.target.value)}
        />

        {/* Quick presets */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, color: '#697386', marginBottom: 6 }}>
            Quick test dates:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {PRESETS.map(p => (
              <button
                key={p.label}
                onClick={() => applyPreset(p)}
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

        <button className="btn btn-purple" onClick={checkForecast} disabled={loading}>
          {loading ? 'Running model…' : "Check Tomorrow's Risk"}
        </button>
      </div>

      {error && <div className="alert alert-fail">{error}</div>}

      {result && (
        <div className="card">
          <h2>{result.spot} · {date}</h2>

          <div style={{ marginBottom: 6, fontWeight: 600, fontSize: 15 }}>
            Tomorrow's work-stop chance:&nbsp;
            <span style={{ color: riskColor(pct), fontSize: 24 }}>{pct}%</span>
          </div>

          {/* Risk bar */}
          <div className="risk-bar-wrap">
            <div
              className="risk-bar"
              style={{ width: `${pct}%`, background: riskColor(pct) }}
            />
          </div>

          <div className="alert" style={{
            background: result.alarm_triggered ? '#fed9cc' : '#dff6dd',
            border: `1px solid ${result.alarm_triggered ? '#f4a070' : '#a8d5a2'}`,
            marginTop: 12
          }}>
            {result.alarm_triggered
              ? `⚠ ALARM — chance above ${result.alarm_cutoff_percent}%. Prepare for a heavy rain day.`
              : `✓ LOW RISK — chance is below ${result.alarm_cutoff_percent}%. Deliveries should be fine.`}
          </div>

          <div className="stats" style={{ marginTop: 14 }}>
            <div className="stat">
              <div className="stat-label">Today's Rain</div>
              <div className="stat-value">{result.today_weather.rain_mm.toFixed(1)} mm</div>
            </div>
            <div className="stat">
              <div className="stat-label">Max Temp</div>
              <div className="stat-value">{result.today_weather.temp_max_c.toFixed(1)} °C</div>
            </div>
          </div>

          {result.next_day_real_outcome && (
            <div className="alert alert-info" style={{ marginTop: 12, fontSize: 13 }}>
              <strong>What really happened next day</strong> ({result.next_day_real_outcome.next_date}):{' '}
              {result.next_day_real_outcome.rain_mm} mm —&nbsp;
              {result.next_day_real_outcome.work_stop_tomorrow
                ? '🌧 Work-stopping rain confirmed.'
                : '☀ No work disruption.'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

