import { useState } from 'react'
import './index.css'
import ScreenPremium from './screens/ScreenPremium'
import ScreenForecast from './screens/ScreenForecast'
import ScreenPayout from './screens/ScreenPayout'

const TABS = [
  { key: 'premium',  label: '① Pay Premium' },
  { key: 'forecast', label: '② Forecast' },
  { key: 'payout',   label: '③ Payout Status' },
]

export default function App() {
  const [tab, setTab] = useState('premium')

  return (
    <>
      {/* Demo banner — shown on every screen */}
      <div className="demo-banner">
        DEMO - PayPal Sandbox. No real money moves.
      </div>

      {/* Nav tabs */}
      <nav className="nav">
        {TABS.map(t => (
          <button
            key={t.key}
            className={tab === t.key ? 'active' : ''}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Screens */}
      {tab === 'premium'  && <ScreenPremium />}
      {tab === 'forecast' && <ScreenForecast />}
      {tab === 'payout'   && <ScreenPayout />}
    </>
  )
}
