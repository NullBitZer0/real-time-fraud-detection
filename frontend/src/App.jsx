import { useState, useEffect, useRef, useCallback } from 'react'
import MetricsBar  from './components/MetricsBar'
import LiveFeed    from './components/LiveFeed'
import FraudChart  from './components/FraudChart'
import DemoResults from './components/DemoResults'
import DriftTab    from './components/DriftTab'

const API  = import.meta.env.VITE_API_URL ?? ''
const WS   = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`
const MAX_EVENTS    = 50
const MAX_CHART_PTS = 60

const TABS = [
  { id: 'live', label: '📊 Live' },
  { id: 'demo', label: '🧪 Demo' },
  { id: 'drift', label: '🌊 Drift' },
]

const EMPTY_METRICS = {
  total_transactions: 0, total_fraud: 0,
  fraud_rate_pct: 0, uptime_seconds: 0,
}

export default function App() {
  const [tab,        setTab]        = useState('live')
  const [metrics,    setMetrics]    = useState(EMPTY_METRICS)
  const [events,     setEvents]     = useState([])
  const [chartData,  setChartData]  = useState([])
  const [wsStatus,   setWsStatus]   = useState('disconnected')
  const [demoResult, setDemoResult] = useState(null)
  const [demoBusy,   setDemoBusy]   = useState(false)
  const wsRef   = useRef(null)
  const tickRef = useRef(0)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const r = await fetch(`${API}/metrics`)
        if (r.ok) setMetrics(await r.json())
      } catch (_) {}
    }
    fetchMetrics()
    const id = setInterval(fetchMetrics, 3000)
    return () => clearInterval(id)
  }, [])

  const connect = useCallback(() => {
    const ws = new WebSocket(WS)
    wsRef.current = ws

    ws.onopen  = () => setWsStatus('connected')
    ws.onclose = () => {
      setWsStatus('disconnected')
      setTimeout(connect, 3000)
    }
    ws.onerror = () => ws.close()

    ws.onmessage = (e) => {
      const txn = JSON.parse(e.data)
      tickRef.current += 1
      setEvents(prev => [txn, ...prev].slice(0, MAX_EVENTS))
      setChartData(prev => {
        const rate = txn.fraud_probability * 100
        const next = [...prev, { t: tickRef.current, rate: +rate.toFixed(3) }]
        return next.slice(-MAX_CHART_PTS)
      })
    }
  }, [])
  useEffect(() => { connect() }, [connect])

  const runDemo = async (n) => {
    setDemoBusy(true)
    try {
      const half = n / 2
      const r = await fetch(`${API}/demo/run-100-tests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_fraud: half, n_legit: half, include_results: n <= 100 }),
      })
      if (r.ok) {
        setDemoResult(await r.json())
        setTab('demo')
      }
    } catch (err) {
      console.error('Demo failed:', err)
    } finally {
      setDemoBusy(false)
    }
  }

  return (
    <div className="dashboard">
      <header className="header">
        <h1>🔍 <span>Fraud</span> Detection Dashboard</h1>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <button
            className="run-demo-btn"
            onClick={() => runDemo(100)}
            disabled={demoBusy}
          >
            {demoBusy ? '⏳ Running…' : '▶ Run 100'}
          </button>
          <button
            className="run-demo-btn large"
            onClick={() => runDemo(1000)}
            disabled={demoBusy}
          >
            {demoBusy ? '⏳ Running…' : '▶ Run 1000'}
          </button>
          <div className="live-badge">
            <div className="live-dot" />
            {wsStatus === 'connected' ? 'LIVE' : 'RECONNECTING…'}
          </div>
          <a
            href={`${API}/docs`}
            target="_blank" rel="noreferrer"
            style={{ fontSize: '12px', color: 'var(--accent)', textDecoration: 'none' }}
          >API Docs ↗</a>
          <a
            href="https://grafana.adeeshaperera.me"
            target="_blank" rel="noreferrer"
            style={{ fontSize: '12px', color: 'var(--accent)', textDecoration: 'none' }}
          >Grafana ↗</a>
          <a
            href="https://airflow.adeeshaperera.me"
            target="_blank" rel="noreferrer"
            style={{ fontSize: '12px', color: 'var(--accent)', textDecoration: 'none' }}
          >Airflow ↗</a>
        </div>
      </header>

      <nav className="tab-bar">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="tab-content">
        {tab === 'live' && (
          <>
            <MetricsBar metrics={metrics} />
            <div className="main-grid">
              <LiveFeed events={events} />
              <FraudChart chartData={chartData} />
            </div>
          </>
        )}

        {tab === 'demo' && (
          <>
            {demoResult
              ? <DemoResults result={demoResult} />
              : (
                <div className="card">
                  <span className="card-title">🧪 Demo Results</span>
                  <p className="monitoring-hint">
                    Click <strong>▶ Run 100</strong> or <strong>▶ Run 1000</strong>
                    in the header to run the demo. The 1000-test run shows
                    summary only (top-100 detail table hidden).
                    It samples evenly from
                    <code>data/raw/fraudTest.csv</code>, scores them, and
                    populates this tab.
                  </p>
                </div>
              )
            }
          </>
        )}

        {tab === 'drift' && <DriftTab />}
      </div>

      <div className="status-bar">
        <div className={`status-dot ${wsStatus}`} />
        <span>WebSocket: {wsStatus}</span>
        <span style={{ marginLeft: 'auto' }}>
          API: {API} · T2: 0.1198 · Model: v3 (PR-AUC 0.8384)
        </span>
      </div>
    </div>
  )
}
