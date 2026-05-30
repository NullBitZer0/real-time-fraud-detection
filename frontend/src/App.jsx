import { useState, useEffect, useRef, useCallback } from 'react'
import MetricsBar from './components/MetricsBar'
import LiveFeed   from './components/LiveFeed'
import FraudChart from './components/FraudChart'

const API  = 'http://localhost:8000'
const WS   = 'ws://localhost:8000/ws'
const MAX_EVENTS    = 50   // keep last 50 in feed
const MAX_CHART_PTS = 60   // last 60 data points in chart

const EMPTY_METRICS = {
  total_transactions: 0,
  total_fraud: 0,
  fraud_rate_pct: 0,
  avg_latency_ms: 0,
  uptime_seconds: 0,
}

export default function App() {
  const [metrics,   setMetrics]   = useState(EMPTY_METRICS)
  const [events,    setEvents]    = useState([])
  const [chartData, setChartData] = useState([])
  const [wsStatus,  setWsStatus]  = useState('disconnected')
  const wsRef   = useRef(null)
  const tickRef = useRef(0)

  // ── Fetch metrics from REST every 3s ──────────────────────────────────────
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

  // ── WebSocket — live transaction stream ────────────────────────────────────
  const connect = useCallback(() => {
    const ws = new WebSocket(WS)
    wsRef.current = ws

    ws.onopen  = () => setWsStatus('connected')
    ws.onclose = () => {
      setWsStatus('disconnected')
      setTimeout(connect, 3000)  // auto-reconnect
    }
    ws.onerror = () => ws.close()

    ws.onmessage = (e) => {
      const txn = JSON.parse(e.data)
      tickRef.current += 1

      // Prepend to feed (newest at top)
      setEvents(prev => [txn, ...prev].slice(0, MAX_EVENTS))

      // Append to chart
      setChartData(prev => {
        const rate = txn.fraud_probability * 100
        const next = [...prev, { t: tickRef.current, rate: +rate.toFixed(3) }]
        return next.slice(-MAX_CHART_PTS)
      })
    }
  }, [])

  useEffect(() => { connect() }, [connect])

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <h1>🔍 <span>Fraud</span> Detection Dashboard</h1>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <div className="live-badge">
            <div className="live-dot" />
            {wsStatus === 'connected' ? 'LIVE' : 'RECONNECTING…'}
          </div>
          <a
            href={`${API}/docs`}
            target="_blank"
            rel="noreferrer"
            style={{ fontSize: '12px', color: 'var(--accent)', textDecoration: 'none' }}
          >
            API Docs ↗
          </a>
        </div>
      </header>

      {/* Metrics */}
      <MetricsBar metrics={metrics} />

      {/* Main Grid */}
      <div className="main-grid">
        <LiveFeed events={events} />
        <FraudChart chartData={chartData} />
      </div>

      {/* Status Bar */}
      <div className="status-bar">
        <div className={`status-dot ${wsStatus}`} />
        <span>WebSocket: {wsStatus}</span>
        <span style={{ marginLeft: 'auto' }}>
          API: {API} · Threshold: 0.35 · Model: LightGBM (PR-AUC 0.888)
        </span>
      </div>
    </div>
  )
}
