import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const STATUS_COLOR = { ok: 'success', true: 'success', false: 'danger' }

function StatusDot({ ok }) {
  return (
    <span className={`status-pill ${ok ? 'ok' : 'fail'}`}>
      <span className="status-dot-inline" /> {ok ? 'OK' : 'FAIL'}
    </span>
  )
}

function Metric({ label, value, sub }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value accent">{value}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  )
}

function parsePromText(text) {
  const out = {}
  for (const line of text.split('\n')) {
    if (!line || line.startsWith('#')) continue
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+\-]+)/)
    if (m) {
      const name = m[1]
      if (!(name in out)) out[name] = 0
      out[name] += parseFloat(m[3])
    }
  }
  return out
}

export default function HealthTab() {
  const [health, setHealth] = useState(null)
  const [ready,  setReady]  = useState(null)
  const [prom,   setProm]   = useState(null)
  const [err,    setErr]    = useState(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const [h, r, p] = await Promise.all([
          fetch(`${API}/health`).then(x => x.json()),
          fetch(`${API}/readyz`).then(x => x.json()),
          fetch(`${API}/metrics/prom`).then(x => x.text()),
        ])
        if (!alive) return
        setHealth(h)
        setReady(r)
        setProm(parsePromText(p))
        setErr(null)
      } catch (e) {
        if (alive) setErr(e.message)
      }
    }
    load()
    const id = setInterval(load, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  if (err && !health) {
    return (
      <div className="card">
        <span className="card-title">❤️ System Health</span>
        <div className="health-error">
          API unreachable: <code>{err}</code>
        </div>
      </div>
    )
  }

  const checks = ready?.checks || {}
  const predTotal = prom?.['fraud_predictions_total'] ?? 0
  const fraudTotal = prom?.['fraud_predictions_fraud_total'] ?? 0
  const fraudPct = predTotal ? (fraudTotal / predTotal * 100).toFixed(2) : '0.00'

  return (
    <div>
      <div className="card">
        <span className="card-title">❤️ System Health</span>
        <div className="health-grid">
          <div className="health-row">
            <span className="health-row-label">API status</span>
            <StatusDot ok={health?.status === 'ok'} />
            <code className="health-row-value">{health?.status ?? '–'}</code>
          </div>
          <div className="health-row">
            <span className="health-row-label">Model loaded</span>
            <StatusDot ok={!!health?.model_loaded} />
            <code className="health-row-value">{health?.model_name ?? '–'}</code>
          </div>
          <div className="health-row">
            <span className="health-row-label">Readiness</span>
            <StatusDot ok={!!ready?.ready} />
            <code className="health-row-value">{ready?.ready ? 'ready' : 'not ready'}</code>
          </div>
          <div className="health-sub">
            <strong>Readiness checks:</strong>
            <ul>
              {Object.entries(checks).map(([k, v]) => (
                <li key={k}>
                  <code>{k}</code>
                  <StatusDot ok={!!v.ok} />
                  {v.error && <span className="health-err"> — {v.error}</span>}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="metrics-bar" style={{ marginTop: 16 }}>
        <Metric
          label="Predictions (Prometheus)"
          value={predTotal.toLocaleString()}
          sub="fraud_predictions_total"
        />
        <Metric
          label="Flagged as fraud"
          value={fraudTotal.toLocaleString()}
          sub="fraud_predictions_fraud_total"
        />
        <Metric
          label="Fraud %"
          value={`${fraudPct}%`}
          sub="flagged / total"
        />
        <Metric
          label="API version"
          value={health?.version ?? ready?.version ?? '–'}
          sub="build info"
        />
      </div>

      <details className="card" style={{ marginTop: 16 }}>
        <summary className="card-title" style={{ cursor: 'pointer' }}>
          🛰️ Raw Prometheus metrics
        </summary>
        <pre className="raw-prom">{JSON.stringify(prom, null, 2)}</pre>
      </details>
    </div>
  )
}
