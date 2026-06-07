import { useState } from 'react'

const GRAFANA_PANEL_TITLES = {
  'totals':    'Total transactions vs fraud detected',
  'fraud-pct': 'Fraud rate %',
  'latency':   'p50 / p95 / p99 latency (ms)',
  'qps':       'Throughput (predictions / sec)',
}

const GRAFANA_DASH = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3001'

export default function MonitoringTab() {
  const [src, setSrc] = useState(`${GRAFANA_DASH}/d/fraud-detection?kiosk`)
  const [errored, setErrored] = useState(false)

  return (
    <div className="card monitoring-card">
      <span className="card-title">📈 Monitoring — Grafana + Prometheus</span>
      <div className="monitoring-bar">
        <span className="monitoring-hint">
          Live metrics scraped from <code>/metrics/prom</code> on the API.
        </span>
        <button
          className="run-demo-btn small"
          onClick={() => setSrc(`${GRAFANA_DASH}/d/fraud-detection?kiosk&t=${Date.now()}`)}
        >
          ↻ Refresh
        </button>
        <a
          className="run-demo-btn small"
          href={GRAFANA_DASH}
          target="_blank" rel="noreferrer"
        >
          Open Grafana ↗
        </a>
      </div>
      {errored ? (
        <div className="monitoring-fallback">
          <h3>Grafana is not reachable at <code>{GRAFANA_DASH}</code></h3>
          <p>Start the monitoring stack with:</p>
          <pre>docker compose up -d prometheus grafana</pre>
          <p>Then open the dashboard manually. The Prometheus scrape config is at
            <code>monitoring/prometheus.yml</code> and the dashboard JSON at
            <code>monitoring/grafana_dashboard.json</code>.</p>
          <p>Panels that would be shown:</p>
          <ul>
            {Object.entries(GRAFANA_PANEL_TITLES).map(([k, v]) => (
              <li key={k}><strong>{k}</strong> — {v}</li>
            ))}
          </ul>
        </div>
      ) : (
        <iframe
          className="monitoring-iframe"
          src={src}
          title="Grafana Fraud Detection Dashboard"
          onError={() => setErrored(true)}
          onLoad={(e) => {
            try {
              const doc = e.target.contentDocument
              if (doc && doc.body && doc.body.innerText.includes('not found')) setErrored(true)
            } catch (_) { /* cross-origin: ignore */ }
          }}
        />
      )}
    </div>
  )
}
