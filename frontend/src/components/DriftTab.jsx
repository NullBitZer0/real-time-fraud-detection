const API  = import.meta.env.VITE_API_URL ?? ''

export default function DriftTab() {
  const src = `${API}/static/drift_report.html`
  return (
    <div className="card drift-card">
      <span className="card-title">🌊 Drift Report — Evidently (train vs test)</span>
      <div className="monitoring-bar">
        <span className="monitoring-hint">
          Generated daily by the <code>fraud_drift_check</code> Airflow DAG.
          Shows distribution shift between <code>fraudTrain.csv</code> and
          <code>fraudTest.csv</code>.
        </span>
        <a
          className="run-demo-btn small"
          href={src}
          target="_blank" rel="noreferrer"
        >
          Open in new tab ↗
        </a>
        <a
          className="run-demo-btn small"
          href={`${API}/static/drift_report.json`}
          target="_blank" rel="noreferrer"
        >
          View JSON ↗
        </a>
      </div>
      <iframe
        className="drift-iframe"
        src={src}
        title="Evidently Drift Report"
      />
    </div>
  )
}
