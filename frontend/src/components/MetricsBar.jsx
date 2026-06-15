export default function MetricsBar({ metrics }) {
  const fmt = (n) => n?.toLocaleString() ?? '–'

  return (
    <div className="metrics-bar">
      <div className="metric-card">
        <span className="metric-label">Total Transactions</span>
        <span className="metric-value accent">{fmt(metrics.total_transactions)}</span>
        <span className="metric-sub">since startup</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Fraud Detected</span>
        <span className="metric-value danger">{fmt(metrics.total_fraud)}</span>
        <span className="metric-sub">flagged transactions</span>
      </div>
      <div className="metric-card">
        <span className="metric-label">Fraud Rate</span>
        <span className="metric-value warning">
          {metrics.fraud_rate_pct?.toFixed(3) ?? '0.000'}%
        </span>
        <span className="metric-sub">of all transactions</span>
      </div>
    </div>
  )
}
