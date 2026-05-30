export default function LiveFeed({ events }) {
  return (
    <div className="card">
      <span className="card-title">⚡ Live Transaction Stream</span>
      <div className="feed">
        {events.length === 0 && (
          <div style={{ color: 'var(--muted)', textAlign: 'center', padding: '40px 0' }}>
            Waiting for transactions…
          </div>
        )}
        {events.map((e) => (
          <div
            key={e.transaction_id}
            className={`feed-item ${e.fraud_prediction === 1 ? 'fraud' : 'normal'}`}
          >
            <span className="feed-icon">
              {e.fraud_prediction === 1 ? '🚨' : '✅'}
            </span>
            <span className="feed-id">
              {e.transaction_id?.slice(0, 8)}…
            </span>
            <span className={`risk-badge risk-${e.risk_level}`}>
              {e.risk_level}
            </span>
            <span className={`feed-score ${e.fraud_probability > 0.5 ? 'high' : 'low'}`}>
              {(e.fraud_probability * 100).toFixed(1)}%
            </span>
            <span style={{ color: 'var(--muted)', fontSize: '11px' }}>
              {e.latency_ms}ms
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
