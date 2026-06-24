const TIER_LABELS = {
  0: { name: 'Approve',       color: 'var(--success)' },
  1: { name: 'Auto Block',    color: 'var(--critical)' },
  2: { name: 'Review Queue',  color: 'var(--warning)' },
  3: { name: 'Soft Signal',   color: '#eab308' },
}

const MODE_LABELS = {
  real_world: { icon: '🌍', name: 'Real World', sub: 'dataset fraud rate' },
  custom:     { icon: '⚖️', name: 'Custom',     sub: 'manual ratio' },
}

export default function DemoResults({ result }) {
  if (!result) return null

  const [[tn, fp], [fn, tp]] = result.confusion_matrix
  const hasDetail = result.results && result.results.length > 0
  const modeInfo = MODE_LABELS[result.mode] ?? MODE_LABELS.real_world

  return (
    <div className="card demo-results">
      <span className="card-title">
        🧪 Demo Results — {result.n_total} Tests
        <span className="demo-mode-badge">
          {modeInfo.icon} {modeInfo.name} ({(result.fraud_rate * 100).toFixed(2)}% fraud)
        </span>
      </span>

      <div className="demo-summary">
        <div className="demo-stat">
          <span className="demo-stat-label">Tested</span>
          <span className="demo-stat-value">{result.n_total}</span>
          <span className="demo-stat-sub">{result.n_fraud} fraud + {result.n_legit} legit</span>
        </div>
        <div className="demo-stat">
          <span className="demo-stat-label">Macro F1</span>
          <span className="demo-stat-value accent">{result.macro_f1.toFixed(3)}</span>
          <span className="demo-stat-sub">vs ground truth</span>
        </div>
        <div className="demo-stat">
          <span className="demo-stat-label">Confusion</span>
          <span className="demo-stat-value mono">
            TN {tn} | FP {fp}<br />FN {fn} | TP {tp}
          </span>
        </div>
      </div>

      <div className="demo-tiers">
        {[1, 2, 3, 0].map(t => {
          const count = t === 1 ? result.tier1_count
                      : t === 2 ? result.tier2_count
                      : t === 3 ? result.tier3_count
                      : result.tier0_count
          const tname = t === 1 ? 'tier1_auto_block'
                      : t === 2 ? 'tier2_review_queue'
                      : t === 3 ? 'tier3_soft_signal'
                      : 'approve'
          const thresh = result.tier_thresholds[tname] ?? 0
          return (
            <div key={t} className={`tier-pill tier-${t}`}>
              <span className="tier-pill-name">T{t} · {TIER_LABELS[t].name}</span>
              <span className="tier-pill-count">{count}</span>
              <span className="tier-pill-thresh">≥ {(thresh * 100).toFixed(2)}%</span>
            </div>
          )
        })}
      </div>

      {hasDetail && (
        <details>
          <summary style={{ cursor: 'pointer', color: 'var(--muted)', fontSize: '12px' }}>
            Show {result.results.length} individual results
          </summary>
          <div className="demo-table-wrap">
            <table className="demo-table">
              <thead>
                <tr>
                  <th>txn</th><th>truth</th><th>tier</th><th>action</th>
                  <th>proba</th><th>amt</th><th>merchant</th><th>category</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map(r => (
                  <tr key={r.trans_num} className={`tier-${r.tier}`}>
                    <td className="mono">{r.trans_num.slice(0, 8)}…</td>
                    <td>
                      <span className={`truth-badge ${r.is_fraud ? 'fraud' : 'legit'}`}>
                        {r.is_fraud ? 'FRAUD' : 'legit'}
                      </span>
                    </td>
                    <td><span className={`tier-badge tier-${r.tier}`}>T{r.tier}</span></td>
                    <td className="mono">{r.action}</td>
                    <td>{(r.fraud_probability * 100).toFixed(2)}%</td>
                    <td>${r.amt.toFixed(2)}</td>
                    <td>{r.merchant}</td>
                    <td>{r.category}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  )
}
