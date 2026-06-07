import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const TIER_NAME = { 0: 'approve', 1: 'auto_block', 2: 'review_queue', 3: 'soft_signal' }
const TIER_COLOR = { 0: 'success', 1: 'critical', 2: 'warning', 3: '#eab308' }

function fmtTime(s) {
  if (!s) return '–'
  try { return new Date(s).toLocaleString() } catch { return s }
}

export default function AuditTab() {
  const [rows,   setRows]   = useState([])
  const [err,    setErr]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')
  const [auto,   setAuto]   = useState(true)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const r = await fetch(`${API}/audit/recent?n=200`)
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
        const data = await r.json()
        if (alive) { setRows(data); setErr(null) }
      } catch (e) {
        if (alive) setErr(e.message)
      } finally {
        if (alive) setLoading(false)
      }
    }
    load()
    let id = null
    if (auto) id = setInterval(load, 5000)
    return () => { alive = false; if (id) clearInterval(id) }
  }, [auto])

  const filtered = filter === 'all' ? rows : rows.filter(r => r.tier === Number(filter))

  const tierCounts = [1, 2, 3, 0].map(t => ({
    tier: t,
    name: TIER_NAME[t],
    count: rows.filter(r => r.tier === t).length,
  }))
  const totalFlagged = rows.filter(r => r.tier >= 1).length
  const totalGround  = rows.filter(r => r.is_fraud_ground_truth !== null).length
  const correct      = rows.filter(r => (r.tier >= 1) === !!r.is_fraud_ground_truth).length
  const backfilledAcc = totalGround ? (correct / totalGround * 100).toFixed(1) : '–'

  return (
    <div>
      <div className="metrics-bar" style={{ marginBottom: 16 }}>
        <div className="metric-card">
          <span className="metric-label">Total decisions</span>
          <span className="metric-value accent">{rows.length}</span>
          <span className="metric-sub">last 200 from decision_log</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Flagged (T1+T2+T3)</span>
          <span className="metric-value danger">{totalFlagged}</span>
          <span className="metric-sub">{(rows.length ? totalFlagged/rows.length*100 : 0).toFixed(1)}% of decisions</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Backfilled accuracy</span>
          <span className="metric-value warning">{backfilledAcc}{backfilledAcc !== '–' && '%'}</span>
          <span className="metric-sub">{totalGround} rows have ground truth</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Avg latency</span>
          <span className="metric-value success">
            {rows.length ? (rows.reduce((s, r) => s + (r.latency_ms || 0), 0) / rows.length).toFixed(1) : '–'}ms
          </span>
          <span className="metric-sub">per /predict call</span>
        </div>
      </div>

      <div className="card">
        <span className="card-title">📜 Audit Log — last 200 /predict decisions</span>
        <div className="monitoring-bar">
          <label>
            Filter tier:&nbsp;
            <select value={filter} onChange={e => setFilter(e.target.value)}>
              <option value="all">all</option>
              <option value="1">T1 auto_block</option>
              <option value="2">T2 review_queue</option>
              <option value="3">T3 soft_signal</option>
              <option value="0">T0 approve</option>
            </select>
          </label>
          <label style={{ marginLeft: 16 }}>
            <input type="checkbox" checked={auto} onChange={e => setAuto(e.target.checked)} />
            &nbsp;auto-refresh (5s)
          </label>
          {loading && <span className="monitoring-hint">loading…</span>}
          {err   && <span className="health-error">⚠ {err}</span>}
        </div>

        <div className="demo-tiers" style={{ marginTop: 12 }}>
          {tierCounts.map(({ tier, name, count }) => (
            <div key={tier} className={`tier-pill tier-${tier}`}>
              <span className="tier-pill-name">T{tier} · {name}</span>
              <span className="tier-pill-count">{count}</span>
            </div>
          ))}
        </div>

        <div className="demo-table-wrap" style={{ marginTop: 12 }}>
          <table className="demo-table">
            <thead>
              <tr>
                <th>id</th><th>trans_num</th><th>proba</th><th>tier</th><th>action</th>
                <th>truth</th><th>latency</th><th>model</th><th>ingested_at</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(r => (
                <tr key={r.id} className={`tier-${r.tier}`}>
                  <td className="mono">{r.id}</td>
                  <td className="mono">{r.trans_num.slice(0, 12)}…</td>
                  <td>{(r.fraud_probability * 100).toFixed(2)}%</td>
                  <td><span className={`tier-badge tier-${r.tier}`}>T{r.tier}</span></td>
                  <td className="mono">{r.action}</td>
                  <td>
                    {r.is_fraud_ground_truth === null
                      ? <span style={{ color: 'var(--muted)' }}>–</span>
                      : r.is_fraud_ground_truth
                        ? <span className="truth-badge fraud">FRAUD</span>
                        : <span className="truth-badge legit">legit</span>}
                  </td>
                  <td>{r.latency_ms?.toFixed(1) ?? '–'}ms</td>
                  <td className="mono" style={{ fontSize: 11 }}>{r.model_version ?? '–'}</td>
                  <td style={{ fontSize: 11, color: 'var(--muted)' }}>{fmtTime(r.ingested_at)}</td>
                </tr>
              ))}
              {filtered.length === 0 && !loading && (
                <tr><td colSpan="9" style={{ textAlign: 'center', color: 'var(--muted)', padding: 20 }}>
                  No decisions match the filter. Run the 100-test demo to populate.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
