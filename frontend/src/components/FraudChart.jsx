import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:'#1a2235', border:'1px solid #1f2d45',
      borderRadius:'8px', padding:'8px 12px', fontSize:'12px'
    }}>
      <p style={{ color:'#4cc9f0' }}>
        Fraud Rate: <strong>{payload[0]?.value?.toFixed(3)}%</strong>
      </p>
      <p style={{ color:'#64748b' }}>t={payload[0]?.payload?.t}</p>
    </div>
  )
}

export default function FraudChart({ chartData }) {
  return (
    <div className="card">
      <span className="card-title">📈 Live Fraud Rate</span>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2d45" />
          <XAxis
            dataKey="t"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#1f2d45' }}
            label={{ value: 'Time', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -2 }}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.17} stroke="#f59e0b" strokeDasharray="4 4"
            label={{ value: 'Dataset avg', fill: '#f59e0b', fontSize: 10 }} />
          <Line
            type="monotone"
            dataKey="rate"
            stroke="#ef4444"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#ef4444' }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div style={{ fontSize: '11px', color: 'var(--muted)', textAlign: 'center' }}>
        Rolling fraud rate (%) — updates on each transaction
      </div>
    </div>
  )
}
