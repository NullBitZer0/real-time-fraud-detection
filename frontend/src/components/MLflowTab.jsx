import { useEffect, useState } from 'react'

const DAGSHUB_OWNER = import.meta.env.VITE_DAGSHUB_OWNER || 'NullBitZer0'
const DAGSHUB_REPO  = import.meta.env.VITE_DAGSHUB_REPO  || 'real-time-fraud-detection'

const URLS = {
  runs:    `https://dagshub.com/${DAGSHUB_OWNER}/${DAGSHUB_REPO}.mlflow`,
  models:  `https://dagshub.com/${DAGSHUB_OWNER}/${DAGSHUB_REPO}/models`,
  repo:    `https://dagshub.com/${DAGSHUB_OWNER}/${DAGSHUB_REPO}`,
}

export default function MLflowTab() {
  const [alias, setAlias] = useState(null)

  useEffect(() => {
    fetch('models/production_alias.json')
      .then(r => r.ok ? r.json() : null)
      .then(j => { if (j) setAlias(j) })
      .catch(() => {})
  }, [])

  return (
    <div>
      <div className="card">
        <span className="card-title">🔗 MLflow + DAGsHub</span>
        <p className="monitoring-hint">
          All training runs, model registry, and artifacts are tracked on
          DAGsHub. The promotion script promotes a new model version to
          <code> @Production</code> when its <code>test_pr_auc</code> is
          greater than the current production model's.
        </p>

        {alias && (
          <div className="mlflow-alias">
            <div className="mlflow-alias-row">
              <span className="mlflow-alias-label">Current Production</span>
              <code className="mlflow-alias-value">{alias.model_name} v{alias.version}</code>
            </div>
            <div className="mlflow-alias-row">
              <span className="mlflow-alias-label">test_pr_auc</span>
              <code className="mlflow-alias-value">{alias.test_pr_auc?.toFixed(4)}</code>
            </div>
            <div className="mlflow-alias-row">
              <span className="mlflow-alias-label">Run ID</span>
              <code className="mlflow-alias-value mono small">{alias.run_id}</code>
            </div>
            <div className="mlflow-alias-row">
              <span className="mlflow-alias-label">Promoted at</span>
              <code className="mlflow-alias-value">{alias.promoted_at}</code>
            </div>
          </div>
        )}

        <div className="mlflow-actions">
          <a className="run-demo-btn"           href={URLS.runs}   target="_blank" rel="noreferrer">📊 Open MLflow runs ↗</a>
          <a className="run-demo-btn"           href={URLS.models} target="_blank" rel="noreferrer">🗂️ Open model registry ↗</a>
          <a className="run-demo-btn secondary" href={URLS.repo}   target="_blank" rel="noreferrer">📁 Open DAGsHub repo ↗</a>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <span className="card-title">📋 Promotion history (from MLflow registry)</span>
        <p className="monitoring-hint">
          Visit the model registry link above for the full list of model versions
          with their stage transitions and metrics. Each version has a run
          attached with the full training metrics (val_pr_auc, test_pr_auc, tier1/2 precision/recall).
        </p>
      </div>
    </div>
  )
}
