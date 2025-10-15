import React, { useState } from 'react'
import { api } from '../lib/api'

export default function SupplierAnalytics() {
  const [supplierId, setSupplierId] = useState('')
  const [data, setData] = useState<any | null>(null)
  async function run() {
    if (!supplierId) return
    const r = await api(`/v1/intelligence/supplier-analysis?supplierId=${encodeURIComponent(supplierId)}`)
    setData(await r.json())
  }
  return (
    <div>
      <h2>Supplier Analytics</h2>
      <div style={{ display:'flex', gap: 8 }}>
        <input value={supplierId} onChange={e => setSupplierId(e.target.value)} placeholder="Supplier ID" />
        <button onClick={run}>Analyze</button>
      </div>
      {data && (
        <div className="card">
          <div><strong>{data.supplier?.name}</strong> — Score {data.analysis?.overallScore}</div>
          <pre style={{ whiteSpace:'pre-wrap' }}>{JSON.stringify(data.analysis?.riskAssessment, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

