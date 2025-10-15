import React, { useEffect, useState } from 'react'
import { api } from '../lib/api'
import Chart from '../components/Chart'

export default function Dashboard() {
  const [rfqs, setRfqs] = useState<any[]>([])
  const [pos, setPos] = useState<any[]>([])
  useEffect(() => {
    ;(async () => {
      const r = await api('/v1/rfqs'); const rd = await r.json(); setRfqs(rd.data || [])
      const p = await api('/v1/purchase-orders'); const pd = await p.json(); setPos(pd.data || [])
    })()
  }, [])
  const rfqByStatus = Object.entries(rfqs.reduce((a: any, r: any) => ((a[r.status || 'open'] = (a[r.status || 'open'] || 0) + 1), a), {}))
  const poByStatus = Object.entries(pos.reduce((a: any, p: any) => ((a[p.status || 'draft'] = (a[p.status || 'draft'] || 0) + 1), a), {}))
  const poTotals = pos.map((p: any) => p.totalValue || 0)
  return (
    <div>
      <h2>Dashboard</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div className="card">
          <div>RFQs</div>
          <strong>{rfqs.length}</strong>
          <ul>{rfqByStatus.map(([s, n]) => <li key={s}>{s}: {n as any}</li>)}</ul>
        </div>
        <div className="card">
          <div>POs</div>
          <strong>{pos.length}</strong>
          <ul>{poByStatus.map(([s, n]) => <li key={s}>{s}: {n as any}</li>)}</ul>
        </div>
        <div className="card">
          <div>PO Totals (trend)</div>
          <Chart data={poTotals} />
        </div>
      </div>
    </div>
  )}

