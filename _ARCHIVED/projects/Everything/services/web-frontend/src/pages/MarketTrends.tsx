import React, { useState } from 'react'
import { api } from '../lib/api'
import Chart from '../components/Chart'

export default function MarketTrends() {
  const [query, setQuery] = useState('STM32F4')
  const [trend, setTrend] = useState<any | null>(null)
  async function run() {
    const compSearch = await api(`/v1/search/components?query=${encodeURIComponent(query)}`)
    const list = await compSearch.json()
    const id = list?.data?.[0]?.id
    const res = await api(`/v1/intelligence/market-trends${id ? `?component=${id}` : ''}`)
    setTrend(await res.json())
  }
  return (
    <div>
      <h2>Market Trends</h2>
      <div style={{ display:'flex', gap: 8 }}>
        <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Part number" />
        <button onClick={run}>Analyze</button>
      </div>
      {trend && (
        <div className="card">
          <div><strong>{trend.component?.manufacturerPartNumber}</strong></div>
          <div>Confidence: {Math.round((trend.confidence || 0) * 100)}%</div>
          <div style={{ marginTop: 8 }}>Availability forecast (index)</div>
          <Chart data={(trend.trends?.availabilityForecast || []).map((_: any, i: number) => i)} />
          <div style={{ marginTop: 8 }}>Recommendations</div>
          <ul>{(trend.recommendations || []).map((r: any, i: number) => <li key={i}>{r.action}: {r.reason}</li>)}</ul>
        </div>
      )}
    </div>
  )
}

