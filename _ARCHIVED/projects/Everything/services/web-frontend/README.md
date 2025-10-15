# SCIP Web Frontend (Analytics)

A lightweight analytics console for SCIP, built with React + Vite + TypeScript. Focused on data flows and insights: market trends, supplier analytics, RFQ and PO dashboards.

## Run

1. Install Node 18+
2. Install deps and start dev server

```
cd services/web-frontend
npm install
npm run dev
```

3. Configure API base (optional)

Copy `.env.example` to `.env` and set `VITE_API_BASE_URL` (defaults to `http://localhost:8000`).

```
VITE_API_BASE_URL=http://localhost:8000
```

4. Open http://localhost:5173

- Log in via the header (dev): email `user@admin`, password `dev`.
- Navigate using the top tabs.

## Pages
- Dashboard: KPI tiles and quick charts from RFQs and POs
- Market Trends: Analyze component trends from `/v1/intelligence/market-trends`
- Supplier Analytics: AI supplier performance from `/v1/intelligence/supplier-analysis`
- RFQ Insights: Status distribution and samples
- PO Analytics: Status distribution and totals trend

## Notes
- No state management library; simple hooks + localStorage token.
- Minimal SVG charts (no heavy chart libs). Easy to replace with Recharts/ECharts later.
- CORS is enabled on the API for dev.
