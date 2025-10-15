import React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './App'
import Dashboard from './pages/Dashboard'
import MarketTrends from './pages/MarketTrends'
import SupplierAnalytics from './pages/SupplierAnalytics'
import RFQInsights from './pages/RFQInsights'
import POAnalytics from './pages/POAnalytics'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'trends', element: <MarketTrends /> },
      { path: 'suppliers', element: <SupplierAnalytics /> },
      { path: 'rfqs', element: <RFQInsights /> },
      { path: 'pos', element: <POAnalytics /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)

