import React, { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { api, setToken, getToken } from './lib/api'

export default function App() {
  const nav = useNavigate()
  const [email, setEmail] = useState('user@admin')
  const [password, setPassword] = useState('dev')
  const [authed, setAuthed] = useState<boolean>(!!getToken())
  const [busy, setBusy] = useState(false)

  async function login() {
    setBusy(true)
    try {
      const res = await api('/v1/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      })
      const data = await res.json()
      if (res.ok && data.accessToken) {
        setToken(data.accessToken)
        setAuthed(true)
        nav('/')
      } else {
        alert('Login failed')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1c2a4d', padding: '10px 16px' }}>
        <div style={{ fontWeight: 700 }}>SCIP Analytics</div>
        <nav style={{ display: 'flex', gap: 12 }}>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>Dashboard</NavLink>
          <NavLink to="/trends" className={({ isActive }) => isActive ? 'active' : ''}>Market Trends</NavLink>
          <NavLink to="/suppliers" className={({ isActive }) => isActive ? 'active' : ''}>Supplier Analytics</NavLink>
          <NavLink to="/rfqs" className={({ isActive }) => isActive ? 'active' : ''}>RFQ Insights</NavLink>
          <NavLink to="/pos" className={({ isActive }) => isActive ? 'active' : ''}>PO Analytics</NavLink>
        </nav>
        {!authed && (
          <div style={{ display: 'flex', gap: 8 }}>
            <input placeholder="email" value={email} onChange={e => setEmail(e.target.value)} />
            <input placeholder="password" type="password" value={password} onChange={e => setPassword(e.target.value)} />
            <button disabled={busy} onClick={login}>Login</button>
          </div>
        )}
      </header>
      <main style={{ padding: 16 }}>
        <Outlet />
      </main>
    </div>
  )
}

