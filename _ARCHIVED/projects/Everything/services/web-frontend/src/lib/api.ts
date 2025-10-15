export function api(path: string, init: RequestInit = {}) {
  const base = import.meta?.env?.VITE_API_BASE_URL || 'http://localhost:8000'
  const token = getToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> | undefined),
  }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return fetch(base + path, { ...init, headers })
}

export function setToken(t: string) {
  localStorage.setItem('scip_token', t)
}

export function getToken(): string | null {
  return localStorage.getItem('scip_token')
}

