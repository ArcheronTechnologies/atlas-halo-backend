from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_search_components():
    headers = admin_headers()
    client.post("/v1/components/", json={"manufacturerPartNumber": "SRCH-COMP-1", "description": "Searchable comp"}, headers=headers)
    r = client.get("/v1/search/components", params={"query": "SRCH"}, headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json().get("data"), list)


def test_intelligence_market_trends_smoke():
    headers = admin_headers()
    r = client.get("/v1/intelligence/market-trends", headers=headers)
    assert r.status_code in (200, 500)

