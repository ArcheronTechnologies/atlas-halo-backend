import json
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_octopart_total_availability():
    headers = admin_headers()
    r = client.get(
        "/v1/market/total-availability",
        params={"q": "STM32", "country": "US", "limit": 3},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("provider") == "octopart"
    assert "parts" in data


def test_octopart_pricing_breaks():
    headers = admin_headers()
    r = client.get(
        "/v1/market/pricing-breaks",
        params={"q": "STM32F429ZIT6", "limit": 3, "currency": "USD"},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("provider") == "octopart"
    assert "parts" in data


def test_octopart_offers():
    headers = admin_headers()
    r = client.get(
        "/v1/market/offers",
        params={"mpn": "STM32F429ZIT6", "country": "US", "currency": "USD"},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("provider") == "octopart"
    assert "sellers" in data


def test_octopart_spec_attributes():
    headers = admin_headers()
    filters = json.dumps({"case_package": ["SSOP"]})
    r = client.get(
        "/v1/market/spec-attributes",
        params={"q": "ADS", "limit": 3, "filters": filters},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("provider") == "octopart"
    assert "parts" in data
