from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_unauthorized_companies():
    r = client.get("/v1/companies/")
    assert r.status_code == 401


def test_login_and_basic_flow():
    # Login
    r = client.post(
        "/v1/auth/login",
        json={"email": "user@admin", "password": "dev"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create company
    r = client.post(
        "/v1/companies/",
        json={"name": "TestCo", "type": "customer", "website": "https://testco.example"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    company_id = r.json()["id"]

    # List companies
    r = client.get("/v1/companies/", headers=headers)
    assert r.status_code == 200
    assert any(c["id"] == company_id for c in r.json()["data"])  # type: ignore[index]

    # Create RFQ referencing the new company
    r = client.post(
        "/v1/rfqs/",
        json={
            "customerId": company_id,
            "rfqNumber": "RFQ-TEST-001",
            "items": [{"quantity": 100}],
            "source": "test",
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text

    # List RFQs
    r = client.get("/v1/rfqs/", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json().get("data"), list)  # type: ignore[union-attr]

    # Update RFQ status (should allow quoted)
    rfq_id = r.json()["data"][0]["id"]
    r2 = client.post(f"/v1/rfqs/{rfq_id}/status", json={"status": "quoted"}, headers=headers)
    assert r2.status_code == 200
    # Fetch detail
    r3 = client.get(f"/v1/rfqs/{rfq_id}", headers=headers)
    assert r3.status_code == 200
