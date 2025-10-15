from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_pricing_ingestion_and_retrieval():
    headers = admin_headers()

    # Create supplier company
    r = client.post(
        "/v1/companies/",
        json={"name": "SupplierCo", "type": "supplier", "website": "https://supplier.example"},
        headers=headers,
    )
    assert r.status_code == 200
    supplier_id = r.json()["id"]

    # Create component
    r = client.post(
        "/v1/components/",
        json={"manufacturerPartNumber": "TEST-INGEST-001", "description": "Test comp"},
        headers=headers,
    )
    assert r.status_code == 200
    component_id = r.json()["id"]

    # Ingest pricing
    r = client.post(
        "/v1/ingestion/pricing",
        json={
            "prices": [
                {
                    "componentId": component_id,
                    "supplierId": supplier_id,
                    "quantityBreak": 1,
                    "unitPrice": 12.34,
                    "currency": "USD",
                }
            ]
        },
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["ingested"] == 1

    # Verify pricing endpoint reflects new data
    r = client.get(f"/v1/components/{component_id}/pricing", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["componentId"] == component_id
    assert data["currentPricing"]
    qb = data["currentPricing"][0]["quantityBreaks"][0]
    assert qb["quantity"] == 1
    assert qb["unitPrice"] == 12.34

