from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_rfq_quotes_listing():
    headers = admin_headers()
    # Company and component
    company = client.post(
        "/v1/companies/",
        json={"name": "Customer Z", "type": "customer"},
        headers=headers,
    ).json()
    comp = client.post(
        "/v1/components/",
        json={"manufacturerPartNumber": "QUOTE-COMP-1"},
        headers=headers,
    ).json()

    # Create RFQ
    rfq = client.post(
        "/v1/rfqs/",
        json={
            "customerId": company["id"],
            "rfqNumber": "RFQ-T-001",
            "items": [{"componentId": comp["id"], "quantity": 100}],
        },
        headers=headers,
    ).json()

    # Submit quote
    r = client.post(
        f"/v1/rfqs/{rfq['id']}/quote",
        json={"quotedItems": [{"rfqItemId": None, "unitPrice": 9.99, "totalPrice": 999.0}]},
        headers=headers,
    )
    assert r.status_code == 200

    # List quotes
    r = client.get(f"/v1/rfqs/{rfq['id']}/quotes", headers=headers)
    assert r.status_code == 200
    data = r.json()["data"]
    assert isinstance(data, list) and len(data) >= 1

