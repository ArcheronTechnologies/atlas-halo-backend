from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_purchase_orders_flow():
    headers = admin_headers()
    # Supplier and component
    supplier = client.post(
        "/v1/companies/",
        json={"name": "Supplier Y", "type": "supplier"},
        headers=headers,
    ).json()
    comp = client.post(
        "/v1/components/",
        json={"manufacturerPartNumber": "PO-COMP-1"},
        headers=headers,
    ).json()

    # Create PO
    r = client.post(
        "/v1/purchase-orders/",
        json={
            "supplierId": supplier["id"],
            "poNumber": "PO-001",
            "items": [
                {"componentId": comp["id"], "quantity": 10, "unitPrice": 5.5},
                {"componentId": comp["id"], "quantity": 2, "unitPrice": 6.0},
            ],
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    po = r.json()
    assert po["poNumber"] == "PO-001"

    # Get details
    r = client.get(f"/v1/purchase-orders/{po['id']}", headers=headers)
    assert r.status_code == 200
    detail = r.json()
    assert len(detail["items"]) == 2

    # Update status
    r = client.post(f"/v1/purchase-orders/{po['id']}/status", json={"status": "sent"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "sent"

    # Add item and then delete it
    r = client.post(
        f"/v1/purchase-orders/{po['id']}/items",
        json={"componentId": comp["id"], "quantity": 3, "unitPrice": 7.0},
        headers=headers,
    )
    assert r.status_code == 200
    item_id = r.json()["id"]
    r = client.delete(f"/v1/purchase-orders/{po['id']}/items/{item_id}", headers=headers)
    assert r.status_code == 200
