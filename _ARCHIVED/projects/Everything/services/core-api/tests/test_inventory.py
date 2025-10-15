from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_inventory_crud():
    headers = admin_headers()
    comp = client.post(
        "/v1/components/",
        json={"manufacturerPartNumber": "INV-COMP-1"},
        headers=headers,
    ).json()

    # Create inventory
    r = client.post(
        "/v1/inventory/",
        json={"componentId": comp["id"], "quantityAvailable": 5, "location": "WH1"},
        headers=headers,
    )
    assert r.status_code == 200
    inv = r.json()

    # List filter by componentId
    r = client.get(f"/v1/inventory?componentId={comp['id']}", headers=headers)
    assert r.status_code == 200
    assert len(r.json()["data"]) >= 1

    # Delete
    r = client.delete(f"/v1/inventory/{inv['id']}", headers=headers)
    assert r.status_code == 200

