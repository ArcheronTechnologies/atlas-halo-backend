from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def get_admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_component_crud():
    headers = get_admin_headers()

    # Create
    r = client.post(
        "/v1/components/",
        json={
            "manufacturerPartNumber": "TEST-123",
            "category": "TestCat",
            "description": "Test component",
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    comp = r.json()
    cid = comp["id"]

    # Update
    r = client.put(
        f"/v1/components/{cid}",
        json={"description": "Updated description"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["description"] == "Updated description"

    # Delete
    r = client.delete(f"/v1/components/{cid}", headers=headers)
    assert r.status_code == 200
    assert r.json().get("deleted") is True

    # Verify 404
    r = client.get(f"/v1/components/{cid}/pricing", headers=headers)
    assert r.status_code == 404

