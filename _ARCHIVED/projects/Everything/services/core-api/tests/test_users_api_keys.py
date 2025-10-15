from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_create_user_and_api_key_and_use_for_ingestion():
    headers = admin_headers()
    # Create user
    r = client.post(
        "/v1/users/",
        json={"email": "apiuser@example.com", "name": "API User"},
        headers=headers,
    )
    assert r.status_code == 200
    user = r.json()

    # Create API key
    r = client.post(f"/v1/users/{user['id']}/api-keys", headers=headers)
    assert r.status_code == 200
    key = r.json()["key"]

    # Use API key to call an endpoint that only requires API key or bearer (no extra scopes)
    r = client.post(
        "/v1/ingestion/web-data",
        json={
            "url": "https://example.com/page",
            "contentType": "product_listing",
            "extractedData": {"components": []},
            "crawledAt": "2024-01-01T00:00:00Z",
        },
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text

