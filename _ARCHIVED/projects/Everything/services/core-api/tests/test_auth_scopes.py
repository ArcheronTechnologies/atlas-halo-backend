from fastapi.testclient import TestClient

from app.main import app
from app.core.security import generate_jwt


client = TestClient(app)


def test_market_requires_scope_for_bearer():
    # Token without read:integrations
    token = generate_jwt({
        "sub": "uuid",
        "email": "user@example.com",
        "permissions": ["read:components"],
    })
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get("/v1/market/total-availability", params={"q": "STM32"}, headers=headers)
    assert r.status_code == 403


def test_intelligence_requires_scope_for_bearer():
    token = generate_jwt({
        "sub": "uuid",
        "email": "user@example.com",
        "permissions": ["read:components"],
    })
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get("/v1/intelligence/market-trends", headers=headers)
    assert r.status_code in (401, 403)

