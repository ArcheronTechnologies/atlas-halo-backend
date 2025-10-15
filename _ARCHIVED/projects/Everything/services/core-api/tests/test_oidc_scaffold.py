from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_oidc_authorize_scaffold():
    r = client.get("/v1/auth/oidc/authorize", params={"redirect_uri": "http://localhost/callback"})
    # May be 400 if not configured, but endpoint should exist
    assert r.status_code in (200, 400)

