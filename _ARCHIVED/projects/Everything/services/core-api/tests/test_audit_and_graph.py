from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def admin_headers():
    r = client.post("/v1/auth/login", json={"email": "user@admin", "password": "dev"})
    assert r.status_code == 200
    token = r.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def test_audit_log_and_graph_neighbors():
    headers = admin_headers()
    # Create manufacturer and component
    mfg = client.post(
        "/v1/companies/",
        json={"name": "MFG Co", "type": "manufacturer"},
        headers=headers,
    ).json()
    comp = client.post(
        "/v1/components/",
        json={"manufacturerPartNumber": "GRAPH-COMP-1", "manufacturerId": mfg["id"]},
        headers=headers,
    ).json()

    # RFQ to generate an audit log via status change
    cust = client.post(
        "/v1/companies/",
        json={"name": "Audit Customer", "type": "customer"},
        headers=headers,
    ).json()
    rfq = client.post(
        "/v1/rfqs/",
        json={"customerId": cust["id"], "items": [{"componentId": comp["id"], "quantity": 1}]},
        headers=headers,
    ).json()
    client.post(f"/v1/rfqs/{rfq['id']}/status", json={"status": "quoted"}, headers=headers)

    # Query audit
    r = client.get(f"/v1/audit?entityType=rfq&entityId={rfq['id']}", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data.get("total", 0) >= 1

    # Graph neighbors
    r = client.get(f"/v1/graph/components/{comp['id']}/neighbors", headers=headers)
    assert r.status_code == 200
    neigh = r.json()["neighbors"]
    assert any(n["type"] == "MANUFACTURER" for n in neigh)

