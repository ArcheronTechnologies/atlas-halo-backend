from typing import Optional
from fastapi import APIRouter, Depends

from ..integrations.visma import VismaClient
from ..integrations.business_central import BusinessCentralClient
from ..integrations.c3 import C3Client
from ..integrations.hubspot import HubSpotClient
from ..core.auth import require_scopes
from ..repositories.inventory_repo import InventoryRepository
from ..db.session import get_session
from sqlalchemy.orm import Session


router = APIRouter()


@router.get("/erp/visma/inventory", dependencies=[Depends(require_scopes(["read:integrations"]))])
def visma_inventory():
    client = VismaClient()
    return {"data": client.get_inventory()}


@router.get("/erp/business-central/items", dependencies=[Depends(require_scopes(["read:integrations"]))])
def bc_items():
    client = BusinessCentralClient()
    return {"data": client.get_items()}


@router.post("/c3/push-recommendations", dependencies=[Depends(require_scopes(["write:integrations"]))])
def c3_push_recommendations(rfqId: str, recommendations: list[dict]):
    client = C3Client()
    return client.push_recommendations(rfqId, recommendations)


@router.post("/hubspot/update-company", dependencies=[Depends(require_scopes(["write:integrations"]))])
def hubspot_update_company(companyId: str, properties: dict):
    client = HubSpotClient()
    return client.update_company_properties(companyId, properties)


@router.post("/erp/visma/sync-inventory", dependencies=[Depends(require_scopes(["write:integrations"]))])
def visma_sync_inventory(session: Session = Depends(get_session)):
    client = VismaClient()
    data = client.get_inventory()
    repo = InventoryRepository(session)
    created = 0
    for item in data:
        try:
            rec = repo.create({
                "componentId": item.get("componentId") or item.get("component_id") or "",
                "location": item.get("location") or item.get("warehouse"),
                "quantityAvailable": int(item.get("quantity", 0)),
                "costPerUnit": float(item.get("cost", 0) or 0),
            })
            created += 1
        except Exception:
            continue
    return {"synced": created}
