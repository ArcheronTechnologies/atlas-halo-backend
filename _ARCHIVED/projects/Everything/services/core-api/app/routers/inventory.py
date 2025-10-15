from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.inventory_repo import InventoryRepository
from ..models.inventory import InventoryCreate, InventoryItem, InventoryListResponse
from ..core.auth import require_scopes


router = APIRouter()


@router.get("/", response_model=InventoryListResponse, dependencies=[Depends(require_scopes(["read:inventory"]))])
def list_inventory(componentId: Optional[str] = Query(default=None), limit: int = 50, offset: int = 0, session: Session = Depends(get_session)):
    repo = InventoryRepository(session)
    items, _ = repo.list(component_id=componentId, limit=limit, offset=offset)
    data: List[InventoryItem] = [
        InventoryItem(
            id=i.id,
            componentId=i.component_id,
            location=i.location,
            quantityAvailable=i.quantity_available,
            quantityReserved=i.quantity_reserved,
            costPerUnit=i.cost_per_unit,
            dateCode=i.date_code,
            lotCode=i.lot_code,
            expiryDate=i.expiry_date.isoformat() if i.expiry_date else None,
            createdAt=i.created_at.isoformat() if i.created_at else None,
            updatedAt=i.updated_at.isoformat() if i.updated_at else None,
        )
        for i in items
    ]
    return InventoryListResponse(data=data)


@router.post("/", response_model=InventoryItem, dependencies=[Depends(require_scopes(["write:inventory"]))])
def create_inventory(body: InventoryCreate, session: Session = Depends(get_session)):
    repo = InventoryRepository(session)
    rec = repo.create(body.model_dump())
    return InventoryItem(
        id=rec.id,
        componentId=rec.component_id,
        location=rec.location,
        quantityAvailable=rec.quantity_available,
        quantityReserved=rec.quantity_reserved,
        costPerUnit=rec.cost_per_unit,
        dateCode=rec.date_code,
        lotCode=rec.lot_code,
        expiryDate=rec.expiry_date.isoformat() if rec.expiry_date else None,
        createdAt=rec.created_at.isoformat() if rec.created_at else None,
        updatedAt=rec.updated_at.isoformat() if rec.updated_at else None,
    )


@router.delete("/{inventory_id}", response_model=dict, dependencies=[Depends(require_scopes(["write:inventory"]))])
def delete_inventory(inventory_id: str, session: Session = Depends(get_session)):
    repo = InventoryRepository(session)
    ok = repo.delete(inventory_id)
    if not ok:
        raise HTTPException(404, detail="Inventory record not found")
    return {"deleted": True}

