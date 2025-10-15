from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.po_repo import PORepository
from ..models.purchase_orders import POCreate, POResponse, POListResponse
from ..core.auth import require_scopes


router = APIRouter()


@router.get("/", response_model=POListResponse, dependencies=[Depends(require_scopes(["read:purchase_orders"]))])
def list_pos(session: Session = Depends(get_session)):
    repo = PORepository(session)
    items = repo.list()
    data: List[POResponse] = []
    for po in items:
        data.append(
            POResponse(
                id=po.id,
                supplierId=po.supplier_id,
                poNumber=po.po_number,
                status=po.status,
                totalValue=po.total_value,
                currency=po.currency,
                paymentTerms=po.payment_terms,
                deliveryAddress=po.delivery_address,
                items=[],
                createdAt=po.created_at.isoformat() if po.created_at else None,
                updatedAt=po.updated_at.isoformat() if po.updated_at else None,
            )
        )
    return POListResponse(data=data)


@router.post("/", response_model=POResponse, dependencies=[Depends(require_scopes(["write:purchase_orders"]))])
def create_po(body: POCreate, session: Session = Depends(get_session)):
    repo = PORepository(session)
    po = repo.create(body.model_dump())
    return POResponse(
        id=po.id,
        supplierId=po.supplier_id,
        poNumber=po.po_number,
        status=po.status,
        totalValue=po.total_value,
        currency=po.currency,
        paymentTerms=po.payment_terms,
        deliveryAddress=po.delivery_address,
        items=[],
        createdAt=po.created_at.isoformat() if po.created_at else None,
        updatedAt=po.updated_at.isoformat() if po.updated_at else None,
    )


@router.post("/{po_id}/items", response_model=dict, dependencies=[Depends(require_scopes(["write:purchase_orders"]))])
def add_po_item(po_id: str, body: dict, session: Session = Depends(get_session)):
    repo = PORepository(session)
    try:
        res = repo.add_item(po_id, body)
    except KeyError:
        raise HTTPException(404, detail="Purchase order not found")
    return res


@router.delete("/{po_id}/items/{item_id}", response_model=dict, dependencies=[Depends(require_scopes(["write:purchase_orders"]))])
def delete_po_item(po_id: str, item_id: str, session: Session = Depends(get_session)):
    repo = PORepository(session)
    ok = repo.delete_item(po_id, item_id)
    if not ok:
        raise HTTPException(404, detail="Item not found")
    return {"deleted": True}


@router.get("/{po_id}", response_model=POResponse, dependencies=[Depends(require_scopes(["read:purchase_orders"]))])
def get_po(po_id: str, session: Session = Depends(get_session)):
    repo = PORepository(session)
    data = repo.get_detail(po_id)
    if not data:
        raise HTTPException(404, detail="Purchase order not found")
    return POResponse(**data)


@router.post("/{po_id}/status", response_model=dict, dependencies=[Depends(require_scopes(["write:purchase_orders"]))])
def update_po_status(po_id: str, body: dict, session: Session = Depends(get_session)):
    repo = PORepository(session)
    status = body.get("status")
    if not status:
        raise HTTPException(400, detail="Missing status")
    try:
        return repo.update_status(po_id, status)
    except KeyError:
        raise HTTPException(404, detail="Purchase order not found")
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
