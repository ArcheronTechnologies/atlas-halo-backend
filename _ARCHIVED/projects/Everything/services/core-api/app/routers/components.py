from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends

from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.components_repo import ComponentsRepository
from ..db.models import Component as ComponentORM
from ..models.components import (
    Component,
    ComponentsListResponse,
    CreateComponentRequest,
    UpdateComponentRequest,
    Pagination,
    ComponentPricingResponse,
    SupplierPricing,
    SupplierRef,
    QuantityBreak,
    PriceHistoryItem,
)
from ..core.auth import require_scopes
from ..search.indexer import index_component, delete_component_index, search_components
from ..repositories.pricing_repo import PricingRepository


router = APIRouter()


@router.get("/", response_model=ComponentsListResponse, dependencies=[Depends(require_scopes(["read:components"]))])
def list_components(
    search: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    manufacturer: Optional[str] = Query(default=None),
    lifecycle: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    repo = ComponentsRepository(session)
    if search:
        ids = search_components(search, limit=limit, offset=offset)
        items = repo.get_by_ids(ids)
        total = len(items)
    else:
        items, total = repo.list_filtered(limit=limit, offset=offset, category=category, manufacturer_id=manufacturer, lifecycle=lifecycle)
    # Ignore category/manufacturer/lifecycle filtering in MVP
    data: List[Component] = [
        Component(
            id=c.id,
            manufacturerPartNumber=c.manufacturer_part_number,
            manufacturer={"id": c.manufacturer_id} if c.manufacturer_id else None,
            category=c.category,
            description=c.description,
            lifecycleStatus=c.lifecycle_status,
            rohsCompliant=c.rohs_compliant,
            datasheet=c.datasheet_url,
            specifications={},
            alternativeParts=[],
            createdAt=c.created_at.isoformat() if c.created_at else None,
            updatedAt=c.updated_at.isoformat() if c.updated_at else None,
        )
        for c in items
    ]
    return ComponentsListResponse(
        data=data,
        pagination=Pagination(total=total, limit=limit, offset=offset, hasMore=offset + limit < total),
    )


@router.post("/", response_model=Component, dependencies=[Depends(require_scopes(["write:components"]))])
def create_component(body: CreateComponentRequest, session: Session = Depends(get_session)):
    repo = ComponentsRepository(session)
    obj = repo.create(
        {
            "manufacturerPartNumber": body.manufacturerPartNumber,
            "manufacturer": {"id": body.manufacturerId} if body.manufacturerId else None,
            "category": body.category,
            "description": body.description,
            "lifecycleStatus": "active",
            "rohsCompliant": body.rohsCompliant,
            "datasheet": body.datasheetUrl,
            "specifications": body.specifications or {},
            "alternativeParts": [],
        }
    )
    res = Component(
        id=obj.id,
        manufacturerPartNumber=obj.manufacturer_part_number,
        manufacturer={"id": obj.manufacturer_id} if obj.manufacturer_id else None,
        category=obj.category,
        description=obj.description,
        lifecycleStatus=obj.lifecycle_status,
        rohsCompliant=obj.rohs_compliant,
        datasheet=obj.datasheet_url,
        specifications={},
        alternativeParts=[],
        createdAt=obj.created_at.isoformat() if obj.created_at else None,
        updatedAt=obj.updated_at.isoformat() if obj.updated_at else None,
    )
    # Best-effort ES indexing
    index_component(res.model_dump())
    return res


@router.put("/{component_id}", response_model=Component, dependencies=[Depends(require_scopes(["write:components"]))])
def update_component(component_id: str, body: UpdateComponentRequest, session: Session = Depends(get_session)):
    repo = ComponentsRepository(session)
    
    # Only include non-None fields in the update data
    update_data = {}
    if body.manufacturerPartNumber is not None:
        update_data["manufacturerPartNumber"] = body.manufacturerPartNumber
    if body.manufacturerId is not None:
        update_data["manufacturer"] = {"id": body.manufacturerId}
    if body.category is not None:
        update_data["category"] = body.category
    if body.subcategory is not None:
        update_data["subcategory"] = body.subcategory
    if body.description is not None:
        update_data["description"] = body.description
    if body.datasheetUrl is not None:
        update_data["datasheet"] = body.datasheetUrl
    if body.rohsCompliant is not None:
        update_data["rohsCompliant"] = body.rohsCompliant
    
    obj = repo.update(component_id, update_data)
    if not obj:
        raise HTTPException(404, detail="Component not found")
    res = Component(
        id=obj.id,
        manufacturerPartNumber=obj.manufacturer_part_number,
        manufacturer={"id": obj.manufacturer_id} if obj.manufacturer_id else None,
        category=obj.category,
        description=obj.description,
        lifecycleStatus=obj.lifecycle_status,
        rohsCompliant=obj.rohs_compliant,
        datasheet=obj.datasheet_url,
        specifications={},
        alternativeParts=[],
        createdAt=obj.created_at.isoformat() if obj.created_at else None,
        updatedAt=obj.updated_at.isoformat() if obj.updated_at else None,
    )
    index_component(res.model_dump())
    return res


@router.delete("/{component_id}", response_model=dict, dependencies=[Depends(require_scopes(["write:components"]))])
def delete_component(component_id: str, session: Session = Depends(get_session)):
    repo = ComponentsRepository(session)
    ok = repo.delete(component_id)
    if not ok:
        raise HTTPException(404, detail="Component not found")
    delete_component_index(component_id)
    return {"deleted": True}


@router.get("/{component_id}/pricing", response_model=ComponentPricingResponse, dependencies=[Depends(require_scopes(["read:components"]))])
def get_pricing(component_id: str, session: Session = Depends(get_session)):
    comp = session.get(ComponentORM, component_id)
    if not comp:
        raise HTTPException(404, detail="Component not found")
    pr = PricingRepository(session)
    current = pr.current_pricing(component_id)
    history = pr.price_history(component_id)
    current_models = [
        SupplierPricing(
            supplier=SupplierRef(**c["supplier"]),
            quantityBreaks=[QuantityBreak(**qb) for qb in c.get("quantityBreaks", [])],
            availability=c.get("availability", "Unknown"),
            leadTimeWeeks=c.get("leadTimeWeeks", 0),
            lastUpdated=c.get("lastUpdated") or "",
        )
        for c in current
    ]
    history_models = [PriceHistoryItem(**h) for h in history]
    return ComponentPricingResponse(componentId=component_id, currentPricing=current_models, priceHistory=history_models)
