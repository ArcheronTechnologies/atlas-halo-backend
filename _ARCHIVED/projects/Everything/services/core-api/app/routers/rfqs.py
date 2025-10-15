from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.rfqs_repo import RFQsRepository
from ..models.rfqs import RFQListResponse, RFQSummary, CreateRFQRequest, CreateQuoteRequest
from ..core.auth import require_scopes
from ..core.auth import require_scopes


router = APIRouter()


@router.get("/", response_model=RFQListResponse, dependencies=[Depends(require_scopes(["read:rfqs"]))])
def list_rfqs(status: str | None = None, customerId: str | None = None, limit: int = 50, offset: int = 0, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    items = repo.list_filtered(status=status, customer_id=customerId, limit=limit, offset=offset)
    data: List[RFQSummary] = []
    for rfq in items:
        data.append(
            RFQSummary(
                id=rfq.id,
                rfqNumber=rfq.rfq_number,
                customer={"id": rfq.customer_id, "name": "Customer"},
                status=rfq.status or "open",
                totalValue=rfq.total_value,
                currency=rfq.currency or "USD",
                requiredDate=rfq.required_date.isoformat() if rfq.required_date else None,
                itemCount=len(rfq.items or []),
                source=rfq.source,
                urgency="high" if len(rfq.items or []) > 3 else "normal",
                aiRecommendations=[
                    {"type": "pricing_strategy", "message": "Historical data suggests 15% markup optimal", "confidence": 0.87}
                ],
                createdAt=rfq.created_at.isoformat() if rfq.created_at else None,
            )
        )
    return RFQListResponse(data=data)


@router.post("/", response_model=dict, dependencies=[Depends(require_scopes(["write:rfqs"]))])
def create_rfq(body: CreateRFQRequest, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    try:
        rfq = repo.create(body.model_dump())
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return rfq


@router.post("/{rfq_id}/quote", response_model=dict, dependencies=[Depends(require_scopes(["write:rfqs"]))])
def create_quote(rfq_id: str, body: CreateQuoteRequest, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    try:
        rfq = repo.add_quote(rfq_id, body.model_dump())
    except KeyError:
        raise HTTPException(404, detail="RFQ not found")
    return rfq


@router.get("/{rfq_id}/quotes", response_model=dict, dependencies=[Depends(require_scopes(["read:rfqs"]))])
def list_quotes(rfq_id: str, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    quotes = repo.list_quotes(rfq_id)
    return {"data": quotes}


@router.post("/{rfq_id}/quotes/{quote_id}/select", response_model=dict, dependencies=[Depends(require_scopes(["write:rfqs"]))])
def select_winning_quote(rfq_id: str, quote_id: str, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    try:
        res = repo.select_winning_quote(rfq_id, quote_id)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    return res


@router.get("/{rfq_id}", response_model=dict, dependencies=[Depends(require_scopes(["read:rfqs"]))])
def get_rfq(rfq_id: str, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    data = repo.get_detail(rfq_id)
    if not data:
        raise HTTPException(404, detail="RFQ not found")
    return data


@router.post("/{rfq_id}/status", response_model=dict, dependencies=[Depends(require_scopes(["write:rfqs"]))])
def update_rfq_status(rfq_id: str, body: dict, session: Session = Depends(get_session)):
    repo = RFQsRepository(session)
    new_status = body.get("status")
    if not new_status:
        raise HTTPException(400, detail="Missing status")
    try:
        data = repo.update_status(rfq_id, new_status)
    except KeyError:
        raise HTTPException(404, detail="RFQ not found")
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return data
