from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..db.session import get_session
from ..core.auth import require_api_key_or_bearer
from ..integrations.market_providers.octopart_total import fetch_total_availability
from ..integrations.market_providers.octopart_pricing import fetch_pricing_breaks
from ..integrations.market_providers.octopart_offers import fetch_offers as fetch_octopart_offers
from ..integrations.market_providers.octopart_specs import fetch_spec_attributes
import json
from typing import Optional, Dict, List, Any
from ..models.market import (
    TotalAvailabilityResponse,
    PricingBreaksResponse,
    OffersResponse,
    SpecAttributesResponse,
)

router = APIRouter()

@router.get("/total-availability", response_model=TotalAvailabilityResponse, dependencies=[Depends(require_api_key_or_bearer)])
async def total_availability(
    q: str = Query(..., min_length=2),
    country: str = Query("US", min_length=2, max_length=2),
    limit: int = Query(5, ge=1, le=20),
    session: Session = Depends(get_session),
):
    return await fetch_total_availability(q=q, country=country, limit=limit)


@router.get("/pricing-breaks", response_model=PricingBreaksResponse, dependencies=[Depends(require_api_key_or_bearer)])
async def pricing_breaks(
    q: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=20),
    currency: str = Query("USD", min_length=3, max_length=3),
    session: Session = Depends(get_session),
):
    return await fetch_pricing_breaks(q=q, limit=limit, currency=currency)


@router.get("/offers", response_model=OffersResponse, dependencies=[Depends(require_api_key_or_bearer)])
async def offers(
    mpn: str = Query(..., min_length=1),
    country: str | None = Query(None, min_length=2, max_length=2),
    currency: str = Query("USD", min_length=3, max_length=3),
    session: Session = Depends(get_session),
):
    return await fetch_octopart_offers(mpn=mpn, country=country, currency=currency)


@router.get("/spec-attributes", response_model=SpecAttributesResponse, dependencies=[Depends(require_api_key_or_bearer)])
async def spec_attributes(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=50),
    filters: Optional[str] = Query(None, description='JSON object of attribute filters, e.g. {"case_package":["SSOP"]}'),
    session: Session = Depends(get_session),
):
    parsed_filters: Optional[Dict[str, List[str]]] = None
    note: Optional[str] = None
    if filters:
        try:
            parsed = json.loads(filters)
            if isinstance(parsed, dict):
                parsed_filters = {k: v for k, v in parsed.items() if isinstance(v, list)}  # simple validation
            else:
                note = "filters must be a JSON object"
        except Exception as e:
            note = f"invalid_filters_json: {e}"
    result = await fetch_spec_attributes(q=q, filters=parsed_filters, limit=limit)
    if note:
        result.setdefault("note", note)
    return result
