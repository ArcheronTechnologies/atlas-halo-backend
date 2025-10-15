from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.companies_repo import CompaniesRepository
from ..models.companies import CompanyCreate, CompanyResponse, CompaniesListResponse
from ..core.auth import require_scopes


router = APIRouter()


@router.get("/", response_model=CompaniesListResponse, dependencies=[Depends(require_scopes(["read:companies"]))])
def list_companies(
    search: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None, alias="type"),
    session: Session = Depends(get_session),
):
    repo = CompaniesRepository(session)
    items = repo.list(search=search, type_=type)
    data: List[CompanyResponse] = [
        CompanyResponse(
            id=c.id,
            name=c.name,
            type=c.type,
            website=c.website,
            createdAt=c.created_at.isoformat() if c.created_at else None,
            updatedAt=c.updated_at.isoformat() if c.updated_at else None,
        )
        for c in items
    ]
    return CompaniesListResponse(data=data)


@router.post("/", response_model=CompanyResponse, dependencies=[Depends(require_scopes(["write:companies"]))])
def create_company(body: CompanyCreate, session: Session = Depends(get_session)):
    repo = CompaniesRepository(session)
    c = repo.create(name=body.name, type_=body.type, website=body.website)
    return CompanyResponse(
        id=c.id,
        name=c.name,
        type=c.type,
        website=c.website,
        createdAt=c.created_at.isoformat() if c.created_at else None,
        updatedAt=c.updated_at.isoformat() if c.updated_at else None,
    )


@router.get("/{company_id}", response_model=CompanyResponse, dependencies=[Depends(require_scopes(["read:companies"]))])
def get_company(company_id: str, session: Session = Depends(get_session)):
    repo = CompaniesRepository(session)
    c = repo.get(company_id)
    if not c:
        raise HTTPException(404, detail="Company not found")
    return CompanyResponse(
        id=c.id,
        name=c.name,
        type=c.type,
        website=c.website,
        createdAt=c.created_at.isoformat() if c.created_at else None,
        updatedAt=c.updated_at.isoformat() if c.updated_at else None,
    )


@router.put("/{company_id}", response_model=CompanyResponse, dependencies=[Depends(require_scopes(["write:companies"]))])
def update_company(company_id: str, body: CompanyCreate, session: Session = Depends(get_session)):
    repo = CompaniesRepository(session)
    c = repo.update(company_id, name=body.name, type_=body.type, website=body.website)
    if not c:
        raise HTTPException(404, detail="Company not found")
    return CompanyResponse(
        id=c.id,
        name=c.name,
        type=c.type,
        website=c.website,
        createdAt=c.created_at.isoformat() if c.created_at else None,
        updatedAt=c.updated_at.isoformat() if c.updated_at else None,
    )


@router.delete("/{company_id}", response_model=dict, dependencies=[Depends(require_scopes(["write:companies"]))])
def delete_company(company_id: str, session: Session = Depends(get_session)):
    repo = CompaniesRepository(session)
    ok = repo.delete(company_id)
    if not ok:
        raise HTTPException(404, detail="Company not found")
    return {"deleted": True}
