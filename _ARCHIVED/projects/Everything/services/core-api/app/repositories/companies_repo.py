from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Company


class CompaniesRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, search: Optional[str] = None, type_: Optional[str] = None) -> List[Company]:
        stmt = select(Company)
        if search:
            like = f"%{search}%"
            stmt = stmt.where(Company.name.ilike(like))
        if type_:
            stmt = stmt.where(Company.type == type_)
        return self.session.execute(stmt.order_by(Company.name.asc())).scalars().all()

    def get(self, company_id: str) -> Company | None:
        return self.session.get(Company, company_id)

    def create(self, name: str, type_: Optional[str] = None, website: Optional[str] = None) -> Company:
        comp = Company(id=str(uuid.uuid4()), name=name, type=type_, website=website)
        self.session.add(comp)
        self.session.commit()
        self.session.refresh(comp)
        return comp

    def update(self, company_id: str, *, name: Optional[str] = None, type_: Optional[str] = None, website: Optional[str] = None) -> Company | None:
        c = self.get(company_id)
        if not c:
            return None
        if name is not None:
            c.name = name
        if type_ is not None:
            c.type = type_
        if website is not None:
            c.website = website
        self.session.add(c)
        self.session.commit()
        self.session.refresh(c)
        return c

    def delete(self, company_id: str) -> bool:
        c = self.get(company_id)
        if not c:
            return False
        self.session.delete(c)
        self.session.commit()
        return True
