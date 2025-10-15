from __future__ import annotations

from typing import List, Optional, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..db.models import AuditLog


class AuditRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, *, entity_type: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> Tuple[List[AuditLog], int]:
        stmt = select(AuditLog)
        if entity_type:
            stmt = stmt.where(AuditLog.entity_type == entity_type)
        if entity_id:
            stmt = stmt.where(AuditLog.entity_id == entity_id)
        total = self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = self.session.execute(stmt.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit)).scalars().all()
        return rows, total

