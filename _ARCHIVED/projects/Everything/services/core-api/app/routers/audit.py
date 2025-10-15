from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..repositories.audit_repo import AuditRepository
from ..core.auth import require_scopes


router = APIRouter()


@router.get("/", dependencies=[Depends(require_scopes(["read:audit"]))])
def list_audit(entityType: Optional[str] = Query(default=None), entityId: Optional[str] = Query(default=None), limit: int = 50, offset: int = 0, session: Session = Depends(get_session)):
    repo = AuditRepository(session)
    rows, total = repo.list(entity_type=entityType, entity_id=entityId, limit=limit, offset=offset)
    data = [
        {
            "id": r.id,
            "entityType": r.entity_type,
            "entityId": r.entity_id,
            "action": r.action,
            "oldValue": r.old_value,
            "newValue": r.new_value,
            "userId": r.user_id,
            "createdAt": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    return {"data": data, "total": total}

