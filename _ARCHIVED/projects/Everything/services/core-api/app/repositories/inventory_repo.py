from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..db.models import Inventory


class InventoryRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, *, component_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> Tuple[List[Inventory], int]:
        stmt = select(Inventory)
        if component_id:
            stmt = stmt.where(Inventory.component_id == component_id)
        total = self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = self.session.execute(stmt.order_by(Inventory.created_at.desc()).offset(offset).limit(limit)).scalars().all()
        return rows, total

    def create(self, payload: dict) -> Inventory:
        rec = Inventory(
            id=str(uuid.uuid4()),
            component_id=payload.get("componentId"),
            location=payload.get("location"),
            quantity_available=payload.get("quantityAvailable", 0),
            quantity_reserved=payload.get("quantityReserved", 0),
            cost_per_unit=payload.get("costPerUnit"),
            date_code=payload.get("dateCode"),
            lot_code=payload.get("lotCode"),
            expiry_date=payload.get("expiryDate"),
        )
        self.session.add(rec)
        self.session.commit()
        self.session.refresh(rec)
        return rec

    def delete(self, inventory_id: str) -> bool:
        rec = self.session.get(Inventory, inventory_id)
        if not rec:
            return False
        self.session.delete(rec)
        self.session.commit()
        return True

