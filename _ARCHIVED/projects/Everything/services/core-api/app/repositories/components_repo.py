from __future__ import annotations

import uuid
from typing import Optional, Tuple, List, Dict, Any

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..db.models import Component


class ComponentsRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, *, search: Optional[str], limit: int, offset: int) -> Tuple[List[Component], int]:
        stmt = select(Component)
        if search:
            like = f"%{search}%"
            stmt = stmt.where(
                (Component.manufacturer_part_number.ilike(like))
                | (Component.description.ilike(like))
            )
        total = self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        items = self.session.execute(stmt.offset(offset).limit(limit)).scalars().all()
        return items, total

    def create(self, data: Dict[str, Any]) -> Component:
        comp = Component(
            id=str(uuid.uuid4()),
            manufacturer_part_number=data.get("manufacturerPartNumber"),
            manufacturer_id=(data.get("manufacturer") or {}).get("id"),
            category=data.get("category"),
            subcategory=data.get("subcategory"),
            description=data.get("description"),
            datasheet_url=data.get("datasheet"),
            lifecycle_status=data.get("lifecycleStatus", "active"),
            rohs_compliant=data.get("rohsCompliant", True),
        )
        self.session.add(comp)
        self.session.commit()
        self.session.refresh(comp)
        return comp

    def list_filtered(
        self,
        *,
        limit: int,
        offset: int,
        category: Optional[str] = None,
        manufacturer_id: Optional[str] = None,
        lifecycle: Optional[str] = None,
    ) -> Tuple[List[Component], int]:
        stmt = select(Component)
        if category:
            stmt = stmt.where(Component.category == category)
        if manufacturer_id:
            stmt = stmt.where(Component.manufacturer_id == manufacturer_id)
        if lifecycle:
            stmt = stmt.where(Component.lifecycle_status == lifecycle)
        total = self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        items = self.session.execute(stmt.offset(offset).limit(limit)).scalars().all()
        return items, total

    def get_by_ids(self, ids: List[str]) -> List[Component]:
        if not ids:
            return []
        stmt = select(Component).where(Component.id.in_(ids))
        items = self.session.execute(stmt).scalars().all()
        # Preserve order of ids
        order = {id_: i for i, id_ in enumerate(ids)}
        items.sort(key=lambda c: order.get(c.id, len(order)))
        return items

    def update(self, component_id: str, data: Dict[str, Any]) -> Optional[Component]:
        obj = self.session.get(Component, component_id)
        if not obj:
            return None
        if "manufacturerPartNumber" in data:
            obj.manufacturer_part_number = data["manufacturerPartNumber"]
        if "manufacturer" in data:
            man = data.get("manufacturer") or {}
            obj.manufacturer_id = man.get("id")
        if "category" in data:
            obj.category = data["category"]
        if "subcategory" in data:
            obj.subcategory = data["subcategory"]
        if "description" in data:
            obj.description = data["description"]
        if "datasheet" in data:
            obj.datasheet_url = data["datasheet"]
        if "lifecycleStatus" in data:
            obj.lifecycle_status = data["lifecycleStatus"]
        if "rohsCompliant" in data:
            obj.rohs_compliant = data["rohsCompliant"]
        self.session.add(obj)
        self.session.commit()
        self.session.refresh(obj)
        return obj

    def delete(self, component_id: str) -> bool:
        obj = self.session.get(Component, component_id)
        if not obj:
            return False
        self.session.delete(obj)
        self.session.commit()
        return True
