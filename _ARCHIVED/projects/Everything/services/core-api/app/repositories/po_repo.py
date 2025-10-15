from __future__ import annotations

import uuid
from typing import List, Dict, Any

from sqlalchemy import select
from sqlalchemy.orm import Session
from ..events.publisher import emit

from ..db.models import PurchaseOrder, POItem


class PORepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self) -> List[PurchaseOrder]:
        return self.session.execute(select(PurchaseOrder).order_by(PurchaseOrder.created_at.desc())).scalars().all()

    def _recalc_total(self, po_id: str) -> None:
        items = self.session.execute(select(POItem).where(POItem.po_id == po_id)).scalars().all()
        total = 0.0
        for it in items:
            if it.unit_price is not None:
                total += (it.unit_price or 0.0) * (it.quantity or 0)
        po = self.session.get(PurchaseOrder, po_id)
        if po:
            po.total_value = total
            self.session.add(po)

    def create(self, payload: dict) -> PurchaseOrder:
        po = PurchaseOrder(
            id=str(uuid.uuid4()),
            supplier_id=payload.get("supplierId"),
            po_number=payload.get("poNumber"),
            payment_terms=payload.get("paymentTerms"),
            delivery_address=payload.get("deliveryAddress"),
            status="draft",
            currency="USD",
        )
        self.session.add(po)
        for item in payload.get("items", []):
            rec = POItem(
                id=str(uuid.uuid4()),
                po_id=po.id,
                component_id=item.get("componentId"),
                quantity=item.get("quantity"),
                unit_price=item.get("unitPrice"),
                lead_time_weeks=item.get("leadTimeWeeks"),
                manufacturer_lot_code=item.get("manufacturerLotCode"),
                date_code=item.get("dateCode"),
                packaging=item.get("packaging"),
            )
            self.session.add(rec)
        self._recalc_total(po.id)
        self.session.commit()
        self.session.refresh(po)
        emit("po.created", {"poId": po.id, "supplierId": po.supplier_id, "totalValue": po.total_value})
        return po

    def get_detail(self, po_id: str) -> Dict[str, Any] | None:
        po = self.session.get(PurchaseOrder, po_id)
        if not po:
            return None
        items = self.session.execute(select(POItem).where(POItem.po_id == po_id)).scalars().all()
        return {
            "id": po.id,
            "supplierId": po.supplier_id,
            "poNumber": po.po_number,
            "status": po.status,
            "totalValue": po.total_value,
            "currency": po.currency,
            "paymentTerms": po.payment_terms,
            "deliveryAddress": po.delivery_address,
            "items": [
                {
                    "id": it.id,
                    "componentId": it.component_id,
                    "quantity": it.quantity,
                    "unitPrice": it.unit_price,
                    "leadTimeWeeks": it.lead_time_weeks,
                    "manufacturerLotCode": it.manufacturer_lot_code,
                    "dateCode": it.date_code,
                    "packaging": it.packaging,
                    "createdAt": it.created_at.isoformat() if it.created_at else None,
                }
                for it in items
            ],
            "createdAt": po.created_at.isoformat() if po.created_at else None,
            "updatedAt": po.updated_at.isoformat() if po.updated_at else None,
        }

    def update_status(self, po_id: str, status: str) -> Dict[str, Any]:
        allowed = {"draft", "sent", "acknowledged", "shipped", "received", "cancelled"}
        if status not in allowed:
            raise ValueError("invalid status")
        po = self.session.get(PurchaseOrder, po_id)
        if not po:
            raise KeyError("po not found")
        po.status = status
        self.session.add(po)
        self.session.commit()
        emit("po.status_changed", {"poId": po.id, "status": po.status})
        return {"id": po.id, "status": po.status}

    def add_item(self, po_id: str, item: dict) -> Dict[str, Any]:
        po = self.session.get(PurchaseOrder, po_id)
        if not po:
            raise KeyError("po not found")
        rec = POItem(
            id=str(uuid.uuid4()),
            po_id=po_id,
            component_id=item.get("componentId"),
            quantity=item.get("quantity"),
            unit_price=item.get("unitPrice"),
            lead_time_weeks=item.get("leadTimeWeeks"),
            manufacturer_lot_code=item.get("manufacturerLotCode"),
            date_code=item.get("dateCode"),
            packaging=item.get("packaging"),
        )
        self.session.add(rec)
        self._recalc_total(po_id)
        self.session.commit()
        emit("po.item_added", {"poId": po_id, "itemId": rec.id})
        return {"id": rec.id}

    def delete_item(self, po_id: str, item_id: str) -> bool:
        rec = self.session.get(POItem, item_id)
        if not rec or rec.po_id != po_id:
            return False
        self.session.delete(rec)
        self._recalc_total(po_id)
        self.session.commit()
        emit("po.item_deleted", {"poId": po_id, "itemId": item_id})
        return True
