from __future__ import annotations

import json
import uuid
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import RFQ, RFQItem, RFQQuote, Company, AuditLog
from datetime import datetime, timezone
import uuid as _uuid
from ..events.publisher import emit


class RFQsRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self) -> List[RFQ]:
        stmt = select(RFQ).order_by(RFQ.created_at.desc())
        return self.session.execute(stmt).scalars().unique().all()

    def create(self, payload: dict) -> dict:
        rfq = RFQ(
            id=str(uuid.uuid4()),
            customer_id=payload.get("customerId"),
            rfq_number=payload.get("rfqNumber"),
            status="open",
            source=payload.get("source"),
        )
        # Verify customer exists
        customer = self.session.get(Company, rfq.customer_id) if rfq.customer_id else None
        if not customer:
            raise ValueError("Invalid customerId")
        self.session.add(rfq)
        for item in payload.get("items", []):
            rfq_item = RFQItem(
                id=str(uuid.uuid4()),
                rfq=rfq,
                component_id=item.get("componentId"),
                customer_part_number=item.get("customerPartNumber"),
                quantity=item.get("quantity"),
                target_price=item.get("targetPrice"),
                lead_time_weeks=item.get("leadTimeWeeks"),
                packaging=item.get("packaging"),
            )
            self.session.add(rfq_item)

        self.session.commit()
        return {
            "id": rfq.id,
            "customerId": rfq.customer_id,
            "rfqNumber": rfq.rfq_number,
            "status": rfq.status,
            "items": [
                {
                    "id": i.id,
                    "componentId": i.component_id,
                    "customerPartNumber": i.customer_part_number,
                    "quantity": i.quantity,
                }
                for i in rfq.items
            ],
        }

    def add_quote(self, rfq_id: str, quote_payload: dict) -> dict:
        rfq = self.session.get(RFQ, rfq_id)
        if not rfq:
            raise KeyError("rfq not found")
        rfq.status = "quoted"
        self.session.add(
            RFQQuote(id=str(uuid.uuid4()), rfq_id=rfq_id, payload=json.dumps(quote_payload))
        )
        self.session.commit()
        emit("rfq.status_changed", {"rfqId": rfq.id, "status": rfq.status})
        return {"id": rfq.id, "status": rfq.status}

    def list_quotes(self, rfq_id: str) -> list[dict]:
        quotes = self.session.execute(select(RFQQuote).where(RFQQuote.rfq_id == rfq_id).order_by(RFQQuote.created_at.desc())).scalars().all()
        return [
            {"id": q.id, "payload": json.loads(q.payload), "createdAt": q.created_at.isoformat() if q.created_at else None}
            for q in quotes
        ]

    def list_filtered(self, *, status: str | None = None, customer_id: str | None = None, limit: int = 50, offset: int = 0) -> List[RFQ]:
        stmt = select(RFQ).order_by(RFQ.created_at.desc())
        if status:
            stmt = stmt.where(RFQ.status == status)
        if customer_id:
            stmt = stmt.where(RFQ.customer_id == customer_id)
        return self.session.execute(stmt.offset(offset).limit(limit)).scalars().unique().all()

    def get_detail(self, rfq_id: str) -> dict | None:
        rfq = self.session.get(RFQ, rfq_id)
        if not rfq:
            return None
        quotes = self.session.execute(select(RFQQuote).where(RFQQuote.rfq_id == rfq_id).order_by(RFQQuote.created_at.desc())).scalars().all()
        return {
            "id": rfq.id,
            "rfqNumber": rfq.rfq_number,
            "customerId": rfq.customer_id,
            "status": rfq.status,
            "requiredDate": rfq.required_date.isoformat() if rfq.required_date else None,
            "items": [
                {
                    "id": i.id,
                    "componentId": i.component_id,
                    "customerPartNumber": i.customer_part_number,
                    "quantity": i.quantity,
                    "targetPrice": i.target_price,
                }
                for i in rfq.items
            ],
            "quotes": [
                {"id": q.id, "payload": json.loads(q.payload), "createdAt": q.created_at.isoformat() if q.created_at else None}
                for q in quotes
            ],
            "createdAt": rfq.created_at.isoformat() if rfq.created_at else None,
        }

    def update_status(self, rfq_id: str, new_status: str) -> dict:
        allowed = {"open", "quoted", "won", "lost", "expired"}
        if new_status not in allowed:
            raise ValueError("invalid status")
        rfq = self.session.get(RFQ, rfq_id)
        if not rfq:
            raise KeyError("rfq not found")
        if rfq.status == "open" and new_status in {"won", "lost"}:
            raise ValueError("cannot transition from open directly to won/lost")
        old = rfq.status
        rfq.status = new_status
        self.session.add(rfq)
        # Audit log
        self.session.add(
            AuditLog(
                id=str(_uuid.uuid4()),
                entity_type="rfq",
                entity_id=rfq_id,
                action="status_change",
                old_value=old,
                new_value=new_status,
                created_at=datetime.now(timezone.utc),
            )
        )
        self.session.commit()
        return {"id": rfq.id, "status": rfq.status}

    def select_winning_quote(self, rfq_id: str, quote_id: str) -> dict:
        rfq = self.session.get(RFQ, rfq_id)
        if not rfq:
            raise KeyError("rfq not found")
        # Reset
        quotes = self.session.execute(select(RFQQuote).where(RFQQuote.rfq_id == rfq_id)).scalars().all()
        found = False
        for q in quotes:
            if q.id == quote_id:
                q.is_winner = True
                found = True
            else:
                q.is_winner = False
            self.session.add(q)
        if not found:
            raise KeyError("quote not found")
        rfq.status = "won"
        self.session.add(rfq)
        self.session.add(
            AuditLog(
                id=str(_uuid.uuid4()),
                entity_type="rfq",
                entity_id=rfq_id,
                action="select_winner",
                old_value=None,
                new_value=quote_id,
                created_at=datetime.now(timezone.utc),
            )
        )
        self.session.commit()
        emit("rfq.winner_selected", {"rfqId": rfq.id, "quoteId": quote_id})
        return {"id": rfq.id, "status": rfq.status, "winner": quote_id}
