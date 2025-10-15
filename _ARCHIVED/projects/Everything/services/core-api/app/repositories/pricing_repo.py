from __future__ import annotations

from typing import Dict, List

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..db.models import PriceHistory, Company


class PricingRepository:
    def __init__(self, session: Session):
        self.session = session

    def current_pricing(self, component_id: str) -> List[dict]:
        subq = (
            select(
                PriceHistory.supplier_id,
                PriceHistory.quantity_break,
                func.max(PriceHistory.created_at).label("latest"),
            )
            .where(PriceHistory.component_id == component_id)
            .group_by(PriceHistory.supplier_id, PriceHistory.quantity_break)
            .subquery()
        )
        stmt = (
            select(PriceHistory, Company)
            .join(subq, (PriceHistory.supplier_id == subq.c.supplier_id) & (PriceHistory.quantity_break == subq.c.quantity_break) & (PriceHistory.created_at == subq.c.latest))
            .join(Company, Company.id == PriceHistory.supplier_id)
            .where(PriceHistory.component_id == component_id)
        )
        rows = self.session.execute(stmt).all()
        grouped: Dict[str, dict] = {}
        for ph, supplier in rows:
            entry = grouped.get(ph.supplier_id)
            if not entry:
                entry = {
                    "supplier": {"id": supplier.id, "name": supplier.name},
                    "quantityBreaks": [],
                    "availability": "Unknown",
                    "leadTimeWeeks": 0,
                    "lastUpdated": ph.created_at.isoformat() if ph.created_at else None,
                }
                grouped[ph.supplier_id] = entry
            entry["quantityBreaks"].append({
                "quantity": ph.quantity_break,
                "unitPrice": float(ph.unit_price),
                "currency": ph.currency,
            })
        return list(grouped.values())

    def price_history(self, component_id: str) -> List[dict]:
        stmt = (
            select(func.date(PriceHistory.created_at).label("date"), func.avg(PriceHistory.unit_price))
            .where(PriceHistory.component_id == component_id)
            .group_by(func.date(PriceHistory.created_at))
            .order_by(func.date(PriceHistory.created_at))
        )
        rows = self.session.execute(stmt).all()
        out: List[dict] = []
        for d, avg in rows:
            out.append({
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "averagePrice": float(avg),
                "priceRange": {"min": None, "max": None},
            })
        return out

