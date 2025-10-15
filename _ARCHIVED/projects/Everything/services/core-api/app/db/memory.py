from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


def _uuid() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class MemoryDB:
    def __init__(self) -> None:
        self.components: Dict[str, Dict[str, Any]] = {}
        self.rfqs: Dict[str, Dict[str, Any]] = {}
        self.rfq_items: Dict[str, Dict[str, Any]] = {}
        self.emails: List[Dict[str, Any]] = []
        self.web_data: List[Dict[str, Any]] = []

    # Components
    def list_components(self, *, search: Optional[str] = None, limit: int = 50, offset: int = 0) -> tuple[List[Dict[str, Any]], int]:
        items = list(self.components.values())
        if search:
            s = search.lower()
            items = [c for c in items if s in c.get("manufacturerPartNumber", "").lower() or s in c.get("description", "").lower()]
        total = len(items)
        return items[offset : offset + limit], total

    def create_component(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cid = _uuid()
        item = {
            "id": cid,
            "createdAt": _now_iso(),
            "updatedAt": _now_iso(),
            **data,
        }
        self.components[cid] = item
        return item

    # RFQs
    def list_rfqs(self, *, limit: int = 50, offset: int = 0) -> tuple[List[Dict[str, Any]], int]:
        items = list(self.rfqs.values())
        total = len(items)
        return items[offset : offset + limit], total

    def create_rfq(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rid = _uuid()
        item = {
            "id": rid,
            "status": "open",
            "createdAt": _now_iso(),
            **data,
        }
        self.rfqs[rid] = item
        return item

    def add_rfq_quote(self, rfq_id: str, quote: Dict[str, Any]) -> Dict[str, Any]:
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise KeyError("rfq not found")
        rfq.setdefault("quotes", []).append({**quote, "createdAt": _now_iso()})
        rfq["status"] = "quoted"
        return rfq

    # Ingestion
    def add_emails(self, emails: List[Dict[str, Any]]) -> int:
        self.emails.extend(emails)
        return len(emails)

    def add_web_data(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        self.web_data.append(doc)
        return doc


db = MemoryDB()

