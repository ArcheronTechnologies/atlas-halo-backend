from __future__ import annotations

import os
from typing import Any, Dict, List
import httpx


class BusinessCentralClient:
    def __init__(self):
        self.base_url = os.getenv("BC_BASE_URL")
        self.tenant = os.getenv("BC_TENANT")
        self.env = os.getenv("BC_ENV", "Production")
        self.token = os.getenv("BC_TOKEN")
        self.timeout = float(os.getenv("BC_TIMEOUT", "10"))

    def _client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return httpx.Client(base_url=self.base_url or "", headers=headers, timeout=self.timeout)

    def get_items(self) -> List[Dict[str, Any]]:
        if not self.base_url:
            return []
        # Example OData endpoint: /api/data/v2.0/companies({companyId})/items
        with self._client() as c:
            r = c.get("/api/data/v2.0/items")
            r.raise_for_status()
            data = r.json()
            return data.get("value", data)

