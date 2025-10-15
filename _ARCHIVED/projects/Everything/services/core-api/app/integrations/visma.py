from __future__ import annotations

import os
from typing import Any, Dict, List
import httpx


class VismaClient:
    def __init__(self):
        self.base_url = os.getenv("VISMA_BASE_URL")
        self.api_key = os.getenv("VISMA_API_KEY")
        self.timeout = float(os.getenv("VISMA_TIMEOUT", "10"))

    def _client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.Client(base_url=self.base_url or "", headers=headers, timeout=self.timeout)

    def get_inventory(self) -> List[Dict[str, Any]]:
        if not self.base_url:
            return []
        with self._client() as c:
            r = c.get("/inventory")
            r.raise_for_status()
            return r.json()

    def get_suppliers(self) -> List[Dict[str, Any]]:
        if not self.base_url:
            return []
        with self._client() as c:
            r = c.get("/suppliers")
            r.raise_for_status()
            return r.json()

