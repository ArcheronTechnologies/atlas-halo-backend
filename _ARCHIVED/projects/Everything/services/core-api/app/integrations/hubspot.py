from __future__ import annotations

from typing import Any, Dict, List


class HubSpotClient:
    def update_company_properties(self, company_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        return {"companyId": company_id, "updated": list(properties.keys())}

    def create_task(self, title: str, description: str, assigned_to: str | None = None) -> Dict[str, Any]:
        return {"created": True, "title": title}

