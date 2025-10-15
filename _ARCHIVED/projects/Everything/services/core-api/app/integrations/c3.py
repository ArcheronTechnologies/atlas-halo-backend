from __future__ import annotations

from typing import Any, Dict, List


class C3Client:
    def push_recommendations(self, rfq_id: str, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"rfqId": rfq_id, "pushed": len(recommendations)}

