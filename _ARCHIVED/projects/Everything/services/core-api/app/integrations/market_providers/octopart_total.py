from __future__ import annotations

import os
from typing import Any, Dict, List

from .octopart_common import graphql_request


async def fetch_total_availability(q: str, country: str = "US", limit: int = 5) -> Dict[str, Any]:
    """Query Octopart for total availability of parts matching query q in a country.

    Returns a dict with parts (mpn, description, totalAvail) and sumTotalAvail.
    If OCTOPART_API_KEY is missing or a request fails, returns an empty result structure.
    """
    api_key = os.getenv("OCTOPART_API_KEY")
    result: Dict[str, Any] = {
        "query": q,
        "country": country,
        "limit": limit,
        "parts": [],
        "sumTotalAvail": 0,
        "provider": "octopart",
    }
    # Allow offline/minimal runs without network
    if os.getenv("SCIP_SKIP_NETWORK") == "1":
        result["note"] = "network_skipped"
        return result
    if not api_key:
        result["note"] = "OCTOPART_API_KEY not configured; returning empty results"
        return result

    gql = (
        "query totalAvailability($q:String!,$country:String!,$limit:Int){\n"
        "  supSearchMpn(q:$q,country:$country,limit:$limit){\n"
        "    results{\n"
        "      description\n"
        "      part{ totalAvail mpn }\n"
        "    }\n"
        "  }\n"
        "}"
    )
    variables = {"q": q, "country": country, "limit": limit}
    try:
        data = await graphql_request("total_availability", gql, variables, cache_key_params={"q": q, "country": country, "limit": limit})
    except Exception as e:
        result["note"] = f"request_failed: {e}"
        return result

    try:
        results = (((data or {}).get("data") or {}).get("supSearchMpn") or {}).get("results") or []
        parts: List[Dict[str, Any]] = []
        total = 0
        for r in results:
            desc = r.get("description")
            p = (r.get("part") or {})
            mpn = p.get("mpn")
            avail = p.get("totalAvail") or 0
            total += int(avail or 0)
            parts.append({"mpn": mpn, "description": desc, "totalAvail": int(avail or 0)})
        result["parts"] = parts
        result["sumTotalAvail"] = total
        return result
    except Exception as e:
        result["note"] = f"parse_failed: {e}"
        return result
