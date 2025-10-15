from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from .octopart_common import graphql_request


async def fetch_spec_attributes(q: str, *, filters: Optional[Dict[str, List[str]]] = None, limit: int = 5) -> Dict[str, Any]:
    """Query Octopart for part specification attributes using supSearchMpn.

    Args:
      q: Search query (e.g., partial MPN)
      filters: Optional dict of attribute shortnames to list of values, e.g. {"case_package": ["SSOP"]}
      limit: Max number of parts to return

    Returns:
      {
        query, limit, provider: "octopart", hits,
        filters, parts: [ { mpn, specs: [ { attribute: { name,id,shortname }, displayValue } ] } ]
      }
    """
    api_key = os.getenv("OCTOPART_API_KEY")
    result: Dict[str, Any] = {
        "query": q,
        "limit": limit,
        "provider": "octopart",
        "hits": 0,
        "filters": filters or {},
        "parts": [],
    }

    if os.getenv("SCIP_SKIP_NETWORK") == "1":
        result["note"] = "network_skipped"
        return result

    if not api_key:
        result["note"] = "OCTOPART_API_KEY not configured; returning empty results"
        return result

    # GraphQL query modeled on Octopart examples; filters is JSON scalar
    gql = (
        "query specAttributes($q:String!,$limit:Int,$filters:JSON){\n"
        "  supSearchMpn(q:$q,limit:$limit,filters:$filters){\n"
        "    hits\n"
        "    results{\n"
        "      part{\n"
        "        mpn\n"
        "        specs{\n"
        "          attribute{ name id shortname }\n"
        "          displayValue\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}"
    )

    variables: Dict[str, Any] = {"q": q, "limit": limit, "filters": filters}
    try:
        data = await graphql_request("spec_attributes", gql, variables, cache_key_params={"q": q, "filters": filters, "limit": limit})
    except Exception as e:
        result["note"] = f"request_failed: {e}"
        return result

    try:
        sup = ((data or {}).get("data") or {}).get("supSearchMpn") or {}
        result["hits"] = sup.get("hits", 0)
        parts_out: List[Dict[str, Any]] = []
        for r in sup.get("results", []) or []:
            part = (r.get("part") or {})
            mpn = part.get("mpn")
            specs_out: List[Dict[str, Any]] = []
            for s in part.get("specs", []) or []:
                attr = s.get("attribute") or {}
                specs_out.append(
                    {
                        "attribute": {
                            "name": attr.get("name"),
                            "id": attr.get("id"),
                            "shortname": attr.get("shortname"),
                        },
                        "displayValue": s.get("displayValue"),
                    }
                )
            parts_out.append({"mpn": mpn, "specs": specs_out})
        result["parts"] = parts_out
        return result
    except Exception as e:
        result["note"] = f"parse_failed: {e}"
        return result
