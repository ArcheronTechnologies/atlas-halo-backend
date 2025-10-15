from __future__ import annotations

import os
from typing import Any, Dict, List

from .octopart_common import graphql_request


async def fetch_pricing_breaks(q: str, limit: int = 5, currency: str = "USD") -> Dict[str, Any]:
    """Return pricing breaks per seller for parts matching q via Octopart supSearchMpn.

    Shape:
    {
      query, limit, currency, provider: "octopart",
      hits, parts: [{ mpn, sellers: [{ company, priceBreaks: [{quantity, price, currency}] }] }]
    }
    """
    api_key = os.getenv("OCTOPART_API_KEY")
    result: Dict[str, Any] = {
        "query": q,
        "limit": limit,
        "currency": currency,
        "provider": "octopart",
        "hits": 0,
        "parts": [],
    }
    if os.getenv("SCIP_SKIP_NETWORK") == "1":
        result["note"] = "network_skipped"
        return result
    if not api_key:
        result["note"] = "OCTOPART_API_KEY not configured; returning empty results"
        return result

    gql = (
        "query pricingByVolumeLevels($q:String!,$limit:Int){\n"
        "  supSearchMpn(q:$q,limit:$limit){\n"
        "    hits\n"
        "    results{\n"
        "      part{\n"
        "        mpn\n"
        "        sellers{\n"
        "          company{name}\n"
        "          offers{prices{quantity price}}\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}"
    )
    variables = {"q": q, "limit": limit}
    try:
        data = await graphql_request("pricing_breaks", gql, variables, cache_key_params={"q": q, "limit": limit})
    except Exception as e:
        result["note"] = f"request_failed: {e}"
        return result

    try:
        sup = ((data or {}).get("data") or {}).get("supSearchMpn") or {}
        result["hits"] = sup.get("hits", 0)
        parts_out: List[Dict[str, Any]] = []
        for r in sup.get("results", []) or []:
            p = (r.get("part") or {})
            mpn = p.get("mpn")
            sellers_out: List[Dict[str, Any]] = []
            for s in p.get("sellers", []) or []:
                name = ((s.get("company") or {}).get("name"))
                breaks = []
                for o in s.get("offers", []) or []:
                    for br in o.get("prices", []) or []:
                        qty = int(br.get("quantity", 0) or 0)
                        price = float(br.get("price", 0) or 0)
                        breaks.append({"quantity": qty, "price": price, "currency": currency})
                sellers_out.append({"company": name, "priceBreaks": breaks})
            parts_out.append({"mpn": mpn, "sellers": sellers_out})
        result["parts"] = parts_out
        return result
    except Exception as e:
        result["note"] = f"parse_failed: {e}"
        return result
