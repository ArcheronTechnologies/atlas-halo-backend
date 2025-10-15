from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .octopart_common import graphql_request


async def fetch_offers(mpn: str, *, country: Optional[str] = None, currency: str = "USD") -> Dict[str, Any]:
    """Fetch detailed offers for a specific MPN from Octopart.

    Response shape:
    {
      mpn, country, currency, provider: "octopart",
      sellers: [
        {
          company: str,
          offers: [
            {
              sku, url, inStockQuantity, moq, orderMultiple, leadTimeDays,
              priceBreaks: [{ quantity, price, currency }]
            }
          ]
        }
      ]
    }
    """
    api_key = os.getenv("OCTOPART_API_KEY")
    result: Dict[str, Any] = {
        "mpn": mpn,
        "country": country,
        "currency": currency,
        "provider": "octopart",
        "sellers": [],
    }

    if os.getenv("SCIP_SKIP_NETWORK") == "1":
        result["note"] = "network_skipped"
        return result

    if not api_key:
        result["note"] = "OCTOPART_API_KEY not configured; returning empty results"
        return result

    # GraphQL query to get offers per seller for an MPN using supSearchMpn
    gql = (
        "query mpnOffers($q:String!,$country:String,$limit:Int){\n"
        "  supSearchMpn(q:$q,country:$country,limit:$limit){\n"
        "    results{\n"
        "      part{\n"
        "        mpn\n"
        "        sellers{\n"
        "          company{name}\n"
        "          offers{\n"
        "            sku\n"
        "            clickUrl\n"
        "            inventoryLevel\n"
        "            moq\n"
        "            orderMultiple\n"
        "            prices{quantity price}\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}"
    )

    variables = {"q": mpn, "country": country, "limit": 1}
    try:
        data = await graphql_request("offers", gql, variables, cache_key_params={"mpn": mpn, "country": country})
    except Exception as e:
        result["note"] = f"request_failed: {e}"
        return result

    try:
        results = (((data or {}).get("data") or {}).get("supSearchMpn") or {}).get("results") or []
        if not results:
            result["sellers"] = []
            return result
        
        part = results[0].get("part") or {}
        sellers = part.get("sellers", []) or []
        sellers_out: List[Dict[str, Any]] = []
        for s in sellers:
            company = ((s.get("company") or {}).get("name"))
            offers_out: List[Dict[str, Any]] = []
            for o in s.get("offers", []) or []:
                price_breaks = []
                for br in o.get("prices", []) or []:
                    qty = int(br.get("quantity", 0) or 0)
                    price = float(br.get("price", 0) or 0)
                    price_breaks.append({"quantity": qty, "price": price, "currency": currency})

                offers_out.append(
                    {
                        "sku": o.get("sku"),
                        "url": o.get("clickUrl"),
                        "inStockQuantity": (
                            int(o.get("inventoryLevel") or 0) if o.get("inventoryLevel") is not None else None
                        ),
                        "moq": int(o.get("moq") or 0) if o.get("moq") is not None else None,
                        "orderMultiple": int(o.get("orderMultiple") or 0)
                        if o.get("orderMultiple") is not None
                        else None,
                        "leadTimeDays": None,
                        "priceBreaks": price_breaks,
                    }
                )
            sellers_out.append({"company": company, "offers": offers_out})

        result["sellers"] = sellers_out
        return result
    except Exception as e:
        result["note"] = f"parse_failed: {e}"
        return result
