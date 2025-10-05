"""
Brå (Brottsförebyggande rådet) PxWeb client and ingestor.

Provides a lightweight wrapper around PxWeb JSON API used at
https://statistik.bra.se/ (e.g., /api/v1/Nationella_brottsstatistik/b1201).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import ssl
try:
    import certifi  # type: ignore
    _CERTIFI = True
except Exception:
    certifi = None  # type: ignore
    _CERTIFI = False

logger = logging.getLogger(__name__)


from ..config.settings import get_settings

# Default base prefers BRA_PXWEB_BASE_URL; if unavailable, falls back to SCB PxWeb
DEFAULT_BASE = get_settings().bra_pxweb_base_url.rstrip("/")


@dataclass
class PxWebQuery:
    table_path: str
    query: Dict[str, Any]
    format: str = "JSON"

    def to_payload(self) -> Dict[str, Any]:
        payload = {"query": self.query, "response": {"format": self.format}}
        return payload


class PxWebClient:
    """Minimal async client for PxWeb endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE, timeout: float | None = None):
        settings = get_settings()
        self.base_url = (base_url or settings.bra_pxweb_base_url).rstrip("/")
        self._timeout = timeout or settings.pxweb_timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        # Use system trust store; optionally allow toggling insecure mode via env if needed later
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def post(self, table_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("PxWebClient not initialized; use 'async with PxWebClient()' context")
        url = f"{self.base_url}/{table_path.lstrip('/')}"
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

    async def fetch(self, pxq: PxWebQuery) -> Dict[str, Any]:
        return await self.post(pxq.table_path, pxq.to_payload())


def make_month_codes(
    start_year: int,
    end_year: int,
    *,
    start_month: int = 1,
    end_month: int = 12,
) -> List[str]:
    """Return PxWeb month period codes like '2023M01'.

    Caps the first and last year to the provided months so callers can avoid
    requesting future periods from PxWeb.
    """
    out: List[str] = []
    for year in range(start_year, end_year + 1):
        month_from = start_month if year == start_year else 1
        month_to = end_month if year == end_year else 12
        for month in range(month_from, month_to + 1):
            out.append(f"{year}M{month:02d}")
    return out


def build_bra_reported_offences_query(
    region_codes: List[str],
    offence_codes: Optional[List[str]] = None,
    periods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct a PxWeb query payload for reported offences (b1201).

    Note: Exact dimension codes must match the Brå table metadata. This builder
    uses typical PxWeb conventions: codes like 'Region', 'Brottskategori', 'Tid'.
    """
    q: List[Dict[str, Any]] = []
    q.append({"code": "Region", "selection": {"filter": "item", "values": region_codes}})
    if offence_codes:
        q.append({
            "code": "Brottskategori",
            "selection": {"filter": "item", "values": offence_codes},
        })
    else:
        # Select all
        q.append({"code": "Brottskategori", "selection": {"filter": "all", "values": ["*"]}})
    if periods:
        q.append({"code": "Tid", "selection": {"filter": "item", "values": periods}})
    else:
        q.append({"code": "Tid", "selection": {"filter": "all", "values": ["*"]}})
    return q


def flatten_pxweb_data(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten PxWeb response into list of dicts with dimension values and value.

    PxWeb returns {"columns": [...], "data": [{"key":[...], "values":["123"]}, ...]}
    """
    columns = result.get("columns", [])
    data = result.get("data", [])
    col_names = [c.get("text") or c.get("code") for c in columns]
    col_codes = [c.get("code") for c in columns]

    flat: List[Dict[str, Any]] = []
    for row in data:
        key_vals = row.get("key", [])
        value_strs = row.get("values", [])
        value = None
        # PxWeb may have single or multiple values; take first
        if value_strs:
            try:
                value = float(value_strs[0])
            except ValueError:
                # Could be missing or non-numeric; store as None
                value = None
        entry: Dict[str, Any] = {"value": value}
        for i, v in enumerate(key_vals):
            code = col_codes[i] if i < len(col_codes) else f"dim{i}"
            entry[code] = v
        flat.append(entry)
    return flat


async def ingest_bra_table(
    table_path: str,
    query: Dict[str, Any],
    table_id: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch and normalize a Brå PxWeb table into generic rows used for DB insert.

    Returns rows with keys: table_id, region_code, region_name, offence_code,
    offence_name, period, value, unit, extra.
    """
    table_id = table_id or table_path
    async with PxWebClient(base_url=base_url or DEFAULT_BASE) as client:
        payload = {"query": query, "response": {"format": "JSON"}}
        result = await client.post(table_path, payload)

    flat = flatten_pxweb_data(result)
    # Guess column codes by typical naming in Brå tables
    # Fallbacks keep unknown columns in 'extra'
    rows: List[Dict[str, Any]] = []
    for e in flat:
        region = e.get("Region") or e.get("region") or e.get("REGION")
        offence = e.get("Brottskategori") or e.get("brottskategori") or e.get("BROTTSKATEGORI")
        period = e.get("Tid") or e.get("tid") or e.get("TID")
        value = e.get("value")
        # Move remaining dims to extra
        extra = {k: v for k, v in e.items() if k not in {"Region", "region", "REGION", "Brottskategori", "brottskategori", "BROTTSKATEGORI", "Tid", "tid", "TID", "value"}}
        rows.append(
            {
                "table_id": table_id,
                "region_code": str(region) if region is not None else "",
                "region_name": None,  # Brå returns names via metadata; keep None for now
                "offence_code": str(offence) if offence is not None and str(offence).strip() else "ALL",
                "offence_name": None,
                "period": str(period) if period is not None else "",
                "value": value if value is not None else 0.0,
                "unit": None,
                "extra": extra,
            }
        )
    logger.info(f"Flattened {len(rows)} Brå rows from {table_path}")
    return rows
