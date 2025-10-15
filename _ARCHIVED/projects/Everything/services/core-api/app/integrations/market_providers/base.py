from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PriceBreak:
    quantity: int
    price: float
    currency: str


@dataclass
class Offer:
    distributor: str
    sku: Optional[str]
    url: Optional[str]
    in_stock_quantity: Optional[int]
    moq: Optional[int]
    order_multiple: Optional[int]
    lead_time_days: Optional[int]
    price_breaks: List[PriceBreak]
    country: Optional[str] = None


class MarketProvider:
    name: str = "base"

    async def fetch_offers(self, mpn: str, *, country: Optional[str] = None, currency: str = "USD") -> List[Offer]:
        raise NotImplementedError

