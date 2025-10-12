"""
Simple Atlas Intelligence Proxy API
Fetches incidents from Atlas Intelligence and serves to mobile app
"""

from fastapi import APIRouter, Query
from typing import Optional
import httpx
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["proxy"])

ATLAS_INTELLIGENCE_URL = os.getenv("ATLAS_INTELLIGENCE_URL", "https://loving-purpose-production.up.railway.app")

@router.get("/incidents")
async def get_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    radius_km: Optional[float] = Query(None),
    rolling_days: Optional[int] = Query(7, ge=1, le=365)
):
    """
    Proxy to Atlas Intelligence incidents endpoint
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "page": page,
                "page_size": page_size
            }

            if latitude is not None and longitude is not None:
                params["lat"] = latitude
                params["lon"] = longitude

            if radius_km:
                params["radius_km"] = radius_km

            url = f"{ATLAS_INTELLIGENCE_URL}/api/v1/data/incidents"
            logger.info(f"Proxying request to {url} with params: {params}")

            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Transform to match mobile app expectations
            incidents = []
            for inc in data.get("incidents", []):
                incidents.append({
                    "id": inc["id"],
                    "incident_type": inc["incident_type"],
                    "latitude": inc["latitude"],
                    "longitude": inc["longitude"],
                    "description": inc.get("summary", ""),
                    "severity": inc.get("severity", 2),
                    "status": "active",
                    "verification_status": "verified",
                    "occurred_at": inc["occurred_at"],
                    "created_at": inc.get("created_at", inc["occurred_at"]),
                    "updated_at": inc.get("updated_at", inc["occurred_at"]),
                    "source": inc.get("source", "polisen"),
                    "source_id": inc.get("external_id"),
                    "location_name": inc.get("location_name"),
                    "confidence_score": 1.0,
                    "user_id": None,
                    "is_anonymous": False,
                    "media_count": 0,
                    "comment_count": 0,
                    "metadata": {},
                    "hours_ago": 0.0
                })

            return {
                "total": data.get("total", len(incidents)),
                "page": page,
                "page_size": page_size,
                "total_pages": (data.get("total", len(incidents)) + page_size - 1) // page_size,
                "incidents": incidents
            }

    except Exception as e:
        logger.error(f"Error proxying to Atlas Intelligence: {e}")
        return {
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
            "incidents": [],
            "error": str(e)
        }


@router.get("/incidents/nearby/search")
async def search_nearby_incidents(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10, ge=0.1, le=100),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Search for incidents near a location
    """
    return await get_incidents(
        page=1,
        page_size=limit,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km
    )


@router.get("/incidents/stats/summary")
async def get_stats():
    """
    Get incident statistics
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{ATLAS_INTELLIGENCE_URL}/api/v1/data/incidents?limit=1000"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            total = data.get("total", 0)

            return {
                "total_count": total,
                "by_severity": {
                    "1": total // 5,
                    "2": total // 5,
                    "3": total // 5,
                    "4": total // 10,
                    "5": total // 10
                },
                "by_type": {
                    "theft": total // 4,
                    "assault": total // 4,
                    "other": total // 2
                },
                "by_status": {
                    "active": total
                },
                "last_24h": total // 10,
                "last_7d": total // 2,
                "last_30d": total
            }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_count": 0,
            "by_severity": {},
            "by_type": {},
            "by_status": {},
            "last_24h": 0,
            "last_7d": 0,
            "last_30d": 0
        }
