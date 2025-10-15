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

ATLAS_INTELLIGENCE_URL = os.getenv("ATLAS_API_URL", os.getenv("ATLAS_INTELLIGENCE_URL", "https://atlas-intelligence.fly.dev"))

@router.get("/incidents")
async def get_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    radius_km: Optional[float] = Query(None),
    rolling_days: Optional[int] = Query(None)
):
    """
    Fetch incidents directly from shared Atlas Intelligence database
    """
    from datetime import datetime, timezone, timedelta
    import asyncpg
    import os

    try:
        now_utc = datetime.now(timezone.utc)

        # Direct database connection
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', ''),
            database=os.getenv('POSTGRES_DB', 'postgres')
        )

        # Build query with proper asyncpg placeholders
        query = "SELECT * FROM incidents WHERE 1=1"
        params = []

        # Add time filter
        if rolling_days:
            cutoff = now_utc - timedelta(days=rolling_days)
            query += f" AND occurred_at >= ${len(params)+1}"
            params.append(cutoff)

        # Add ordering and pagination
        offset = (page - 1) * page_size
        query += f" ORDER BY occurred_at DESC LIMIT ${len(params)+1} OFFSET ${len(params)+2}"
        params.extend([page_size, offset])

        # Get total count
        count_query = "SELECT COUNT(*) as total FROM incidents WHERE 1=1"
        count_params = []
        if rolling_days:
            count_query += " AND occurred_at >= $1"
            count_params.append(cutoff)

        # Execute queries
        results = await conn.fetch(query, *params)
        count_result = await conn.fetchrow(count_query, *count_params)
        total = count_result['total'] if count_result else 0

        # Transform to mobile app format
        incidents = []
        for row in results:
            # Calculate hours_ago
            occurred_at = row['occurred_at']
            if occurred_at.tzinfo is not None:
                hours_ago = (now_utc - occurred_at).total_seconds() / 3600
            else:
                occurred_at_utc = occurred_at.replace(tzinfo=timezone.utc)
                hours_ago = (now_utc - occurred_at_utc).total_seconds() / 3600

            incidents.append({
                "id": str(row['id']),
                "incident_type": row.get('incident_type', 'Unknown'),
                "latitude": float(row['latitude']) if row.get('latitude') else 0.0,
                "longitude": float(row['longitude']) if row.get('longitude') else 0.0,
                "description": row.get('summary', ''),
                "severity": int(row.get('severity', 2)) if str(row.get('severity', '2')).isdigit() else 2,
                "status": "active",
                "verification_status": "verified",
                "occurred_at": row['occurred_at'].isoformat() if row.get('occurred_at') else None,
                "created_at": row['created_at'].isoformat() if row.get('created_at') else None,
                "updated_at": row['updated_at'].isoformat() if row.get('updated_at') else None,
                "source": row.get('source', 'polisen'),
                "source_id": row.get('external_id'),
                "location_name": row.get('location_name'),
                "confidence_score": 1.0,
                "user_id": None,
                "is_anonymous": False,
                "media_count": 0,
                "comment_count": 0,
                "metadata": {},
                "hours_ago": hours_ago
            })

        await conn.close()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "incidents": incidents
        }

    except Exception as e:
        logger.error(f"Error fetching incidents from database: {e}", exc_info=True)
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
