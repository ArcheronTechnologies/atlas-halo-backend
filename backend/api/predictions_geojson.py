"""
Predictions GeoJSON API
Serves pre-computed predictions from the predictions table with OSM boundary data
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncpg
import os
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


async def get_db_connection():
    """Get database connection"""
    return await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
        database=os.getenv('POSTGRES_DB', 'postgres')
    )


@router.get("/neighborhoods")
async def get_neighborhood_predictions(
    lat: Optional[float] = Query(None, description="Center latitude"),
    lon: Optional[float] = Query(None, description="Center longitude"),
    radius_km: Optional[float] = Query(50.0, description="Radius in kilometers"),
    hour_offset: int = Query(0, description="Hours from now (0-23)"),
    min_risk: float = Query(0.0, description="Minimum risk score (0-1)"),
    limit: int = Query(100, description="Maximum predictions to return")
):
    """
    Get neighborhood predictions with OSM boundary data

    Returns predictions from the pre-computed predictions table, including:
    - Real neighborhood names
    - Risk scores for specific hour
    - OSM polygon boundaries (when available)
    - Fallback to rectangular bounds
    """

    conn = await get_db_connection()

    try:
        # Build query based on parameters
        if lat is not None and lon is not None:
            # Spatial query within radius
            query = """
            SELECT
                id,
                grid_lat,
                grid_lon,
                latitude,
                longitude,
                bounds_north,
                bounds_south,
                bounds_east,
                bounds_west,
                location_name,
                neighborhood_name,
                hourly_predictions,
                historical_count,
                incident_types,
                avg_severity,
                computed_at,
                expires_at,
                boundary_geojson
            FROM predictions
            WHERE SQRT(POWER((latitude - $1) * 111.0, 2) + POWER((longitude - $2) * 111.0 * COS(RADIANS($1)), 2)) <= $3
            ORDER BY computed_at DESC
            LIMIT $4
            """

            results = await conn.fetch(query, lat, lon, radius_km, limit)
        else:
            # Get all predictions (fallback to latest until new ones generated)
            query = """
            SELECT
                id,
                grid_lat,
                grid_lon,
                latitude,
                longitude,
                bounds_north,
                bounds_south,
                bounds_east,
                bounds_west,
                location_name,
                neighborhood_name,
                hourly_predictions,
                historical_count,
                incident_types,
                avg_severity,
                computed_at,
                expires_at,
                boundary_geojson
            FROM predictions
            ORDER BY computed_at DESC
            LIMIT $1
            """

            results = await conn.fetch(query, limit)

        # Format predictions
        predictions = []

        for row in results:
            # Parse hourly predictions
            hourly_preds = json.loads(row['hourly_predictions']) if isinstance(row['hourly_predictions'], str) else row['hourly_predictions']

            # Get prediction for requested hour
            hour_key = str(hour_offset)
            hour_data = hourly_preds.get(hour_key, hourly_preds.get('0', {}))

            risk_score = hour_data.get('risk_score', 0.0)

            # Filter by minimum risk
            if risk_score < min_risk:
                continue

            # Parse boundary_geojson if present
            boundary_geojson = None
            if row['boundary_geojson']:
                if isinstance(row['boundary_geojson'], str):
                    boundary_geojson = json.loads(row['boundary_geojson'])
                else:
                    boundary_geojson = row['boundary_geojson']

            # Determine boundary source
            boundary_source = "osm" if boundary_geojson else "generated"

            prediction = {
                "id": str(row['id']),
                "name": row['neighborhood_name'],
                "city": row['location_name'],
                "lat": float(row['latitude']),
                "lng": float(row['longitude']),
                "intensity": risk_score,
                "risk_level": (
                    "High" if risk_score >= 0.7 else
                    "Medium-High" if risk_score >= 0.5 else
                    "Medium" if risk_score >= 0.3 else
                    "Low"
                ),
                "confidence": hour_data.get('confidence', 0.5),
                "estimated_incidents": hour_data.get('estimated_incidents', 0),
                "prediction": risk_score,
                "neighborhood": row['neighborhood_name'],
                "radius": 1000,  # Default 1km radius
                "bounds": {
                    "north": float(row['bounds_north']),
                    "south": float(row['bounds_south']),
                    "east": float(row['bounds_east']),
                    "west": float(row['bounds_west'])
                },
                "boundary": boundary_geojson,  # OSM GeoJSON polygon (optional)
                "boundary_source": boundary_source,
                "historical_count": row['historical_count'],
                "incident_types": row['incident_types'],
                "avg_severity": float(row['avg_severity']),
                "computed_at": row['computed_at'].isoformat(),
                "hour_offset": hour_offset
            }

            predictions.append(prediction)

        return {
            "predictions": predictions[:limit],
            "count": len(predictions[:limit]),
            "hour_offset": hour_offset,
            "timestamp": datetime.now().isoformat()
        }

    finally:
        await conn.close()


@router.get("/neighborhoods/stats")
async def get_neighborhood_stats():
    """Get statistics about neighborhood predictions"""

    conn = await get_db_connection()

    try:
        query = """
        SELECT
            COUNT(*) as total_predictions,
            COUNT(DISTINCT location_name) as total_cities,
            COUNT(DISTINCT neighborhood_name) as total_neighborhoods,
            COUNT(CASE WHEN boundary_geojson IS NOT NULL THEN 1 END) as with_boundaries,
            COUNT(CASE WHEN boundary_geojson->'geometry'->>'type' = 'Polygon' THEN 1 END) as with_polygons,
            COUNT(CASE WHEN boundary_geojson->'geometry'->>'type' = 'Point' THEN 1 END) as with_points,
            MIN(computed_at) as oldest_prediction,
            MAX(computed_at) as newest_prediction
        FROM predictions
        """

        result = await conn.fetchrow(query)

        return {
            "total_predictions": result['total_predictions'],
            "total_cities": result['total_cities'],
            "total_neighborhoods": result['total_neighborhoods'],
            "with_osm_boundaries": result['with_boundaries'],
            "with_polygon_boundaries": result['with_polygons'],
            "with_point_boundaries": result['with_points'],
            "coverage_percentage": round((result['with_boundaries'] / result['total_predictions'] * 100) if result['total_predictions'] > 0 else 0, 1),
            "oldest_prediction": result['oldest_prediction'].isoformat() if result['oldest_prediction'] else None,
            "newest_prediction": result['newest_prediction'].isoformat() if result['newest_prediction'] else None,
            "timestamp": datetime.now().isoformat()
        }

    finally:
        await conn.close()
