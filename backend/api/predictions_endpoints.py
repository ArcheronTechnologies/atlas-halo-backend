"""
Predictions API Endpoints
Serves ML-generated crime predictions to mobile app
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from datetime import datetime
from ..database.postgis_database import get_database
from ..services.prediction_scheduler import get_prediction_scheduler
import os
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.get("/hotspots")
async def get_hotspot_predictions(
    lat: Optional[float] = Query(None, description="Center latitude"),
    lon: Optional[float] = Query(None, description="Center longitude"),
    radius_km: Optional[float] = Query(None, description="Radius in kilometers"),
    min_risk: Optional[float] = Query(0.0, description="Minimum risk score (0-1)"),
    limit: int = Query(100, description="Maximum predictions to return")
):
    """
    Get ML-generated hotspot predictions

    - If lat/lon/radius provided: Returns predictions within radius
    - Otherwise: Returns all active predictions
    """

    db = await get_database()

    if lat is not None and lon is not None and radius_km is not None:
        # Get predictions within radius
        query = """
        SELECT
            id,
            latitude,
            longitude,
            risk_score,
            prediction_time,
            valid_until,
            model_version,
            metadata
        FROM hotspot_predictions
        WHERE valid_until > NOW()
        AND risk_score >= $1
        AND ST_DWithin(
            location::geography,
            ST_MakePoint($2, $3)::geography,
            $4
        )
        ORDER BY risk_score DESC
        LIMIT $5
        """

        results = await db.pool.fetch(
            query,
            min_risk,
            lon,
            lat,
            radius_km * 1000,  # Convert km to meters
            limit
        )
    else:
        # Get all active predictions
        query = """
        SELECT
            id,
            latitude,
            longitude,
            risk_score,
            prediction_time,
            valid_until,
            model_version,
            metadata
        FROM hotspot_predictions
        WHERE valid_until > NOW()
        AND risk_score >= $1
        ORDER BY risk_score DESC
        LIMIT $2
        """

        results = await db.pool.fetch(query, min_risk, limit)

    # Format response
    predictions = []
    for row in results:
        # Parse metadata if it's a JSON string
        metadata = row['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        predictions.append({
            "id": str(row['id']),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "risk_score": float(row['risk_score']),
            "prediction_time": row['prediction_time'].isoformat(),
            "valid_until": row['valid_until'].isoformat(),
            "model_version": row['model_version'],
            "metadata": metadata or {}
        })

    return {
        "predictions": predictions,
        "count": len(predictions),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/hotspots/stats")
async def get_prediction_stats():
    """Get statistics about current predictions"""

    db = await get_database()

    query = """
    SELECT
        COUNT(*) as total_predictions,
        AVG(risk_score) as avg_risk,
        MIN(risk_score) as min_risk,
        MAX(risk_score) as max_risk,
        COUNT(DISTINCT model_version) as model_versions,
        MIN(prediction_time) as oldest_prediction,
        MAX(valid_until) as newest_expiry
    FROM hotspot_predictions
    WHERE valid_until > NOW()
    """

    result = await db.pool.fetchrow(query)

    if not result:
        return {
            "total_predictions": 0,
            "avg_risk": 0,
            "min_risk": 0,
            "max_risk": 0
        }

    return {
        "total_predictions": result['total_predictions'],
        "avg_risk": float(result['avg_risk']) if result['avg_risk'] else 0,
        "min_risk": float(result['min_risk']) if result['min_risk'] else 0,
        "max_risk": float(result['max_risk']) if result['max_risk'] else 0,
        "model_versions": result['model_versions'],
        "oldest_prediction": result['oldest_prediction'].isoformat() if result['oldest_prediction'] else None,
        "newest_expiry": result['newest_expiry'].isoformat() if result['newest_expiry'] else None,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/hotspots/by-city/{city_name}")
async def get_predictions_by_city(
    city_name: str,
    min_risk: float = Query(0.0, description="Minimum risk score")
):
    """Get all predictions for a specific city"""

    db = await get_database()

    query = """
    SELECT
        id,
        latitude,
        longitude,
        risk_score,
        prediction_time,
        valid_until,
        model_version,
        metadata
    FROM hotspot_predictions
    WHERE valid_until > NOW()
    AND risk_score >= $1
    AND metadata->>'city' = $2
    ORDER BY risk_score DESC
    """

    results = await db.pool.fetch(query, min_risk, city_name)

    predictions = []
    for row in results:
        # Parse metadata if it's a JSON string
        metadata = row['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        predictions.append({
            "id": str(row['id']),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "risk_score": float(row['risk_score']),
            "prediction_time": row['prediction_time'].isoformat(),
            "valid_until": row['valid_until'].isoformat(),
            "model_version": row['model_version'],
            "metadata": metadata or {},
            "neighborhood": metadata.get('neighborhood') if metadata else None
        })

    return {
        "city": city_name,
        "predictions": predictions,
        "count": len(predictions),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/validation/accuracy")
async def get_validation_accuracy(days: int = Query(7, description="Number of days to analyze")):
    """
    Get prediction accuracy metrics for the last N days
    Shows how well predictions performed against actual incidents
    """
    from ..ml_training.prediction_validator import PredictionValidator

    db = await get_database()
    validator = PredictionValidator(db)

    metrics = await validator.get_accuracy_metrics(days)

    return metrics


@router.get("/validation/adjustments")
async def get_location_adjustments():
    """
    Get current location-specific adjustment factors
    Shows how predictions are being refined based on validation feedback
    """
    from ..ml_training.prediction_refiner import PredictionRefiner

    db = await get_database()
    refiner = PredictionRefiner(db)
    await refiner.load_adjustments()

    summary = await refiner.get_adjustment_summary()

    return summary


@router.post("/validation/run")
async def trigger_validation():
    """
    Manually trigger validation and refinement process
    This normally runs automatically at 3 AM daily
    """
    db = await get_database()
    model_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'models',
        'production',
        'crime_prediction_model.joblib'
    )

    try:
        scheduler = await get_prediction_scheduler(db, model_path)
        await scheduler.run_now()

        return {
            "status": "success",
            "message": "Validation and refinement completed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/temporal-risk")
async def get_temporal_risk_data():
    """
    Get hourly crime risk scores for all neighborhoods

    Returns ML-computed risk scores (0-1) for each neighborhood × hour combination.
    This data powers the mobile app's temporal slider feature.

    Recomputed daily at 02:00 UTC via Celery Beat scheduler.
    This endpoint serves pre-computed cached data for fast response times.
    """

    from pathlib import Path

    # Try to serve from cache first (fast path)
    cache_file = Path(__file__).parent.parent.parent / "data" / "cache" / "temporal_risk_data.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if cache is recent (less than 25 hours old)
            from datetime import timedelta
            computed_at = datetime.fromisoformat(cached_data['computed_at'])
            age = datetime.now() - computed_at

            if age < timedelta(hours=25):
                logger.info(f"✅ Serving cached temporal risk data (age: {age.seconds // 3600}h)")
                return cached_data
            else:
                logger.warning(f"⚠️ Cached data is stale (age: {age.days}d {age.seconds // 3600}h)")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")

    # Cache miss or stale - compute on-demand (fallback)
    logger.info("🔄 Computing temporal risk data on-demand...")

    from ..database.postgis_database import get_database
    from datetime import timedelta
    from collections import defaultdict
    import math

    db = await get_database()

    # Get historical incidents from last 90 days
    ninety_days_ago = datetime.now() - timedelta(days=90)

    query = """
    SELECT
        incident_type,
        occurred_at,
        latitude,
        longitude,
        severity,
        metadata
    FROM crime_incidents
    WHERE occurred_at >= $1
    AND latitude IS NOT NULL
    AND longitude IS NOT NULL
    ORDER BY occurred_at DESC
    LIMIT 10000
    """

    results = await db.pool.fetch(query, ninety_days_ago)

    if not results:
        return {
            "neighborhoods": {},
            "computed_at": datetime.now().isoformat(),
            "message": "No historical data available"
        }

    # Load neighborhood definitions from mobile constants
    # This ensures we use the exact same neighborhoods as the mobile app
    try:
        import sys
        from pathlib import Path
        mobile_path = Path(__file__).parent.parent.parent / "mobile"
        sys.path.insert(0, str(mobile_path))
        from constants.data.swedishLocations import NEIGHBORHOODS
        sys.path.remove(str(mobile_path))
    except Exception as e:
        # Fallback: use city-level grouping if neighborhood data unavailable
        logger.warning(f"Could not load neighborhood data: {e}")
        NEIGHBORHOODS = []

    # Helper function to find nearest neighborhood for a lat/lng
    def find_nearest_neighborhood(lat: float, lng: float) -> Optional[dict]:
        if not NEIGHBORHOODS:
            return None

        min_distance = float('inf')
        nearest = None

        for neighborhood in NEIGHBORHOODS:
            n_lat = neighborhood.get('latitude')
            n_lng = neighborhood.get('longitude')

            if n_lat is None or n_lng is None:
                continue

            # Haversine distance approximation (good enough for small areas)
            lat_diff = (lat - n_lat) * 111.0  # km per degree latitude
            lng_diff = (lng - n_lng) * 111.0 * math.cos(math.radians(lat))  # km per degree longitude
            distance = math.sqrt(lat_diff**2 + lng_diff**2)

            # Check if within neighborhood radius
            radius_km = neighborhood.get('estimated_radius_km', 2.0)

            if distance < radius_km and distance < min_distance:
                min_distance = distance
                nearest = neighborhood

        return nearest

    # Count incidents by neighborhood and hour
    neighborhood_hour_counts = defaultdict(lambda: defaultdict(float))
    neighborhood_total_counts = defaultdict(float)
    neighborhood_names = {}  # Map neighborhood keys to display names

    for row in results:
        lat = float(row['latitude'])
        lng = float(row['longitude'])
        hour = row['occurred_at'].hour
        severity = row['severity'] or 3

        # Weight by severity (1-5 scale)
        weight = severity / 5.0

        # Find nearest neighborhood
        neighborhood = find_nearest_neighborhood(lat, lng)

        if neighborhood:
            # Use neighborhood name + city as unique key
            neighborhood_key = f"{neighborhood['name']}, {neighborhood['city']}"
            neighborhood_names[neighborhood_key] = {
                'name': neighborhood['name'],
                'city': neighborhood['city'],
                'latitude': neighborhood['latitude'],
                'longitude': neighborhood['longitude']
            }
        else:
            # Fallback to city from metadata
            metadata = row['metadata'] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            city = metadata.get('city') or 'Unknown'
            neighborhood_key = city
            neighborhood_names[neighborhood_key] = {
                'name': city,
                'city': city,
                'latitude': lat,
                'longitude': lng
            }

        neighborhood_hour_counts[neighborhood_key][hour] += weight
        neighborhood_total_counts[neighborhood_key] += weight

    # Normalize to risk scores (0-1)
    temporal_risk_data = {
        "neighborhoods": {},
        "computed_at": datetime.now().isoformat()
    }

    # Find global max for normalization
    global_max = 0
    for neighborhood_key in neighborhood_hour_counts:
        for hour in range(24):
            count = neighborhood_hour_counts[neighborhood_key][hour]
            global_max = max(global_max, count)

    # Build output structure
    for neighborhood_key in sorted(neighborhood_hour_counts.keys())[:100]:  # Top 100 neighborhoods
        hourly_risk = []

        for hour in range(24):
            count = neighborhood_hour_counts[neighborhood_key][hour]

            # Normalize to 0-1 scale
            if global_max > 0:
                normalized_risk = count / global_max
                # Apply smoothing to spread values (0.1-0.9 range)
                # Use exponential scaling to emphasize differences
                smoothed_risk = 0.1 + (normalized_risk ** 0.7) * 0.8
            else:
                smoothed_risk = 0.5  # Default neutral

            hourly_risk.append(round(smoothed_risk, 3))

        neighborhood_info = neighborhood_names.get(neighborhood_key, {'name': neighborhood_key, 'city': 'Unknown'})

        temporal_risk_data["neighborhoods"][neighborhood_key] = {
            "name": neighborhood_info['name'],
            "city": neighborhood_info['city'],
            "hourly_risk": hourly_risk,
            "total_incidents": int(neighborhood_total_counts[neighborhood_key])
        }

    # Save to cache for future requests
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(temporal_risk_data, f, indent=2)
        logger.info(f"💾 Cached on-demand temporal risk computation to {cache_file}")
    except Exception as e:
        logger.error(f"❌ Failed to cache temporal risk data: {e}")

    return temporal_risk_data