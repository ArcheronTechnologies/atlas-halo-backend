"""
Dynamic Risk Score Updates
Updates prediction risk scores when new incidents are reported
"""

import logging
from typing import Dict
from datetime import datetime, timedelta
import math


logger = logging.getLogger(__name__)


async def update_nearby_risk_scores(
    incident_location: Dict[str, float],
    incident_category: str,
    db_connection,
    radius_km: float = 2.0,
    risk_increase: float = 0.10
):
    """
    Update risk scores for predictions near a new incident

    Args:
        incident_location: {'latitude': float, 'longitude': float}
        incident_category: Polisen.se category
        db_connection: Database connection
        radius_km: Radius to update (default 2km)
        risk_increase: Base risk increase (default 10%)

    How it works:
    1. Find predictions within radius_km
    2. Calculate distance decay (closer = higher impact)
    3. Increase risk score by: base_increase * decay_factor
    4. Only update next 3 hours (immediate impact)
    """
    try:
        logger.info(f"Updating risk scores near {incident_location} (radius: {radius_km}km)")

        # Find predictions within radius and next 3 hours
        query = """
            SELECT
                id,
                risk_score,
                latitude,
                longitude,
                hours_ahead
            FROM predictions
            WHERE
                ST_DWithin(
                    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
                    ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                    $3
                )
                AND hours_ahead <= 3
        """

        predictions = await db_connection.fetch(
            query,
            incident_location['longitude'],
            incident_location['latitude'],
            radius_km * 1000  # Convert to meters
        )

        if not predictions:
            logger.info("No predictions found nearby")
            return

        logger.info(f"Found {len(predictions)} predictions to update")

        # Update each prediction
        updated_count = 0
        for pred in predictions:
            # Calculate distance for decay factor
            distance_km = calculate_distance(
                incident_location['latitude'],
                incident_location['longitude'],
                pred['latitude'],
                pred['longitude']
            )

            # Distance decay: 100% impact at 0km, 0% impact at radius_km
            decay_factor = 1.0 - (distance_km / radius_km)

            # Calculate new risk score
            current_risk = pred['risk_score']
            risk_delta = risk_increase * decay_factor

            new_risk = min(1.0, current_risk + risk_delta)

            # Update database
            update_query = """
                UPDATE predictions
                SET
                    risk_score = $1,
                    updated_at = NOW()
                WHERE id = $2
            """

            await db_connection.execute(update_query, new_risk, pred['id'])
            updated_count += 1

            logger.debug(
                f"Updated prediction {pred['id']}: "
                f"{current_risk:.2f} → {new_risk:.2f} "
                f"(+{risk_delta:.2f}, distance: {distance_km:.1f}km)"
            )

        logger.info(f"✅ Updated {updated_count} predictions")

    except Exception as e:
        logger.error(f"Failed to update risk scores: {e}", exc_info=True)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula

    Returns distance in kilometers
    """
    # Earth radius in km
    R = 6371.0

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance
