"""
Background worker to pre-compute crime predictions for all of Sweden
Runs hourly to generate predictions on a fixed 0.03° grid (~3.3km cells)

This makes the API scalable by:
- Computing once, serving many times
- Fast lookups instead of expensive calculations
- Handles thousands of concurrent users
"""

import asyncio
import asyncpg
import os
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math
import httpx
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Geocoding cache to avoid repeated API calls
GEOCODE_CACHE: Dict[Tuple[float, float], str] = {}

# Sweden bounding box (approximate)
SWEDEN_BOUNDS = {
    'min_lat': 55.0,
    'max_lat': 69.5,
    'min_lon': 10.5,
    'max_lon': 24.5
}

async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Reverse geocode lat/lon to neighborhood name using Nominatim (OpenStreetMap)
    Returns suburb/neighbourhood if available, otherwise city/town
    """
    # Round to 2 decimals for caching (~1km precision)
    cache_key = (round(lat, 2), round(lon, 2))

    if cache_key in GEOCODE_CACHE:
        return GEOCODE_CACHE[cache_key]

    try:
        async with httpx.AsyncClient() as client:
            # Nominatim API with Swedish preference
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "format": "json",
                "lat": lat,
                "lon": lon,
                "zoom": 16,  # Neighborhood level
                "accept-language": "sv,en"  # Swedish first, English fallback
            }
            headers = {
                "User-Agent": "Halo-Crime-Prediction-App/1.0"
            }

            response = await client.get(url, params=params, headers=headers, timeout=5.0)

            if response.status_code == 200:
                data = response.json()
                address = data.get("address", {})

                # Priority: suburb > neighbourhood > quarter > city_district > town > city
                neighborhood_name = (
                    address.get("suburb") or
                    address.get("neighbourhood") or
                    address.get("quarter") or
                    address.get("city_district") or
                    address.get("town") or
                    address.get("city") or
                    address.get("municipality") or
                    f"Area_{abs(int(lat*100))}{abs(int(lon*100))}"
                )

                GEOCODE_CACHE[cache_key] = neighborhood_name

                # Rate limiting: 1 request per second for Nominatim
                await asyncio.sleep(1.1)

                return neighborhood_name

    except Exception as e:
        logger.warning(f"Geocoding failed for {lat},{lon}: {e}")

    # Fallback to generic name
    fallback = f"Area_{abs(int(lat*100))}{abs(int(lon*100))}"
    GEOCODE_CACHE[cache_key] = fallback
    return fallback

GRID_SIZE = 0.03  # ~3.3km grid cells


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km using Haversine formula"""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def calculate_risk_score(incidents: List[Dict], target_hour: int) -> float:
    """Calculate risk score with temporal patterns"""
    if not incidents:
        return 0.0

    hour_counts = defaultdict(int)
    severity_sum = 0

    for inc in incidents:
        try:
            occurred_at = datetime.fromisoformat(inc["occurred_at"].replace('Z', '+00:00'))
            hour_counts[occurred_at.hour] += 1
            severity_sum += inc.get("severity", 2)
        except:
            continue

    # Base risk from density
    base_risk = min(1.0, len(incidents) / 50)

    # Severity weight
    avg_severity = severity_sum / len(incidents) if incidents else 2
    severity_weight = avg_severity / 5

    # Temporal pattern
    temporal_weight = 1.0
    if hour_counts:
        target_count = hour_counts.get(target_hour, 0)
        max_count = max(hour_counts.values())
        if max_count > 0:
            temporal_weight = 0.5 + (0.8 * target_count / max_count)

    # Combined score
    risk_score = base_risk * 0.4 + severity_weight * 0.3 + (temporal_weight * base_risk) * 0.3
    return min(1.0, risk_score)


async def generate_country_predictions(conn: asyncpg.Connection):
    """Generate predictions for entire country on a fixed grid"""
    logger.info("Starting country-wide prediction generation...")

    # Fetch all recent incidents (last 30 days)
    query = """
        SELECT id, incident_type, summary, latitude, longitude,
               occurred_at, severity, source, external_id, location_name
        FROM incidents
        WHERE occurred_at >= NOW() - INTERVAL '30 days'
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
        ORDER BY occurred_at DESC
    """

    results = await conn.fetch(query)
    logger.info(f"Loaded {len(results)} incidents from database")

    # Convert to dict format
    all_incidents = []
    for row in results:
        all_incidents.append({
            "id": str(row['id']),
            "incident_type": row.get('incident_type', 'Unknown'),
            "summary": row.get('summary', ''),
            "latitude": float(row['latitude']),
            "longitude": float(row['longitude']),
            "occurred_at": row['occurred_at'].isoformat() if row.get('occurred_at') else None,
            "severity": int(row.get('severity', 2)),
            "source": row.get('source', 'polisen'),
            "external_id": row.get('external_id'),
            "location_name": row.get('location_name', '')
        })

    # Grid-based clustering
    logger.info("Clustering incidents into grid cells...")
    neighborhood_clusters = defaultdict(list)

    for inc in all_incidents:
        grid_lat = round(inc["latitude"] / GRID_SIZE) * GRID_SIZE
        grid_lon = round(inc["longitude"] / GRID_SIZE) * GRID_SIZE
        cluster_key = (grid_lat, grid_lon)
        neighborhood_clusters[cluster_key].append(inc)

    logger.info(f"Created {len(neighborhood_clusters)} grid cells with incidents")

    # Generate predictions for each grid cell and all 24 hours
    current_time = datetime.now(timezone.utc)
    predictions_to_insert = []

    for cluster_key, cluster_incidents in neighborhood_clusters.items():
        # Include all cells with at least 1 incident for complete neighborhood coverage
        if len(cluster_incidents) < 1:
            continue

        grid_lat, grid_lon = cluster_key

        # Calculate center (weighted by severity)
        total_weight = sum(inc.get("severity", 2) for inc in cluster_incidents)
        weighted_lat = sum(inc["latitude"] * inc.get("severity", 2) for inc in cluster_incidents) / total_weight
        weighted_lon = sum(inc["longitude"] * inc.get("severity", 2) for inc in cluster_incidents) / total_weight

        # Calculate bounds
        lats = [inc["latitude"] for inc in cluster_incidents]
        lons = [inc["longitude"] for inc in cluster_incidents]
        bounds = {
            "north": max(lats) + 0.015,
            "south": min(lats) - 0.015,
            "east": max(lons) + 0.015,
            "west": min(lons) - 0.015
        }

        # Get neighborhood name via reverse geocoding
        neighborhood_name = await reverse_geocode(weighted_lat, weighted_lon)
        logger.info(f"Geocoded {grid_lat},{grid_lon} → {neighborhood_name}")

        # Get top incident types
        incident_types = defaultdict(int)
        for inc in cluster_incidents:
            incident_types[inc["incident_type"]] += 1
        top_types = [t[0] for t in sorted(incident_types.items(), key=lambda x: x[1], reverse=True)[:3]]

        # Calculate avg severity
        avg_severity = sum(inc.get("severity", 2) for inc in cluster_incidents) / len(cluster_incidents)

        # Calculate hourly predictions (0-23 hours ahead)
        hourly_predictions = {}
        for hour_offset in range(24):
            target_hour = (current_time.hour + hour_offset) % 24
            risk_score = calculate_risk_score(cluster_incidents, target_hour)

            # Calculate confidence
            recency_scores = []
            for inc in cluster_incidents:
                try:
                    occurred_at = datetime.fromisoformat(inc["occurred_at"].replace('Z', '+00:00'))
                    days_ago = (current_time - occurred_at).days
                    recency_score = max(0, 1 - (days_ago / 30))
                    recency_scores.append(recency_score)
                except:
                    recency_scores.append(0.5)

            avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
            data_quality = min(1.0, len(cluster_incidents) / 20)
            confidence = (avg_recency * 0.6 + data_quality * 0.4)

            hourly_predictions[str(hour_offset)] = {
                "risk_score": round(risk_score, 3),
                "confidence": round(confidence, 2),
                "estimated_incidents": round(risk_score * 8)
            }

        # Only store if at least one hour has predictions
        if hourly_predictions:
            predictions_to_insert.append({
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "latitude": weighted_lat,
                "longitude": weighted_lon,
                "bounds_north": bounds["north"],
                "bounds_south": bounds["south"],
                "bounds_east": bounds["east"],
                "bounds_west": bounds["west"],
                "neighborhood_name": neighborhood_name,
                "hourly_predictions": hourly_predictions,
                "historical_count": len(cluster_incidents),
                "incident_types": top_types,
                "avg_severity": round(avg_severity, 1),
                "computed_at": current_time,
                "expires_at": current_time + timedelta(hours=1)
            })

    logger.info(f"Generated {len(predictions_to_insert)} prediction cells")

    # Insert into database (upsert to handle duplicates)
    if predictions_to_insert:
        logger.info("Inserting predictions into database...")

        # Clear old predictions
        await conn.execute("DELETE FROM predictions WHERE expires_at < NOW()")

        # Insert new predictions
        insert_query = """
            INSERT INTO predictions (
                grid_lat, grid_lon, latitude, longitude,
                bounds_north, bounds_south, bounds_east, bounds_west,
                neighborhood_name, hourly_predictions,
                historical_count, incident_types, avg_severity,
                computed_at, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (grid_lat, grid_lon, computed_at) DO UPDATE SET
                hourly_predictions = EXCLUDED.hourly_predictions,
                expires_at = EXCLUDED.expires_at
        """

        import json
        for pred in predictions_to_insert:
            await conn.execute(
                insert_query,
                pred["grid_lat"],
                pred["grid_lon"],
                pred["latitude"],
                pred["longitude"],
                pred["bounds_north"],
                pred["bounds_south"],
                pred["bounds_east"],
                pred["bounds_west"],
                pred["neighborhood_name"],
                json.dumps(pred["hourly_predictions"]),
                pred["historical_count"],
                pred["incident_types"],
                pred["avg_severity"],
                pred["computed_at"],
                pred["expires_at"]
            )

        logger.info(f"✅ Successfully inserted {len(predictions_to_insert)} predictions")


async def main():
    """Main worker function"""
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
        database=os.getenv('POSTGRES_DB', 'postgres')
    )

    try:
        # Generate predictions
        await generate_country_predictions(conn)

        logger.info("✅ Prediction generation complete")

    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}", exc_info=True)
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
