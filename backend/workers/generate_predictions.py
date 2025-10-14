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
import re
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Geocoding cache to avoid repeated API calls
GEOCODE_CACHE: Dict[Tuple[float, float], str] = {}

# OSM neighborhood boundaries cache
OSM_BOUNDARIES: Dict[str, Dict[str, Dict]] = {}

# Sweden bounding box (approximate)
SWEDEN_BOUNDS = {
    'min_lat': 55.0,
    'max_lat': 69.5,
    'min_lon': 10.5,
    'max_lon': 24.5
}


def load_osm_boundaries() -> Dict[str, Dict[str, Dict]]:
    """
    Load OSM neighborhood boundaries from constants file
    Returns: {"CityName": {"NeighborhoodName": {"lat": 59.0, "lon": 18.0, "type": "suburb"}}}
    """
    global OSM_BOUNDARIES

    if OSM_BOUNDARIES:
        return OSM_BOUNDARIES

    boundaries_file = Path(__file__).parent.parent / "constants" / "neighborhood_boundaries.json"

    if not boundaries_file.exists():
        logger.warning(f"OSM boundaries file not found: {boundaries_file}")
        return {}

    try:
        with open(boundaries_file, 'r', encoding='utf-8') as f:
            OSM_BOUNDARIES = json.load(f)
        logger.info(f"✅ Loaded OSM boundaries for {len(OSM_BOUNDARIES)} cities, {sum(len(v) for v in OSM_BOUNDARIES.values())} neighborhoods")
        return OSM_BOUNDARIES
    except Exception as e:
        logger.error(f"Failed to load OSM boundaries: {e}")
        return {}


def find_matching_osm_neighborhood(city: str, neighborhood: str, lat: float, lon: float) -> Optional[Dict]:
    """
    Find matching OSM neighborhood boundary for a given neighborhood name and location
    Returns boundary data with coordinates if found, None otherwise
    """
    boundaries = load_osm_boundaries()

    if not boundaries or city not in boundaries:
        return None

    city_neighborhoods = boundaries[city]

    # Exact match
    if neighborhood in city_neighborhoods:
        return city_neighborhoods[neighborhood]

    # Fuzzy match (case-insensitive, handle variations)
    neighborhood_lower = neighborhood.lower()
    for osm_name, osm_data in city_neighborhoods.items():
        osm_name_lower = osm_name.lower()

        # Check various match strategies:
        # 1. Exact lowercase match
        if osm_name_lower == neighborhood_lower:
            return osm_data

        # 2. One name contains the other
        if neighborhood_lower in osm_name_lower or osm_name_lower in neighborhood_lower:
            return osm_data

        # 3. Remove common suffixes/prefixes
        cleaned_neighborhood = neighborhood_lower.replace(' centrum', '').replace('området', '').strip()
        cleaned_osm = osm_name_lower.replace(' centrum', '').replace('området', '').strip()
        if cleaned_neighborhood == cleaned_osm:
            return osm_data

    # No match found - could fall back to nearest neighborhood by distance
    return None


def extract_neighborhood_from_summary(summary: str, location_name: str) -> Optional[str]:
    """
    Extract neighborhood name from Polisen.se incident summary text.

    Patterns used by Polisen:
    - "Description, Neighborhood." (comma before neighborhood)
    - "Description i Neighborhood" (after 'i')
    - "Description på Neighborhood" (after 'på')
    - "Description, Neighborhood, City" (middle comma-separated part)

    Returns neighborhood name or None if not found
    """
    if not summary:
        return None

    summary = summary.strip()

    # Blacklist of common words/phrases that aren't neighborhoods
    blacklist = {
        'brand', 'lägenhet', 'butik', 'matbutik', 'kiosk', 'bil', 'bilar',
        'stöld', 'ringa stöld', 'inbrott', 'olycka', 'bråk', 'sjukhus',
        'fastnat', 'ett hus', 'en bil', 'en lokal'
    }

    # Phrases that indicate this isn't a neighborhood name
    invalid_patterns = ['och ', 'indikerade', 'misstänkt', 'anledning']

    # Pattern 1: After "i " or "på " (most reliable, captures specific locations)
    # "Inbrott i lägenhet på Rosengård" → "Rosengård"
    # "Tre bilar i brand i Oxie" → "Oxie"
    # "Polis larmas till Värnhem" → "Värnhem"
    match = re.search(r'\b(i|på|till)\s+([A-ZÅÄÖ][a-zåäöA-ZÅÄÖ\s-]+?)(?:\.|,|$|\s+med\s+)', summary)
    if match:
        potential = match.group(2).strip()
        potential_lower = potential.lower()

        # Check if any invalid pattern appears in the extracted text
        has_invalid_pattern = any(pattern in potential_lower for pattern in invalid_patterns)

        # Filter out blacklisted words and common patterns
        if (len(potential) < 30 and
            potential != location_name and
            potential_lower not in blacklist and
            not potential_lower.startswith('stöld') and
            not potential_lower.startswith('inbrott') and
            not potential_lower.startswith('brand') and
            not has_invalid_pattern):
            return potential

    # Pattern 2: Text after last comma (fallback)
    # "Inbrott i lägenhet, Mariastaden."
    if ', ' in summary:
        parts = summary.split(', ')
        if len(parts) >= 2:
            potential_neighborhood = parts[-1].strip(' .').strip()
            potential_lower = potential_neighborhood.lower()

            # Only accept if it's not too long, doesn't match city, and not blacklisted
            if (len(potential_neighborhood) < 30 and
                potential_neighborhood != location_name and
                potential_lower not in blacklist):
                return potential_neighborhood

    return None

async def reverse_geocode(lat: float, lon: float, summary: str = "", location_name: str = "") -> Optional[str]:
    """
    Get neighborhood name, prioritizing summary extraction over geocoding.
    1. First tries to extract from Polisen.se summary text (fast, accurate)
    2. Falls back to reverse geocoding via Nominatim
    """
    # Try extracting from summary first (fast, no API calls)
    neighborhood_from_summary = extract_neighborhood_from_summary(summary, location_name)
    if neighborhood_from_summary:
        logger.info(f"Extracted neighborhood from summary: {neighborhood_from_summary}")
        return neighborhood_from_summary

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
                    (f"{location_name} Area" if location_name else None) or
                    f"Area_{abs(int(lat*100))}{abs(int(lon*100))}"
                )

                GEOCODE_CACHE[cache_key] = neighborhood_name

                # Rate limiting: 1 request per second for Nominatim
                await asyncio.sleep(1.1)

                return neighborhood_name

    except Exception as e:
        logger.warning(f"Geocoding failed for {lat},{lon}: {e}")

    # Fallback to location_name + "Area" or generic code
    fallback = (f"{location_name} Area" if location_name else None) or f"Area_{abs(int(lat*100))}{abs(int(lon*100))}"
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

    # Base risk from density (adjusted for smaller grid size)
    # With 0.03 degree grid (~3km), 5+ incidents = high risk
    base_risk = min(1.0, len(incidents) / 10)

    # Severity weight (higher impact)
    avg_severity = severity_sum / len(incidents) if incidents else 2
    severity_weight = min(1.0, avg_severity / 3.5)

    # Temporal pattern
    temporal_weight = 1.0
    if hour_counts:
        target_count = hour_counts.get(target_hour, 0)
        max_count = max(hour_counts.values())
        if max_count > 0:
            temporal_weight = 0.5 + (0.8 * target_count / max_count)

    # Combined score (increased weighting on severity for known high-crime areas)
    risk_score = base_risk * 0.3 + severity_weight * 0.4 + (temporal_weight * base_risk) * 0.3
    return min(1.0, max(0.0, risk_score))


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

    # Extract neighborhood names FIRST, then cluster by (grid + neighborhood)
    # This allows Rosengård, Värnhem, Oxie to be separate despite sharing GPS coordinates
    logger.info("Extracting neighborhoods from incident summaries...")
    for inc in all_incidents:
        neighborhood = extract_neighborhood_from_summary(inc.get("summary", ""), inc.get("location_name", ""))
        inc["extracted_neighborhood"] = neighborhood or ""  # Store for clustering

    # City + Neighborhood-aware grid-based clustering
    logger.info("Clustering incidents by grid + city + neighborhood...")
    neighborhood_clusters = defaultdict(list)

    for inc in all_incidents:
        grid_lat = round(inc["latitude"] / GRID_SIZE) * GRID_SIZE
        grid_lon = round(inc["longitude"] / GRID_SIZE) * GRID_SIZE
        # Cluster by grid + location (city) + neighborhood name
        # This ensures "Centrum" in Malmö is separate from "Centrum" in Stockholm
        cluster_key = (grid_lat, grid_lon, inc.get("location_name", ""), inc["extracted_neighborhood"])
        neighborhood_clusters[cluster_key].append(inc)

    logger.info(f"Created {len(neighborhood_clusters)} neighborhood clusters")

    # Generate predictions for each grid cell and all 24 hours
    current_time = datetime.now(timezone.utc)
    predictions_to_insert = []

    for cluster_key, cluster_incidents in neighborhood_clusters.items():
        # Include all cells with at least 1 incident for complete neighborhood coverage
        if len(cluster_incidents) < 1:
            continue

        grid_lat, grid_lon, location_name, extracted_neighborhood = cluster_key

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

        # Use extracted neighborhood if available, otherwise use grid-based fallback
        if extracted_neighborhood:
            neighborhood_name = extracted_neighborhood
            logger.info(f"Using extracted neighborhood: {grid_lat},{grid_lon} → {neighborhood_name}")
        else:
            # Fallback to grid-based name (skip slow geocoding)
            neighborhood_name = f"{location_name} Area" if location_name else f"Area_{abs(int(grid_lat*100))}{abs(int(grid_lon*100))}"
            logger.info(f"Using grid fallback: {grid_lat},{grid_lon} → {neighborhood_name}")

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
            # Try to find OSM boundary data for this neighborhood
            osm_boundary = find_matching_osm_neighborhood(location_name, neighborhood_name, grid_lat, grid_lon)

            # Assign unique coordinates to each neighborhood to avoid overlaps
            # Use a hash of neighborhood name to create consistent offsets
            neighborhood_hash = hash(f"{location_name}_{neighborhood_name}") % 100
            lat_offset = (neighborhood_hash % 10) * 0.01  # 0-0.09 degrees
            lon_offset = ((neighborhood_hash // 10) % 10) * 0.01  # 0-0.09 degrees

            # If we have OSM data, use its coordinates; otherwise use grid + offset
            if osm_boundary:
                display_lat = osm_boundary["lat"]
                display_lon = osm_boundary["lon"]
                logger.debug(f"Using OSM coordinates for {neighborhood_name}: {display_lat}, {display_lon}")
            else:
                display_lat = grid_lat + lat_offset
                display_lon = grid_lon + lon_offset

            # Recalculate bounds to center around display coordinates (pin at center of box)
            bounds_size = 0.01  # ~1km box around pin (reduced from 0.03 for better accuracy)
            offset_bounds = {
                "north": display_lat + bounds_size,
                "south": display_lat - bounds_size,
                "east": display_lon + bounds_size,
                "west": display_lon - bounds_size
            }

            # Create GeoJSON boundary if we have OSM data
            boundary_geojson = None
            if osm_boundary:
                # Check if we have polygon geometry data
                if "geometry" in osm_boundary:
                    # Use actual polygon boundary from OSM
                    boundary_geojson = {
                        "type": "Feature",
                        "properties": {
                            "name": neighborhood_name,
                            "type": osm_boundary.get("type", "neighborhood")
                        },
                        "geometry": osm_boundary["geometry"]
                    }
                    logger.debug(f"Using polygon boundary for {neighborhood_name}: {osm_boundary['geometry']['type']}")
                else:
                    # Fallback to point if no polygon available
                    boundary_geojson = {
                        "type": "Feature",
                        "properties": {
                            "name": neighborhood_name,
                            "type": osm_boundary.get("type", "neighborhood")
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [osm_boundary["lon"], osm_boundary["lat"]]
                        }
                    }

            predictions_to_insert.append({
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "latitude": display_lat,  # Pin coordinate
                "longitude": display_lon,  # Pin coordinate
                "bounds_north": offset_bounds["north"],  # Box centered on pin
                "bounds_south": offset_bounds["south"],
                "bounds_east": offset_bounds["east"],
                "bounds_west": offset_bounds["west"],
                "location_name": location_name or "Unknown",
                "neighborhood_name": neighborhood_name,
                "hourly_predictions": hourly_predictions,
                "historical_count": len(cluster_incidents),
                "incident_types": top_types,
                "avg_severity": round(avg_severity, 1),
                "computed_at": current_time,
                "expires_at": current_time + timedelta(hours=1),
                "boundary_geojson": boundary_geojson
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
                location_name, neighborhood_name, hourly_predictions,
                historical_count, incident_types, avg_severity,
                computed_at, expires_at, boundary_geojson
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            ON CONFLICT (grid_lat, grid_lon, location_name, neighborhood_name, computed_at) DO UPDATE SET
                hourly_predictions = EXCLUDED.hourly_predictions,
                expires_at = EXCLUDED.expires_at,
                boundary_geojson = EXCLUDED.boundary_geojson
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
                pred["location_name"],
                pred["neighborhood_name"],
                json.dumps(pred["hourly_predictions"]),
                pred["historical_count"],
                pred["incident_types"],
                pred["avg_severity"],
                pred["computed_at"],
                pred["expires_at"],
                json.dumps(pred["boundary_geojson"]) if pred["boundary_geojson"] else None
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
