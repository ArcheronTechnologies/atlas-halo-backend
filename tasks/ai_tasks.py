"""
Celery Tasks for AI Analysis
Async processing of photos, videos, and audio
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .celery_app import celery_app
from ..ai_processing.photo_analyzer import PhotoAnalyzer
from ..ai_processing.video_analyzer import VideoAnalyzer
from ..ai_processing.audio_analyzer import AudioAnalyzer
from ..database.postgis_database import PostGISDatabase, DatabaseConfig

logger = logging.getLogger(__name__)

# Initialize analyzers (singleton pattern)
photo_analyzer = PhotoAnalyzer()
video_analyzer = VideoAnalyzer()
audio_analyzer = AudioAnalyzer()


@celery_app.task(name='analyze_photo', bind=True, max_retries=3)
def analyze_photo_task(self, media_id: str, file_path: str) -> Dict[str, Any]:
    """
    Async photo analysis task

    Args:
        media_id: Database ID of media file
        file_path: Path to photo file

    Returns:
        AI analysis results
    """
    try:
        logger.info(f"Starting photo analysis for media_id={media_id}")

        # Run AI analysis
        result = photo_analyzer.analyze(file_path)

        # Update database with results
        # (This would normally use async, but Celery tasks are sync)
        # db = get_sync_database()
        # db.update_media_analysis(media_id, result)

        logger.info(f"Photo analysis complete for media_id={media_id}")
        return {
            'media_id': media_id,
            'status': 'success',
            'analysis': result,
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Photo analysis failed for media_id={media_id}: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@celery_app.task(name='analyze_video', bind=True, max_retries=3)
def analyze_video_task(self, media_id: str, file_path: str) -> Dict[str, Any]:
    """
    Async video analysis task (keyframe extraction + object detection)

    Args:
        media_id: Database ID of media file
        file_path: Path to video file

    Returns:
        AI analysis results
    """
    try:
        logger.info(f"Starting video analysis for media_id={media_id}")

        # Run AI analysis (2-5s for 60s video)
        result = video_analyzer.analyze(file_path)

        logger.info(f"Video analysis complete for media_id={media_id}")
        return {
            'media_id': media_id,
            'status': 'success',
            'analysis': result,
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Video analysis failed for media_id={media_id}: {e}")
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@celery_app.task(name='analyze_audio', bind=True, max_retries=3)
def analyze_audio_task(self, media_id: str, file_path: str) -> Dict[str, Any]:
    """
    Async audio analysis task (Whisper speech-to-text + threat detection)

    Args:
        media_id: Database ID of media file
        file_path: Path to audio file

    Returns:
        AI analysis results
    """
    try:
        logger.info(f"Starting audio analysis for media_id={media_id}")

        # Run AI analysis (5-15s for 60s audio)
        result = audio_analyzer.analyze(file_path)

        logger.info(f"Audio analysis complete for media_id={media_id}")
        return {
            'media_id': media_id,
            'status': 'success',
            'analysis': result,
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Audio analysis failed for media_id={media_id}: {e}")
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@celery_app.task(name='analyze_incident_bundle', bind=True)
def analyze_incident_bundle_task(
    self,
    incident_id: str,
    photo_ids: list[str] = None,
    video_ids: list[str] = None,
    audio_ids: list[str] = None
) -> Dict[str, Any]:
    """
    Analyze complete incident with all media

    Args:
        incident_id: Database ID of incident
        photo_ids: List of photo media IDs
        video_ids: List of video media IDs
        audio_ids: List of audio media IDs

    Returns:
        Combined analysis results
    """
    try:
        logger.info(f"Analyzing incident bundle for incident_id={incident_id}")

        results = {
            'incident_id': incident_id,
            'photos': [],
            'videos': [],
            'audio': [],
            'status': 'success'
        }

        # Queue individual analysis tasks
        if photo_ids:
            for photo_id in photo_ids:
                # Would fetch file path from DB
                # task = analyze_photo_task.delay(photo_id, file_path)
                # results['photos'].append(task.id)
                pass

        if video_ids:
            for video_id in video_ids:
                # task = analyze_video_task.delay(video_id, file_path)
                # results['videos'].append(task.id)
                pass

        if audio_ids:
            for audio_id in audio_ids:
                # task = analyze_audio_task.delay(audio_id, file_path)
                # results['audio'].append(task.id)
                pass

        return results

    except Exception as e:
        logger.error(f"Incident bundle analysis failed for incident_id={incident_id}: {e}")
        raise


@celery_app.task(name='compute_temporal_risk_scores', bind=True)
def compute_temporal_risk_scores_task(self) -> Dict[str, Any]:
    """
    Daily task to recompute temporal risk scores based on rolling 90-day historical data

    This task:
    1. Analyzes last 90 days of crime incidents
    2. Computes hourly risk scores per neighborhood
    3. Caches results for fast API serving
    4. Runs automatically at 02:00 UTC daily

    Returns:
        Task completion status and statistics
    """
    try:
        logger.info("🔄 Starting daily temporal risk computation...")

        import asyncio
        from datetime import timedelta
        from collections import defaultdict
        import math
        import json

        # Get database connection (async wrapper)
        async def compute_risk_scores():
            from ..database.postgis_database import get_database

            db = await get_database()
            ninety_days_ago = datetime.now() - timedelta(days=90)

            # Query historical incidents
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
                logger.warning("⚠️ No historical data found for temporal risk computation")
                return None

            # Load neighborhood definitions
            try:
                import sys
                from pathlib import Path
                mobile_path = Path(__file__).parent.parent.parent / "mobile"
                sys.path.insert(0, str(mobile_path))
                from constants.data.swedishLocations import NEIGHBORHOODS
                sys.path.remove(str(mobile_path))
            except Exception as e:
                logger.warning(f"Could not load neighborhood data: {e}")
                NEIGHBORHOODS = []

            # Helper: find nearest neighborhood
            def find_nearest_neighborhood(lat: float, lng: float):
                if not NEIGHBORHOODS:
                    return None

                min_distance = float('inf')
                nearest = None

                for neighborhood in NEIGHBORHOODS:
                    n_lat = neighborhood.get('latitude')
                    n_lng = neighborhood.get('longitude')

                    if n_lat is None or n_lng is None:
                        continue

                    lat_diff = (lat - n_lat) * 111.0
                    lng_diff = (lng - n_lng) * 111.0 * math.cos(math.radians(lat))
                    distance = math.sqrt(lat_diff**2 + lng_diff**2)

                    radius_km = neighborhood.get('estimated_radius_km', 2.0)

                    if distance < radius_km and distance < min_distance:
                        min_distance = distance
                        nearest = neighborhood

                return nearest

            # Count incidents by neighborhood and hour
            neighborhood_hour_counts = defaultdict(lambda: defaultdict(float))
            neighborhood_total_counts = defaultdict(float)
            neighborhood_names = {}

            for row in results:
                lat = float(row['latitude'])
                lng = float(row['longitude'])
                hour = row['occurred_at'].hour
                severity = row['severity'] or 3
                weight = severity / 5.0

                neighborhood = find_nearest_neighborhood(lat, lng)

                if neighborhood:
                    neighborhood_key = f"{neighborhood['name']}, {neighborhood['city']}"
                    neighborhood_names[neighborhood_key] = {
                        'name': neighborhood['name'],
                        'city': neighborhood['city'],
                        'latitude': neighborhood['latitude'],
                        'longitude': neighborhood['longitude']
                    }
                else:
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

            # Normalize to risk scores
            temporal_risk_data = {
                "neighborhoods": {},
                "computed_at": datetime.now().isoformat()
            }

            global_max = 0
            for neighborhood_key in neighborhood_hour_counts:
                for hour in range(24):
                    count = neighborhood_hour_counts[neighborhood_key][hour]
                    global_max = max(global_max, count)

            for neighborhood_key in sorted(neighborhood_hour_counts.keys())[:100]:
                hourly_risk = []

                for hour in range(24):
                    count = neighborhood_hour_counts[neighborhood_key][hour]

                    if global_max > 0:
                        normalized_risk = count / global_max
                        smoothed_risk = 0.1 + (normalized_risk ** 0.7) * 0.8
                    else:
                        smoothed_risk = 0.5

                    hourly_risk.append(round(smoothed_risk, 3))

                neighborhood_info = neighborhood_names.get(neighborhood_key, {'name': neighborhood_key, 'city': 'Unknown'})

                temporal_risk_data["neighborhoods"][neighborhood_key] = {
                    "name": neighborhood_info['name'],
                    "city": neighborhood_info['city'],
                    "hourly_risk": hourly_risk,
                    "total_incidents": int(neighborhood_total_counts[neighborhood_key])
                }

            # Cache results to file for fast API serving
            from pathlib import Path
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir / "temporal_risk_data.json"
            with open(cache_file, 'w') as f:
                json.dump(temporal_risk_data, f, indent=2)

            logger.info(f"💾 Cached temporal risk data to {cache_file}")

            return temporal_risk_data

        # Run async computation
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(compute_risk_scores())

        if result:
            neighborhood_count = len(result.get('neighborhoods', {}))
            logger.info(f"✅ Temporal risk computation complete: {neighborhood_count} neighborhoods")

            return {
                'status': 'success',
                'neighborhoods_analyzed': neighborhood_count,
                'computed_at': result['computed_at'],
                'task_completed_at': datetime.utcnow().isoformat()
            }
        else:
            logger.warning("⚠️ Temporal risk computation returned no data")
            return {
                'status': 'no_data',
                'message': 'No historical incidents found'
            }

    except Exception as e:
        logger.error(f"❌ Temporal risk computation failed: {e}")
        raise self.retry(exc=e, countdown=3600)  # Retry in 1 hour
