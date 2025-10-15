"""
Sensor Fusion Service - Multi-User Incident Correlation
Detects when multiple users report the same incident and merges them
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from backend.services.atlas_client import get_atlas_client

logger = logging.getLogger(__name__)


class SensorFusionService:
    """
    Correlates incidents from multiple users based on:
    1. Location proximity (50m radius)
    2. Time window (5 minute window)
    3. Category match
    4. Visual/audio similarity (via Atlas Intelligence)
    """

    def __init__(self):
        self.atlas_client = get_atlas_client()
        self.location_threshold_meters = 50
        self.time_threshold_minutes = 5
        self.similarity_threshold = 0.7  # 70% visual similarity
        logger.info("SensorFusionService initialized")

    async def correlate_incident(
        self,
        new_incident: Dict,
        db_connection
    ) -> Optional[Dict]:
        """
        Find and merge matching incidents

        Args:
            new_incident: {
                'latitude': float,
                'longitude': float,
                'occurred_at': datetime,
                'polisen_category': str,
                'video_id': str (optional),
                'user_id': str
            }
            db_connection: Database connection

        Returns:
            Merged incident if match found, None otherwise
        """
        try:
            # Find candidate incidents within location + time window
            candidates = await self._find_candidates(new_incident, db_connection)

            if not candidates:
                logger.info("No correlation candidates found")
                return None

            logger.info(f"Found {len(candidates)} potential correlations")

            # Check each candidate for similarity
            for candidate in candidates:
                is_match = await self._check_similarity(new_incident, candidate)

                if is_match:
                    logger.info(f"Correlation match found! Incident {candidate['id']}")
                    merged = await self._merge_incidents(
                        new_incident,
                        candidate,
                        db_connection
                    )
                    return merged

            logger.info("No matching correlations found")
            return None

        except Exception as e:
            logger.error(f"Correlation failed: {e}", exc_info=True)
            return None

    async def _find_candidates(
        self,
        new_incident: Dict,
        db_connection
    ) -> List[Dict]:
        """
        Find incidents within location + time proximity

        Uses PostGIS ST_DWithin for geographic distance
        """
        try:
            # Calculate time window
            occurred_at = new_incident['occurred_at']
            time_start = occurred_at - timedelta(minutes=self.time_threshold_minutes)
            time_end = occurred_at + timedelta(minutes=self.time_threshold_minutes)

            # Query using PostGIS
            query = """
                SELECT
                    id,
                    incident_type,
                    latitude,
                    longitude,
                    occurred_at,
                    reporter_count,
                    corroborating_reports,
                    video_ids,
                    user_id
                FROM incidents
                WHERE
                    ST_DWithin(
                        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
                        ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                        $3
                    )
                    AND occurred_at BETWEEN $4 AND $5
                    AND incident_type = $6
                    AND user_id != $7
                ORDER BY occurred_at DESC
                LIMIT 5
            """

            candidates = await db_connection.fetch(
                query,
                new_incident['longitude'],
                new_incident['latitude'],
                self.location_threshold_meters,
                time_start,
                time_end,
                new_incident.get('incident_type', 'unknown'),
                new_incident['user_id']
            )

            return [dict(c) for c in candidates]

        except Exception as e:
            logger.error(f"Failed to find candidates: {e}")
            return []

    async def _check_similarity(
        self,
        new_incident: Dict,
        candidate: Dict
    ) -> bool:
        """
        Check if incidents are visually/audibly similar using Atlas Intelligence

        If either incident lacks media, use location+time+category match only
        """
        # If no video available, rely on location+time+category
        if not new_incident.get('video_id') or not candidate.get('video_ids'):
            logger.info("No media available, using location+time+category match")
            return True  # Already matched on these criteria

        try:
            # Compare videos using Atlas Intelligence
            video1_id = new_incident['video_id']
            video2_ids = candidate['video_ids']

            if not video2_ids or len(video2_ids) == 0:
                return True

            # Compare with first video from candidate
            video2_id = video2_ids[0]

            similarity = await self._compare_media(video1_id, video2_id)

            logger.info(f"Media similarity: {similarity:.2f}")
            return similarity >= self.similarity_threshold

        except Exception as e:
            logger.error(f"Similarity check failed: {e}")
            # On error, be conservative - assume match based on location+time+category
            return True

    async def _compare_media(
        self,
        media_id_1: str,
        media_id_2: str
    ) -> float:
        """
        Compare two media files for visual/audio similarity

        Uses Atlas Intelligence object detection to compare what's in frame
        Returns similarity score 0.0-1.0
        """
        try:
            # Get analysis for both videos from Atlas
            # This is a placeholder - Atlas would need a comparison endpoint
            # For now, we'll use a simplified approach

            # In production, you would:
            # 1. Fetch both video analyses from database
            # 2. Compare detected objects (Jaccard similarity)
            # 3. Compare people count
            # 4. Compare weapon detections

            # Simplified placeholder:
            logger.info(f"Comparing media {media_id_1} with {media_id_2}")

            # TODO: Implement actual media comparison
            # For MVP, return high similarity if same category (already checked)
            return 0.8

        except Exception as e:
            logger.error(f"Media comparison failed: {e}")
            return 0.0

    async def _merge_incidents(
        self,
        new_incident: Dict,
        existing_incident: Dict,
        db_connection
    ) -> Dict:
        """
        Merge new incident into existing incident

        Updates:
        - reporter_count += 1
        - corroborating_reports array
        - video_ids array
        """
        try:
            # Prepare corroboration data
            corroboration = {
                'user_id': new_incident['user_id'],
                'reported_at': datetime.utcnow().isoformat(),
                'video_id': new_incident.get('video_id'),
                'confidence': new_incident.get('confidence', 1.0)
            }

            # Update existing incident
            query = """
                UPDATE incidents
                SET
                    reporter_count = reporter_count + 1,
                    corroborating_reports = corroborating_reports || $1::jsonb,
                    video_ids = CASE
                        WHEN $2 IS NOT NULL THEN array_append(video_ids, $2)
                        ELSE video_ids
                    END,
                    updated_at = NOW()
                WHERE id = $3
                RETURNING *
            """

            import json
            updated = await db_connection.fetchrow(
                query,
                json.dumps(corroboration),
                new_incident.get('video_id'),
                existing_incident['id']
            )

            logger.info(
                f"Merged incidents: {existing_incident['id']} now has "
                f"{updated['reporter_count']} reporters"
            )

            return dict(updated)

        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)
            raise

    async def resolve_conflict(
        self,
        reports: List[Dict]
    ) -> str:
        """
        Resolve conflicting incident classifications

        Strategy:
        1. Count votes for each category
        2. If >50% majority, use that
        3. Otherwise, use AI tiebreaker

        Args:
            reports: List of incident reports with categories

        Returns:
            Resolved category (polisen.se)
        """
        from collections import Counter

        # Count votes
        categories = [r.get('incident_type', 'Övrigt') for r in reports]
        vote_counts = Counter(categories)

        total_votes = len(reports)

        # Check for majority
        for category, count in vote_counts.most_common(1):
            if count / total_votes > 0.5:
                logger.info(f"Consensus reached: {category} ({count}/{total_votes})")
                return category

        # No majority - use AI tiebreaker
        logger.info("No consensus, using AI tiebreaker")

        try:
            # Combine all descriptions
            combined_description = " | ".join([
                r.get('description', '') for r in reports
            ])

            # Ask Atlas Intelligence to classify
            result = await self.atlas_client.classify_threat(
                combined_description,
                context={'conflicting_reports': len(reports)}
            )

            suggested_category = result.get('product_mappings', {}).get(
                'halo_incident_type',
                'Övrigt'
            )

            logger.info(f"AI tiebreaker: {suggested_category}")
            return suggested_category

        except Exception as e:
            logger.error(f"AI tiebreaker failed: {e}")
            # Fallback to most common category
            return vote_counts.most_common(1)[0][0]


# Singleton instance
_sensor_fusion_service = None


def get_sensor_fusion_service() -> SensorFusionService:
    """Get or create sensor fusion service singleton"""
    global _sensor_fusion_service
    if _sensor_fusion_service is None:
        _sensor_fusion_service = SensorFusionService()
    return _sensor_fusion_service
