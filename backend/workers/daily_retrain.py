"""
Daily Retraining Worker
Retrains prediction model daily using new incident data
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from backend.database.postgis_database import get_database
from backend.services.atlas_client import get_atlas_client

logger = logging.getLogger(__name__)


async def daily_retrain_job():
    """
    Run daily at 02:00 UTC

    Steps:
    1. Collect incidents from past 24 hours
    2. Extract features for training
    3. Send to Atlas Intelligence for retraining
    4. Regenerate predictions for all neighborhoods
    """
    logger.info("="*60)
    logger.info("DAILY RETRAINING JOB STARTED")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
    logger.info("="*60)

    try:
        db = get_database()
        atlas = get_atlas_client()

        # Step 1: Collect incidents from past 24 hours
        logger.info("Step 1: Collecting new incidents...")
        new_incidents = await collect_recent_incidents(db)
        logger.info(f"Collected {len(new_incidents)} new incidents")

        if len(new_incidents) == 0:
            logger.info("No new incidents to process. Skipping retraining.")
            return

        # Step 2: Extract features
        logger.info("Step 2: Extracting training features...")
        training_data = extract_features(new_incidents)
        logger.info(f"Extracted features from {len(training_data)} incidents")

        # Step 3: Send to Atlas for retraining
        logger.info("Step 3: Sending data to Atlas Intelligence for retraining...")
        retrain_result = await atlas.retrain_model(training_data)
        logger.info(f"Retraining result: {retrain_result.get('status')}")

        # Step 4: Regenerate predictions
        logger.info("Step 4: Regenerating predictions...")
        await regenerate_predictions(db)

        logger.info("="*60)
        logger.info("DAILY RETRAINING JOB COMPLETED SUCCESSFULLY")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Daily retraining job failed: {e}", exc_info=True)
        # TODO: Send alert to monitoring system
        raise


async def collect_recent_incidents(db) -> List[Dict]:
    """
    Collect incidents from past 24 hours for training

    Returns list of incidents with:
    - location (lat/lon)
    - incident_type (polisen category)
    - occurred_at (timestamp)
    - severity
    - reporter_count (for weighting)
    """
    try:
        query = """
            SELECT
                id,
                incident_type,
                latitude,
                longitude,
                occurred_at,
                severity,
                reporter_count,
                metadata
            FROM incidents
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            AND source != 'prediction'
            ORDER BY occurred_at DESC
        """

        incidents = await db.execute_query(query)
        return [dict(inc) for inc in incidents]

    except Exception as e:
        logger.error(f"Failed to collect incidents: {e}")
        return []


def extract_features(incidents: List[Dict]) -> List[Dict]:
    """
    Extract ML features from incidents for training

    Features:
    - location (lat, lon)
    - hour_of_day (0-23)
    - day_of_week (0-6)
    - incident_type (categorical)
    - severity (1-5)
    - corroboration_weight (based on reporter_count)
    """
    features = []

    for inc in incidents:
        occurred_at = inc['occurred_at']

        feature_dict = {
            'incident_id': str(inc['id']),
            'latitude': inc['latitude'],
            'longitude': inc['longitude'],
            'incident_type': inc['incident_type'],
            'severity': inc['severity'],
            'hour_of_day': occurred_at.hour,
            'day_of_week': occurred_at.weekday(),
            'timestamp': occurred_at.isoformat(),
            'corroboration_weight': inc.get('reporter_count', 1)
        }

        features.append(feature_dict)

    return features


async def regenerate_predictions(db):
    """
    Regenerate predictions for all neighborhoods using updated model

    This is expensive - only run after retraining
    """
    try:
        # Get all neighborhoods with historical incidents
        query = """
            SELECT DISTINCT
                location_name,
                latitude,
                longitude,
                COUNT(*) as incident_count
            FROM incidents
            WHERE location_name IS NOT NULL
            GROUP BY location_name, latitude, longitude
            HAVING COUNT(*) >= 3
            ORDER BY incident_count DESC
        """

        neighborhoods = await db.execute_query(query)
        logger.info(f"Regenerating predictions for {len(neighborhoods)} neighborhoods")

        # For each neighborhood, generate 24-hour predictions
        from backend.workers.generate_predictions import generate_predictions_for_location

        prediction_count = 0
        for neighborhood in neighborhoods:
            try:
                await generate_predictions_for_location(
                    db,
                    neighborhood['latitude'],
                    neighborhood['longitude'],
                    neighborhood['location_name']
                )
                prediction_count += 24  # 24 hours

            except Exception as e:
                logger.error(f"Failed to generate predictions for {neighborhood['location_name']}: {e}")
                continue

        logger.info(f"Generated {prediction_count} predictions")

    except Exception as e:
        logger.error(f"Failed to regenerate predictions: {e}")
        raise


# For testing/manual runs
async def main():
    """Run retraining job manually"""
    await daily_retrain_job()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
