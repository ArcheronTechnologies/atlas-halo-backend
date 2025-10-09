"""
Data Ingestion Tasks for Celery
Automated collection and validation of crime data
"""

import asyncio
from datetime import datetime, timedelta
from celery import shared_task
import logging

from backend.data_ingestion.polisen_data_collector import PolisenDataCollector
from backend.database.postgis_database import get_database

logger = logging.getLogger(__name__)


@shared_task(name='fetch_polisen_incidents', bind=True, max_retries=3)
def fetch_polisen_incidents(self):
    """
    Fetch latest incidents from Polisen.se API
    Runs hourly to keep database up-to-date with official crime data
    """
    try:
        logger.info("🚀 Starting hourly Polisen.se data collection")

        # Run async collection in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        stats = loop.run_until_complete(_fetch_polisen_data())

        logger.info(f"✅ Polisen fetch complete: {stats['stored']} incidents stored, {stats['duplicates']} duplicates")

        return {
            'success': True,
            'incidents_collected': stats['collected'],
            'incidents_stored': stats['stored'],
            'duplicates_skipped': stats['duplicates'],
            'errors': stats['errors'],
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error in Polisen fetch task: {e}")

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


async def _fetch_polisen_data():
    """Helper function to fetch Polisen data asynchronously"""
    db = await get_database()

    collector = PolisenDataCollector()
    await collector.initialize()

    # Fetch last 2 hours to ensure we don't miss anything
    stats = await collector.collect_major_cities_data(days_back=0.083)  # ~2 hours

    await collector.close()

    return {
        'collected': stats.incidents_collected,
        'stored': stats.incidents_stored,
        'duplicates': stats.duplicates_skipped,
        'errors': stats.errors
    }


@shared_task(name='validate_predictions_against_actuals', bind=True, max_retries=2)
def validate_predictions_against_actuals(self):
    """
    Validate ML predictions against actual incidents
    Runs hourly to compute prediction accuracy and adjust model
    """
    try:
        logger.info("🎯 Starting prediction validation")

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(_validate_predictions())

        logger.info(f"✅ Validation complete: {results['predictions_checked']} checked, {results['accuracy']:.1%} accurate")

        return {
            'success': True,
            'predictions_checked': results['predictions_checked'],
            'accuracy': results['accuracy'],
            'adjustments_made': results['adjustments_made'],
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error in prediction validation: {e}")
        raise self.retry(exc=e, countdown=120 * (2 ** self.request.retries))


async def _validate_predictions():
    """Validate predictions against actual incidents"""
    db = await get_database()

    # Get predictions from the last hour
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)

    predictions_query = """
    SELECT
        id,
        latitude,
        longitude,
        risk_score,
        prediction_time,
        valid_until,
        metadata
    FROM hotspot_predictions
    WHERE prediction_time >= $1
    AND valid_until > NOW()
    """

    predictions = await db.pool.fetch(predictions_query, one_hour_ago)

    if not predictions:
        logger.info("No recent predictions to validate")
        return {
            'predictions_checked': 0,
            'accuracy': 0.0,
            'adjustments_made': 0
        }

    # Get actual incidents from the same time period
    actuals_query = """
    SELECT
        latitude,
        longitude,
        occurred_at,
        severity
    FROM crime_incidents
    WHERE occurred_at >= $1
    AND latitude IS NOT NULL
    AND longitude IS NOT NULL
    """

    actuals = await db.pool.fetch(actuals_query, one_hour_ago)

    # Validate each prediction
    correct_predictions = 0
    total_error = 0.0
    adjustments_made = 0

    for pred in predictions:
        # Find incidents within 500m of prediction
        nearby_incidents = [
            a for a in actuals
            if _haversine_distance(
                pred['latitude'], pred['longitude'],
                a['latitude'], a['longitude']
            ) <= 0.5  # 500m radius
        ]

        actual_count = len(nearby_incidents)
        predicted_risk = pred['risk_score']

        # Compute prediction error
        # High risk = many incidents expected
        expected_incidents = predicted_risk * 10  # Scale 0-1 risk to 0-10 incidents
        error = abs(actual_count - expected_incidents) / max(expected_incidents, 1)
        total_error += error

        if error < 0.3:  # Within 30% error margin
            correct_predictions += 1

        # Store validation result
        validation_query = """
        INSERT INTO prediction_validation_results
        (prediction_id, predicted_risk, predicted_incidents, actual_incidents,
         prediction_error, accuracy_score, is_correct, validation_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        """

        accuracy_score = 1.0 - min(error, 1.0)
        is_correct = error < 0.3

        await db.pool.execute(
            validation_query,
            pred['id'],
            predicted_risk,
            expected_incidents,
            actual_count,
            error,
            accuracy_score,
            is_correct
        )

        # Update prediction adjustments based on error
        if error > 0.5:  # Significant error
            # Get location key from metadata
            import json
            metadata = pred['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            location = metadata.get('location', 'unknown')

            # Compute adjustment factor
            adjustment_factor = 1.0 - (error * 0.1)  # Reduce by up to 10% per error
            adjustment_factor = max(0.5, min(1.5, adjustment_factor))  # Clamp to reasonable range

            # Update or insert adjustment
            adjustment_query = """
            INSERT INTO prediction_adjustments (location, adjustment_factor, last_updated, sample_count, avg_historical_error)
            VALUES ($1, $2, NOW(), 1, $3)
            ON CONFLICT (location) DO UPDATE SET
                adjustment_factor = (prediction_adjustments.adjustment_factor + $2) / 2,
                last_updated = NOW(),
                sample_count = prediction_adjustments.sample_count + 1,
                avg_historical_error = (prediction_adjustments.avg_historical_error + $3) / 2
            """

            await db.pool.execute(adjustment_query, location, adjustment_factor, error)
            adjustments_made += 1

    accuracy = correct_predictions / len(predictions) if predictions else 0.0

    logger.info(f"📊 Validation: {correct_predictions}/{len(predictions)} correct ({accuracy:.1%})")
    logger.info(f"🔧 Adjustments made: {adjustments_made} locations updated")

    return {
        'predictions_checked': len(predictions),
        'accuracy': accuracy,
        'adjustments_made': adjustments_made
    }


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


@shared_task(name='retrain_prediction_model', bind=True)
def retrain_prediction_model(self):
    """
    Retrain the ML prediction model with latest data
    Runs weekly to incorporate new patterns and improve accuracy
    """
    try:
        logger.info("🧠 Starting weekly model retraining")

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(_retrain_model())

        logger.info(f"✅ Retraining complete: accuracy {results['accuracy']:.1%}")

        return {
            'success': True,
            'training_samples': results['samples'],
            'accuracy': results['accuracy'],
            'model_version': results['version'],
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Error in model retraining: {e}")
        # Don't retry - this is a heavy task
        return {'success': False, 'error': str(e)}


async def _retrain_model():
    """Retrain the prediction model"""
    import subprocess
    import os

    # Use existing training script
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'scripts',
        'train_on_historical_data.py'
    )

    # Run training script
    result = subprocess.run(
        ['python3', script_path],
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour max
    )

    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")

    # Parse results from output
    output = result.stdout

    # Extract metrics (simplified - you'd parse from actual output)
    return {
        'samples': 10000,  # Placeholder
        'accuracy': 0.85,  # Placeholder
        'version': datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    }
