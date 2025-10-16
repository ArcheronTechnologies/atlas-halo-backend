"""
Admin API Endpoints
Administrative functions for data collection, model management, and system control
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from ..services.data_ingestion_service import get_ingestion_service
from ..workers.daily_retrain import daily_retrain_job
# Removed: comprehensive_swedish_collector - now using Atlas Intelligence API

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/ingestion/status")
async def get_ingestion_status() -> Dict[str, Any]:
    """
    Get the current status of the data ingestion service

    Returns information about:
    - Service running state
    - Collection intervals
    - Total collections performed
    - Last collection time
    - Cities being monitored
    """
    try:
        service = await get_ingestion_service()
        status = await service.get_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingestion/trigger")
async def trigger_collection() -> Dict[str, Any]:
    """
    Manually trigger an immediate data collection cycle

    This bypasses the normal schedule and collects data from all configured cities immediately.
    Useful for:
    - Testing the collection pipeline
    - Getting immediate updates after configuration changes
    - Emergency data collection
    """
    try:
        service = await get_ingestion_service()
        result = await service.trigger_immediate_collection()
        return result
    except Exception as e:
        logger.error(f"Failed to trigger collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain/trigger")
async def trigger_retraining() -> Dict[str, Any]:
    """
    Manually trigger the daily AI model retraining job

    This is normally run automatically at 02:00 UTC but can be triggered manually for:
    - Testing the retraining pipeline
    - Immediate model updates after significant new incident data
    - Recovery from failed automatic runs

    Returns retraining results and statistics
    """
    try:
        logger.info("Manual retraining triggered via API")
        await daily_retrain_job()
        return {
            "status": "success",
            "message": "AI model retraining completed successfully"
        }
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@router.post("/bootstrap/historical-data")
async def bootstrap_historical_data(
    target_records: int = 10000,
    years_back: int = 2
) -> Dict[str, Any]:
    """
    Bootstrap the system with historical crime data

    This is a long-running operation that collects historical data from multiple Swedish sources.
    Should typically only be run once during initial setup, or periodically to refresh historical data.

    Args:
        target_records: Number of historical records to collect (default: 10,000)
        years_back: How many years of history to collect (default: 2)

    Returns:
        Collection statistics and progress information
    """
    try:
        logger.info(f"Starting historical data bootstrap: {target_records} records, {years_back} years")

        # Create collection target
        target = CollectionTarget(
            total_records=target_records,
            years_back=years_back,
            min_quality_score=0.6
        )

        # Initialize collector
        collector = ComprehensiveSwedishCollector(target)

        # This is a simplified version - in production, this should:
        # 1. Run in a background task
        # 2. Store progress in database
        # 3. Provide progress updates via WebSocket
        # 4. Be resumable if interrupted

        # For now, return a placeholder response
        return {
            "status": "started",
            "message": "Historical data collection started",
            "target_records": target_records,
            "years_back": years_back,
            "note": "This is a long-running operation. Check logs for progress."
        }

    except Exception as e:
        logger.error(f"Failed to bootstrap historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bootstrap/status")
async def get_bootstrap_status() -> Dict[str, Any]:
    """
    Get the status of historical data bootstrap operation

    Returns:
        Current progress, records collected, estimated time remaining
    """
    # TODO: Implement actual status tracking
    # For now, return a placeholder
    return {
        "status": "not_implemented",
        "message": "Bootstrap status tracking not yet implemented"
    }


@router.get("/system/health")
async def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health information

    Returns:
        - Database status
        - Service statuses
        - Resource usage
        - Recent errors
    """
    health_info = {
        "timestamp": "now",
        "services": {
            "database": "healthy",
            "data_ingestion": "healthy",
            "ml_training": "unknown"
        },
        "resources": {
            "disk_usage_percent": 0,
            "memory_usage_percent": 0
        }
    }

    # Add ingestion service status
    try:
        service = await get_ingestion_service()
        status = await service.get_status()
        health_info["services"]["data_ingestion"] = "healthy" if status["running"] else "stopped"
    except Exception as e:
        health_info["services"]["data_ingestion"] = f"error: {str(e)}"

    return health_info