"""
ML Monitoring Dashboard API
Provides endpoints for monitoring ML training and model performance
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from backend.monitoring.ml_training_monitor import ml_monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml-monitoring", tags=["ml_monitoring"])


class TrainingRunResponse(BaseModel):
    """Training run summary"""
    run_id: str
    model_type: str
    version: str
    trigger: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    training_duration_seconds: Optional[float]
    error_message: Optional[str]


class PerformanceTrendResponse(BaseModel):
    """Performance trend data point"""
    hour: datetime
    avg_accuracy: float
    avg_precision: float
    avg_recall: float
    avg_f1_score: float
    total_predictions: int


class DriftAlertResponse(BaseModel):
    """Data drift alert"""
    feature_name: str
    drift_score: float
    is_drifted: bool
    timestamp: datetime
    baseline_mean: float
    current_mean: float


class ModelHealthResponse(BaseModel):
    """Overall model health status"""
    model_type: str
    version: str
    status: str  # "healthy", "warning", "critical"
    current_accuracy: float
    accuracy_trend: str  # "improving", "stable", "declining"
    drift_alerts: int
    last_training: datetime
    total_predictions_24h: int
    avg_latency_ms: float


@router.get("/training/history", response_model=List[TrainingRunResponse])
async def get_training_history(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    limit: int = Query(10, ge=1, le=100, description="Number of runs to return")
):
    """
    Get recent training runs with metrics.

    Returns training history including:
    - Training duration
    - Performance metrics (accuracy, precision, recall, F1)
    - Training samples count
    - Status (completed, failed, in progress)
    """
    try:
        runs = await ml_monitor.get_training_history(
            model_type=model_type,
            limit=limit
        )

        return [
            TrainingRunResponse(
                run_id=run.run_id,
                model_type=run.model_type,
                version=run.version,
                trigger=run.trigger,
                started_at=run.started_at,
                completed_at=run.completed_at,
                status=run.status,
                accuracy=run.accuracy,
                precision=run.precision,
                recall=run.recall,
                f1_score=run.f1_score,
                training_duration_seconds=run.training_duration_seconds,
                error_message=run.error_message
            )
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/{run_id}")
async def get_training_details(run_id: str):
    """
    Get detailed information about a specific training run.

    Includes:
    - Full metrics (accuracy, precision, recall, F1, AUC-ROC)
    - Feature importance
    - Training configuration
    - Resource usage (memory, CPU)
    - Loss curves
    """
    try:
        if not ml_monitor.pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with ml_monitor.pool.acquire() as conn:
            run = await conn.fetchrow("""
                SELECT * FROM ml_training_runs WHERE run_id = $1
            """, run_id)

            if not run:
                raise HTTPException(status_code=404, detail="Training run not found")

            return {
                "run_id": run['run_id'],
                "model_type": run['model_type'],
                "version": run['version'],
                "trigger": run['trigger'],
                "started_at": run['started_at'].isoformat(),
                "completed_at": run['completed_at'].isoformat() if run['completed_at'] else None,
                "status": run['status'],
                "training_samples": run['training_samples'],
                "validation_samples": run['validation_samples'],
                "test_samples": run['test_samples'],
                "accuracy": run['accuracy'],
                "precision": run['precision'],
                "recall": run['recall'],
                "f1_score": run['f1_score'],
                "auc_roc": run['auc_roc'],
                "training_loss": run['training_loss'],
                "validation_loss": run['validation_loss'],
                "training_duration_seconds": run['training_duration_seconds'],
                "peak_memory_mb": run['peak_memory_mb'],
                "avg_cpu_percent": run['avg_cpu_percent'],
                "model_path": run['model_path'],
                "config": run['config'],
                "feature_importance": run['feature_importance'],
                "error_message": run['error_message']
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/trend", response_model=List[PerformanceTrendResponse])
async def get_performance_trend(
    model_type: str = Query(..., description="Model type"),
    days: int = Query(7, ge=1, le=30, description="Number of days")
):
    """
    Get model performance trend over time.

    Returns hourly aggregated metrics for the specified time period:
    - Average accuracy
    - Average precision/recall
    - Total predictions count
    """
    try:
        trend = await ml_monitor.get_model_performance_trend(
            model_type=model_type,
            days=days
        )

        return [
            PerformanceTrendResponse(
                hour=point['hour'],
                avg_accuracy=point['avg_accuracy'] or 0,
                avg_precision=point['avg_precision'] or 0,
                avg_recall=point['avg_recall'] or 0,
                avg_f1_score=point['avg_f1_score'] or 0,
                total_predictions=point['total_predictions'] or 0
            )
            for point in trend
        ]
    except Exception as e:
        logger.error(f"Failed to get performance trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/alerts", response_model=List[DriftAlertResponse])
async def get_drift_alerts(
    feature_name: Optional[str] = Query(None, description="Filter by feature"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours")
):
    """
    Get data drift alerts.

    Returns drift detection results for features:
    - Drift score (KL divergence)
    - Baseline vs current statistics
    - Alert status (drifted or not)
    """
    try:
        if not ml_monitor.pool:
            raise HTTPException(status_code=503, detail="Database not available")

        cutoff = datetime.now() - timedelta(hours=hours)

        async with ml_monitor.pool.acquire() as conn:
            query = """
                SELECT
                    feature_name,
                    drift_score,
                    is_drifted,
                    timestamp,
                    baseline_stats,
                    current_stats
                FROM ml_data_drift
                WHERE timestamp >= $1
                AND ($2::text IS NULL OR feature_name = $2)
                ORDER BY timestamp DESC
            """
            rows = await conn.fetch(query, cutoff, feature_name)

            return [
                DriftAlertResponse(
                    feature_name=row['feature_name'],
                    drift_score=row['drift_score'],
                    is_drifted=row['is_drifted'],
                    timestamp=row['timestamp'],
                    baseline_mean=row['baseline_stats'].get('mean', 0),
                    current_mean=row['current_stats'].get('mean', 0)
                )
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to get drift alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=ModelHealthResponse)
async def get_model_health(
    model_type: str = Query(..., description="Model type")
):
    """
    Get overall model health status.

    Provides a comprehensive health check:
    - Current performance metrics
    - Performance trend (improving/declining)
    - Active drift alerts
    - Recent training information
    - Prediction volume and latency
    """
    try:
        if not ml_monitor.pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with ml_monitor.pool.acquire() as conn:
            # Get latest training run
            latest_training = await conn.fetchrow("""
                SELECT version, completed_at, accuracy
                FROM ml_training_runs
                WHERE model_type = $1 AND status = 'completed'
                ORDER BY completed_at DESC NULLS LAST
                LIMIT 1
            """, model_type)

            if not latest_training:
                raise HTTPException(
                    status_code=404,
                    detail=f"No completed training runs found for {model_type}"
                )

            # Get recent performance metrics
            recent_performance = await conn.fetch("""
                SELECT accuracy
                FROM ml_performance_metrics
                WHERE model_type = $1
                ORDER BY timestamp DESC
                LIMIT 10
            """, model_type)

            # Get drift alerts count
            drift_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT feature_name)
                FROM ml_data_drift
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                AND is_drifted = true
            """)

            # Get prediction stats (last 24h)
            prediction_stats = await conn.fetchrow("""
                SELECT
                    SUM(total_predictions) as total,
                    AVG(avg_latency_ms) as avg_latency
                FROM ml_performance_metrics
                WHERE model_type = $1
                AND timestamp >= NOW() - INTERVAL '24 hours'
            """, model_type)

            # Calculate accuracy trend
            accuracies = [row['accuracy'] for row in recent_performance if row['accuracy']]
            accuracy_trend = "stable"
            if len(accuracies) >= 3:
                recent_avg = sum(accuracies[:3]) / 3
                older_avg = sum(accuracies[3:6]) / 3 if len(accuracies) >= 6 else recent_avg
                if recent_avg > older_avg + 0.02:
                    accuracy_trend = "improving"
                elif recent_avg < older_avg - 0.02:
                    accuracy_trend = "declining"

            # Determine health status
            current_accuracy = latest_training['accuracy'] or 0
            status = "healthy"
            if current_accuracy < 0.7 or drift_count > 3:
                status = "critical"
            elif current_accuracy < 0.8 or drift_count > 1 or accuracy_trend == "declining":
                status = "warning"

            return ModelHealthResponse(
                model_type=model_type,
                version=latest_training['version'],
                status=status,
                current_accuracy=current_accuracy,
                accuracy_trend=accuracy_trend,
                drift_alerts=drift_count or 0,
                last_training=latest_training['completed_at'],
                total_predictions_24h=prediction_stats['total'] or 0,
                avg_latency_ms=prediction_stats['avg_latency'] or 0
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summary of all ML metrics.

    Provides high-level overview:
    - Total training runs
    - Active models
    - Overall performance
    - Resource usage
    """
    try:
        if not ml_monitor.pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with ml_monitor.pool.acquire() as conn:
            # Training runs summary
            training_summary = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN status = 'completed' THEN accuracy END) as avg_accuracy,
                    AVG(CASE WHEN status = 'completed' THEN training_duration_seconds END) as avg_duration
                FROM ml_training_runs
                WHERE started_at >= NOW() - INTERVAL '30 days'
            """)

            # Model types
            model_types = await conn.fetch("""
                SELECT DISTINCT model_type, MAX(version) as latest_version
                FROM ml_training_runs
                WHERE status = 'completed'
                GROUP BY model_type
            """)

            # Predictions summary (last 24h)
            prediction_summary = await conn.fetchrow("""
                SELECT
                    SUM(total_predictions) as total_predictions,
                    AVG(avg_confidence) as avg_confidence,
                    AVG(avg_latency_ms) as avg_latency_ms
                FROM ml_performance_metrics
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            """)

            # Drift alerts
            drift_summary = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN is_drifted THEN 1 ELSE 0 END) as drifted_features
                FROM ml_data_drift
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            """)

            return {
                "training": {
                    "total_runs_30d": training_summary['total_runs'] or 0,
                    "completed": training_summary['completed'] or 0,
                    "failed": training_summary['failed'] or 0,
                    "avg_accuracy": round(training_summary['avg_accuracy'] or 0, 3),
                    "avg_duration_seconds": round(training_summary['avg_duration'] or 0, 1)
                },
                "models": [
                    {
                        "type": row['model_type'],
                        "latest_version": row['latest_version']
                    }
                    for row in model_types
                ],
                "predictions_24h": {
                    "total": prediction_summary['total_predictions'] or 0,
                    "avg_confidence": round(prediction_summary['avg_confidence'] or 0, 3),
                    "avg_latency_ms": round(prediction_summary['avg_latency_ms'] or 0, 1)
                },
                "drift_24h": {
                    "total_checks": drift_summary['total_checks'] or 0,
                    "drifted_features": drift_summary['drifted_features'] or 0
                },
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance/{run_id}")
async def get_feature_importance(run_id: str):
    """
    Get feature importance from a training run.

    Returns sorted list of features by importance score.
    """
    try:
        if not ml_monitor.pool:
            raise HTTPException(status_code=503, detail="Database not available")

        async with ml_monitor.pool.acquire() as conn:
            run = await conn.fetchrow("""
                SELECT feature_importance, model_type
                FROM ml_training_runs
                WHERE run_id = $1
            """, run_id)

            if not run:
                raise HTTPException(status_code=404, detail="Training run not found")

            if not run['feature_importance']:
                return {
                    "run_id": run_id,
                    "model_type": run['model_type'],
                    "features": []
                }

            # Sort by importance
            features = [
                {"name": name, "importance": score}
                for name, score in run['feature_importance'].items()
            ]
            features.sort(key=lambda x: x['importance'], reverse=True)

            return {
                "run_id": run_id,
                "model_type": run['model_type'],
                "features": features
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
