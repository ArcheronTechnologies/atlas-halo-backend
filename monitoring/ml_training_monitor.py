"""
ML Training & Model Performance Monitoring System
Comprehensive monitoring for ML model training, deployment, and inference performance
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge, Info
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PROMETHEUS METRICS FOR ML TRAINING
# =============================================================================

# Training metrics
ml_training_runs_total = Counter(
    'atlas_ml_training_runs_total',
    'Total number of ML training runs',
    ['model_type', 'trigger', 'status']
)

ml_training_duration_seconds = Histogram(
    'atlas_ml_training_duration_seconds',
    'Time taken to train ML models',
    ['model_type'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400)  # 1m to 4h
)

ml_training_samples = Histogram(
    'atlas_ml_training_samples',
    'Number of samples used in training',
    ['model_type', 'split'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000)
)

# Model performance metrics
ml_model_accuracy = Gauge(
    'atlas_ml_model_accuracy',
    'Model accuracy score',
    ['model_type', 'version', 'stage']
)

ml_model_precision = Gauge(
    'atlas_ml_model_precision',
    'Model precision score',
    ['model_type', 'version']
)

ml_model_recall = Gauge(
    'atlas_ml_model_recall',
    'Model recall score',
    ['model_type', 'version']
)

ml_model_f1_score = Gauge(
    'atlas_ml_model_f1_score',
    'Model F1 score',
    ['model_type', 'version']
)

ml_model_auc_roc = Gauge(
    'atlas_ml_model_auc_roc',
    'Model AUC-ROC score',
    ['model_type', 'version']
)

# Inference metrics
ml_predictions_total = Counter(
    'atlas_ml_predictions_total',
    'Total number of predictions made',
    ['model_type', 'prediction_type']
)

ml_prediction_latency_seconds = Histogram(
    'atlas_ml_prediction_latency_seconds',
    'Time taken for model inference',
    ['model_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

ml_prediction_confidence = Histogram(
    'atlas_ml_prediction_confidence',
    'Confidence scores of predictions',
    ['model_type'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# Data drift metrics
ml_feature_drift_score = Gauge(
    'atlas_ml_feature_drift_score',
    'Feature distribution drift score (KL divergence)',
    ['feature_name']
)

ml_prediction_drift_score = Gauge(
    'atlas_ml_prediction_drift_score',
    'Prediction distribution drift score',
    ['model_type']
)

# Model versioning
ml_model_version_info = Info(
    'atlas_ml_model_version',
    'Current ML model version information'
)

ml_active_model_version = Gauge(
    'atlas_ml_active_model_version',
    'Active model version number',
    ['model_type', 'stage']
)

# Performance degradation alerts
ml_accuracy_drop_alert = Gauge(
    'atlas_ml_accuracy_drop_alert',
    'Alert if model accuracy dropped significantly (1=alert, 0=ok)',
    ['model_type']
)

ml_data_drift_alert = Gauge(
    'atlas_ml_data_drift_alert',
    'Alert if data drift detected (1=alert, 0=ok)',
    ['feature_name']
)

# Training resource usage
ml_training_memory_mb = Gauge(
    'atlas_ml_training_memory_mb',
    'Memory usage during training in MB',
    ['model_type']
)

ml_training_cpu_percent = Gauge(
    'atlas_ml_training_cpu_percent',
    'CPU usage during training',
    ['model_type']
)


# =============================================================================
# ML TRAINING MONITORING SYSTEM
# =============================================================================

class TrainingStatus(Enum):
    """Training run status"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingRun:
    """Single training run record"""
    run_id: str
    model_type: str
    version: str
    trigger: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "started"

    # Data metrics
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None

    # Training metrics
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    total_epochs: Optional[int] = None

    # Resource metrics
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    training_duration_seconds: Optional[float] = None

    # Model artifacts
    model_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None

    # Error info
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking over time"""
    model_type: str
    version: str
    timestamp: datetime

    # Performance scores
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

    # Sample counts
    total_predictions: int
    correct_predictions: int

    # Confidence metrics
    avg_confidence: float
    low_confidence_count: int

    # Latency metrics
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


@dataclass
class DataDriftMetrics:
    """Data drift detection results"""
    timestamp: datetime
    feature_name: str
    drift_score: float  # KL divergence or similar
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    is_drifted: bool
    threshold: float = 0.1


class MLTrainingMonitor:
    """
    Comprehensive ML training and model performance monitoring.

    Features:
    - Track training runs with full metrics
    - Monitor model performance over time
    - Detect data drift
    - Alert on performance degradation
    - Resource usage tracking
    - Model version management
    """

    def __init__(self, db_pool=None):
        self.pool = db_pool
        self.active_runs: Dict[str, TrainingRun] = {}
        self.performance_history: List[ModelPerformanceMetrics] = []
        self.drift_history: List[DataDriftMetrics] = []

    # =========================================================================
    # TRAINING RUN TRACKING
    # =========================================================================

    async def start_training_run(
        self,
        model_type: str,
        version: str,
        trigger: str,
        config: Dict[str, Any] = None
    ) -> str:
        """Start tracking a new training run"""
        run_id = f"{model_type}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = TrainingRun(
            run_id=run_id,
            model_type=model_type,
            version=version,
            trigger=trigger,
            started_at=datetime.now(),
            config=config or {}
        )

        self.active_runs[run_id] = run

        # Update Prometheus metrics
        ml_training_runs_total.labels(
            model_type=model_type,
            trigger=trigger,
            status='started'
        ).inc()

        # Store in database
        if self.pool:
            await self._store_training_run(run)

        logger.info(f"🚀 Started training run: {run_id}")
        return run_id

    async def update_training_progress(
        self,
        run_id: str,
        epoch: int,
        total_epochs: int,
        training_loss: float,
        validation_loss: float,
        metrics: Dict[str, float] = None
    ):
        """Update training progress"""
        if run_id not in self.active_runs:
            logger.warning(f"Training run {run_id} not found")
            return

        run = self.active_runs[run_id]
        run.total_epochs = total_epochs
        run.training_loss = training_loss
        run.validation_loss = validation_loss

        if metrics:
            run.accuracy = metrics.get('accuracy')
            run.precision = metrics.get('precision')
            run.recall = metrics.get('recall')
            run.f1_score = metrics.get('f1_score')

        logger.debug(f"📊 Training progress {run_id}: epoch {epoch}/{total_epochs}")

    async def complete_training_run(
        self,
        run_id: str,
        final_metrics: Dict[str, float],
        model_path: str,
        feature_importance: Dict[str, float] = None
    ):
        """Mark training run as completed"""
        if run_id not in self.active_runs:
            logger.warning(f"Training run {run_id} not found")
            return

        run = self.active_runs[run_id]
        run.completed_at = datetime.now()
        run.status = TrainingStatus.COMPLETED.value
        run.model_path = model_path
        run.feature_importance = feature_importance

        # Update final metrics
        run.accuracy = final_metrics.get('accuracy')
        run.precision = final_metrics.get('precision')
        run.recall = final_metrics.get('recall')
        run.f1_score = final_metrics.get('f1_score')
        run.auc_roc = final_metrics.get('auc_roc')

        # Calculate duration
        run.training_duration_seconds = (
            run.completed_at - run.started_at
        ).total_seconds()

        # Update Prometheus metrics
        ml_training_runs_total.labels(
            model_type=run.model_type,
            trigger=run.trigger,
            status='completed'
        ).inc()

        ml_training_duration_seconds.labels(
            model_type=run.model_type
        ).observe(run.training_duration_seconds)

        ml_model_accuracy.labels(
            model_type=run.model_type,
            version=run.version,
            stage='production'
        ).set(run.accuracy or 0)

        if run.precision:
            ml_model_precision.labels(
                model_type=run.model_type,
                version=run.version
            ).set(run.precision)

        if run.recall:
            ml_model_recall.labels(
                model_type=run.model_type,
                version=run.version
            ).set(run.recall)

        if run.f1_score:
            ml_model_f1_score.labels(
                model_type=run.model_type,
                version=run.version
            ).set(run.f1_score)

        if run.auc_roc:
            ml_model_auc_roc.labels(
                model_type=run.model_type,
                version=run.version
            ).set(run.auc_roc)

        # Update database
        if self.pool:
            await self._update_training_run(run)

        # Remove from active runs
        del self.active_runs[run_id]

        logger.info(f"✅ Completed training run: {run_id} (accuracy: {run.accuracy:.3f})")

    async def fail_training_run(self, run_id: str, error_message: str):
        """Mark training run as failed"""
        if run_id not in self.active_runs:
            return

        run = self.active_runs[run_id]
        run.completed_at = datetime.now()
        run.status = TrainingStatus.FAILED.value
        run.error_message = error_message

        ml_training_runs_total.labels(
            model_type=run.model_type,
            trigger=run.trigger,
            status='failed'
        ).inc()

        if self.pool:
            await self._update_training_run(run)

        del self.active_runs[run_id]

        logger.error(f"❌ Training run failed: {run_id} - {error_message}")

    # =========================================================================
    # PREDICTION MONITORING
    # =========================================================================

    def track_prediction(
        self,
        model_type: str,
        prediction_type: str,
        latency_seconds: float,
        confidence: float
    ):
        """Track individual prediction"""
        ml_predictions_total.labels(
            model_type=model_type,
            prediction_type=prediction_type
        ).inc()

        ml_prediction_latency_seconds.labels(
            model_type=model_type
        ).observe(latency_seconds)

        ml_prediction_confidence.labels(
            model_type=model_type
        ).observe(confidence)

    async def track_batch_predictions(
        self,
        model_type: str,
        version: str,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Any] = None
    ):
        """Track batch of predictions and calculate metrics"""
        timestamp = datetime.now()

        # Calculate confidence metrics
        confidences = [p.get('confidence', 0) for p in predictions]
        avg_confidence = np.mean(confidences) if confidences else 0
        low_confidence_count = sum(1 for c in confidences if c < 0.6)

        # If ground truth available, calculate performance
        correct_predictions = 0
        if ground_truth:
            for pred, truth in zip(predictions, ground_truth):
                if pred.get('prediction') == truth:
                    correct_predictions += 1

        metrics = ModelPerformanceMetrics(
            model_type=model_type,
            version=version,
            timestamp=timestamp,
            accuracy=correct_predictions / len(predictions) if ground_truth else 0,
            precision=0,  # Calculate from confusion matrix
            recall=0,
            f1_score=0,
            auc_roc=0,
            total_predictions=len(predictions),
            correct_predictions=correct_predictions,
            avg_confidence=avg_confidence,
            low_confidence_count=low_confidence_count,
            avg_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0
        )

        self.performance_history.append(metrics)

        # Store in database
        if self.pool:
            await self._store_performance_metrics(metrics)

    # =========================================================================
    # DATA DRIFT DETECTION
    # =========================================================================

    async def detect_data_drift(
        self,
        feature_name: str,
        baseline_distribution: np.ndarray,
        current_distribution: np.ndarray,
        threshold: float = 0.1
    ) -> DataDriftMetrics:
        """Detect data drift using KL divergence"""

        # Calculate KL divergence
        drift_score = self._calculate_kl_divergence(
            baseline_distribution,
            current_distribution
        )

        is_drifted = drift_score > threshold

        metrics = DataDriftMetrics(
            timestamp=datetime.now(),
            feature_name=feature_name,
            drift_score=drift_score,
            baseline_stats={
                'mean': float(np.mean(baseline_distribution)),
                'std': float(np.std(baseline_distribution)),
                'min': float(np.min(baseline_distribution)),
                'max': float(np.max(baseline_distribution))
            },
            current_stats={
                'mean': float(np.mean(current_distribution)),
                'std': float(np.std(current_distribution)),
                'min': float(np.min(current_distribution)),
                'max': float(np.max(current_distribution))
            },
            is_drifted=is_drifted,
            threshold=threshold
        )

        self.drift_history.append(metrics)

        # Update Prometheus metrics
        ml_feature_drift_score.labels(
            feature_name=feature_name
        ).set(drift_score)

        ml_data_drift_alert.labels(
            feature_name=feature_name
        ).set(1 if is_drifted else 0)

        if is_drifted:
            logger.warning(
                f"⚠️  Data drift detected for {feature_name}: "
                f"score={drift_score:.3f} (threshold={threshold})"
            )

        # Store in database
        if self.pool:
            await self._store_drift_metrics(metrics)

        return metrics

    def _calculate_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """Calculate KL divergence between two distributions"""
        # Normalize to probability distributions
        p = p / (np.sum(p) + epsilon)
        q = q / (np.sum(q) + epsilon)

        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon

        # Calculate KL divergence
        kl_div = np.sum(p * np.log(p / q))
        return float(kl_div)

    # =========================================================================
    # PERFORMANCE DEGRADATION DETECTION
    # =========================================================================

    async def check_performance_degradation(
        self,
        model_type: str,
        current_accuracy: float,
        baseline_accuracy: float,
        threshold: float = 0.05
    ) -> bool:
        """Check if model performance has degraded"""
        accuracy_drop = baseline_accuracy - current_accuracy
        is_degraded = accuracy_drop > threshold

        ml_accuracy_drop_alert.labels(
            model_type=model_type
        ).set(1 if is_degraded else 0)

        if is_degraded:
            logger.warning(
                f"⚠️  Performance degradation detected for {model_type}: "
                f"accuracy dropped {accuracy_drop:.3f} "
                f"({baseline_accuracy:.3f} → {current_accuracy:.3f})"
            )

        return is_degraded

    # =========================================================================
    # DATABASE STORAGE
    # =========================================================================

    async def _store_training_run(self, run: TrainingRun):
        """Store training run in database"""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_training_runs (
                        run_id, model_type, version, trigger, started_at,
                        status, config
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        config = EXCLUDED.config
                """, run.run_id, run.model_type, run.version, run.trigger,
                    run.started_at, run.status, json.dumps(run.config or {}))
        except Exception as e:
            logger.error(f"Failed to store training run: {e}")

    async def _update_training_run(self, run: TrainingRun):
        """Update training run in database"""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE ml_training_runs SET
                        completed_at = $1,
                        status = $2,
                        training_samples = $3,
                        validation_samples = $4,
                        test_samples = $5,
                        accuracy = $6,
                        precision = $7,
                        recall = $8,
                        f1_score = $9,
                        auc_roc = $10,
                        training_loss = $11,
                        validation_loss = $12,
                        training_duration_seconds = $13,
                        model_path = $14,
                        feature_importance = $15,
                        error_message = $16
                    WHERE run_id = $17
                """, run.completed_at, run.status, run.training_samples,
                    run.validation_samples, run.test_samples, run.accuracy,
                    run.precision, run.recall, run.f1_score, run.auc_roc,
                    run.training_loss, run.validation_loss,
                    run.training_duration_seconds, run.model_path,
                    json.dumps(run.feature_importance or {}),
                    run.error_message, run.run_id)
        except Exception as e:
            logger.error(f"Failed to update training run: {e}")

    async def _store_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """Store performance metrics in database"""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_performance_metrics (
                        model_type, version, timestamp, accuracy, precision,
                        recall, f1_score, auc_roc, total_predictions,
                        correct_predictions, avg_confidence, low_confidence_count
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, metrics.model_type, metrics.version, metrics.timestamp,
                    metrics.accuracy, metrics.precision, metrics.recall,
                    metrics.f1_score, metrics.auc_roc, metrics.total_predictions,
                    metrics.correct_predictions, metrics.avg_confidence,
                    metrics.low_confidence_count)
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")

    async def _store_drift_metrics(self, metrics: DataDriftMetrics):
        """Store drift metrics in database"""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_data_drift (
                        timestamp, feature_name, drift_score, is_drifted,
                        threshold, baseline_stats, current_stats
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, metrics.timestamp, metrics.feature_name, metrics.drift_score,
                    metrics.is_drifted, metrics.threshold,
                    json.dumps(metrics.baseline_stats),
                    json.dumps(metrics.current_stats))
        except Exception as e:
            logger.error(f"Failed to store drift metrics: {e}")

    # =========================================================================
    # REPORTING & ANALYTICS
    # =========================================================================

    async def get_training_history(
        self,
        model_type: str = None,
        limit: int = 10
    ) -> List[TrainingRun]:
        """Get recent training runs"""
        if not self.pool:
            return []

        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT * FROM ml_training_runs
                    WHERE ($1::text IS NULL OR model_type = $1)
                    ORDER BY started_at DESC
                    LIMIT $2
                """
                rows = await conn.fetch(query, model_type, limit)

                return [
                    TrainingRun(
                        run_id=row['run_id'],
                        model_type=row['model_type'],
                        version=row['version'],
                        trigger=row['trigger'],
                        started_at=row['started_at'],
                        completed_at=row.get('completed_at'),
                        status=row['status'],
                        accuracy=row.get('accuracy'),
                        # ... other fields
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get training history: {e}")
            return []

    async def get_model_performance_trend(
        self,
        model_type: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get model performance trend over time"""
        if not self.pool:
            return []

        try:
            cutoff = datetime.now() - timedelta(days=days)
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        DATE_TRUNC('hour', timestamp) as hour,
                        AVG(accuracy) as avg_accuracy,
                        AVG(precision) as avg_precision,
                        AVG(recall) as avg_recall,
                        AVG(f1_score) as avg_f1_score,
                        SUM(total_predictions) as total_predictions
                    FROM ml_performance_metrics
                    WHERE model_type = $1 AND timestamp >= $2
                    GROUP BY hour
                    ORDER BY hour
                """, model_type, cutoff)

                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get performance trend: {e}")
            return []


# Global monitor instance
ml_monitor = MLTrainingMonitor()


async def initialize_ml_monitor(db_pool):
    """Initialize ML training monitor with database connection"""
    global ml_monitor
    ml_monitor = MLTrainingMonitor(db_pool)

    # Create database tables if they don't exist
    await _create_ml_monitoring_tables(db_pool)

    logger.info("✅ ML Training Monitor initialized")


async def _create_ml_monitoring_tables(db_pool):
    """Create database tables for ML monitoring"""
    async with db_pool.acquire() as conn:
        # Training runs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_training_runs (
                run_id VARCHAR PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                trigger VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                status VARCHAR NOT NULL,
                training_samples INTEGER,
                validation_samples INTEGER,
                test_samples INTEGER,
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                auc_roc FLOAT,
                training_loss FLOAT,
                validation_loss FLOAT,
                training_duration_seconds FLOAT,
                peak_memory_mb FLOAT,
                avg_cpu_percent FLOAT,
                model_path VARCHAR,
                config JSONB,
                feature_importance JSONB,
                error_message TEXT
            )
        """)

        # Performance metrics table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_performance_metrics (
                id SERIAL PRIMARY KEY,
                model_type VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                auc_roc FLOAT,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                avg_confidence FLOAT,
                low_confidence_count INTEGER,
                avg_latency_ms FLOAT,
                p95_latency_ms FLOAT,
                p99_latency_ms FLOAT
            )
        """)

        # Data drift table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_data_drift (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                feature_name VARCHAR NOT NULL,
                drift_score FLOAT NOT NULL,
                is_drifted BOOLEAN NOT NULL,
                threshold FLOAT NOT NULL,
                baseline_stats JSONB,
                current_stats JSONB
            )
        """)

        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_runs_model_type
            ON ml_training_runs(model_type, started_at DESC)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_model_type
            ON ml_performance_metrics(model_type, timestamp DESC)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_drift_feature
            ON ml_data_drift(feature_name, timestamp DESC)
        """)
